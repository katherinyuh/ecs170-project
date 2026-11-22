'''
Concrete MethodModule class for Stage 4 text classification (IMDB sentiment) with a
recurrent network. One class covers all three cell types asked for in step 4-5:

    cell_type = 'rnn'   plain (Elman) RNN
    cell_type = 'lstm'  LSTM
    cell_type = 'gru'   GRU

Model: word IDs -> Embedding -> RNN/LSTM/GRU over the words of the review ->
final hidden state -> Dropout -> Linear -> 2 scores (negative, positive).
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD
# Input comes from Part A's Dataset_Loader_Classification: X is (N, max_len) word IDs with
# 0 = <pad> placed BEFORE the words ("pre" padding). Inside the model the padding is moved
# to the end and the sequences are packed, so the network only reads real words and the
# final hidden state comes right after each review's last word (in both directions when
# bidirectional). That matters most for the plain RNN, which otherwise has to carry its
# memory across a long run of padding.

from local_code.base_class.method import method
import random
import time
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

PAD_ID = 0


def pick_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def pre_to_post_padding(X):
    '''Move each row's padding (zeros) from the start to the end. Returns (X_post, lengths).'''
    X = np.asarray(X)
    lengths = (X != PAD_ID).sum(axis=1)
    out = np.zeros_like(X)
    for i, n in enumerate(lengths):
        if n:
            row = X[i]
            out[i, :n] = row[row != PAD_ID]
    return out, np.maximum(lengths, 1)


class Method_RNN_Classification(method, nn.Module):
    data = None

    # ---- configurable settings (baseline defaults) ----
    cell_type = 'lstm'          # 'rnn', 'lstm' or 'gru'
    embedding_dim = 100
    hidden_size = 128
    num_layers = 1
    bidirectional = False
    dropout = 0.3               # after the embedding and before the output layer
    rnn_nonlinearity = 'tanh'   # plain RNN only: 'tanh' or 'relu'
    # pretrained GloVe vectors: set embedding_matrix to a (vocab_size, dim) array before run()
    # (see Glove_Utils.load_glove_matrix). None = learn the embedding from scratch.
    embedding_matrix = None
    freeze_embeddings = False   # True keeps the GloVe vectors fixed; False fine-tunes them
    max_epoch = 10
    learning_rate = 1e-3
    weight_decay = 0.0
    optimizer_name = 'adam'     # 'adam', 'sgd' or 'rmsprop'
    batch_size = 64
    grad_clip = 5.0             # caps exploding gradients, which plain RNNs are prone to
    eval_batch_size = 1000
    track_test_acc = True       # test accuracy per epoch, for the learning curves only
    train_subset = None         # use only the first N training reviews (quick checks)
    print_every = 1
    seed = 2
    device_name = 'auto'        # 'auto' picks cuda / mps / cpu

    # set by run() from the data
    vocab_size = None
    num_classes = 2

    def __init__(self, mName, mDescription, config=None):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        if config:
            for key, value in config.items():
                if not hasattr(self, key):
                    raise AttributeError(f'Unknown RNN config key: {key}')
                setattr(self, key, value)
        if self.cell_type not in ('rnn', 'lstm', 'gru'):
            raise ValueError(f"cell_type must be 'rnn', 'lstm' or 'gru', got {self.cell_type}")
        self.device = pick_device() if self.device_name == 'auto' else torch.device(self.device_name)
        self.embedding = None
        self.rnn = None
        self.classifier = None
        self.loss_history = []
        self.train_acc_history = []
        self.test_acc_history = []
        self.epoch_times = []

    def set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def build_model(self, vocab_size, num_classes=2):
        self.set_seed()
        self.vocab_size = int(vocab_size)
        self.num_classes = int(num_classes)
        if self.embedding_matrix is not None:
            # torch.tensor COPIES the array, so fine-tuning this model never changes the shared
            # GloVe matrix that the next model (LSTM / GRU) starts from
            weights = torch.tensor(np.asarray(self.embedding_matrix), dtype=torch.float32)
            if weights.shape[0] != self.vocab_size:
                raise ValueError(f'embedding_matrix has {weights.shape[0]} rows, vocab has {self.vocab_size}')
            self.embedding_dim = int(weights.shape[1])
            self.embedding = nn.Embedding.from_pretrained(weights, freeze=self.freeze_embeddings,
                                                          padding_idx=PAD_ID)
        else:
            self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=PAD_ID)
        rnn_kwargs = dict(input_size=self.embedding_dim, hidden_size=self.hidden_size,
                          num_layers=self.num_layers, batch_first=True,
                          bidirectional=self.bidirectional,
                          dropout=self.dropout if self.num_layers > 1 else 0.0)
        if self.cell_type == 'rnn':
            self.rnn = nn.RNN(nonlinearity=self.rnn_nonlinearity, **rnn_kwargs)
        elif self.cell_type == 'lstm':
            self.rnn = nn.LSTM(**rnn_kwargs)
        else:
            self.rnn = nn.GRU(**rnn_kwargs)
        self.drop = nn.Dropout(self.dropout)
        directions = 2 if self.bidirectional else 1
        # raw scores; CrossEntropyLoss applies softmax itself
        self.classifier = nn.Linear(self.hidden_size * directions, self.num_classes)
        self.to(self.device)

    def forward(self, x, lengths):
        '''x: (B, T) post-padded word IDs on the device; lengths: (B,) CPU int64.'''
        emb = self.drop(self.embedding(x))
        packed = pack_padded_sequence(emb, lengths, batch_first=True, enforce_sorted=False)
        _, h_n = self.rnn(packed)
        if self.cell_type == 'lstm':
            h_n = h_n[0]                      # LSTM returns (hidden, cell)
        if self.bidirectional:
            final = torch.cat([h_n[-2], h_n[-1]], dim=1)   # last layer, both directions
        else:
            final = h_n[-1]                   # last layer
        return self.classifier(self.drop(final))

    def get_optimizer(self):
        if self.optimizer_name == 'sgd':
            return torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9,
                                   weight_decay=self.weight_decay)
        if self.optimizer_name == 'rmsprop':
            return torch.optim.RMSprop(self.parameters(), lr=self.learning_rate,
                                       weight_decay=self.weight_decay)
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate,
                                weight_decay=self.weight_decay)

    def _predict_logits(self, X_cpu, L_cpu):
        outs = []
        for s in range(0, X_cpu.shape[0], self.eval_batch_size):
            xb = X_cpu[s:s + self.eval_batch_size].to(self.device)
            outs.append(self.forward(xb, L_cpu[s:s + self.eval_batch_size]).cpu())
        return torch.cat(outs)

    def fit(self, X, y, X_test=None, y_test=None):
        '''Train on (X, y). Test data, if given, is only scored for the learning curves.'''
        self.set_seed()
        X_post, lengths = pre_to_post_padding(X)
        X_cpu = torch.as_tensor(X_post, dtype=torch.int64)
        L_cpu = torch.as_tensor(lengths, dtype=torch.int64)
        y_cpu = torch.as_tensor(np.asarray(y), dtype=torch.int64)
        Xt_cpu = Lt_cpu = yt_cpu = None
        if X_test is not None and y_test is not None:
            Xt_post, Lt = pre_to_post_padding(X_test)
            Xt_cpu = torch.as_tensor(Xt_post, dtype=torch.int64)
            Lt_cpu = torch.as_tensor(Lt, dtype=torch.int64)
            yt_cpu = torch.as_tensor(np.asarray(y_test), dtype=torch.int64)

        optimizer = self.get_optimizer()
        loss_function = nn.CrossEntropyLoss()
        n = X_cpu.shape[0]
        self.loss_history, self.train_acc_history = [], []
        self.test_acc_history, self.epoch_times = [], []

        for epoch in range(self.max_epoch):
            start = time.time()
            self.train()            # turn Dropout on
            order = torch.randperm(n)
            total_loss, correct = 0.0, 0
            for s in range(0, n, self.batch_size):
                idx = order[s:s + self.batch_size]
                xb = X_cpu[idx].to(self.device)
                yb = y_cpu[idx].to(self.device)
                logits = self.forward(xb, L_cpu[idx])
                loss = loss_function(logits, yb)
                optimizer.zero_grad()
                loss.backward()
                if self.grad_clip:
                    nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
                optimizer.step()
                total_loss += loss.item() * len(idx)
                # running train accuracy on the batches as seen during training
                correct += (logits.argmax(1) == yb).sum().item()
            epoch_loss, train_acc = total_loss / n, correct / n

            self.eval()             # Dropout off for the test score
            test_acc = None
            if self.track_test_acc and Xt_cpu is not None:
                with torch.no_grad():
                    test_acc = (self._predict_logits(Xt_cpu, Lt_cpu).argmax(1) == yt_cpu).float().mean().item()
                self.test_acc_history.append(test_acc)
            self.loss_history.append(epoch_loss)
            self.train_acc_history.append(train_acc)
            self.epoch_times.append(time.time() - start)
            if epoch % self.print_every == 0 or epoch == self.max_epoch - 1:
                msg = f'Epoch: {epoch} Accuracy: {train_acc:.4f} Loss: {epoch_loss:.4f}'
                if test_acc is not None:
                    msg += f' TestAcc: {test_acc:.4f}'
                print(msg + f' ({self.epoch_times[-1]:.1f}s)', flush=True)

    def test(self, X):
        self.eval()
        X_post, lengths = pre_to_post_padding(X)
        with torch.no_grad():
            logits = self._predict_logits(torch.as_tensor(X_post, dtype=torch.int64),
                                          torch.as_tensor(lengths, dtype=torch.int64))
        return logits.argmax(1).numpy()

    def parameter_table(self):
        '''(part, shape description, parameters) rows for the report.'''
        rows = [
            ('Embedding' + (' (GloVe' + (', frozen)' if self.freeze_embeddings else ', fine-tuned)')
                            if self.embedding_matrix is not None else ''),
             f'{self.vocab_size} words x {self.embedding_dim}',
             sum(p.numel() for p in self.embedding.parameters())),
            (self.cell_type.upper() + (' (bidirectional)' if self.bidirectional else ''),
             f'{self.num_layers} layer(s), hidden {self.hidden_size}',
             sum(p.numel() for p in self.rnn.parameters())),
            ('Linear output', f'{self.classifier.in_features} -> {self.num_classes}',
             sum(p.numel() for p in self.classifier.parameters())),
        ]
        return rows

    def print_parameter_table(self):
        rows = self.parameter_table()
        print(f'{"part":<26} {"shape":<32} {"params"}')
        for name, shape, n in rows:
            print(f'{name:<26} {shape:<32} {n}')
        print('total params:', sum(r[2] for r in rows))

    def run(self):
        print('method running...')
        print('--device:', self.device, '| cell:', self.cell_type)
        X_train = np.asarray(self.data['train']['X'])
        y_train = np.asarray(self.data['train']['y'])
        X_test = np.asarray(self.data['test']['X'])
        y_test = np.asarray(self.data['test']['y'])
        if self.train_subset:
            X_train, y_train = X_train[:self.train_subset], y_train[:self.train_subset]
        if self.embedding is None:
            vocab_size = self.vocab_size or int(max(X_train.max(), X_test.max())) + 1
            self.build_model(vocab_size, int(max(y_train.max(), y_test.max())) + 1)
        print('--model:', self.embedding, self.rnn, self.classifier, sep='\n  ')
        print('--start training...')
        self.fit(X_train, y_train, X_test=X_test, y_test=y_test)
        print('--start testing...')
        pred_y = self.test(X_test)
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}
