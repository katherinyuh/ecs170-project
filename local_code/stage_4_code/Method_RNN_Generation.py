'''
Concrete MethodModule class for Stage 4 joke generation.

One class covers all three cell types, same switch as Method_RNN_Classification:

    cell_type = 'rnn'   plain (Elman) RNN
    cell_type = 'lstm'  LSTM
    cell_type = 'gru'   GRU

Model: word IDs -> Embedding -> RNN/LSTM/GRU -> Linear over the vocabulary.
Training data comes from Dataset_Loader_Generation as-is: each example is a
window of seq_len word IDs (left-padded with <pad>) and the next word, including
<eos> at the end of every joke. Nothing about the cell type changes the rest of
the model, the loss, or the training loop.

Vocabulary from Part A: 0 = <pad>, 1 = <unk>, 2 = <eos>, 3+ = words.
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.method import method
import random
import time
import numpy as np
import torch
from torch import nn

from local_code.stage_4_code.Text_Utils import (
    EOS, PAD, UNK, PAD_ID, UNK_ID, encode, tokenize,
)


def pick_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


class Method_RNN_Generation(method, nn.Module):
    data = None

    # ---- configurable settings (baseline defaults; Part D can override all of these) ----
    cell_type = 'lstm'          # 'rnn', 'lstm' or 'gru'
    embedding_dim = 100
    hidden_size = 128           # Part B's name. 'hidden_dim' is accepted as an alias.
    num_layers = 1
    seq_len = 3                 # words in the context window; Part A's context_len default
    dropout = 0.2               # after the embedding and before the output layer
    rnn_nonlinearity = 'tanh'   # plain RNN only: 'tanh' or 'relu'
    max_epoch = 15
    learning_rate = 1e-3
    weight_decay = 0.0
    optimizer_name = 'adam'     # 'adam', 'sgd' or 'rmsprop'
    batch_size = 64
    grad_clip = 5.0             # caps exploding gradients, which plain RNNs are prone to
    eval_batch_size = 1024
    train_subset = None         # use only the first N training pairs (quick checks)
    print_every = 1
    seed = 2
    device_name = 'auto'        # 'auto' picks cuda / mps / cpu

    vocab_size = None

    def __init__(self, mName, mDescription, config=None):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        if config:
            config = dict(config)
            # Part B names this hidden_size. Accept hidden_dim so a sweep can use either.
            if 'hidden_dim' in config:
                hidden_dim = config.pop('hidden_dim')
                if 'hidden_size' in config and config['hidden_size'] != hidden_dim:
                    raise ValueError('hidden_size and hidden_dim were both set and they differ')
                config['hidden_size'] = hidden_dim
            for key, value in config.items():
                if not hasattr(self, key):
                    raise AttributeError(f'Unknown RNN config key: {key}')
                setattr(self, key, value)
        if self.cell_type not in ('rnn', 'lstm', 'gru'):
            raise ValueError(f"cell_type must be 'rnn', 'lstm' or 'gru', got {self.cell_type}")
        self.device = pick_device() if self.device_name == 'auto' else torch.device(self.device_name)
        self.embedding = None
        self.rnn = None
        self.output_layer = None
        self.word_to_id = None
        self.id_to_word = None
        self.eos_id = None
        self.loss_history = []
        self.train_acc_history = []
        self.epoch_times = []
        self.final_eval_acc = None

    def set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def build_model(self, vocab_size):
        self.set_seed()
        self.vocab_size = int(vocab_size)
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=PAD_ID)
        rnn_kwargs = dict(input_size=self.embedding_dim, hidden_size=self.hidden_size,
                          num_layers=self.num_layers, batch_first=True,
                          dropout=self.dropout if self.num_layers > 1 else 0.0)
        if self.cell_type == 'rnn':
            self.rnn = nn.RNN(nonlinearity=self.rnn_nonlinearity, **rnn_kwargs)
        elif self.cell_type == 'lstm':
            self.rnn = nn.LSTM(**rnn_kwargs)
        else:
            self.rnn = nn.GRU(**rnn_kwargs)
        self.drop = nn.Dropout(self.dropout)
        # one score per vocabulary word; CrossEntropyLoss applies log-softmax itself
        self.output_layer = nn.Linear(self.hidden_size, self.vocab_size)
        self.to(self.device)

    def forward(self, x):
        '''x: (B, seq_len) word IDs. Returns logits (B, vocab) for the next word.'''
        emb = self.drop(self.embedding(x))
        out, _ = self.rnn(emb)
        # The label is the word that follows the LAST position of the window.
        return self.output_layer(self.drop(out[:, -1, :]))

    def get_optimizer(self):
        if self.optimizer_name == 'sgd':
            return torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9,
                                   weight_decay=self.weight_decay)
        if self.optimizer_name == 'rmsprop':
            return torch.optim.RMSprop(self.parameters(), lr=self.learning_rate,
                                       weight_decay=self.weight_decay)
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate,
                                weight_decay=self.weight_decay)

    def _training_pairs(self):
        '''Windows from Part A's loader. Rebuild them if seq_len differs from context_len.'''
        X = np.asarray(self.data['train']['X'])
        y = np.asarray(self.data['train']['y'])
        if X.ndim == 2 and X.shape[1] == self.seq_len:
            return X, y
        joke_ids = self.data.get('joke_ids')
        if not joke_ids:
            raise ValueError(
                f'seq_len is {self.seq_len} but train X has width {X.shape[1] if X.ndim == 2 else "?"}, '
                'and joke_ids was not passed in. Set Dataset_Loader_Generation.context_len before load(), '
                'or pass the full loader dict (it includes joke_ids).')
        # Same construction as Dataset_Loader_Generation.make_training_pairs.
        rows_x, rows_y = [], []
        for ids in joke_ids:
            padded = [PAD_ID] * self.seq_len + list(ids)
            for t in range(len(ids)):
                rows_x.append(padded[t:t + self.seq_len])
                rows_y.append(ids[t])
        return np.asarray(rows_x, dtype=np.int64), np.asarray(rows_y, dtype=np.int64)

    def _prepare_vocab(self):
        vocab = self.data['vocab']
        self.word_to_id = vocab['word_to_id']
        self.id_to_word = vocab['id_to_word']
        self.eos_id = self.word_to_id[EOS]
        if self.embedding is None:
            self.build_model(vocab['size'])

    def fit(self, X, y):
        '''Train next-word prediction. Accuracy is the fraction of windows whose argmax is y.'''
        self.set_seed()
        X_cpu = torch.as_tensor(np.asarray(X), dtype=torch.int64)
        y_cpu = torch.as_tensor(np.asarray(y), dtype=torch.int64)
        optimizer = self.get_optimizer()
        loss_function = nn.CrossEntropyLoss()
        n = X_cpu.shape[0]
        self.loss_history, self.train_acc_history, self.epoch_times = [], [], []

        for epoch in range(self.max_epoch):
            start = time.time()
            self.train()            # turn Dropout on
            order = torch.randperm(n)
            total_loss, correct = 0.0, 0
            for s in range(0, n, self.batch_size):
                idx = order[s:s + self.batch_size]
                xb = X_cpu[idx].to(self.device)
                yb = y_cpu[idx].to(self.device)
                logits = self.forward(xb)
                loss = loss_function(logits, yb)
                optimizer.zero_grad()
                loss.backward()
                if self.grad_clip:
                    nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
                optimizer.step()
                total_loss += loss.item() * len(idx)
                correct += (logits.argmax(1) == yb).sum().item()
            epoch_loss, train_acc = total_loss / n, correct / n
            self.loss_history.append(epoch_loss)
            self.train_acc_history.append(train_acc)
            self.epoch_times.append(time.time() - start)
            if epoch % self.print_every == 0 or epoch == self.max_epoch - 1:
                print(f'Epoch: {epoch} Accuracy: {train_acc:.4f} Loss: {epoch_loss:.4f} '
                      f'({self.epoch_times[-1]:.1f}s)', flush=True)

    def test(self, X):
        '''Next-word predictions (greedy) for windows X of shape (N, seq_len).'''
        self.eval()
        X = np.asarray(X)
        preds = []
        with torch.no_grad():
            for s in range(0, X.shape[0], self.eval_batch_size):
                xb = torch.as_tensor(X[s:s + self.eval_batch_size], dtype=torch.int64, device=self.device)
                preds.append(self.forward(xb).argmax(1).cpu())
        if not preds:
            return np.zeros(0, dtype=np.int64)
        return torch.cat(preds).numpy()

    def _prompt_ids(self, start_words):
        '''Same cleaning as Dataset_Loader_Generation.words_to_ids. Unknown words become <unk>.'''
        if isinstance(start_words, str):
            words = tokenize(start_words, keep_punctuation=True)
        else:
            words = [str(w).lower() for w in start_words]
        unk = [w for w in words if w not in self.word_to_id]
        return words, encode(words, self.word_to_id), unk

    def generate(self, start_words, max_len=40, temperature=None):
        '''Write one word at a time, feeding each prediction back in.

        start_words: three words, as a list or a string ("what did the").
        max_len: most new words to add after the prompt.
        temperature: None or <= 0 is greedy (argmax). A positive value samples
        from softmax(logits / temperature). Sampling is seeded so a rerun matches.

        Stops when the model predicts <eos> or max_len new words have been added.
        <pad> and <unk> are never chosen as the next word (<unk> can still appear
        in the prompt, when a starting word is not in the vocabulary).
        '''
        if self.word_to_id is None or self.embedding is None:
            raise RuntimeError('Call run() before generate(), so the vocabulary and weights exist.')
        prompt_words, prompt_ids, unknown = self._prompt_ids(start_words)
        if not prompt_ids:
            raise ValueError('start_words did not contain any tokens')
        context = [PAD_ID] * self.seq_len + prompt_ids
        context = context[-self.seq_len:]
        generated = []
        stopped_on = 'max_len'
        self.eval()
        # A local generator keeps sampling reproducible without touching the training RNG.
        generator = torch.Generator(device='cpu')
        generator.manual_seed(self.seed)
        with torch.no_grad():
            for _ in range(max_len):
                xb = torch.tensor([context], dtype=torch.int64, device=self.device)
                logits = self.forward(xb)[0].detach().float().cpu()
                logits[PAD_ID] = float('-inf')
                logits[UNK_ID] = float('-inf')
                if temperature is None or temperature <= 0:
                    next_id = int(torch.argmax(logits).item())
                else:
                    probs = torch.softmax(logits / float(temperature), dim=-1)
                    next_id = int(torch.multinomial(probs, 1, generator=generator).item())
                if next_id == self.eos_id:
                    stopped_on = 'eos'
                    break
                generated.append(next_id)
                context = context[1:] + [next_id]
        ids = prompt_ids + generated
        words = [self.id_to_word[i] if 0 <= i < len(self.id_to_word) else UNK for i in ids]
        return {
            'ids': ids,
            'words': words,
            'prompt_words': prompt_words,
            'prompt_ids': prompt_ids,
            'generated_ids': generated,
            'unknown_words': unknown,
            'stopped_on': stopped_on,
            'temperature': None if temperature is None or temperature <= 0 else float(temperature),
        }

    def parameter_table(self):
        rows = [
            ('Embedding', f'{self.vocab_size} words x {self.embedding_dim}',
             sum(p.numel() for p in self.embedding.parameters())),
            (self.cell_type.upper(), f'{self.num_layers} layer(s), hidden {self.hidden_size}',
             sum(p.numel() for p in self.rnn.parameters())),
            ('Linear output', f'{self.output_layer.in_features} -> {self.vocab_size}',
             sum(p.numel() for p in self.output_layer.parameters())),
        ]
        return rows

    def print_parameter_table(self):
        rows = self.parameter_table()
        print(f'{"part":<26} {"shape":<32} {"params"}')
        for name, shape, n in rows:
            print(f'{name:<26} {shape:<32} {n}')
        print('total params:', sum(r[2] for r in rows))

    def run(self):
        '''Train on the loader's next-word pairs. There is no test split in that loader.

        self.data is the dict Dataset_Loader_Generation.load() returns
        (train, vocab, joke_ids, jokes, context_len), not the stripped
        train/test dict the classification Setting builds.
        '''
        print('method running...')
        print('--device:', self.device, '| cell:', self.cell_type, '| seq_len:', self.seq_len)
        self._prepare_vocab()
        X, y = self._training_pairs()
        if self.train_subset:
            X, y = X[:self.train_subset], y[:self.train_subset]
        print('--model:', self.embedding, self.rnn, self.output_layer, sep='\n  ')
        print(f'--training pairs: {len(y)}, vocab: {self.vocab_size}')
        print('--start training...')
        self.fit(X, y)
        print('--start testing (next-word accuracy on the training windows, dropout off)...')
        pred_y = self.test(X)
        self.final_eval_acc = float((pred_y == y).mean()) if len(y) else 0.0
        print(f'--train next-word accuracy: {self.final_eval_acc:.4f}')
        return {'pred_y': pred_y, 'true_y': y,
                'loss_history': list(self.loss_history),
                'train_acc_history': list(self.train_acc_history)}
