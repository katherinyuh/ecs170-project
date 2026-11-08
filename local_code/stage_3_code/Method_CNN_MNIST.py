'''
Concrete MethodModule class for the Stage 3 MNIST handwritten-digit CNN.
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD
# MNIST CNN: configurable Conv blocks -> flatten -> FC -> dropout -> 10 logits.
# Same interface as Method_CNN_ORL (config dict, fit/test, loss/accuracy histories)
# so the experiment scripts and report tables line up across datasets.
# Uses fit() for the training loop so nn.Module.train()/eval() still control Dropout.

from local_code.base_class.method import method
import random
import torch
from torch import nn
import numpy as np


def pick_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    # Apple Silicon GPU when CUDA is absent
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


class Method_CNN_MNIST(method, nn.Module):
    data = None

    # ---- configurable experiment settings (baseline defaults) ----
    max_epoch = 10
    learning_rate = 1e-3
    # one entry per conv block: output channels after that block
    conv_channels = [32, 64]
    kernel_size = 3
    # None -> same padding that keeps H/W stable before pooling (kernel_size // 2)
    padding = None
    stride = 1
    # 'max', 'avg' or 'none'
    pool_type = 'max'
    pool_kernel = 2
    pool_stride = 2
    fc_hidden = 128
    dropout = 0.25
    weight_decay = 0.0
    # 'adam', 'sgd' or 'rmsprop'
    optimizer_name = 'adam'
    # 'cross_entropy', 'nll' or 'multi_margin'
    loss_name = 'cross_entropy'
    batch_size = 128
    eval_batch_size = 2000
    print_every = 1
    seed = 2
    # log test accuracy each epoch for curves only (never used to pick weights)
    track_test_acc = True

    # MNIST from Dataset_Loader_MNIST: (N, 1, 28, 28), already scaled to [0, 1]
    in_channels = 1
    num_classes = 10
    image_size = (28, 28)

    def __init__(self, mName, mDescription, config=None):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        # optional dict overrides class defaults, e.g. {'kernel_size': 5, 'dropout': 0.0}
        if config:
            for key, value in config.items():
                if not hasattr(self, key):
                    raise AttributeError(f'Unknown CNN config key: {key}')
                setattr(self, key, value)

        self.device = pick_device()
        self.features = None
        self.classifier = None
        self.flatten_dim = None
        self.loss_history = []
        self.train_acc_history = []
        self.test_acc_history = []

    def set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def _pool_layer(self):
        if self.pool_type == 'avg':
            return nn.AvgPool2d(kernel_size=self.pool_kernel, stride=self.pool_stride)
        if self.pool_type == 'max':
            return nn.MaxPool2d(kernel_size=self.pool_kernel, stride=self.pool_stride)
        raise ValueError(f"pool_type must be 'max', 'avg' or 'none', got {self.pool_type}")

    def build_model(self):
        pad = self.kernel_size // 2 if self.padding is None else self.padding
        layers = []
        prev_c = self.in_channels
        for out_c in self.conv_channels:
            layers.append(nn.Conv2d(prev_c, out_c, kernel_size=self.kernel_size,
                                    stride=self.stride, padding=pad))
            layers.append(nn.ReLU(inplace=True))
            if self.pool_type != 'none':
                layers.append(self._pool_layer())
            prev_c = out_c
        self.features = nn.Sequential(*layers)

        # infer flatten size from a dummy image instead of hard-coding H*W math
        with torch.no_grad():
            dummy = torch.zeros(1, self.in_channels, *self.image_size)
            self.flatten_dim = int(self.features(dummy).view(1, -1).shape[1])

        self.classifier = nn.Sequential(
            nn.Linear(self.flatten_dim, self.fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            # raw logits; CrossEntropyLoss applies log-softmax internally
            nn.Linear(self.fc_hidden, self.num_classes),
        )
        self.to(self.device)

    def forward(self, x):
        '''Forward propagation, returns raw class scores (logits).'''
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    def get_optimizer(self):
        params = self.parameters()
        if self.optimizer_name == 'sgd':
            return torch.optim.SGD(params, lr=self.learning_rate,
                                   momentum=0.9, weight_decay=self.weight_decay)
        if self.optimizer_name == 'rmsprop':
            return torch.optim.RMSprop(params, lr=self.learning_rate,
                                       weight_decay=self.weight_decay)
        return torch.optim.Adam(params, lr=self.learning_rate,
                                weight_decay=self.weight_decay)

    def get_loss_function(self):
        if self.loss_name == 'nll':
            log_softmax = nn.LogSoftmax(dim=1)
            nll = nn.NLLLoss()
            return lambda logits, y: nll(log_softmax(logits), y)
        if self.loss_name == 'multi_margin':
            # hinge-style (SVM-like) loss: a genuinely different objective from cross-entropy.
            # Same formula as nn.MultiMarginLoss (margin 1), written out by hand because
            # PyTorch's built-in version does not run on the Mac (MPS) GPU.
            def multi_margin(logits, y):
                correct = logits.gather(1, y.unsqueeze(1))           # score of the true class
                margins = torch.clamp(1 - correct + logits, min=0)   # how far each wrong class gets too close
                margins = margins.scatter(1, y.unsqueeze(1), 0.0)    # the true class itself does not count
                return margins.sum(dim=1).mean() / logits.size(1)
            return multi_margin
        return nn.CrossEntropyLoss()

    def _to_tensor(self, X, y=None):
        X_t = torch.as_tensor(np.asarray(X), dtype=torch.float32, device=self.device)
        if y is None:
            return X_t
        y_t = torch.as_tensor(np.asarray(y), dtype=torch.int64, device=self.device)
        return X_t, y_t

    def _predict_tensor(self, X_tensor):
        # in chunks: 60,000 images through the conv layers at once would not fit in memory
        preds = []
        with torch.no_grad():
            for start in range(0, X_tensor.shape[0], self.eval_batch_size):
                logits = self.forward(X_tensor[start:start + self.eval_batch_size])
                preds.append(logits.argmax(1))
        return torch.cat(preds)

    def _accuracy(self, X_tensor, y_tensor):
        return (self._predict_tensor(X_tensor) == y_tensor).float().mean().item()

    def fit(self, X, y, X_test=None, y_test=None):
        '''Train on (X, y). Optional test tensors are only scored for learning curves.'''
        self.set_seed()
        X_tensor, y_true = self._to_tensor(X, y)
        X_test_t = y_test_t = None
        if X_test is not None and y_test is not None:
            X_test_t, y_test_t = self._to_tensor(X_test, y_test)

        optimizer = self.get_optimizer()
        loss_function = self.get_loss_function()
        n = X_tensor.shape[0]
        self.loss_history = []
        self.train_acc_history = []
        self.test_acc_history = []

        for epoch in range(self.max_epoch):
            self.train()  # enable Dropout
            order = torch.randperm(n, device=self.device)
            epoch_loss = 0.0
            for start in range(0, n, self.batch_size):
                idx = order[start:start + self.batch_size]
                logits = self.forward(X_tensor[idx])
                train_loss = loss_function(logits, y_true[idx])
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                epoch_loss += train_loss.item() * len(idx)
            epoch_loss /= n

            # eval mode so Dropout does not scramble the reported train/test accuracy
            self.eval()
            train_acc = self._accuracy(X_tensor, y_true)
            test_acc = None
            if self.track_test_acc and X_test_t is not None:
                test_acc = self._accuracy(X_test_t, y_test_t)
                self.test_acc_history.append(test_acc)
            self.loss_history.append(epoch_loss)
            self.train_acc_history.append(train_acc)

            if epoch % self.print_every == 0 or epoch == self.max_epoch - 1:
                msg = f'Epoch: {epoch} Accuracy: {train_acc:.4f} Loss: {epoch_loss:.4f}'
                if test_acc is not None:
                    msg += f' TestAcc: {test_acc:.4f}'
                print(msg)

    def test(self, X):
        self.eval()
        return self._predict_tensor(self._to_tensor(X)).cpu().numpy()

    def layer_shape_table(self):
        '''Return rows of (name, output_shape, num_params) for the report.'''
        if self.features is None or self.classifier is None:
            raise RuntimeError('Call build_model() before layer_shape_table().')
        x = torch.zeros(1, self.in_channels, *self.image_size, device=self.device)
        rows = [('input', tuple(x.shape), 0)]
        with torch.no_grad():
            for name, module in list(self.features.named_children()) + list(self.classifier.named_children()):
                if isinstance(module, nn.Linear):
                    x = x.view(x.size(0), -1)
                x = module(x)
                n_params = sum(p.numel() for p in module.parameters())
                rows.append((f'{name}:{module.__class__.__name__}', tuple(x.shape), n_params))
        return rows

    def print_shape_table(self):
        print(f'{"layer":<28} {"output_shape":<22} {"params"}')
        for name, shape, n_params in self.layer_shape_table():
            print(f'{name:<28} {str(shape):<22} {n_params}')

    def run(self):
        print('method running...')
        print('--device:', self.device)
        X_train = np.asarray(self.data['train']['X'])
        y_train = np.asarray(self.data['train']['y'])
        X_test = np.asarray(self.data['test']['X'])
        y_test = np.asarray(self.data['test']['y'])

        if self.features is None or self.classifier is None:
            self.build_model()
        print('--model features:', self.features)
        print('--model classifier:', self.classifier)
        print('--flatten_dim:', self.flatten_dim)
        print('--start training...')
        self.fit(X_train, y_train, X_test=X_test, y_test=y_test)
        print('--start testing...')
        pred_y = self.test(X_test)
        # Evaluate_Metrics / Result_Saver expect exactly these two keys
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}
