'''
Concrete MethodModule class for the Stage 3 CIFAR-10 colored-object CNN.
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD
# CIFAR CNN: configurable Conv blocks (Conv -> BatchNorm -> ReLU, x convs_per_block)
# -> pool -> flatten -> FC -> dropout -> 10 logits.
# Same structure and config style as Method_CNN_ORL so results line up in the report.
# Differences for CIFAR: 3 input channels, mini-batched evaluation (50,000 images
# do not fit through the network in one pass), optional BatchNorm and optional
# flip/crop augmentation, which help a lot on CIFAR.

from local_code.base_class.method import method
import random
import time
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


def pick_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    # Apple Silicon GPU when CUDA is absent
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


class Method_CNN_CIFAR(method, nn.Module):
    data = None

    # ---- configurable experiment settings (baseline defaults) ----
    max_epoch = 20
    learning_rate = 1e-3
    # one entry per conv block: output channels of that block
    conv_channels = [32, 64, 128]
    # conv layers inside each block before pooling (2 = VGG-style)
    convs_per_block = 2
    kernel_size = 3
    # None -> same padding that keeps H/W stable before pooling (kernel_size // 2)
    padding = None
    stride = 1
    # 'max' or 'avg'
    pool_type = 'max'
    pool_kernel = 2
    pool_stride = 2
    use_batchnorm = True
    fc_hidden = 256
    dropout = 0.5
    weight_decay = 5e-4
    optimizer_name = 'adam'
    loss_name = 'cross_entropy'
    batch_size = 128
    # random horizontal flip + random 32x32 crop from a 4-pixel padded image (train only)
    augment = True
    print_every = 1
    seed = 2
    # log test accuracy each epoch for curves only (never used to pick weights)
    track_test_acc = True
    # batch size used only for evaluation passes
    eval_batch_size = 1000
    # use only the first N training images (None = all); handy for quick trial runs
    train_subset = None
    # input already [0,1] from Dataset_Loader_CIFAR; leave False unless using a raw loader
    normalize_input = False
    # 'auto' picks cuda/mps/cpu; set 'cpu' to force CPU
    device_name = 'auto'

    # filled by build_model / run; set ahead of time only if you build before run()
    in_channels = 3
    num_classes = 10
    image_size = (32, 32)

    def __init__(self, mName, mDescription, config=None):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        # optional dict overrides class defaults, e.g. {'kernel_size': 5, 'dropout': 0.3}
        if config:
            for key, value in config.items():
                if not hasattr(self, key):
                    raise AttributeError(f'Unknown CNN config key: {key}')
                setattr(self, key, value)

        self.device = pick_device() if self.device_name == 'auto' else torch.device(self.device_name)
        self.features = None
        self.classifier = None
        self.flatten_dim = None
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

    def _pool_layer(self):
        if self.pool_type == 'avg':
            return nn.AvgPool2d(kernel_size=self.pool_kernel, stride=self.pool_stride)
        if self.pool_type != 'max':
            raise ValueError(f"pool_type must be 'max' or 'avg', got {self.pool_type}")
        return nn.MaxPool2d(kernel_size=self.pool_kernel, stride=self.pool_stride)

    def build_model(self, in_channels=None, num_classes=None, image_size=None):
        if in_channels is not None:
            self.in_channels = in_channels
        if num_classes is not None:
            self.num_classes = num_classes
        if image_size is not None:
            self.image_size = tuple(image_size)

        pad = self.kernel_size // 2 if self.padding is None else self.padding
        layers = []
        prev_c = self.in_channels
        for out_c in self.conv_channels:
            for i in range(self.convs_per_block):
                # stride only on the first conv of a block so later convs keep the size
                layers.append(nn.Conv2d(prev_c, out_c, kernel_size=self.kernel_size,
                                        stride=self.stride if i == 0 else 1, padding=pad,
                                        bias=not self.use_batchnorm))
                if self.use_batchnorm:
                    layers.append(nn.BatchNorm2d(out_c))
                layers.append(nn.ReLU(inplace=True))
                prev_c = out_c
            layers.append(self._pool_layer())
        self.features = nn.Sequential(*layers)

        # infer flatten size from a dummy image instead of hard-coding H*W math
        with torch.no_grad():
            dummy = torch.zeros(1, self.in_channels, *self.image_size)
            out = self.features.eval()(dummy)
            if out.shape[2] < 1 or out.shape[3] < 1:
                raise ValueError(f'Feature map shrank to {tuple(out.shape)}; use fewer blocks, '
                                 'more padding or a smaller stride.')
            self.flatten_dim = int(out.view(1, -1).shape[1])

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
        if self.loss_name == 'label_smoothing':
            return nn.CrossEntropyLoss(label_smoothing=0.1)
        return nn.CrossEntropyLoss()

    def _augment(self, x):
        '''Random horizontal flip and random 32x32 crop from a 4-pixel zero-padded image.'''
        b, c, h, w = x.shape
        flip = torch.rand(b, device=x.device) < 0.5
        x = torch.where(flip[:, None, None, None], x.flip(3), x)
        pad = 4
        x_p = F.pad(x, (pad, pad, pad, pad))
        dy = torch.randint(0, 2 * pad + 1, (b,), device=x.device)
        dx = torch.randint(0, 2 * pad + 1, (b,), device=x.device)
        rows = (dy[:, None] + torch.arange(h, device=x.device)[None, :])  # (b, h)
        cols = (dx[:, None] + torch.arange(w, device=x.device)[None, :])  # (b, w)
        x_p = torch.gather(x_p, 2, rows[:, None, :, None].expand(b, c, h, w + 2 * pad))
        return torch.gather(x_p, 3, cols[:, None, None, :].expand(b, c, h, w))

    def _predict_logits(self, X_cpu):
        '''Logits for a CPU tensor, run in evaluation-sized batches.'''
        outs = []
        for start in range(0, X_cpu.shape[0], self.eval_batch_size):
            xb = X_cpu[start:start + self.eval_batch_size].to(self.device)
            outs.append(self.forward(xb).cpu())
        return torch.cat(outs)

    def _accuracy(self, X_cpu, y_cpu):
        return (self._predict_logits(X_cpu).argmax(1) == y_cpu).float().mean().item()

    def fit(self, X, y, X_test=None, y_test=None):
        '''Train on (X, y). Optional test data is only scored for learning curves.'''
        self.set_seed()
        # data stays on the CPU; each mini-batch is moved to the device
        X_cpu = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        y_cpu = torch.as_tensor(np.asarray(y), dtype=torch.int64)
        if self.normalize_input:
            X_cpu = X_cpu / 255.0
        X_test_cpu = y_test_cpu = None
        if X_test is not None and y_test is not None:
            X_test_cpu = torch.as_tensor(np.asarray(X_test), dtype=torch.float32)
            y_test_cpu = torch.as_tensor(np.asarray(y_test), dtype=torch.int64)
            if self.normalize_input:
                X_test_cpu = X_test_cpu / 255.0

        optimizer = self.get_optimizer()
        loss_function = self.get_loss_function()
        n = X_cpu.shape[0]
        batch_size = self.batch_size or n
        self.loss_history = []
        self.train_acc_history = []
        self.test_acc_history = []
        self.epoch_times = []

        for epoch in range(self.max_epoch):
            start_time = time.time()
            self.train()  # enable Dropout and BatchNorm batch statistics
            order = torch.randperm(n)
            epoch_loss = 0.0
            correct = 0
            for start in range(0, n, batch_size):
                idx = order[start:start + batch_size]
                xb = X_cpu[idx].to(self.device)
                yb = y_cpu[idx].to(self.device)
                if self.augment:
                    xb = self._augment(xb)
                logits = self.forward(xb)
                train_loss = loss_function(logits, yb)
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                epoch_loss += train_loss.item() * len(idx)
                # running train accuracy on the batches as seen during training
                # (a full extra pass over 50,000 images each epoch would cost too much time)
                correct += (logits.argmax(1) == yb).sum().item()
            epoch_loss /= n
            train_acc = correct / n

            # eval mode so Dropout/BatchNorm do not change the reported test accuracy
            self.eval()
            test_acc = None
            if self.track_test_acc and X_test_cpu is not None:
                with torch.no_grad():
                    test_acc = self._accuracy(X_test_cpu, y_test_cpu)
                self.test_acc_history.append(test_acc)
            self.loss_history.append(epoch_loss)
            self.train_acc_history.append(train_acc)
            self.epoch_times.append(time.time() - start_time)

            if epoch % self.print_every == 0 or epoch == self.max_epoch - 1:
                msg = f'Epoch: {epoch} Accuracy: {train_acc:.4f} Loss: {epoch_loss:.4f}'
                if test_acc is not None:
                    msg += f' TestAcc: {test_acc:.4f}'
                msg += f' ({self.epoch_times[-1]:.1f}s)'
                print(msg, flush=True)

    def test(self, X):
        self.eval()
        X_cpu = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        if self.normalize_input:
            X_cpu = X_cpu / 255.0
        with torch.no_grad():
            logits = self._predict_logits(X_cpu)
        return logits.argmax(1).numpy()

    def layer_shape_table(self, image_size=None):
        '''Return rows of (name, output_shape, num_params) for the report.'''
        if self.features is None or self.classifier is None:
            raise RuntimeError('Call build_model() before layer_shape_table().')
        h, w = image_size or self.image_size
        was_training = self.training
        self.eval()
        x = torch.zeros(1, self.in_channels, h, w, device=self.device)
        rows = [('input', tuple(x.shape), 0)]
        with torch.no_grad():
            for name, module in (list(self.features.named_children())
                                 + list(self.classifier.named_children())):
                if isinstance(module, nn.Linear):
                    x = x.view(x.size(0), -1)
                x = module(x)
                n_params = sum(p.numel() for p in module.parameters())
                rows.append((f'{name}:{module.__class__.__name__}', tuple(x.shape), n_params))
        self.train(was_training)
        return rows

    def print_shape_table(self):
        rows = self.layer_shape_table()
        print(f'{"layer":<28} {"output_shape":<22} {"params"}')
        for name, shape, n_params in rows:
            print(f'{name:<28} {str(shape):<22} {n_params}')
        print('total params:', sum(r[2] for r in rows))

    def run(self):
        print('method running...')
        print('--device:', self.device)
        X_train = np.asarray(self.data['train']['X'])
        y_train = np.asarray(self.data['train']['y'])
        X_test = np.asarray(self.data['test']['X'])
        y_test = np.asarray(self.data['test']['y'])
        if self.train_subset:
            X_train = X_train[:self.train_subset]
            y_train = y_train[:self.train_subset]

        if self.features is None or self.classifier is None:
            # X is (N, C, H, W) from Dataset_Loader_CIFAR
            self.build_model(
                in_channels=X_train.shape[1],
                num_classes=int(max(y_train.max(), y_test.max())) + 1,
                image_size=X_train.shape[2:],
            )
        print('--model features:', self.features)
        print('--model classifier:', self.classifier)
        print('--flatten_dim:', self.flatten_dim)
        print('--start training...')
        self.fit(X_train, y_train, X_test=X_test, y_test=y_test)
        print('--start testing...')
        pred_y = self.test(X_test)
        # Evaluate_Metrics / Result_Saver expect exactly these two keys
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}
