'''
Concrete MethodModule class for the stage 2 MLP model
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD
# Stage 2 changes: layer sizes are set from the data, multiple hidden layers,
# raw logits out (CrossEntropyLoss applies softmax itself), optional mini-batches,
# and per-epoch loss/accuracy history for the learning curve plots.

from local_code.base_class.method import method
import torch
from torch import nn
import numpy as np


class Method_MLP(method, nn.Module):
    data = None

    # ---- settings the experiments person can change ----
    # max rounds to train the model
    max_epoch = 200
    # learning rate for the optimizer
    learning_rate = 1e-3
    # sizes of the hidden layers, e.g. [256] for one layer, [512, 256, 128] for three
    hidden_dims = [256, 128]
    # 'relu', 'tanh' or 'sigmoid'
    activation = 'relu'
    # 'adam', 'sgd' or 'rmsprop'
    optimizer_name = 'adam'
    # 'cross_entropy' or 'nll'
    loss_name = 'cross_entropy'
    # None = whole training set as one batch (like the template), or a number like 256
    batch_size = None
    # print progress every N epochs
    print_every = 10
    # divide inputs by the largest training value (pixels 0-255 become 0-1), which trains much better
    normalize_input = True
    input_scale = 1.0

    # input size and class count are read from the data in run(),
    # set them here only if you want to build the model before run()
    input_dim = None
    num_classes = None

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.model = None
        # filled in during training, used by the plotting script for the learning curves
        self.loss_history = []
        self.train_acc_history = []
        if self.input_dim is not None and self.num_classes is not None:
            self.build_model(self.input_dim, self.num_classes)

    # builds the layers: input -> hidden layers -> output (one score per class)
    def build_model(self, input_dim, num_classes):
        self.input_dim = input_dim
        self.num_classes = num_classes
        activations = {'relu': nn.ReLU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid}
        layers = []
        prev_dim = input_dim
        for h in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(activations[self.activation]())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, num_classes))
        # no softmax at the end: CrossEntropyLoss already applies it
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        '''Forward propagation, returns raw class scores (logits)'''
        return self.model(x)

    def get_optimizer(self):
        if self.optimizer_name == 'sgd':
            return torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)
        if self.optimizer_name == 'rmsprop':
            return torch.optim.RMSprop(self.parameters(), lr=self.learning_rate)
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def get_loss_function(self):
        if self.loss_name == 'nll':
            log_softmax = nn.LogSoftmax(dim=1)
            nll = nn.NLLLoss()
            return lambda logits, y: nll(log_softmax(logits), y)
        return nn.CrossEntropyLoss()

    def train(self, X, y):
        X_tensor = torch.FloatTensor(np.array(X)) / self.input_scale
        y_true = torch.LongTensor(np.array(y))
        optimizer = self.get_optimizer()
        loss_function = self.get_loss_function()
        n = X_tensor.shape[0]
        batch_size = self.batch_size or n

        for epoch in range(self.max_epoch):
            order = torch.randperm(n) if batch_size < n else torch.arange(n)
            epoch_loss = 0.0
            for start in range(0, n, batch_size):
                idx = order[start:start + batch_size]
                y_pred = self.forward(X_tensor[idx])
                train_loss = loss_function(y_pred, y_true[idx])
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                epoch_loss += train_loss.item() * len(idx)
            epoch_loss /= n

            # training accuracy on the whole training set for the learning curve
            with torch.no_grad():
                train_acc = (self.forward(X_tensor).argmax(1) == y_true).float().mean().item()
            self.loss_history.append(epoch_loss)
            self.train_acc_history.append(train_acc)

            if epoch % self.print_every == 0 or epoch == self.max_epoch - 1:
                print('Epoch:', epoch, 'Accuracy:', round(train_acc, 4), 'Loss:', round(epoch_loss, 4))

    def test(self, X):
        with torch.no_grad():
            y_pred = self.forward(torch.FloatTensor(np.array(X)) / self.input_scale)
        # each instance gets the label with the highest score
        return y_pred.argmax(1).numpy()

    def run(self):
        print('method running...')
        X_train = np.array(self.data['train']['X'])
        y_train = np.array(self.data['train']['y'])
        if self.normalize_input:
            self.input_scale = float(max(np.abs(X_train).max(), 1))
        if self.model is None:
            # labels must be 0..C-1 for CrossEntropyLoss
            num_classes = int(max(y_train.max(), np.array(self.data['test']['y']).max())) + 1
            self.build_model(X_train.shape[1], num_classes)
        print('--model:', self.model)
        print('--start training...')
        self.train(X_train, y_train)
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}
