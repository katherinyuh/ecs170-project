from local_code.base_class.method import method
import numpy as np
import torch
from torch import nn


class Method_CNN_MNIST(method, nn.Module):
    data = None

    # --- settings to change for experiments (step 3-5) ---
    max_epoch = 10
    learning_rate = 1e-3
    batch_size = 128
    conv_channels = (32, 64)   # one conv block per entry: more entries = deeper model
    kernel_size = 3
    padding = 1
    stride = 1
    hidden_dim = 128
    use_pooling = True

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        # use the Mac GPU if it's available, otherwise the CPU
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

        # convolution blocks: conv -> ReLU -> (optional) 2x2 max pooling
        layers = []
        in_channels = 1  # MNIST is grayscale
        for out_channels in self.conv_channels:
            layers.append(nn.Conv2d(in_channels, out_channels, self.kernel_size,
                                    stride=self.stride, padding=self.padding))
            layers.append(nn.ReLU())
            if self.use_pooling:
                layers.append(nn.MaxPool2d(2))
            in_channels = out_channels
        self.features = nn.Sequential(*layers)

        # work out the flattened size by passing one fake 28x28 image through
        with torch.no_grad():
            flat_dim = self.features(torch.zeros(1, 1, 28, 28)).numel()

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 10),  # 10 digit classes
        )

        self.loss_history = []  # average training loss per epoch, for the learning curve
        self.to(self.device)

    def forward(self, x):
        return self.classifier(self.features(x))

    # Named train_model/test_model (not train/test) on purpose: nn.Module already
    # has a train() method that PyTorch uses internally, so overriding it can break things.
    def train_model(self, X, y):
        # X is already float32, shape (N, 1, 28, 28), scaled to 0-1 by the loader: no extra normalization
        X = torch.from_numpy(X)
        y = torch.from_numpy(y)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss()
        n = len(X)

        for epoch in range(self.max_epoch):
            nn.Module.train(self, True)
            order = torch.randperm(n)  # shuffle each epoch
            total_loss, correct = 0.0, 0
            for start in range(0, n, self.batch_size):
                idx = order[start:start + self.batch_size]
                X_batch = X[idx].to(self.device)
                y_batch = y[idx].to(self.device)

                optimizer.zero_grad()
                output = self.forward(X_batch)
                loss = loss_function(output, y_batch)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(idx)
                correct += (output.argmax(dim=1) == y_batch).sum().item()

            epoch_loss = total_loss / n
            self.loss_history.append(epoch_loss)
            print(f'Epoch {epoch + 1}/{self.max_epoch}  loss: {epoch_loss:.4f}  '
                  f'train accuracy: {correct / n:.4f}')

    def test_model(self, X):
        nn.Module.train(self, False)  # evaluation mode
        X = torch.from_numpy(X)
        predictions = []
        with torch.no_grad():
            for start in range(0, len(X), 1000):
                X_batch = X[start:start + 1000].to(self.device)
                predictions.append(self.forward(X_batch).argmax(dim=1).cpu())
        return torch.cat(predictions).numpy()

    def run(self):
        print('method running...')
        print('--start training...')
        self.train_model(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test_model(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}