import torch
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
import local_datasets

class Model(nn.Module):
    def __init__(self, d_i, d_k, d_o, num_hidden, dropout):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(d_i, d_k))
        self.layers.append(nn.ReLU())

        for _ in range(num_hidden):
            if dropout:
                self.layers.append(nn.Dropout(p=0.2))
            self.layers.append(nn.Linear(d_k, d_k))
            self.layers.append(nn.ReLU())

        self.layers.append(nn.Linear(d_k, d_o))
        # Note: softmax is handled by CrossEntropyLoss
        self.init_weights()

    def init_weights(self):
        for layer in self.layers:
            if type(layer) is nn.Linear:
                init.kaiming_normal_(layer.weight)
                init.zeros_(layer.bias)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

def train_model(model, loader, x_test, y_test, epochs, lr):
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, gamma=0.5, step_size=10)
    errors = np.zeros((epochs, 1))

    for epoch in range(epochs):
        model.train()
        for batch in loader:
            x_batch, y_batch = batch
            prediction = model(x_batch)
            loss = loss_function(prediction, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        prediction = model(x_test)

        num_correct = 0
        loss = loss_function(prediction, y_test)
        max_indices = torch.argmax(prediction, dim=-1)
        predicted_class = F.one_hot(max_indices, num_classes=prediction.shape[-1])

        num_correct = (predicted_class == y_test).all(dim=1).sum()
        accuracy = 100 * num_correct / y_test.shape[0]
        errors[epoch] = 100 - accuracy

        print(f"Epoch {epoch + 1}/{epochs} | Loss: {loss:.3f} | Accuracy: {accuracy:.3f}%")
        scheduler.step()

    return errors

x_train, y_train, x_test, y_test = local_datasets.load_mnist(
    "../data/mnist/train-images.idx3-ubyte",
    "../data/mnist/train-labels.idx1-ubyte",
    "../data/mnist/t10k-images.idx3-ubyte",
    "../data/mnist/t10k-labels.idx1-ubyte"
)
data_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=500, shuffle=True)
epochs = 50

dropout_model = Model(x_train.shape[1], 80, y_train.shape[1], 8, True)
dropout_errors = train_model(dropout_model, data_loader, x_test, y_test, epochs, 0.006)

regular_model = Model(x_train.shape[1], 80, y_train.shape[1], 8, False)
regular_errors = train_model(regular_model, data_loader, x_test, y_test, epochs, 0.001)

fig, ax = plt.subplots()
x_axis = np.arange(0, epochs, 1)
ax.plot(x_axis, regular_errors, "r-", label="Without dropout")
ax.plot(x_axis, dropout_errors, "b-", label="With dropout")
ax.set_xlabel("Epoch"); ax.set_ylabel("Error (%)")
ax.set_xlim(0, epochs); ax.set_ylim(0, 100)
plt.legend()
plt.show()