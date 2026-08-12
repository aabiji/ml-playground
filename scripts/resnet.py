import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
from IPython.display import clear_output
from IPython import display
import local_datasets


def random_pad_crop(img_batch, pad, flip_percentage):
    padded = F.pad(img_batch, (pad, pad, pad, pad), mode="constant", value=0)

    img_size = (img_batch.shape[-2], img_batch.shape[-1])
    padded_size = (padded.shape[-2], padded.shape[-1])

    for sample in range(img_batch.shape[0]):
        xy = (
            torch.randint(padded_size[0] - img_size[0], ()).item(),
            torch.randint(padded_size[1] - img_size[1], ()).item())

        img_batch[sample] = padded[sample, :, xy[0]:xy[0]+img_size[0], xy[1]:xy[1]+img_size[1]]

        flip = torch.randint(100, ()).item() < (100 * flip_percentage)
        if flip:
            img_batch[sample] = torch.flip(img_batch[sample], dims=[2])


def random_cutout(img_batch, percentage, num_cuts, cut_size):
    img_size = (img_batch.shape[-2], img_batch.shape[-1])

    for sample in range(img_batch.shape[0]):
        cutout = torch.randint(100, ()).item() < (100 * percentage)
        if not cutout:
            continue

        for _ in range(num_cuts):
            xy = (
                torch.randint(img_size[0] - cut_size, ()).item(),
                torch.randint(img_size[1] - cut_size, ()).item())
            img_batch[sample, :, xy[0]:xy[0]+cut_size, xy[1]:xy[1]+cut_size] = 0.0


def classification_error(prediction, actual):
    probabilities = torch.softmax(prediction, dim=1)
    answer = torch.argmax(probabilities, dim=1)
    num_correct = (answer == actual).sum().item()
    batch_size = answer.shape[0]
    return 100 * (batch_size - num_correct) / batch_size


def plot_stats(fig, ax, title, errors=None, losses=None):
    clear_output(wait=True)
    ax.clear()

    if losses is not None:
        ax.plot(losses, "b-", label=f"Loss {losses[-1]:.3f}")

    if errors is not None:
        ax.plot(errors, "r-", label=f"Error {errors[-1]:.2f}%")

    ax.autoscale_view(scalex=True, scaley=True)
    ax.set_title(title)
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Stats")
    plt.legend()
    display.display(fig)


def init_layer(layer):
    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
        nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)

    if isinstance(layer, nn.BatchNorm2d):
        nn.init.zeros_(layer.bias)
        nn.init.ones_(layer.weight)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, downsample):
        super().__init__()

        stride1 = 2 if downsample else 1
        out_channels = in_channels * 2 if downsample else in_channels

        self.layers = nn.ModuleList([
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, 3, stride=stride1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
        ])

        self.projection = None
        self.projection_norm = None
        if downsample:
            self.projection = nn.Conv2d(in_channels, out_channels, 1, stride=stride1, padding=0)
            self.projection_norm = nn.BatchNorm2d(out_channels)
            init_layer(self.projection)
            init_layer(self.projection_norm)

        for layer in self.layers:
            init_layer(layer)

    def forward(self, x):
        prev_x = x.clone()
        for layer in self.layers:
            x = layer(x)

        if self.projection is not None and self.projection_norm is not None:
            prev_x = self.projection(prev_x)
            prev_x = self.projection_norm(prev_x)

        return x + prev_x

"""
Resnet architecture in the paper:
Convolution on 3x32x32 input, 16x3x3 kernel

2 blocks (in = 16x32x32, out = 16x32x32), 16x3x3 kernel, padding = 1, stride = 1
2 blocks (in = 16x32x32, out = 32x16x16), 32x3x3 kernel, padding = 1, stride = 2, 1
2 blocks (in = 32x16x16, out =   64x8x8), 64x3x3 kernel, padding = 1, stride = 2, 1

Global average pooling (64x8x8 -> 64x1)
10 activation linear + softmax

Each block:
Batchnorm
ReLU
Convolution
Batchnorm
ReLU
Convolution
Residual
* Projection = 1x1 convolution with a stride of 2 (second and third block) applied to input, followed by BatchNorm
"""
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        channels = 32
        self.in_conv = nn.Conv2d(3, channels, 3, stride=1, padding=1)

        self.blocks = nn.ModuleList()
        for i in range(4):
            downsample = i % 2 == 0
            self.blocks.append(ResidualBlock(channels, downsample))
            if downsample:
                channels *= 2

        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.out_linear = nn.Linear(128, 10)

        init_layer(self.in_conv)
        init_layer(self.out_linear)

    def forward(self, x):
        x = self.in_conv(x)
        for block in self.blocks:
            x = block(x)
        x = self.pooling(x)
        x = torch.flatten(x, 1)
        x = self.out_linear(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

batches, classes = local_datasets.load_cifar10(device)
training_batches, test_batch = batches[:5], batches[5]

model = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = SGD(model.parameters(), lr=0.05, weight_decay=1e-4, momentum=0.9)
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
batch_size, minibatch_size, epochs = 10000, 128, 50
minibatch_iters = math.ceil(batch_size / minibatch_size)

losses = []
errors = []
fig, ax = plt.subplots()

# Train model
model.train()
for epoch in range(epochs):
    for i, batch in enumerate(training_batches):
        all_imgs, all_labels = batch
        total_loss, total_error = 0, 0
        random_indexes = torch.randperm(batch_size).to(device)

        for j in range(0, batch_size, minibatch_size):
            idx = random_indexes[j:j+minibatch_size]
            imgs, labels = all_imgs[idx].clone(), all_labels[idx].clone()
            random_pad_crop(imgs, 4, 0.5)
            random_cutout(imgs, 0.5, 3, 3)

            prediction = model(imgs)
            loss = criterion(prediction, labels)
            total_loss += loss.item()
            total_error += classification_error(prediction, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        # Visualize the training loss curve in real time
        losses.append(total_loss / minibatch_iters)
        errors.append(total_error / minibatch_iters)

        plot_stats(fig, ax, "Training stats", errors=errors, losses=losses)
        print(f"Epoch {epoch+1}/{epochs}, Batch: {i+1}/5, LR: {scheduler.get_last_lr()[0]:.3f}")

    scheduler.step()

# Test performance
model.eval()
total_error = 0
for j in range(0, batch_size, minibatch_size):
    all_imgs, all_labels = test_batch
    imgs, labels = all_imgs[j:j+minibatch_size], all_labels[j:j+minibatch_size]
    prediction = model(imgs)
    total_error += classification_error(prediction, labels)

print(f"Mean test error: {(total_error / minibatch_size):.3f}%")
