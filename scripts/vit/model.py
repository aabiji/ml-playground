import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
import matplotlib.pyplot as plt

def plot_training_progress(num_steps, total_steps, loss, bar_length):
  num_bars = float(num_steps) * bar_length / (total_steps - 1)
  progress_bar = '█' * int(num_bars) + ' ' * int(bar_length - num_bars)
  end = "\n" if num_bars == bar_length else "\r"
  steps_count = f"{num_steps + 1} / {total_steps} steps"
  print(f"{steps_count} | [{progress_bar}] | Loss: {loss} \033[K", end=end, flush=True)

def change_img_view(event, fig, ax, ax_img, loader):
  if event.key != " ":
    return

  size = len(loader.dataset) # type: ignore
  i = torch.randint(0, size, (1, )).item()
  img = loader.dataset[i][0].permute(1, 2, 0)
  label = classes[loader.dataset[i][1]]

  ax_img.set_data((img + 1) * 0.5)
  ax.set_title(label)
  fig.canvas.draw_idle()

pipeline = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # [0, 255] -> [-1, 1]
])

classes = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
train_dataset = CIFAR10(root=".cache/train/cifar10", train=True, download=True, transform=pipeline)
test_dataset = CIFAR10(root=".cache/test/cifar10", train=False, download=True, transform=pipeline)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

fig, ax = plt.subplots()
ax.axis("off")
img_size = test_dataset.data[0][0].shape[0]
ax_img = ax.imshow(torch.zeros((img_size, img_size, 3)))

fig.canvas.mpl_connect(
  "key_press_event",
  lambda ev: change_img_view(ev, fig, ax, ax_img, test_loader))
plt.show()