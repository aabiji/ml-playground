import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2
import matplotlib.pyplot as plt

def plot_training_progress(num_steps, total_steps, loss, bar_length):
  num_bars = float(num_steps) * bar_length / (total_steps - 1)
  progress_bar = '█' * int(num_bars) + ' ' * int(bar_length - num_bars)
  end = "\n" if num_bars == bar_length else "\r"
  steps_count = f"{num_steps + 1} / {total_steps} steps"
  print(f"{steps_count} | [{progress_bar}] | Loss: {loss} \033[K", end=end, flush=True)

def patchify_img(img_batch, B, C, P, G):
  # Convert image into a sequence of patches
  patches = img_batch.unfold(2, P, P).unfold(3, P, P)
  patches = patches.contiguous().view(B, C, G ** 2, P, P)
  patches = patches.permute(0, 2, 3, 4, 1)
  patches = patches.flatten(start_dim=2, end_dim=4)
  return patches

"""
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

fig, ax = plt.subplots()
ax.axis("off")
img_size = test_dataset.data[0][0].shape[0]
ax_img = ax.imshow(torch.zeros((img_size, img_size, 3)))

fig.canvas.mpl_connect(
  "key_press_event",
  lambda ev: change_img_view(ev, fig, ax, ax_img, test_loader))
plt.show()
"""

class MSA(nn.Module):
  def __init__(self, attn_heads, embed_dim):
    super().__init__()
    self.Q_proj = nn.Parameter(torch.rand(embed_dim, embed_dim))
    self.K_proj = nn.Parameter(torch.rand(embed_dim, embed_dim))
    self.V_proj = nn.Parameter(torch.rand(embed_dim, embed_dim))
    self.O_proj = nn.Parameter(torch.rand(embed_dim, embed_dim))
    self.attn_softmax = nn.Softmax(dim=-1)
    self.H = attn_heads

  # Input and output shape: (B, N + 1, D)
  # D = embedding dimension, H = number of attention heads
  def forward(self, x):
    B, N, D = x.shape
    sqrt_dmodel = 1 / torch.sqrt(torch.tensor(D // self.H)).item()

    Q = (x @ self.Q_proj).view((B, N, self.H, D // self.H)).permute(0, 2, 1, 3)
    K = (x @ self.K_proj).view((B, N, self.H, D // self.H)).permute(0, 2, 1, 3)
    V = (x @ self.V_proj).view((B, N, self.H, D // self.H)).permute(0, 2, 1, 3)

    scores = (Q @ K.permute(0, 1, 3, 2)) * sqrt_dmodel
    output = self.attn_softmax(scores) @ V
    output = output.permute(0, 2, 1, 3).reshape(B, N, D)
    return output @ self.O_proj

class ViT(nn.Module):
  def __init__(self, patch_size, grid_size, channels, embed_dim,
               layers, attn_heads, num_classes):
    super().__init__()
    size = channels * patch_size * patch_size
    elements = grid_size * grid_size + 1

    self.class_token = nn.Parameter(torch.zeros(embed_dim))
    self.embed_filters = nn.Parameter(torch.zeros(size, embed_dim))
    self.pos_embed = nn.Parameter(torch.zeros(elements, embed_dim))

    self.layers = nn.ModuleList([
      nn.ModuleList([
        nn.LayerNorm(embed_dim),
        MSA(attn_heads, embed_dim),
        nn.LayerNorm(embed_dim),
        nn.Sequential(
          nn.Linear(embed_dim, embed_dim * 4),
          nn.GELU(),
          nn.Linear(embed_dim * 4, embed_dim)),
      ])
      for _ in range(layers)
    ])

    self.class_norm = nn.LayerNorm(embed_dim)
    self.class_proj = nn.Linear(embed_dim, num_classes)

  # P = patch size, N = patch count, C = channel count
  # B = batch size, O = number of output classes
  # Input shape: (B, N, P * P * C), output shape: (B, O)
  def forward(self, x):
    embeddings = x @ self.embed_filters
    B, N, D = embeddings.shape

    tokens = torch.zeros(B, N + 1, D)
    tokens[:, 0] = self.class_token
    tokens[:, 1:] = embeddings
    z = tokens + self.pos_embed

    for layer in self.layers:
      prev_norm, msa, msa_norm, mlp = layer
      z1 = msa(prev_norm(z)) + z
      z = mlp(msa_norm(z1)) + z1

    class_token = self.class_norm(z[:, 0])
    return self.class_proj(class_token)

torch.manual_seed(67)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hp = {
  "batch_size": 64, "embedding_dim": 256, "attn_heads": 8,
  "layers": 8, "num_classes": 10, "img_size": 32, "patch_size": 8,
}

load_pipeline = v2.Compose([
  v2.RGB(), v2.ToImage(),
  v2.ToDtype(torch.float32, scale=True), v2.Resize(size=(32, 32)),
  v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # [0, 255] -> [-1, 1]
])

augment_pipeline = v2.Compose([
  v2.RGB(), v2.ToImage(),
  v2.ToDtype(torch.float32, scale=True), v2.Resize(size=(32, 32)),
  v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # [0, 255] -> [-1, 1]
  v2.RandomHorizontalFlip(),
  v2.RandomRotation(90),
  v2.RandomErasing(p=0.3),
])

classes = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
train_dataset = CIFAR10(root=".cache/train/cifar10", train=True,
                        download=True, transform=augment_pipeline)
train_loader = DataLoader(train_dataset, batch_size=hp["batch_size"], shuffle=True)

images, labels = next(iter(train_loader)) # (B, C, H, W)
images, labels = images.to(device), labels.to(device)

patches = patchify_img(
  images, images.shape[0], images.shape[1], hp["patch_size"], hp["img_size"] // hp["patch_size"])

model = ViT(
  hp["patch_size"], hp["img_size"] // hp["patch_size"], images.shape[1],
  hp["embedding_dim"], hp["layers"], hp["attn_heads"], hp["num_classes"]).to(device)

cls_embedding = model(patches)
print(cls_embedding.shape)
