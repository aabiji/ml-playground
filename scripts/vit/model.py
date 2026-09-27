import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
import matplotlib.pyplot as plt

"""
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

fig, ax = plt.subplots()
ax.axis("off")
img_size = test_dataset.data[0][0].shape[0]
ax_img = ax.imshow(torch.zeros((img_size, img_size, 3)))

fig.canvas.mpl_connect(
  "key_press_event",
  lambda ev: change_img_view(ev, fig, ax, ax_img, test_loader))
plt.show()

# TODO: how to initialize embeddings and parameters in this model
"""

class MSA(nn.Module):
  def __init__(self, attn_heads, embed_dim, dropout):
    super().__init__()
    self.Q_proj = nn.Parameter(torch.rand(embed_dim, embed_dim))
    self.K_proj = nn.Parameter(torch.rand(embed_dim, embed_dim))
    self.V_proj = nn.Parameter(torch.rand(embed_dim, embed_dim))
    self.O_proj = nn.Parameter(torch.rand(embed_dim, embed_dim))
    self.attn_softmax = nn.Softmax(dim=-1)
    self.attn_dropout = nn.Dropout(p=dropout)
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
    scores = self.attn_dropout(self.attn_softmax(scores))

    output = (scores @ V).permute(0, 2, 1, 3).reshape(B, N, D)
    return output @ self.O_proj


class ViT(nn.Module):
  def __init__(self, patch_size, grid_size, channels,
               embed_dim, mlp_dim, layers, attn_heads, dropout):
    super().__init__()
    size = channels * patch_size * patch_size
    count = grid_size * grid_size + 1 

    self.cls_token = nn.Parameter(torch.zeros(embed_dim))
    self.embed_filters = nn.Parameter(torch.zeros(size, embed_dim))
    self.pos_embed = nn.Parameter(torch.zeros(count, embed_dim))

    self.layers = nn.ModuleList([
      nn.ModuleList([
        nn.LayerNorm(embed_dim),
        MSA(attn_heads, embed_dim, dropout),
        nn.LayerNorm(embed_dim),
        nn.Sequential(
          nn.Linear(count * embed_dim, mlp_dim),
          nn.GELU(),
          nn.Linear(mlp_dim, count * embed_dim)),
      ])
      for _ in range(layers)
    ])

    self.cls_norm = nn.LayerNorm(embed_dim)

  # P = patch size, N = patch count, C = channel count, B = batch size
  # Input shape: (B, N, P * P * C), output shape: (B, D)
  def forward(self, x):
    embeddings = x @ self.embed_filters
    B, N, D = embeddings.shape

    tokens = torch.zeros(B, N + 1, D)
    tokens[:, 0] = self.cls_token
    tokens[:, 1:] = embeddings
    z = tokens + self.pos_embed
 
    # TODO: are all tokens flattened into one before being passed into mlp or is each token processed in parllel? parlell, right? 
    for layer in self.layers:
      prev_norm, msa, msa_norm, mlp = layer
      z1 = msa(prev_norm(z)) + z

      mlp_in = msa_norm(z1).flatten(start_dim=1)
      z = mlp(mlp_in) + z1

    return self.cls_norm(z[0])


hp = {
  "batch_size": 64, "patch_size": 8, "embedding_dim": 256, "mlp_dim": 512,
  "attn_heads": 8, "attn_dropout": 0.5, "layers": 8,
}

pipeline = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # [0, 255] -> [-1, 1]
])

classes = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
train_dataset = CIFAR10(root=".cache/train/cifar10", train=True, download=True, transform=pipeline)
train_loader = DataLoader(train_dataset, batch_size=hp["batch_size"], shuffle=True)

images, labels = next(iter(train_loader)) # (B, C, H, W)

B, C, H, _ = images.shape
P, G = hp["patch_size"], H // hp["patch_size"]

# Convert image into a sequence of patches
patches = images.unfold(2, P, P).unfold(3, P, P)
patches = patches.contiguous().view(B, C, G ** 2, P, P)
patches = patches.permute(0, 2, 3, 4, 1)
patches = patches.flatten(start_dim=2, end_dim=4)

model = ViT(P, G, C, hp["embedding_dim"], hp["mlp_dim"],
            hp["layers"], hp["attn_heads"], hp["attn_dropout"])
cls_embedding = model(patches)
print(cls_embedding.shape)
