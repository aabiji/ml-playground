%matplotlib inline
import matplotlib.pyplot as plt
from IPython.display import clear_output
from IPython import display

import math
import torch
import torch.nn as nn
from torchvision.transforms import v2
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import LinearLR

class Encoder(nn.Module):
  def __init__(self, D, H):
    super().__init__()
    self.norm1 = nn.LayerNorm(D)
    self.Q_embed = nn.Linear(D, D, bias=False)
    self.K_embed = nn.Linear(D, D, bias=False)
    self.V_embed = nn.Linear(D, D, bias=False)
    self.O_embed = nn.Linear(D, D, bias=False)

    self.norm2 = nn.LayerNorm(D)
    self.mlp_hidden = nn.Linear(D, D * 4)
    self.mlp_act = nn.GELU()
    self.mlp_out = nn.Linear(D * 4, D)

    self.norm3 = nn.LayerNorm(D)
    self.h = H

  def forward(self, x):
    # MultiHead Self-Attention, where Q = K = V
    x = self.norm1(x)

    # Project each attention headnum_tokens into a smaller vector space
    b, n, d = x.shape[0], x.shape[1], x.shape[2]
    Q_proj = self.Q_embed(x).view(b, n, self.h, d // self.h)
    K_proj = self.Q_embed(x).view(b, n, self.h, d // self.h)
    V_proj = self.Q_embed(x).view(b, n, self.h, d // self.h)

    # Self-attention for each attention head
    scores = torch.softmax(Q_proj @ K_proj.T / self.sqrt_dim, dim=-1)

    # Concatenate the scaled values (H x N x D_k) -> (H x N * D_k)
    concat = torch.permute(scores @ V_proj, (1, 0, 2)).reshape(-1)

    # Linearly project back into the original vector space and add a residual connection
    z = self.O_embed(concat) + x

    # Feed forward network
    hidden = self.mlp_hidden(self.norm2(z))
    output = self.mlp_out(self.mlp_act(hidden))
    return self.norm3(output + z)

class ViT(nn.Module):
  def __init__(self, config):
    super().__init__()
    D, H = config["embedding_dim"], config["attn_heads"]
    self.L, self.P = config["encoder_layers"], config["patch_size"]

    grid_size = config["image_size"] // config["patch_size"]
    num_tokens = 1 + grid_size * grid_size
    self.sqrt_dim = math.sqrt(D // H)

    self.class_embedding = nn.Parameter(torch.rand(1, 1, D))
    self.patch_embedding = nn.Parameter(torch.rand(num_tokens, D))
    self.pos_embedding = nn.Parameter(torch.rand(num_tokens, D))

    self.encoders = nn.ModuleList([Encoder(D, H) for _ in range(self.L)])

    # NOTE: these layers are only for pretraining, see the paper for more details
    self.head_hidden = nn.Linear(D, D * 2)
    self.head_act = nn.GELU()
    self.head_out = nn.Linear(D * 2, config["out_classes"])

  def tokenize_images(self, x):
    # Split the image into a series of patches in row orderx
    # [B, C, H, W] -> [B, C, H / P, W / P, P, P]
    patches = x.unfold(2, self.P, self.P).unfold(3, self.P, self.P)

    # Flatten the patch grid into a flat list and prepend the [class] token
    patches = patches.permute(0, 2, 3, 1, 4, 5)
    B, m, c, p = \
      patches.shape[0], patches.shape[1], patches.shape[3], patches.shape[4]
    patches = patches.reshape(B, m * m, c * p * p)
    tokens = torch.cat([self.class_embedding.expand(B, -1, -1), patches], dim=1)

    # Convert the tokens into embeddings and add their positional embeddings
    return tokens @ self.patch_embedding + self.pos_embedding

  def forward(self, image_batch):
    x = self.tokenize_images(image_batch)

    for l in range(self.L):
      x = self.encoders[l](x)

    print(x.shape)

    # Run the classification head on the output [class] token
    y = self.head_hidden(x[:, 0])
    return self.head_out(self.head_act(y))

def plot_stats(fig, axs, errors=None, losses=None):
  clear_output(wait=True)
  for ax in axs:
      ax.clear()
  ax_loss, ax_error = axs[0], axs[1]

  if losses is not None and len(losses) > 0:
    ax_loss.plot(losses, "b-")
    ax_loss.autoscale_view(scalex=True, scaley=True)
    ax_loss.set_title(f"Mean Validation Loss: {losses[-1]:.2f}")
    ax_loss.set_xlabel("Iterations")
    ax_loss.set_ylabel("Loss")

  if errors is not None and len(errors) > 0:
    ax_error.plot(errors, "r-")
    ax_error.autoscale_view(scalex=True, scaley=True)
    ax_error.set_title(f"Mean Validation Error {errors[-1]:.2f}")
    ax_error.set_xlabel("Iterations")
    ax_error.set_ylabel("Error")

  fig.tight_layout() # Prevents overlapping labels
  display.display(fig)

def main(root_dir, config):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  augmentation = v2.Compose([
    v2.ToImage(),
    v2.RandomResizedCrop(size=(config["image_size"], config["image_size"]), antialias=True),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ToDtype(dtype=torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # imagenet-1k mean and std
  ])

  train_dataset = datasets.ImageFolder(root=root_dir, transform=augmentation)
  train_loader = DataLoader(train_dataset, batch_size=32,
                            shuffle=True, num_workers=4, pin_memory=True)

  model = ViT(config).to(device)
  criterion = nn.CrossEntropyLoss()
  optimizer = Adam(model.parameters(), lr=8e-4, betas=(0.9, 0.999), weight_decay=0.1)
  scheduler = LinearLR(optimizer, total_iters=10000)

  losses, errors = [], []
  fig, axs = plt.subplots(1, 2, figsize=(12, 5))

  #for _ in range(epochs):
  model.train()
  with torch.enable_grad():
    for images, labels in train_loader:
      images, labels = images.to(device), labels.to(device)
      prediction = model(images)

      print(prediction.shape, labels.shape)

      loss = criterion(prediction, labels)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

    scheduler.step()

if __name__ == "__main__":
  config = { "embedding_dim": 768, "attn_heads": 12, "encoder_layers": 12,
             "image_size": 224, "patch_size": 16, "out_classes": 200, "epochs": 7 }
  main("/kaggle/input/datasets/sautkin/imagenet1k0", config)

# TODO: pretrain model on ImageNet-1k
# TODO: Visualize the attention maps
# TODO: fine tune on another dataset
# TODO: evaluate fine tuned performance