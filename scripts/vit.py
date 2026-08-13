import torch
import torch.nn as nn
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import LinearLR

from datasets import load_dataset
from huggingface_hub import login

import matplotlib.pyplot as plt
import math


class ViT(nn.Module):
  def __init__(self, D, L, H):
    super().__init__()
    self.sqrt_dim = math.sqrt(D // H)
    self.num_layers, self.num_sublayers = L, 9

    self.embeddings = nn.ParameterList([
      l for l in [
        nn.Parameter(torch.rand(H, D, D // H)),
        nn.Parameter(torch.rand(H, D, D // H)),
        nn.Parameter(torch.rand(H, D, D // H)),
        nn.Parameter(torch.rand(D, D)),
      ] for _ in range(L)
    ])

    self.layers = nn.ModuleList([l for l in [
      nn.LayerNorm(D),
      nn.LayerNorm(D),
      nn.Linear(D, D * 4),
      nn.Linear(D * 4, D),
      nn.LayerNorm(D)
    ] for _ in range(L)])

  # MultiHead Self Attention, where Q = K = V
  def MSA(self, X, Q_embed, K_embed, V_embed, O_embed):
    # Project each attention head into a smaller vector space
    Q_proj = X.unsqueeze(1) @ Q_embed.unsqueeze(0)
    K_proj = X.unsqueeze(1) @ K_embed.unsqueeze(0)
    V_proj = X.unsqueeze(1) @ V_embed.unsqueeze(0)

    # Self-attention for each attention head
    scores = torch.softmax(Q_proj @ K_proj.T / self.sqrt_dim, dim=-1)

    # Concatenate the scaled values (H x N x D_k) -> (H x N * D_k)
    concat = torch.permute(scores @ V_proj, (1, 0, 2)).reshape(-1)

    # Linearly project back into the original vector space
    return concat @ O_embed

  def forward(self, x):
    for l in range(self.num_layers):
      Q_embed, K_embed, V_embed, O_embed = self.embeddings[l * 4: (l + 1) * 4]
      norm1, norm2, linear1, linear2, norm3 = self.layers[l * 5: (l + 1) * 5]

      msa_out = self.MSA(norm1(x), Q_embed, K_embed, V_embed, O_embed)
      z = msa_out + x

      hidden = linear1(norm2(z))
      mlp_out = linear2(hidden)
      x = norm3(mlp_out + z)

    return x


def run_model(model, loader, training, device):
  if training:
    model.train()
  else:
    model.eval()

  context = torch.enable_grad() if training else torch.no_grad()
  with context:
    for batch in loader:
      images, labels = batch["image"].to(device), batch["label"].to(device)

      prediction = model(images)
      loss = criterion(prediction, labels)
      if training:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ViT-8/16
img_size, patch_size = 224, 16
num_tokens = 1 + img_size / patch_size
model = ViT(768, 12, 12).to(device)
warmup_steps = 10000

criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=8e-4, betas=(0.9, 0.999), weight_decay=0.1)
scheduler = LinearLR(optimizer, total_iters=warmup_steps)

transform = None
def process_transform(sample):
  global transform
  if transform is None:
    transform = v2.Compose([
      v2.RandomResizedCrop(size=(224, 224), antialias=True), # Note: size=384 for fine-tuning
      v2.RandomHorizontalFlip(p=0.5),
      v2.ToImage(),
      v2.ToDtype(dtype=torch.float32, scale=True),
      #v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # imagenet-1k mean and std
    ])
  return { "augmented": transform(sample["image"]).clone(), "label": sample["label"] }

def main():
  data_stream = load_dataset("ILSVRC/imagenet-1k", split="train", streaming=True)
  data_stream = data_stream.map(process_transform, remove_columns=["image"])
  train_loader = DataLoader(data_stream, batch_size=32, num_workers=4) # type: ignore[arg-type]

  for _, batch in enumerate(train_loader, start=1):
    images, labels = batch["augmented"], batch["label"]
    plt.imshow(images[0].permute(*torch.arange(images[0].ndim - 1, -1, -1)))

if __name__ == "__main__":
  main()

# TODO: Visualize an image using matplotlib
# TODO: pretrain model on ImageNet-1k
# TODO: Visualize the attention maps
# TODO: fine tune on another dataset
# TODO: evaluate fine tuned performance