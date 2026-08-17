import matplotlib.pyplot as plt
from IPython.display import clear_output
from IPython import display

import torch
import torch.nn as nn
from torchvision.transforms import v2
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import LinearLR
from torch.amp import autocast, GradScaler

# Pre train on a large imagenet dataset, save the weights, load the weights, replace the classification head and fine tune on cifar10 dataset.
# --> Compare test performance on self supervised pretraining vs supervised pretraining.
# --> Compare test performance on cifar10 when pretrained on imagenet, when not pretrained on anything and when pretrained on cifar10.
# What is Xavier initialization or He initialization or Layer Normalization or another init schemes supposed to be doing mathmatically?
# What is GELU? Understand the CDF to better intuit where the percentages on the z-score table are coming from.
# Why Warmup the Learning Rate? Underlying Mechanisms and Improvements

class Encoder(nn.Module):
  def __init__(self, d_e, d_h, h):
    super().__init__()
    self.h = h
    self.sqrt_dim = torch.sqrt(torch.tensor(d_e // h)).item()

    self.norm1 = nn.LayerNorm(d_e)
    self.Q_proj = nn.Linear(d_e, d_e, bias=False)
    self.K_proj = nn.Linear(d_e, d_e, bias=False)
    self.V_proj = nn.Linear(d_e, d_e, bias=False)
    self.O_proj = nn.Linear(d_e, d_e, bias=False)

    self.norm2 = nn.LayerNorm(d_e)
    self.mlp_hidden = nn.Linear(d_e, d_h)
    self.mlp_act = nn.GELU()
    self.mlp_out = nn.Linear(d_h, d_e)

  def forward(self, x):
    # MultiHead Self-Attention
    x = self.norm1(x)

    # Project each attention headnum_tokens into a smaller vector space
    b, n, d_e = x.shape[0], x.shape[1], x.shape[2]
    q = self.Q_proj(x).view(b, n, self.h, d_e // self.h).transpose(1, 2)
    k = self.K_proj(x).view(b, n, self.h, d_e // self.h).transpose(1, 2)
    v = self.V_proj(x).view(b, n, self.h, d_e // self.h).transpose(1, 2)

    # Self-attention for each attention head
    k_T = k.transpose(-2, -1)
    scores = torch.softmax(q @ k_T / self.sqrt_dim, dim=-1)

    # Concatenate the scaled values (H x N x D_k) -> (H x N * D_k)
    concat = (scores @ v).transpose(1, 2).contiguous().view(b, n, d_e)

    # Linearly project back into the original vector space and add a residual connection
    z = self.O_proj(concat) + x

    # Feed forward network
    hidden = self.mlp_hidden(self.norm2(z))
    output = self.mlp_out(self.mlp_act(hidden))
    return output + z


class ViT(nn.Module):
  def __init__(self, config):
    super().__init__()
    d_e, d_h, h = config["embedding_dim"], config["hidden_dim"], config["attn_heads"]
    self.layers, self.P = config["encoder_layers"], config["patch_size"]

    grid_size = config["image_size"] // config["patch_size"]
    num_tokens = 1 + grid_size * grid_size

    self.patch_proj = nn.Linear(3 * self.P * self.P, d_e)
    self.class_embedding = nn.Parameter(torch.zeros(1, 1, d_e))
    self.pos_embedding = nn.Parameter(torch.zeros(1, num_tokens, d_e))

    nn.init.trunc_normal_(self.class_embedding, std=0.02)
    nn.init.trunc_normal_(self.pos_embedding, std=0.02)

    self.encoders = nn.ModuleList([Encoder(d_e, d_h, h) for _ in range(self.layers)])

    # NOTE: these layers are only for pretraining, see the paper for more details
    self.out_norm = nn.LayerNorm(d_e)
    self.head_hidden = nn.Linear(d_e, d_e * 2)
    self.head_act = nn.GELU()
    self.head_out = nn.Linear(d_e * 2, config["out_classes"])

  def tokenize_images(self, x):
    # Split the image into a series of patches in row orderx
    # [B, C, H, W] -> [B, C, H / P, W / P, P, P]
    patches = x.unfold(2, self.P, self.P).unfold(3, self.P, self.P)

    # Flatten the patch grid into a flat list and project the patches into embeddings
    patches = patches.permute(0, 2, 3, 1, 4, 5)
    B, m, c, p = \
      patches.shape[0], patches.shape[1], patches.shape[3], patches.shape[4]
    patch_tokens = self.patch_proj(patches.reshape(B, m * m, c * p * p))

    # Add the spetial [class] embedding
    class_token = self.class_embedding.expand(B, -1, -1)
    tokens = torch.cat([class_token, patch_tokens], dim=1)

    # Convert the tokens into embeddings and add their positional embeddings
    return tokens + self.pos_embedding

  def forward(self, image_batch):
    x = self.tokenize_images(image_batch)

    for l in range(self.layers):
      x = self.encoders[l](x)

    # Run the classification head off of the output [class] token
    class_token = self.out_norm(x[:, 0])
    y = self.head_hidden(class_token)
    return self.head_out(self.head_act(y))


def plot_stats(fig, ax_loss, ax_error, losses, errors, epoch):
  clear_output(wait=True)

  if losses is not None and len(losses) > 0:
    ax_loss.clear()
    ax_loss.plot(losses, "b-")
    ax_loss.autoscale_view(scalex=True, scaley=True)
    ax_loss.set_title(f"Mean Training Loss for Epoch {epoch}: {losses[-1]:.2f}")
    ax_loss.set_xlabel("Iterations")
    ax_loss.set_ylabel("Loss")

  if errors is not None and len(errors) > 0:
    ax_error.clear()
    ax_error.plot(errors, "r-")
    ax_error.autoscale_view(scalex=True, scaley=True)
    ax_error.set_title(f"Mean Validation Error for Epoch {epoch}: {errors[-1]:.2f}")
    ax_error.set_xlabel("Iterations")
    ax_error.set_ylabel("Error")

  fig.tight_layout() # Prevents overlapping labels
  display.display(fig)


def mean_error(model, loader, device, norm):
  num_correct, total = 0, 0

  with torch.inference_mode():
    model.eval()
    for images, labels in loader:
      images = norm(images.to(device))
      labels = labels.to(device)

      with autocast(device.type):
          answers = model(images).argmax(dim=1)
          num_correct += (answers == labels).sum().item()
          total += labels.shape[0]

  return 100 - 100 * num_correct / total


def main(train_dir, val_dir, config):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  scaler = GradScaler(device.type)

  imagenet_mean, imagenet_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
  cpu_transforms = v2.Compose([ v2.ToImage(), v2.ToDtype(dtype=torch.float32, scale=True) ])
  gpu_augmentations = nn.Sequential(
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomResizedCrop(size=(config["image_size"], config["image_size"]), antialias=True),
    v2.Normalize(mean=imagenet_mean, std=imagenet_std)
  )

  train_dataset = datasets.ImageFolder(root=train_dir, transform=cpu_transforms)
  train_loader = DataLoader(train_dataset, batch_size=32, persistent_workers=True,
                            shuffle=True, num_workers=4, pin_memory=True)

  val_dataset = datasets.ImageFolder(root=val_dir, transform=cpu_transforms)
  val_loader = DataLoader(val_dataset, batch_size=32, persistent_workers=True,
                          num_workers=4, pin_memory=True)

  model = ViT(config).to(device)
  criterion = nn.CrossEntropyLoss()
  optimizer = Adam(model.parameters(), lr=8e-4, betas=(0.9, 0.999), weight_decay=0.1)
  scheduler = LinearLR(optimizer)

  losses, errors = [], []
  fig, (ax_loss, ax_error) = plt.subplots(1, 2, figsize=(12, 5))

  print("Training started...")
  for epoch in range(config["epochs"]):
    model.train()
    total_loss = 0

    with torch.enable_grad():
      for images, labels in train_loader:
        images = gpu_augmentations(images.to(device))
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        with autocast(device.type):
            loss = criterion(model(images), labels)
            total_loss += loss.item()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

    errors.append(mean_error(model, val_loader, device, gpu_augmentations))
    losses.append(total_loss / len(train_loader))
    plot_stats(fig, ax_loss, ax_error, losses, errors, epoch)

  fig.savefig("pretrain_curves.png", dpi=300, bbox_inches="tight")
  torch.save(model.state_dict(), "vit_pretrain_weights.pth")


if __name__ == "__main__":
  main(
    "/kaggle/input/datasets/akash2sharma/tiny-imagenet/tiny-imagenet-200/train",
    "/kaggle/input/datasets/akash2sharma/tiny-imagenet/tiny-imagenet-200/val",
    {
      "embedding_dim": 324,
      "hidden_dim": 1536,
      "attn_heads": 12,
      "encoder_layers": 12,
      "image_size": 224,
      "patch_size": 16,
      "out_classes": 200,
      "epochs": 7
    })