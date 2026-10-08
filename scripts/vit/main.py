import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, random_split
import torchvision.datasets as datasets
from torchvision.transforms import v2
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import copy, humanize, math

def log_training_progress(batch_idx, epoch, total_epochs, loss, bar_length, loader):
  num_bars = (batch_idx + 1) * bar_length / len(loader)
  progress_bar = '█' * int(num_bars) + ' ' * int(bar_length - num_bars)
  end = "\n" if (batch_idx + 1) == len(loader) else "\r"
  info = f"Batch {batch_idx + 1} / {len(loader)} | Epoch {epoch + 1} / {total_epochs}"
  print(f"{info} | [{progress_bar}] | Loss: {loss:.3f} \033[K", end=end, flush=True)

def plot_loss_curves(train_losses, val_losses, output_path):
  train_losses = torch.tensor(train_losses)
  val_losses = torch.tensor(val_losses)

  fig, ax = plt.subplots()
  formatter = ScalarFormatter()
  formatter.set_scientific(False)

  ax.set_title("Loss Curves")
  ax.set_xlabel("Epochs")
  ax.set_ylabel("Loss")
  ax.set_yscale("log")
  ax.yaxis.set_major_formatter(formatter)

  ax.plot(train_losses, "b-")
  ax.plot(val_losses, "o-")

  stats = f"min: {train_losses.min().item()}, max: {train_losses.max().item()}, " + \
          f"mean: {train_losses.mean().item()}, std: {train_losses.std().item()}, "
  ax.text(0.5, -0.18, f"Train loss | {stats}",
          transform=ax.transAxes, ha="center", va="top", fontsize=10)

  stats = f"min: {val_losses.min().item()}, max: {val_losses.max().item()}, " + \
          f"mean: {val_losses.mean().item()}, std: {val_losses.std().item()}, "
  ax.text(0.5, -0.36, f"Validation loss | {stats}",
          transform=ax.transAxes, ha="center", va="top", fontsize=10)

  fig.tight_layout()
  fig.savefig(output_path, bbox_inches="tight")
  plt.show()

def custom_lr_scheduler(step, warmup, total):
  if step < warmup:
    return float(step) / float(max(1, warmup))

  # Cosine decay phase
  progress = float(step - warmup) / float(max(1, total - warmup))
  progress = min(1.0, max(0.0, progress))
  return 0.5 * (1.0 + math.cos(math.pi * progress))

def patchify_img(img_batch, B, C, P, G):
  # Convert image into a sequence of patches
  patches = img_batch.unfold(2, P, P).unfold(3, P, P)
  patches = patches.contiguous().view(B, C, G * G, P, P)
  patches = patches.permute(0, 2, 3, 4, 1)
  patches = patches.flatten(start_dim=2, end_dim=4)
  return patches

def init_weights(module):
  if isinstance(module, nn.Linear):
    nn.init.kaiming_uniform_(module.weight)
    if module.bias is not None:
      nn.init.zeros_(module.bias)

class ViT(nn.Module):
  def __init__(self, patch_size, grid_size, embed_dim,
               layers, num_heads, num_classes):
    super().__init__()
    size = 3 * patch_size * patch_size
    elements = grid_size * grid_size + 1

    self.embed_filters = nn.Linear(size, embed_dim, bias=False)
    self.class_token = nn.Parameter(torch.zeros(embed_dim))
    self.pos_embed = nn.Parameter(torch.zeros(elements, embed_dim))
    nn.init.normal_(self.class_token, std=0.02)
    nn.init.normal_(self.pos_embed, std=0.02)

    self.layers = nn.ModuleList([
      nn.ModuleList([
        nn.LayerNorm(embed_dim),
        nn.MultiheadAttention(embed_dim, num_heads, batch_first=True),
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
    embeddings = self.embed_filters(x)
    tok = self.class_token.unsqueeze(0).unsqueeze(0)
    tok = tok.expand(x.shape[0], -1, -1)
    tokens = torch.cat((tok, embeddings), dim=1)
    z = tokens + self.pos_embed

    for layer in self.layers:
      prev_norm, msa, msa_norm, mlp = layer
      n = prev_norm(z)
      z1, _ = msa(n, n, n)
      z1 = z1 + z
      z = mlp(msa_norm(z1)) + z1

    class_token = self.class_norm(z[:, 0])
    return self.class_proj(class_token)

class Harness:
  def __init__(self, hyperparams, img_size):
    self.h = hyperparams
    self.patches_per_side = img_size // self.h["patch_size"]

    self.model = ViT(
      self.h["patch_size"],
      self.patches_per_side,
      self.h["embedding_dim"],
      self.h["layers"],
      self.h["num_heads"],
      self.h["num_classes"],
    ).apply(init_weights).to(device)
    self.num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    self.optimizer = AdamW(self.model.parameters(),
                           lr=self.h["learning_rate"],
                           betas=self.h["betas"])
    self.criterion = nn.CrossEntropyLoss()

  def run_model(self, epoch, train, loader, scheduler=None):
    losses = []
    num_correct = 0

    for batch, (images, labels) in enumerate(loader):
      # Images shape: (B, C, H, W), Patches shape: (B, N, P, P, C)
      images, labels = images.to(device), labels.to(device)
      patches = patchify_img(
        images, images.shape[0],
        images.shape[1], self.h["patch_size"],
        self.patches_per_side
      )

      prediction = self.model(patches)
      predicted_labels = prediction.argmax(dim=-1)
      loss = self.criterion(prediction, labels)

      log_training_progress(batch, epoch, self.h["epochs"], loss.item(), 20, loader)
      losses.append(loss.item())
      num_correct += (predicted_labels == labels).sum().item()

      if train:
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=False)
        if scheduler is not None:
          scheduler.step()

    return losses, num_correct

torch.manual_seed(67)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hyperparams = {
  "epochs": 25,
  "warmup_epochs": 3,
  "batch_size": 64,
  "learning_rate": 1e-3,
  "betas": [0.9, 0.999],
  "embedding_dim": 256,
  "patch_size": 7,
  "num_heads": 8,
  "layers": 12,
  "num_classes": 10,
}

# NOTE: This is not grid search. Look into Optuna to understand how it does its hyperparameter sweeps
sweep_tweaks = {
  "epochs": [10, 25, 50, 100],
  "embedding_dim": [256, 384, 512],
  "num_heads": [4, 8, 16],
  "layers": [12, 18, 24],
  "patch_size": [4, 7]
}

p = v2.Compose([ v2.RGB(), v2.ToImage(), v2.ToDtype(torch.float32, scale=True) ])
test_set = datasets.MNIST(root=".cache/mnist", train=False, download=True, transform=p)
dataset = datasets.MNIST(root=".cache/mnist", train=True, download=True, transform=p)
train_set, val_set = random_split(dataset, [0.8, 0.2])
img_size = train_set[0][0].shape[-1]

train_loader = DataLoader(train_set, shuffle=True, batch_size=hyperparams["batch_size"])
val_loader = DataLoader(val_set, shuffle=False, batch_size=hyperparams["batch_size"])
test_loader = DataLoader(test_set, shuffle=False, batch_size=hyperparams["batch_size"])

# Test different hyperparameters
for key in sweep_tweaks:
  hp = copy.deepcopy(hyperparams)

  for v in sweep_tweaks[key]:
    print(f"Sweep: {key} = {v}")
    hp[key] = v
    if key == "epochs":
      hp["warmup_epochs"] = int(v * 0.1)

    harness = Harness(hp, img_size)
    ts = hp["epochs"] * len(train_loader)
    ws = hp["warmup_epochs"] * len(train_loader)
    scheduler = LambdaLR(harness.optimizer,
                        lambda step: custom_lr_scheduler(step, ws, ts))
    print(f"Training a {humanize.metric(harness.num_params)} model using {device}")
    train_losses, val_losses = [], []

    # Training loop
    for epoch in range(hp["epochs"]):
      harness.model.train()
      print("Training...")
      with torch.enable_grad():
        batch_losses, _ = harness.run_model(epoch, True, train_loader, scheduler=scheduler)
        train_losses.append(sum(batch_losses) / len(batch_losses))

      print("Validating...")
      harness.model.eval()
      with torch.no_grad():
        batch_losses, _ = harness.run_model(epoch, False, val_loader)
        val_losses.append(sum(batch_losses) / len(batch_losses))

    plot_loss_curves(train_losses, val_losses, f"curves_{key}_{v}.png")

    # Test model accuracy
    print("Testing...")
    harness.model.eval()
    with torch.no_grad():
      _, num_correct = harness.run_model(0, False, test_loader)
      accuracy = 100.0 * num_correct / len(test_loader.dataset)
      print(f"Top 1 test accuracy: {accuracy:.2f}%")

    del harness
    torch.cuda.empty_cache()
