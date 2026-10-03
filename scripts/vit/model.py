import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from torchvision.transforms import v2
import humanize, pathlib
import viz

class Hyperparams:
  def __init__(self):
    self.epochs = 50
    self.batch_size = 128
    self.learning_rate = 1e-3
    self.betas = [0.9, 0.999]
    self.label_smoothing = 0.1
    self.img_size = 96
    self.embedding_dim = 512
    self.patch_size = 16
    self.attn_heads = 8
    self.layers = 32
    self.class_dim = 128
    self.classes = ["airplane", "bird", "car", "cat", "deer", "dog", "horse", "monkey", "ship", "truck"]

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

class MSA(nn.Module):
  def __init__(self, attn_heads, embed_dim):
    super().__init__()
    self.Q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
    self.K_proj = nn.Linear(embed_dim, embed_dim, bias=False)
    self.V_proj = nn.Linear(embed_dim, embed_dim, bias=False)
    self.O_proj = nn.Linear(embed_dim, embed_dim, bias=False)
    self.attn_softmax = nn.Softmax(dim=-1)
    self.variance_scale = 1 / torch.sqrt(torch.tensor(embed_dim // attn_heads)).item()
    self.H = attn_heads

  # Input and output shape: (B, N + 1, D)
  # D = embedding dimension, H = number of attention heads
  def forward(self, x):
    B, N, D = x.shape
    Q = self.Q_proj(x).view((B, N, self.H, D // self.H)).permute(0, 2, 1, 3)
    K = self.K_proj(x).view((B, N, self.H, D // self.H)).permute(0, 2, 1, 3)
    V = self.V_proj(x).view((B, N, self.H, D // self.H)).permute(0, 2, 1, 3)

    scores = (Q @ K.permute(0, 1, 3, 2))
    norm = self.attn_softmax(scores * self.variance_scale)

    output = (norm @ V).permute(0, 2, 1, 3).reshape(B, N, D)
    return self.O_proj(output), norm

class ViT(nn.Module):
  def __init__(self, patch_size, grid_size, embed_dim,
               layers, attn_heads, class_dim, num_classes):
    super().__init__()
    size = 3 * patch_size * patch_size
    elements = grid_size * grid_size + 1

    self.embed_filters = nn.Linear(size, embed_dim, bias=False)
    self.class_token = nn.Parameter(torch.zeros(embed_dim))
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
    self.class_head = nn.Sequential(
      nn.Linear(embed_dim, class_dim),
      nn.Tanh(),
      nn.Linear(class_dim, num_classes)
    )

  # P = patch size, N = patch count, C = channel count
  # B = batch size, O = number of output classes
  # Input shape: (B, N, P * P * C), output shape: (B, O)
  def forward(self, x):
    layer_scores = []

    embeddings = self.embed_filters(x)
    tok = self.class_token.unsqueeze(0).unsqueeze(0)
    tok = tok.expand(x.shape[0], -1, -1)
    tokens = torch.cat((tok, embeddings), dim=1)
    z = tokens + self.pos_embed

    for layer in self.layers:
      prev_norm, msa, msa_norm, mlp = layer
      z1, scores = msa(prev_norm(z))
      z1 = z1 + z
      z = mlp(msa_norm(z1)) + z1
      layer_scores.append(scores)

    class_token = self.class_norm(z[:, 0])
    prediction = self.class_head(class_token)
    layer_scores = torch.stack(layer_scores, dim=0)
    return prediction, layer_scores

torch.manual_seed(67)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

h, out_folder = Hyperparams(), "trial7"
pathlib.Path(out_folder).mkdir(parents=True, exist_ok=True)

augment_pipeline = v2.Compose([
  v2.RGB(), v2.ToImage(),
  v2.ToDtype(torch.float32, scale=True),
  v2.Resize(size=(h.img_size, h.img_size)),
  v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # [0, 255] -> [-1, 1]
  v2.RandomHorizontalFlip(),
  v2.RandomRotation(90),
  v2.RandomErasing(p=0.3),
])

train_dataset = datasets.STL10(root=".cache/train/stl10", split="train",
                        download=True, transform=augment_pipeline)
train_loader = DataLoader(train_dataset, batch_size=h.batch_size, shuffle=True)

test_dataset = datasets.STL10(root=".cache/train/stl10", split="test",
                        download=True, transform=augment_pipeline)
test_loader = DataLoader(test_dataset, batch_size=h.batch_size, shuffle=True)

model = ViT(
  h.patch_size, h.img_size // h.patch_size,
  h.embedding_dim, h.layers, h.attn_heads,
  h.class_dim, len(h.classes)
).apply(init_weights).to(device)
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Training a {humanize.metric(trainable_params)} model using {device}")

optimizer = AdamW(model.parameters(), lr=h.learning_rate, betas=h.betas)
scheduler = CosineAnnealingLR(optimizer, T_max=h.epochs)
criterion = nn.CrossEntropyLoss(label_smoothing=h.label_smoothing)
losses = []

# Training loop
model.train()

with torch.enable_grad():
  for epoch in range(h.epochs):
    batch_losses = torch.zeros((len(train_loader), 1), device=device)

    for batch, (images, labels) in enumerate(train_loader):
      # Images shape: (B, C, H, W), Patches shape: (B, N, P, P, C)
      images, labels = images.to(device), labels.to(device)
      patches = patchify_img(
        images, images.shape[0],
        images.shape[1], h.patch_size,
        h.img_size // h.patch_size
      )

      prediction, scores = model(patches)
      loss = criterion(prediction, labels)

      batch_losses[batch] = loss.item()
      viz.log_training_progress(batch, epoch, h.epochs, loss.item(), 50, train_loader)

      loss.backward()
      optimizer.step()
      optimizer.zero_grad(set_to_none=False)

      # Visualize the change in attention scores as training progresses
      last_batch = epoch > 0 and batch == len(train_loader) - 1
      first_batch = epoch == 0 and batch == 0
      if first_batch or last_batch:
        pos_embeds = model.pos_embed.cpu().detach()
        viz.viz_model_features(prediction, patches, labels, scores, pos_embeds, h,
                              f"{out_folder}/scores_{epoch + 1}.png",
                              f"{out_folder}/patches_{epoch + 1}.png",
                              f"{out_folder}/pos_similarity_{epoch + 1}.png")

    losses.append(batch_losses.mean().item())
    scheduler.step()

viz.plot_loss_curve(losses, f"{out_folder}/loss.png")
torch.save(model.state_dict(), f"{out_folder}/weights.pth")

# Compute model accuracy
model.eval()
num_correct = 0

with torch.no_grad():
  for images, labels in test_loader:
    images, labels = images.to(device), labels.to(device)
    patches = patchify_img(
      images, images.shape[0],
      images.shape[1], h.patch_size,
      h.img_size // h.patch_size
    )
    prediction, _ = model(patches)

    predicted_labels = prediction.argmax(dim=-1)
    num_correct += (predicted_labels == labels).sum().item()

  accuracy = 100.0 * num_correct / len(test_loader.dataset)
  print(f"Top 1 Accuracy: {accuracy:.2f}%")
