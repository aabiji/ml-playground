import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2
import pathlib, tomllib, sys
import viz

def load_model_hyperparams():
  if len(sys.argv) != 2:
    print("This script requires one path to a toml file listing model hyperparameters.")
    sys.exit()

  filename = sys.argv[1]
  with open(filename, "rb") as file:
    data = tomllib.load(file)
  return data

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

hp = load_model_hyperparams()
out_folder = hp["experiment"]
pathlib.Path(out_folder).mkdir(parents=True, exist_ok=True)

augment_pipeline = v2.Compose([
  v2.RGB(), v2.ToImage(),
  v2.ToDtype(torch.float32, scale=True), v2.Resize(size=(32, 32)),
  v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # [0, 255] -> [-1, 1]
  #v2.RandomHorizontalFlip(),
  #v2.RandomRotation(90),
  #v2.RandomErasing(p=0.3),
])

train_dataset = CIFAR10(root=".cache/train/cifar10", train=True,
                        download=True, transform=augment_pipeline)
train_loader = DataLoader(train_dataset, batch_size=hp["batch_size"], shuffle=True)

model = ViT(
  hp["patch_size"], hp["img_size"] // hp["patch_size"],
  hp["embedding_dim"], hp["layers"], hp["attn_heads"],
  hp["class_dim"], len(hp["classes"])
).apply(init_weights).to(device)

optimizer = AdamW(model.parameters(), lr=hp["learning_rate"], betas=hp["betas"])
scheduler = CosineAnnealingLR(optimizer, T_max=hp["epochs"])
criterion = nn.CrossEntropyLoss(label_smoothing=hp["label_smoothing"])
losses = []

for epoch in range(hp["epochs"]):
  for batch, (images, labels) in enumerate(train_loader):
    # Images shape: (B, C, H, W), Patches shape: (B, N, P, P, C)
    images, labels = images.to(device), labels.to(device)
    patches = patchify_img(
      images, images.shape[0],
      images.shape[1], hp["patch_size"],
      hp["img_size"] // hp["patch_size"]
    )

    prediction, scores = model(patches)
    loss = criterion(prediction, labels)

    losses.append(loss.item())
    viz.log_training_progress(batch, epoch, hp["epochs"], loss.item(), 50, train_loader)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=False)

    # Visualize the change in attention scores as training progresses
    if batch == 0:
      G = hp["img_size"] // hp["patch_size"]
      patch_shape = [G * G, hp["patch_size"], hp["patch_size"], 3]

      i = torch.randint(hp["batch_size"], size=(1,)).item()
      label = hp["classes"][labels[i]]
      p = patches[i].cpu().detach().numpy()
      s = scores[:, i].cpu().detach().numpy()

      viz.visualize_patches(p, patch_shape, [G, G], label,
                        f"{out_folder}/patches_{epoch + 1}.png")

      viz.visualize_attention_scores(s, [hp["layers"], hp["attn_heads"]],
                                 f"{out_folder}/scores_{epoch + 1}.png")

      viz.visualize_positional_embeddings(model.pos_embed.cpu().detach(),
                                      f"{out_folder}/pos_similarity_{epoch + 1}.png",
                                      f"{out_folder}/pos_embedding_{epoch + 1}.png")

  scheduler.step()

experiment = "Experiment 2"
viz.plot_loss_curve(losses, f"{out_folder}/loss.png")
torch.save(model.state_dict(), f"{out_folder}/weights.pth")
