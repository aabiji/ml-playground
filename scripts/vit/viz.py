import torch
import matplotlib.pyplot as plt

def log_training_progress(batch_idx, epoch, total_epochs, loss, bar_length, loader):
  num_batches = len(loader)
  batch_count = epoch * num_batches + batch_idx
  progress = batch_count / (total_epochs * num_batches)
  num_bars = progress * bar_length
  progress_bar = '█' * int(num_bars) + ' ' * int(bar_length - num_bars)
  end = "\n" if num_bars == bar_length else "\r"
  info = f"Batch {batch_idx + 1} / {num_batches} | Epoch {epoch + 1} / {total_epochs}"
  print(f"{info} | [{progress_bar}] | Loss: {loss:.3f} \033[K", end=end, flush=True)

def plot_loss_curve(data, output_path):
  vmin, vmax = data.min().item(), data.max().item()
  mean, std = data.mean().item(), data.std().item()
  fig, ax = plt.subplots()
  ax.set_title("Training loss curve")
  ax.set_xlabel("Batches")
  ax.set_ylabel("Loss")
  ax.plot(data, "b-")
  ax.text(0.5, -0.18, f"Min: {vmin} Max: {vmax} Mean: {mean} Std: {std}",
          transform=ax.transAxes, ha="center", va="top", fontsize=10)
  ax.set_yscale("log")
  fig.tight_layout()
  fig.savefig(output_path, bbox_inches="tight")
  plt.show()

def visualize_patches(patches, patch_shape, fig_grid, label, fig_path):
  N, Px, Py, C = patch_shape
  reshaped = patches.reshape(N, Px, Py, C)

  fig, axs = plt.subplots(nrows=fig_grid[0], ncols=fig_grid[1],
                            figsize=(4, 4), layout="constrained")
  axs = axs.flatten()
  fig.suptitle(f"{label} image patches")

  for i in range(N):
    img = (reshaped[i] + 1) / 2
    axs[i].imshow(img)
    axs[i].axis("off")
  fig.savefig(fig_path)

def visualize_attention_scores(layer_scores, fig_grid, fig_path):
  fig, axs = plt.subplots(nrows=fig_grid[0], ncols=fig_grid[1],
                          figsize=(10, 10), layout="constrained")
  axs = axs.flatten()

  vmin, vmax = layer_scores.min().item(), layer_scores.max().item()
  layers, heads, N = layer_scores.shape[0:3]
  fig.suptitle(f"{N}x{N} attention scores: {layers} layers, {heads} heads", fontsize=20)

  for l in range(layers):
    for h in range(heads):
      idx = l * heads + h
      image = axs[idx].imshow(layer_scores[l, h], vmin=vmin, vmax=vmax, cmap="magma")
      axs[idx].axis("off")

  cbar = fig.colorbar(image, ax=axs, location="right", shrink=0.7)
  cbar.ax.set_ylabel("Similarity", va="bottom", rotation=-90)
  fig.savefig(fig_path)

def visualize_positional_embeddings(pos_embed, fig1_path, fig2_path):
  fig1, ax1 = plt.subplots()
  ax1.set_title("Patch position similarity")
  ax1.set_xlabel("Patch index")
  ax1.set_ylabel("Patch index")

  logits = pos_embed @ pos_embed.T
  similarity = torch.nn.functional.softmax(logits, dim=-1)
  heatmap = ax1.imshow(similarity, cmap="magma")

  cbar = fig1.colorbar(heatmap, ax=ax1, location="right", shrink=0.7)
  cbar.ax.set_ylabel("Similarity", va="bottom", rotation=-90)
  fig1.savefig(fig1_path)

  fig2, ax2 = plt.subplots(figsize=(24, 8), dpi=120)
  ax2.imshow(pos_embed, cmap="magma", aspect="auto")
  ax2.set_title("Positional embeddings", fontsize=24)
  ax2.set_yticks(torch.arange(0, pos_embed.shape[0] + 1, 1))
  ax2.set_xlabel("Embedding dimension", fontsize=18)
  ax2.set_ylabel("Path index", fontsize=18)

  fig2.tight_layout()
  fig2.savefig(fig2_path)
