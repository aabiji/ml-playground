import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

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
  data = torch.tensor(data)
  vmin, vmax = data.min().item(), data.max().item()
  mean, std = data.mean().item(), data.std().item()
  fig, ax = plt.subplots()
  ax.set_title("Training loss curve")
  ax.set_xlabel("Epochs")
  ax.set_ylabel("Loss")
  ax.plot(data, "b-")
  ax.text(0.5, -0.18, f"Min: {vmin:.2f} Max: {vmax:.2f} Mean: {mean:.2f} Std: {std:.2f}",
          transform=ax.transAxes, ha="center", va="top", fontsize=10)
  ax.set_yscale("log")
  ax.yaxis.set_major_formatter(ScalarFormatter())
  fig.tight_layout()
  fig.savefig(output_path, bbox_inches="tight")
  plt.show()

def viz_patches(patches, patch_shape, fig_grid,
                      true_label, predicted_label, fig_path):
  N, Px, Py, C = patch_shape
  reshaped = patches.reshape(N, Px, Py, C)

  fig, axs = plt.subplots(nrows=fig_grid[0], ncols=fig_grid[1],
                            figsize=(4, 4), layout="constrained")
  axs = axs.flatten()
  fig.suptitle(f"{true_label} image patches, predicted: {predicted_label}")

  for i in range(N):
    img = (reshaped[i] + 1) / 2
    axs[i].imshow(img)
    axs[i].axis("off")
  fig.savefig(fig_path)

def viz_attn_scores(layer_scores, fig_grid, fig_path):
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

def viz_pos_embeds(pos_embed, fig_path):
  fig, ax1 = plt.subplots()
  ax1.set_title("Patch position similarity")
  ax1.set_xlabel("Patch index")
  ax1.set_ylabel("Patch index")

  logits = pos_embed @ pos_embed.T
  similarity = torch.nn.functional.softmax(logits, dim=-1)
  heatmap = ax1.imshow(similarity, cmap="magma")

  cbar = fig.colorbar(heatmap, ax=ax1, location="right", shrink=0.7)
  cbar.ax.set_ylabel("Similarity", va="bottom", rotation=-90)
  fig.savefig(fig_path)

def viz_model_features(prediction, patches, labels, scores, pos_embeds,
                       h, score_fig_path, patch_fig_path, pos_fig_path):
  G = h.img_size // h.patch_size
  patch_shape = [G * G, h.patch_size, h.patch_size, 3]

  i = torch.randint(h.batch_size, size=(1,)).item()
  label = h.classes[labels[i]]
  p = patches[i].cpu().detach().numpy()
  s = scores[:, i].cpu().detach().numpy()
  max_idx = prediction[i].argmax(dim=-1).item()
  predicted_label = h.classes[max_idx]

  viz_patches(p, patch_shape, [G, G], label, predicted_label, patch_fig_path)
  viz_attn_scores(s, [h.layers, h.attn_heads], score_fig_path)
  viz_pos_embeds(pos_embeds, pos_fig_path)
