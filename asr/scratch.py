import torch
import librosa
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torch.optim import Optimizer

def spec_augment(spec_batch, max_freq_range, n_freq_masks,
                 max_time_range, n_time_masks, mask_value):
  _, spec_height, spec_width = spec_batch.shape

  for _ in range(n_freq_masks):
    f_range = torch.randint(0, max_freq_range, size=(1,))
    y = torch.randint(0, spec_height - f_range, size=(1,))
    spec_batch[:, y:y+f_range, :] = mask_value

  for _ in range(n_time_masks):
    t_range = torch.randint(0, max_time_range, size=(1,))
    x = torch.randint(0, spec_width - t_range, size=(1,))
    spec_batch[:, :, x:x+t_range] = mask_value

class LibriSpeech(Dataset):
  def __init__(self, folder_path):
    files = [file for file in Path(folder_path).rglob("*") if file.is_file()]
    self.audio_files = list(filter(lambda f: f.suffix == ".flac", files))
    self.transcript_cache = {}

    # Cache transcripts
    transcript_files = list(filter(lambda f: f.suffix == ".txt", files))
    for tfile in transcript_files:
      key = tfile.stem.split(".")[0]
      with open(tfile, "r") as f:
        lines = f.read().split("\n")
        self.transcript_cache[key] = lines

  def process_all_samples(self, all_samples):
    # Convert a batch of audio samples into a batch of padded spectrograms with a
    # framerate of 100 kHz. For example, with 1000 ms of audio, a 16 kHz sampling
    # rate and a hop length of 160 samples, that's (1000 * 16000) / 160 + 1 = 101
    # time frames. Adding 1 since each frame is centered on its timestamp rather
    # than starting on it.
    hop_length = 160
    samples_lengths = list(map(lambda x: x.shape[-1], all_samples))

    frame_lengths = (torch.tensor(samples_lengths) // hop_length) + 1

    # Padding samples before applying the spectrogram for efficiency
    padded = pad_sequence(all_samples, batch_first=True, padding_value=0)
    padded_np = padded.detach().cpu().numpy()

    spectrograms = librosa.feature.melspectrogram(y=padded_np, sr=16000, hop_length=hop_length,
                                                  n_mels=80, n_fft=400, win_length=400)
    spectrograms = librosa.power_to_db(spectrograms, amin=1e-5, ref=np.max)
    spectrograms = torch.from_numpy(spectrograms).float()
    return spectrograms, frame_lengths

  def __len__(self):
    return len(self.audio_files)

  def __getitem__(self, raw_idx):
    all_samples = []
    transcripts = []
    tok_counts = []

    if torch.is_tensor(raw_idx):
      idx = raw_idx.tolist()
    elif isinstance(raw_idx, slice):
      idx = torch.arange(len(self.audio_files))[raw_idx].tolist()
    else:
      idx = [raw_idx]

    for i in idx:
      # Load audio sample
      f = self.audio_files[i]
      np_samples, _ = librosa.load(f, mono=True, sr=16000)
      all_samples.append(torch.from_numpy(np_samples).float())

      # Get the corresponding transcript and apply codepoint-level tokenization
      last_dash = f.stem.rfind("-")
      key = f.stem[:last_dash]
      line_index =  int(f.stem[last_dash + 1:])
      words = self.transcript_cache[key][line_index].split(" ")

      tokens = list(map(ord, " ".join(words[1:])))
      transcripts.append(torch.tensor(tokens))
      tok_counts.append(len(tokens))

    transcripts = pad_sequence(transcripts, batch_first=True, padding_value=0)
    transcript_lengths = torch.tensor(tok_counts)
    spectrograms, frame_lengths = self.process_all_samples(all_samples)
    return spectrograms, frame_lengths, transcripts, transcript_lengths


def eden_lr_scheduler(global_step, start_lr, base_lr, warmup_steps,
                      step_threshold, epoch_threshold, steps_per_epoch):
  step_dampen = (global_step ** 2 + step_threshold ** 2) / (step_threshold ** 2)
  step_dampen = math.pow(step_dampen, -0.25)

  epoch = global_step // steps_per_epoch
  epoch_dampen = (epoch ** 2 + epoch_threshold ** 2) / (epoch_threshold ** 2)
  epoch_dampen = math.pow(epoch_dampen, -0.25)

  slope = (1 - start_lr) / warmup_steps
  warmup_lr = slope * global_step + start_lr

  # Considering the step index and the epoch index keeps the amount of parameter change
  # roughly constant after a certain amount of training data when the batch size is changed.
  # However, this does not make this scheduler batch size invariant.
  return base_lr * step_dampen * epoch_dampen * warmup_lr

class ScaledAdam(Optimizer):
  """
  Although Adam normalizes the first and second order moments, it doesn't account
  for the scale of the parameters themselves. This leads to some parameters being
  stepped too much and others being stepped too little. ScaledAdam solves this
  by tracking and adjusting the step size relative to parameter scale.

  lr             : The learning rate controls the stepping magnitude.
  (beta1, beta2) : Specify the exponential decay rates for the first and second order moment estimates.
  eta            : Scaling factor on the learning rate for parameter scale.
  eps            : Epsilon value to avoid dvision by zero.
  """
  def __init__(self, params, lr=0.025, beta1=0.9,
               beta2=0.98, eta=0.1, eps=1e-8, min_rms=1e-9):
    if lr <= 0: raise ValueError(f"Invalid learning rate lr={lr}")
    if eps <= 0: raise ValueError(f"eps must be greater than 0")

    defaults = dict(lr=lr, step=0, beta1=beta1, beta2=beta2,
                    eta=eta, eps=eps, min_rms=min_rms)
    super(ScaledAdam, self).__init__(params, defaults)

  @torch.no_grad()
  def step(self, closure=None): # type: ignore
    loss = None
    if closure is not None:
      with torch.enable_grad():
        loss = closure()

    for group in self.param_groups:
      lr, b1, b2 = group["lr"], group["beta1"], group["beta2"]
      eta, eps, min_rms = group["eta"], group["eps"], group["min_rms"]

      for p in group["params"]:
        g = p.grad
        if g is None:
          continue

        state = self.state[p]
        if len(state) == 0:
          state["t"] = 0 # Time step
          # First and second-order scale moments (scalar)
          state["n"] = torch.zeros(())
          state["w"] = torch.zeros(())
          # First and second-order gradient moments
          state["m"] = torch.zeros_like(p, memory_format=torch.preserve_format)
          state["v"] = torch.zeros_like(p, memory_format=torch.preserve_format)

        state["t"] += 1
        t = state["t"]

        # The bias-correction term amplifies the moment when t is small, and that amplification diminishes as t increases
        c = math.sqrt(1 - b2 ** t) / (1 - b1 ** t)

        # The root mean square of p (parameter scale) should never be zero,
        # since that would make grad_delta zero forever and break stepping.
        r = p.square().mean().sqrt().clamp(min=min_rms)

        # Calculate the first and second order gradient moments
        m, v, n, w = state["m"], state["v"], state["n"], state["w"]
        m.mul_(b1).add_(g, alpha=1 - b1)
        v.mul_(b2).addcmul_(g, g, alpha=1 - b2)

        # Calculate the first and second order scale moments
        h = (g * p).sum() # Scalar gradient with respect to the parameter's scale
        n.mul_(b1).add_(h, alpha=1 - b1)
        w.mul_(b2).addcmul_(h, h, alpha=1 - b2)

        # Step the parameter
        grad_delta = -lr * r * c * m / (v.sqrt() + eps)
        scale_delta = -eta * lr * c * n / (w.sqrt() + eps)
        p.add_(grad_delta + scale_delta * p)

    return loss


# Activation functions: using softplus to nicely handle larger values of x
class SwooshR(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, x):
    return F.softplus(x - 1) - 0.08 * x - 0.313261687

class SwooshL(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, x):
    return F.softplus(x - 4) - 0.08 * x - 0.035

# Truncate or pad the tensor to match the target final dimension
def convert_embedding_dim(x, target_dim):
  B, T, input_embed = x.shape

  if input_embed > target_dim:
    return x[:, :, :target_dim]

  pad = torch.zeros(B, T, target_dim - input_embed,
                    device=x.device, dtype=x.dtype)
  return torch.cat((x, pad), dim=-1)

class Bypass(nn.Module):
  def __init__(self, embed_dim):
    super().__init__()
    self.c = nn.Parameter(0.9 + 0.1 * torch.rand(embed_dim))

  def forward(self, mod_in, mod_out):
    # TODO: enforce the [0.9 to 1.0] range here and change to a low of 0.2 after 20k steps
    return (1 - self.c)[None, None, :] * mod_in + self.c[None, None, :] * mod_out

class BiasNorm(nn.Module):
  def __init__(self, embed_dim):
    super().__init__()
    self.bias = nn.Parameter(torch.rand(embed_dim))
    self.gamma = nn.Parameter(torch.tensor(0.0))

  def forward(self, x):
    # x.shape = [Batch, Time, Embedding Dim]
    term = x - self.bias[None, None, :]
    rms = torch.sqrt(torch.mean(term ** 2, dim=-1)) # Channel-wise
    return x / rms[:, :, None] * torch.exp(self.gamma)

class Sampler(nn.Module):
  def __init__(self, factor):
    super().__init__()
    self.weights = nn.Parameter(torch.zeros(factor))
    self.f = factor

  def forward(self, x, downsample):
    B, T, E = x.shape

    # Upsample: [B, T, E] -> [B, T * factor, E]
    if not downsample:
      return torch.repeat_interleave(x, repeats=self.f, dim=1)

    # Downsample: [B, T, E] -> [B, T // factor, E]
    if T % self.f != 0:
      next_multiple = math.ceil(T / self.f) * self.f
      pad = torch.zeros(B, next_multiple - T, E, device=x.device, dtype=x.dtype)
      x = torch.cat((x, pad), dim=1)

    # Weighted averaging of groups of twp consecutive time frames
    norm_weights = torch.softmax(self.weights[None, None, :, None], dim=-1)
    grouped = x.reshape(B, T // self.f, self.f, E)
    return (grouped * norm_weights).sum(dim=2)

class ConvEmbed(nn.Module):
  def __init__(self, n_mels, embed_dim):
    super().__init__()
    # [B, C, T, F] -> [B, 128, (T / 2) - 3, (F / 4) - 1]
    # We can afford to drop frames and frequencies since
    # they're negligible and probably redundant anyways
    self.downsample = nn.Sequential(
      nn.Conv2d(1, 8, 3, stride=1, padding=(0, 1), bias=False),
      SwooshR(),
      nn.Conv2d(8, 32, 3, stride=2, bias=False),
      SwooshR(),
      nn.Conv2d(32, 128, 3, stride=(1, 2), bias=False),
      SwooshR(),
    )

    # ConvNeXT block: 1 depth-wise conv followed by 2 point-wise convs
    self.conv_next_block = nn.Sequential(
      nn.Conv2d(128, 128, 7, padding=3, groups=128, bias=False),
      nn.Conv2d(128, 384, 1, bias=False),
      SwooshL(),
      nn.Conv2d(384, 128, 1, bias=False),
    )

    # Linear projection
    # [B, 128, (T / 2) - 3, (F / 4) - 1] -> [B, (T / 2) - 3, E]
    new_freq_dim = (((n_mels - 1) // 2) - 1) // 2
    self.proj = nn.Linear(128 * new_freq_dim, embed_dim)
    self.norm = BiasNorm(embed_dim)

  def forward(self, spectrograms):
    x = spectrograms.unsqueeze(1) # x.shape = [B, C, T, F]
    x = self.downsample(x)
    y = self.conv_next_block(x)

    # [B, C, T, F] -> [B, C, F, T] -> [B, C * F, T] -> [B, T, C * F]
    r = x.clone() + y
    r = r.tranpose(-1, -2)
    r = r.flatten(start_dim=1, end_dim=2)
    r = r.transpose(-1, -2)

    # [B, T, C * F] -> [B, T, E]
    return self.norm(self.proj(r))

class NonLinearAttention(nn.Module):
  def __init__(self, embed_dim):
    super().__init__()
    # "...it first projects the input with 3 linear layers A, B, and C,
    # each being of 3/4 input dimension" (p. 4)
    proj_dim = int(embed_dim * 0.75)
    self.a_proj = nn.Linear(embed_dim, proj_dim, bias=False)
    self.b_proj = nn.Linear(embed_dim, proj_dim, bias=False)
    self.c_proj = nn.Linear(embed_dim, proj_dim, bias=False)
    self.linear = nn.Linear(proj_dim, embed_dim)

  def forward(self, x, attn_weights):
    # Non-linear attention is single headed
    a, b, c = self.a_proj(x), self.b_proj(x), self.c_proj(x)
    head_weights = attn_weights[:, 0, :, :] # Take the first head
    attention = head_weights @ (torch.tanh(b) * c)
    return self.linear(a * attention)

# TODO: where's the relative positional encoding???
class EncoderBlock(nn.Module):
  def __init__(self, embed_dim, ff_dim, conv_kernel, n_heads):
    super().__init__()
    self.d_k, self.d_v = 32, 12
    self.H, self.E = n_heads, embed_dim
    self.Q_proj = nn.Linear(embed_dim, self.H * self.d_k)
    self.K_proj = nn.Linear(embed_dim, self.H * self.d_k)

    # "...For the three feed-forward modules in each Zipformer block, the hidden dimensions
    # in the first one and the last one are 3/4 and 5/4 of that in the middle one."
    self.ff1 = nn.Sequential(
      nn.Linear(embed_dim, int(ff_dim * 0.75)),
      SwooshL(),
      nn.Linear(int(ff_dim * 0.75), embed_dim)
    )

    self.nla = NonLinearAttention(embed_dim)

    self.V_proj1 = nn.Linear(embed_dim, self.H * self.d_v)
    self.O_proj1 = nn.Linear(self.H * self.d_v, embed_dim)

    self.conv1 = nn.Conv2d(embed_dim, embed_dim, conv_kernel, stride=1,
                           padding=(conv_kernel - 1) // 2, bias=False)

    self.ff2 = nn.Sequential(
      nn.Linear(embed_dim, ff_dim),
      nn.Linear(ff_dim, embed_dim)
    )

    self.mid_bypass = Bypass(embed_dim)

    self.V_proj2 = nn.Linear(embed_dim, self.H * self.d_v)
    self.O_proj2 = nn.Linear(self.H * self.d_v, embed_dim)

    self.conv2 = nn.Conv2d(embed_dim, embed_dim, conv_kernel, stride=1,
                           padding=(conv_kernel - 1) // 2, bias=False)

    self.ff3 = nn.Sequential(
      nn.Linear(embed_dim, int(ff_dim * 1.25)),
      SwooshL(),
      nn.Linear(int(ff_dim * 1.25), embed_dim)
    )

    self.norm = BiasNorm(embed_dim)

    self.bypass = Bypass(embed_dim)

  # input shape and output shape is [Batches, Time frames, Embedding dimension]
  def forward(self, x):
    x = convert_embedding_dim(x, self.E)
    B, T, _ = x.shape
    r = x.clone() # Residual stream

    # Zipformer splits multihead attention into computing shared weights and
    # computing scaled values in order to amortize the cost of attention.
    queries = self.Q_proj(r).view(B, T, self.H, self.d_k).transpose(1, 2)
    keys = self.K_proj(r).view(B, T, self.H, self.d_k).transpose(1, 2)
    scale = torch.sqrt(torch.tensor(self.d_k))
    logits = (queries @ keys.transpose(-2, -1)) / scale # [B, H, N, N]
    attn_weights = torch.softmax(logits, dim=-1)

    r = r + self.ff1(x)
    r = r + self.nla(r, attn_weights)

    # First Self-Attention
    v1 = self.V_proj1(r).view(B, T, self.H, self.d_v).transpose(1, 2)
    result1 = (attn_weights @ v1).transpose(1, 2).reshape(B, T, self.H * self.d_v)
    r = r + self.O_proj1(result1)

    r = r + self.conv1(r)
    r = r + self.ff2(r)
    r = self.mid_bypass(x.clone(), r)

    # Second Self-Attention
    v2 = self.V_proj2(r).view(B, T, self.H, self.d_v).transpose(1, 2)
    result2 = (attn_weights @ v2).transpose(1, 2).reshape(B, T, self.H * self.d_v)
    r = r + self.O_proj2(result2)

    r = r + self.conv2(r)
    r = r + self.ff3(r)
    r = self.norm(r)
    return self.bypass(x.clone(), r)

class Zipformer(nn.Module):
  def __init__(self, n_mels, conv_kernel_sizes, layer_heads, ff_dims,
               embed_dims, layer_counts, sampling_factors):
    super().__init__()
    self.n_layers = len(layer_counts)

    self.conv_embed = ConvEmbed(n_mels, embed_dims[0])

    self.stacks = nn.ModuleList([
      nn.Sequential(*[
        EncoderBlock(embed_dims[i], ff_dims[i],
                     conv_kernel_sizes[i], layer_heads[i])
        for _ in range(layer_counts[i])
      ])
      for i in range(self.n_layers)
    ])

    self.samplers = nn.ModuleList([
      Sampler(sampling_factors[i]) for i in range(self.n_layers - 1)
    ])

    self.bypasses = nn.ModuleList([
      Bypass(embed_dims[i]) for i in range(1, self.n_layers)
    ])

    self.final_sampler = Sampler(sampling_factors[-1])

  def forward(self, x):
    x = self.conv_embed(x)
    x = self.stacks[0](x)

    for i in range(self.n_layers - 1):
      x1 = self.samplers[i](x, True)
      x1 = self.stacks[i + 1](x1)
      x1 = self.samplers[i](x1, False)
      x  = self.bypasses[i](x, x1)

    return self.final_sampler(x, True)

