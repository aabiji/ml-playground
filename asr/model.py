# Model implementation based on Zipformer (Yao et al., 2024) and icefall's Zipformer implementation.
import torch
import torch.nn as nn

# Activation functions
def swooshr(x):
  return torch.log(1 + torch.exp(x - 1)) - 0.08 * x - 0.313261687

def swooshl(x):
  return torch.log(1 + torch.exp(x - 4)) - 0.08 * x - 0.035

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
    #Starting range of [0.9 to 1.0] TODO: change to a low of 0.2 after 20k steps
    self.c = nn.Parameter(0.9 + 0.1 * torch.rand(embed_dim))

  def forward(self, mod_in, mod_out):
    return (1 - self.c)[None, None, :] * mod_in + self.c[None, None, :] * mod_out

class BiasNorm(nn.Module):
  def __init__(self, embed_dim):
    super().__init__()
    self.bias = nn.Parameter(torch.rand(embed_dim))
    self.gamma = torch.rand(1)

  def forward(self, x):
    # x.shape = [Batch, Time, Embedding Dim]
    s = torch.square(x - self.bias[None, None, :])
    rms = (1 / s.shape[-1]) * torch.sum(s, dim=-1) # Channel-wise RMS
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
      next_multiple = T // self.f + self.f
      pad = torch.zeros(B, next_multiple - T, E, device=x.device, dtype=x.dtype)
      x = torch.cat((x, pad), dim=1)

    grouped = x.reshape(B, T // self.f, self.f, E)
    weighted = (grouped * self.weights[None, :, None, None])
    return weighted.sum(dim=1)

class ConvEmbed(nn.Module):
  def __init__(self, n_mels, embed_dim):
    super().__init__()
    # [B, C, T, F] -> [B, 128, (T / 2) - 3, (F / 8) - 1]
    # We can afford to drop frames and frequencies since
    # they're negligible and probably redundant anyways
    self.downsample1 = nn.Conv2d(1, 8, 3, stride=1, padding=(0, 1), bias=False)
    self.downsample2 = nn.Conv2d(8, 32, 3, stride=2, bias=False)
    self.downsample3 = nn.Conv2d(32, 128, 3, stride=(1, 2), bias=False)

    # ConvNeXT block
    self.depth_wise1 = nn.Conv2d(128, 128, 7, padding=3, groups=128, bias=False)
    self.point_wise1 = nn.Conv2d(128, 384, 1, bias=False)
    self.point_wise2 = nn.Conv2d(384, 128, 1, bias=False)

    # Linear projection
    # [B, 128, (T / 2) - 3, (F / 8) - 1] -> [B, (T / 2) - 3, E]
    new_freq_dim = ((n_mels / 8) - 1)
    self.proj = nn.Linear(128 * new_freq_dim, embed_dim)
    self.norm = BiasNorm(embed_dim)

  def forward(self, spectrograms):
    x = spectrograms.unsqueeze(1) # x.shape = [B, C, T, F]
    x = swooshr(self.downsample1(x))
    x = swooshr(self.downsample2(x))
    x = swooshr(self.downsample3(x))

    y = self.depth_wise1(x)
    y = self.point_wise1(y)
    y = swooshl(y)
    y = self.point_wise2(y)

    # [B, C, T, F] -> [B, C, F, T] -> [B, C * F, T] -> [B, T, C * F]
    r = x.clone() + y
    r = r.permute(-1, -2)
    r = r.flatten(start_dim=1, end_dim=2)
    r = r.permute(-1, -2)

    # [B, T, C * F] -> [B, T, E]
    return self.norm(self.proj(r))

class NonLinearAttention(nn.Module):
  def __init__(self, embed_dim):
    super().__init__()
    # "...it first projects the input with 3 linear layers A, B, and C,
    # each being of 3/4 input dimension" (p. 4)
    self.a_proj = nn.Linear(embed_dim, embed_dim * 0.75, bias=False)
    self.b_proj = nn.Linear(embed_dim, embed_dim * 0.75, bias=False)
    self.c_proj = nn.Linear(embed_dim, embed_dim * 0.75, bias=False)
    self.linear = nn.Linear(embed_dim * 0.75, embed_dim)

  def forward(self, x, attn_weights):
    # Non-linear attention is single headed
    a, b, c = self.a_proj(x), self.b_proj(x), self.c_proj(x)
    head_weights = attn_weights[:, 0, :, :] # Take the first head
    return self.linear(head_weights @ (torch.tanh(b) * c))

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
      nn.Linear(embed_dim, ff_dim * 0.75),
      nn.Linear(ff_dim * 0.75, embed_dim)
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
      nn.Linear(embed_dim, ff_dim * 1.75),
      nn.Linear(ff_dim * 1.75, embed_dim)
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
    queries = self.Q_proj(r).view(B, T, self.H, self.d_k).permute(1, 2)
    keys = self.K_proj(r).view(B, T, self.H, self.d_k).permute(1, 2)
    scale = torch.sqrt(torch.tensor(self.d_k))
    logits = (queries.permute(-2, -1) @ keys) / scale # [B, H, N, N]
    attn_weights = torch.softmax(logits, dim=-1)

    r = r + self.ff1(x)
    r = r + self.nla(r, attn_weights)

    # First Self-Attention
    v1 = self.V_proj1(r).view(B, T, self.H, self.d_v).permute(1, 2)
    result1 = (attn_weights @ v1).permute(1, 2).reshape(B, T, self.H * self.d_v)
    r = r + self.O_proj1(result1)

    r = r + self.conv1(r)
    r = r + self.ff2(r)
    r = r + self.mid_bypass(x.clone(), r)

    # Second Self-Attention
    v2 = self.V_proj2(r).view(B, T, self.H, self.d_v).permute(1, 2)
    result2 = (attn_weights @ v2).permute(1, 2).reshape(B, T, self.H * self.d_v)
    r = r + self.O_proj2(result2)

    r = r + self.conv2(r)
    r = r + self.ff3(r)
    r = self.norm(r)
    return r + self.bypass(r)

class Zipformer(nn.Module):
  def __init__(self, n_mels, conv_kernel_sizes, layer_heads, ff_dims,
               embed_dims, layer_counts, sampling_factors):
    super().__init__()
    self.n_layers = len(layer_counts)

    self.conv_embed = ConvEmbed(n_mels, embed_dims[0])

    self.stacks = nn.ModuleList([
      nn.ModuleList([
        EncoderBlock(embed_dims[i], ff_dims[i],
                     conv_kernel_sizes[i], layer_heads[i])
        for _ in layer_counts[i]
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
