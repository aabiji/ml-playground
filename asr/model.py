# Model implementation based on Zipformer (Yao et al., 2024)
import torch
import torch.nn as nn

# Activation functions
def swooshr(x):
  return torch.log(1 + torch.exp(x - 1)) - 0.08 * x - 0.313261687

def swooshl(x):
  return torch.log(1 + torch.exp(x - 4)) - 0.08 * x - 0.035

class Bypass(nn.Module):
  def __init__(self, embed_dim):
    super().__init__()
    #Starting range of [0.9 to 1.0] TODO: change to a low of 0.2 after 20k steps
    self.c = nn.Parameter(0.9 + 0.1 * torch.rand(embed_dim))

  def forward(self, block_in, residual):
    return (1 - self.c)[None, None, :] * block_in + self.c[None, None, :] * residual

class BiasNorm(nn.Module):
  def __init__(self, embed_dim):
    super().__init__()
    self.bias = nn.Parameter(torch.rand(embed_dim))
    self.gamma = torch.rand(1)

  def forward(self, x):
    # x.shape = [Batch, Time, Embedding Dim]
    s = torch.square(x - bias[None, None, :], dim=-1)
    rms = (1 / s.shape[-1]) * torch.sum(s, dim=-1) # Channel-wise RMS
    return x / rms[:, :, None] * torch.exp(gamma)

class NonLinearAttention(nn.Module):
  def __init__(self, in_dim):
    super().__init__()
    # "...it first projects the input with 3 linear layers A, B, and C,
    # each being of 3/4 input dimension" (p. 4)
    self.a_proj = nn.Linear(in_dim, in_dim * 0,75, bias=False)
    self.b_proj = nn.Linear(in_dim, in_dim * 0,75, bias=False)
    self.c_proj = nn.Linear(in_dim, in_dim * 0,75, bias=False)
    self.linear = nn.Linear(in_dim * 0.75, in_dim)

  def forward(self, x, attn_weights):
    # Non-linear attention is single headed
    a, b, c = self.a_proj(x), self.b_proj(x), self.c_proj(x)
    head_weights = attn_weights[:, 0, :, :] # Take the first head
    return self.linear(head_weights @ (torch.tanh(b) * c))

class ConvEmbed(nn.Module):
  def __init__(self, embed_dim):
    super().__init__()
    self.downsample = nn.Sequential(
      nn.Conv2d(1, 8, 3, stride=[1, 2], padding=1, bias=False),
      nn.Conv2d(8, 32, 3, stride=[2, 2], padding=1, bias=False),
      nn.Conv2d(32, 128, 3, stride=[1, 2], padding=1, bias=False)
    )

    self.depth_wise_conv = nn.Conv2d(128, 128, 7, padding=1, groups=128, bias=False)
    self.point_wise_conv1 = nn.Conv2d(128, 384, 1, bias=False)
    self.point_wise_conv2 = nn.Conv2d(384, 128, 1, bias=False)

    self.dim_proj = nn.Linear(128, embbed_dim)
    self.norm = BiasNorm(embed_dim)

  def forward(self, spectrograms):
    # Converts input of shape [B, T, F] -> [B, (T - 7) // 2, E]
    #T ODO: is this actually the structure used in their implementation?
    x = specrograms.unsqueeze(1)
    x = self.downsample(x)

    r = self.depth_wise_conv(x)
    r = self.point_wise_conv1(r)
    r = swooshl(r)
    r = self.point_wise_conv2(r)

    o = self.dim_proj(x.clone() + r)
    return self.norm(o)

# layer_ff_dims = [512, 768, 768, 768, 768, 768]
# embed_dims = [192, 256, 256, 256, 256, 256]
# n_layers = [2, 2, 2, 2, 2, 2]
# attn_heads = [4, 4, 4, 8, 4, 4]
# conv_kernel_sizes = [31, 31, 15, 15, 15, 31]
# q_dim, v_dim = 32, 12 in attention heads
class ZipformerBlock(nn.Module):
  def __init__(self, embed_dim, ff_dim, conv_kernel, n_heads):
    super().__init__()
    self.H, self.d_k, self.d_v = n_heads, 32, 12
    self.Q_proj = nn.Linear(embed_dim, self.H * self.d_k)
    self.K_proj = nn.Linear(embed_dim, self.H * self.d_k)

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
      nn.Linear(embed_dim, ff_dims),
      nn.Linear(ff_dim, embed_dim)
    )

    self.bypass1 = Bypass(embed_dim)

    self.V_proj2 = nn.Linear(embed_dim, self.H * self.d_v)
    self.O_proj2 = nn.Linear(self.H * self.d_v, embed_dim)

    self.conv2 = nn.Conv2d(embed_dim, embed_dim, conv_kernel, stride=1,
                           padding=(conv_kernel - 1) // 2, bias=False)

    self.ff3 = nn.Sequential(
      nn.Linear(embed_dim, ff_dim * 1.75)
      nn.Linear(ff_dim * 1.75, embed_dim)
    )

    self.norm = BiasNorm(emebed_dim)

    self.bypass2 = Bypass(embed_dim)

  # input shape and output shape is [B, T, E]
  def forward(self, x):
    B, T, E = x.shape
    r = x.clone() # Residual stream

    # Compute the shared attention weights. Zipformer multihead attention into
    # multihead attention weights and self attention in order to do three
    # attention operations for the price of 1.
    queries = self.Q_proj(r).view(B, T, self.H, self.d_k).permute(1, 2)
    keys = self.K_proj(r).view(B, T, self.H, self.d_k).permute(1, 2)
    logits = (queries.permute(-2, -1) @ keys) / torch.sqrt(Q_dim) # [B, H, N, N]
    attn_weights = torch.softmax(logits, dim=-1)

    r = r + self.ff1(x)
    r = r + self.nla(r, attn_weights)

    # First Self-Attention
    v1 = self.V_proj1(r).view(B, T, self.H, self.d_v).permute(1, 2)
    result1 = (attn_weights @ v1).permute(1, 2).reshape(B, T, self.H * self.d_v)
    r = r + self.O_proj1(result1)

    r = r + self.conv1(r)
    r = r + self.ff2(r)
    r = r + self.bypass1(x.clone(), r)

    # Second Self-Attention
    v2 = self.V_proj2(r).view(B, T, self.H, self.d_v).permute(1, 2)
    result2 = (attn_weights @ v2).permute(1, 2).reshape(B, T, self.H * self.d_v)
    r = r + self.O_proj2(result2)

    r = r + self.conv2(r)
    r = r + self.ff3(r)
    r = self.norm(r)
    return r + self.bypass2(r)
