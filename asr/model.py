# Model implementation based on Zipformer (Yao et al., 2024)
import torch
import torch.nn as nn

# Activation functions
def swooshr(x):
  return torch.log(1 + torch.exp(x - 1)) - 0.08 * x - 0.313261687

def swooshl(x):
  return torch.log(1 + torch.exp(x - 4)) - 0.08 * x - 0.035

# Simpler version of LayerNorm
# x.shape = [Batch, Time, Channels], bias.shape = [Channels], gamma is a scalar
def biasnorm(x, bias, gamma):
  s = torch.square(x - bias[None, None, :], dim=-1)
  rms = (1 / s.shape[-1]) * torch.sum(s, dim=-1) # Channel-wise RMS
  return x / rms[:, :, None] * torch.exp(gamma)

class NonLinearAttention(nn.Module):
  def __init__(self, in_dim, n_heads):
    super().__init__()
    proj_dim = 0.75 * in_dim
    self.a_proj = nn.Linear(in_dim, proj_dim, bias=False)
    self.b_proj = nn.Linear(in_dim, proj_dim, bias=False)
    self.c_proj = nn.Linear(in_dim, proj_dim, bias=False)
    self.linear = nn.Linear(proj_dim, in_dim)

  def forward(self, x, attn_weights):
    # Non-linear attention is single headed
    a, b, c = self.a_proj(x), self.b_proj(x), self.c_proj(x)
    head_weights = attn_weights[:, 0, :, :] # Take the first head
    return self.linear(head_weights @ (torch.tanh(b) * c))

# layer_ff_dims = [512, 768, 768, 768, 768, 768]
# embed_dims = [192, 256, 256, 256, 256, 256]
# n_layers = [2, 2, 2, 2, 2, 2]
# attn_heads = [4, 4, 4, 8, 4, 4]
# conv_kernel_sizes = [31, 31, 15, 15, 15, 31]
# q_dim, v_dim = 32, 12 in attention heads
class ZipformerBlock(nn.Module):
  def __init__(self, embed_dim, ff_dim, conv_kernel, n_heads):
    super().__init__()

    # Input shape: [Batch, Time, Channels]

    Q_dim, K_dim = 32, 12
    self.Q_proj = nn.Linear(embed_dim, n_heads * Q_dim)
    self.K_proj = nn.Linear(embed_dim, n_heads * Q_dim)

    self.ff1 = nn.Sequential(
      nn.Linear(embed_dim, ff_dim * 0.75),
      nn.Linear(ff_dim * 0.75, embed_dim)
    )

    self.nla = NonLinearAttention(embed_dim)

    self.V_proj1 = nn.Linear(embed_dim, n_heads * V_dim)
    self.O_proj1 = nn.Linear(n_heads * V_dim, embed_dim)

    self.conv1 = nn.Conv2d(embed_dim, embed_dim, conv_kernel,
                           stride=1, padding=(conv_kernel - 1) // 2)

    self.ff2 = nn.Sequential(
      nn.Linear(embed_dim, ff_dims),
      nn.Linear(ff_dim, embed_dim)
    )

    self.bypass1 = Bypass()

    self.V_proj2 = nn.Linear(embed_dim, n_heads * V_dim)
    self.O_proj2 = nn.Linear(n_heads * V_dim, embed_dim)

    self.conv2 = nn.Conv2d(embed_dim, embed_dim, conv_kernel,
                           stride=1, padding=(conv_kernel - 1) // 2)

    self.ff3 = nn.Sequential(
      nn.Linear(embed_dim, ff_dim * 1.75)
      nn.Linear(ff_dim * 1.75, embed_dim)
    )

    self.norm = BiasNorm()

    self.bypass1 = Bypass()

  def forward(self, x):
    pass
