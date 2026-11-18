import torch
from torch.optim import Optimizer

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
  def __init__(self, params, lr=0.025, beta1=0.9, beta2=0.98, eta=0.1, eps=1e-8):
    assert lr > 0, ValueError(f"Invalid learning rate lr={lr}")
    assert eps > 0, ValueError(f"eps must be greater than 0")

    defaults = dict(lr=lr, step=0, beta1=beta1, beta2=beta2, eta=eta, eps=eps)
    super(ScaledAdam, self).__init__(params, defaults)

  @torch.no_grad()
  def step(self, closure=None):
    loss = None
    if closure is not None:
      with torch.enable_grad():
        loss = closure()

    for group in self.param_groups:
      lr, b1, b2 = group["lr"], group["beta1"], group["beta2"]
      eta, eps = group["eta"], group["eps"]

      for p in group["params"]:
        g = p.grad
        if g is None:
          continue

        # Init tracking state for the parameter
        state = self.state[p]
        if len(state) == 0:
          state["param_scale"] = 0
          state["moment1"] = torch.zeros_like(p, memory_format=torch.preserve_format)
          state["moment2"] = torch.zeros_like(p, memory_format=torch.preserve_format)
          state["scale_moment1"] = torch.zeros_like(p, memory_format=torch.preserve_format)
          state["scale_moment2"] = torch.zeros_like(p, memory_format=torch.preserve_format)

        state["step"] += 1
        state["param_scale"] = root_mean_square(p)
        r = group["param_scale"]
        t = state["step"]

        m, v = group["moment1"], group["moment2"]
        m.mul_(b1).add_(g, alpha=1 - b1)
        v.mul_(b2).add_(g ** 2, alpha=1 - b2)

        n, w = group["scaled_moment1"], group["scaled_moment2"]
        h = g * p # Parameter scale gradient
        n.mul_(b1).add_(h, alpha=1 - b1)
        w.mul_(b2).add_(h ** 2, alpha=1 - b2)

        c = torch.sqrt((1 - b2) ** t) / ((1 - b1) ** t) # Bias-correction term
        grad_delta = -lr * r * c * m / (torch.sqrt(v) + eps)
        scale_delta = (-eta * lr * c * n / (torch.sqrt(w) + eps)) * p

        p.add_(grad_delta).add_(scale_delta)

    return loss