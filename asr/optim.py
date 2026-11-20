import torch
import math
from torch.optim import Optimizer

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
