"""
Flow matching training utilities.

The flow between noise x_0 ~ N(0, I) and data x_1 (using straight line path):
    x_t = (1 - t) * x_0 + t * x_1, t in [0, 1]
    dx_t / dt = x_1 - x_0

Training minimizes MSE between the following:
- network's predicted velocity v_model(x_t, t; y),
- the target velocity v = x_1 - x_0.

Timestep sampling: Uniform[0, 1] (baseline).
"""

import torch


def sample_triple(x_1):
    """
    Inputs
    x_1: input image(s)

    Calculate
    x_0: random noise ~ N(0, 1)
    t: random time sample ~ U[0, 1]
    x_t: linear interpolation between image x_1 and random noise x_0 at time t

    Return
    x_t, t, v_t
    """

    x_0 = torch.randn_like(x_1)
    t = torch.rand(x_1.shape[0], device=x_1.device, dtype=x_1.dtype)

    return (1 - t[:, None, None, None]) * x_0 + t[:, None, None, None] * x_1, t, x_1 - x_0


def flow_matching_loss(v_pred, v_target):
    """
    mse_loss = || v_model(x_t, t; y) - (x_1 - x_0) ||^2
    """

    return ((v_pred - v_target) ** 2).mean()
