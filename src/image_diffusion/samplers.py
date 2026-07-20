"""
Solving dx/dt = v(x, t; y) using three methods:
- Euler method
- Heun method
- Runge-Kutta 4
"""

import torch


@torch.no_grad()
def euler(velocity_fn, x, y, num_steps, t0=0.0, t1=1.0) -> torch.Tensor:
    """
    x_{t + 1} = x_t + h v(x_t, t; y)
    """

    h, one_vec = (t1 - t0) / num_steps, torch.ones(x.shape[0], device=x.device, dtype=x.dtype)
    for i in range(num_steps):
        x = x + h * velocity_fn(x, t0 + one_vec * (i * h), y)

    return x


@torch.no_grad()
def heun(velocity_fn, x, y, num_steps, t0=0.0, t1=1.0) -> torch.Tensor:
    """
    k1 = v(x_t, t; y)
    k2 = v(x_{t + 1}, t + h; y)
    xprediction_{t + 1} = x_t + h k1
    xcorrection_{t + 1} = x_t + (h / 2) [k1 + k2]
    """

    h, one_vec = (t1 - t0) / num_steps, torch.ones(x.shape[0], device=x.device, dtype=x.dtype)
    for i in range(num_steps):
        # Prediction step
        k1 = velocity_fn(x, t0 + one_vec * (i * h), y)
        x_pred = x + h * k1

        # Correction step
        k2 = velocity_fn(x_pred, t0 + one_vec * ((i + 1) * h), y)
        x = x + (h / 2) * (k1 + k2)

    return x


@torch.no_grad()
def rk4(velocity_fn, x, y, num_steps, t0=0.0, t1=1.0) -> torch.Tensor:
    """
    k1 = v(x_t, t; y): slope at the beginning of the interval
    k2 = v(x_t + k1 * h/2, t + h/2; y): slope at the midpoint (using y and k1)
    k3 = v(x_t + k2 * h/2, t + h/2; y): slope at the midpoint (using y and k2)
    k4 = v(x_t + k3 * h, t + h; y): slope at the end of the interval
    x_{t + 1} = x_t + h / 6 (k1 + 2k2 + 2k3 + k4)
    """

    h, one_vec = (t1 - t0) / num_steps, torch.ones(x.shape[0], device=x.device, dtype=x.dtype)
    for i in range(num_steps):
        # RK4 slopes
        k1 = velocity_fn(x, t0 + one_vec * (i * h), y)
        k2 = velocity_fn(x + k1 * (h / 2), t0 + one_vec * ((i + 0.5) * h), y)
        k3 = velocity_fn(x + k2 * (h / 2), t0 + one_vec * ((i + 0.5) * h), y)
        k4 = velocity_fn(x + k3 * h, t0 + one_vec * ((i + 1) * h), y)

        x = x + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    return x
