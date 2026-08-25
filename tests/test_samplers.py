"""
Tests for the samplers: euler, heun, and rk4.

Three velocities to test:
- constant case:            v(x, t; y) = c0                 x(t) = x0 + c0 (t - t0)
- linear case:              v(x, t; y) = lambda * x         x(t) = x0 * exp[lambda * (t - t0)]
- non-autonomous case:      v(x, t; y) = cos(lambda * t)    x(t) = x0 + (sin(lambda * t) - sin(lambda * t0)) / lambda

Testing based on number of iterations needed to achieve error bound.
The tolerances are measured, not derived.
"""
import pytest
import torch
import math

from image_diffusion import REGISTRY

# Hyperparameters of the tests
X0 = torch.ones(1)
C0 = 2
T0 = 0
T1 = 1
LMD = 3.5
# Budget - (number of steps, tolerance)
BUDGETS = {
    "linear":         {"euler": (int(1e4), 3e-2), "heun": (int(1e3), 1e-3), "rk4": (int(1e2), 1e-4)},
    "non_autonomous": {"euler": (int(1e4), 1e-2), "heun": (int(1e3), 1e-3), "rk4": (int(1e2), 1e-4)},
}

def constant_case(x, t, y):
    return C0


def linear_case(x, t, y, lmd=LMD):
    return lmd * x


def non_autonomous_case(x, t, y, lmd=LMD):
    return torch.cos(lmd * t)


@pytest.mark.parametrize("name,solver", REGISTRY.items(), ids=list(REGISTRY))
def test_constant(name, solver):
    assert torch.allclose(solver(constant_case, x=X0, y=-1, num_steps=1, t0=T0, t1=T1), X0 + C0 * (T1 - T0))


@pytest.mark.parametrize("name,solver", REGISTRY.items(), ids=list(REGISTRY))
def test_linear_case(name, solver):
    num_steps, tol = BUDGETS["linear"][name]
    assert torch.norm(solver(linear_case, x=X0, y=-1, num_steps=num_steps, t0=T0, t1=T1) - 
                      X0 * math.exp(LMD * (T1 - T0))) < tol


@pytest.mark.parametrize("name,solver", REGISTRY.items(), ids=list(REGISTRY))
def test_non_autonomous_case(name, solver):
    num_steps, tol = BUDGETS["non_autonomous"][name]
    assert torch.norm(solver(non_autonomous_case, x=X0, y=-1, num_steps=num_steps, t0=T0, t1=T1) 
                      - (X0 + (1 / LMD) * (math.sin(LMD * T1) - math.sin(LMD * T0)))) < tol
