"""Heston Monte Carlo - pricing under stochastic volatility

Heston SDE:
    dS = r * S * dt + sqrt(v) * S * dW1
    dv = kappa * (theta - v) * dt + xi * sqrt(v) * dW2
    Corr(dW1, dW2) = rho"""

import numpy as np


def generate_correlated_normals(n_paths, rho, seed = None):
    """Two arrays of standard normals with correlation rho

    Cholesky trick: Cholesky decomposition of 2x2 correlation matrix
        z1 = randn()
        z2 = rho * z1 + sqrt(1 - rho ** 2) * randn()"""
    
    rng = np.random.default_rng(seed)
    z1 = rng.standard_normal(size = n_paths)
    z2 = rho * z1 + np.sqrt(1 - rho ** 2) * rng.standard_normal(size = n_paths)

    return z1, z2


def heston_step(S, v, r, dt, kappa, theta, xi, z1, z2):
    """Advance (S, v) forward by one Euler step"""

    v_pos = np.maximum(v, 0)
    v_new = v + kappa * (theta - v) * dt + xi * np.sqrt(v_pos) * np.sqrt(dt) * z1
    S_new = S * np.exp((r - v_pos / 2) * dt + np.sqrt(v_pos) * np.sqrt(dt) * z2)

    return S_new, v_new


def heston_simulate(S_0, v_0, kappa, theta, xi, rho, r, T, n_paths = 5000, n_steps = 100, seed = None):
    """Simulate `n_paths` correlated (S, v) paths over `n_steps` from 0 to T

    Returns (S, v) both arrays of shape (n_paths, n_steps + 1). Column 0 holds initial values"""

    rng = np.random.default_rng(seed = seed)
    dt = T / n_steps
    S = np.empty((n_paths, n_steps + 1))
    v = np.empty_like(S)
    S[:, 0] = S_0
    v[:, 0] = v_0
    v1 = rng.standard_normal((n_steps, n_paths))
    v2 = rng.standard_normal((n_steps, n_paths))
    z1 = v1
    z2 = rho * v1 + np.sqrt(1 - rho ** 2) * v2

    for t in range(n_steps):
        S[:, t + 1], v[:, t + 1] = heston_step(S[:, t], v[:, t], r, dt, kappa, theta, xi, z1[t], z2[t])
    
    return S, v


def heston_call_price(S_0, K, T, r, v_0, kappa, theta, xi, rho, n_paths = 5000, n_steps = 100, seed = None):
    """Price a European call option via Heston Monte Carlo"""

    S, _v = heston_simulate(S_0, v_0, kappa, theta, xi, rho, r, T, n_paths = n_paths, n_steps = n_steps, seed = seed)
    payoffs = np.maximum(S[:, -1] - K, 0)

    return np.exp(-r * T) * payoffs.mean()


def heston_put_price(S_0, K, T, r, v_0, kappa, theta, xi, rho, n_paths = 5000, n_steps = 100, seed = None):
    """Price a European put option via Heston Monte Carlo"""

    S, _v = heston_simulate(S_0, v_0, kappa, theta, xi, rho, r, T, n_paths = n_paths, n_steps = n_steps, seed = seed)
    payoffs = np.maximum(K - S[:, -1], 0)

    return np.exp(-r * T) * payoffs.mean()