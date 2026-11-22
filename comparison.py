"""Black-Scholes vs Heston fits - quantify how much better a stochastic volatility
model reproduces the market smile"""

import numpy as np
from black_scholes import bs_call_price, generate_market_smile, implied_volatility
from heston import heston_call_price


def bs_fit(S_0, strikes, T, r, c, atm_vol):
    """Price the option chain under Black-Scholes with a single sigma"""

    prices = np.array([
        bs_call_price(S_0, k, T, r, c, atm_vol)
        for k in strikes
    ])
    iv = np.array([
        implied_volatility(cp, S_0, k, T, r, c, "call")
        for cp, k in zip(prices, strikes)
    ])

    return prices, iv


def heston_fit(S_0, strikes, T, r, v_0, kappa, theta, xi, rho, n_paths = 4000, n_steps = 80, seed = 42):
    """Price the option chain under Heston, them re-imply sigma per strike"""

    prices = np.array([
        heston_call_price(S_0, k, T, r, v_0, kappa, theta, xi, rho, n_paths = n_paths, n_steps = n_steps, seed = seed)
        for i, k in enumerate(strikes)
    ])
    iv = np.array([
        implied_volatility(cp, S_0, k, T, r, c = 0, option_type = "call")
        for cp, k in zip(prices, strikes)
    ])

    return prices, iv


def rmse(a, b):
    """Root-mean-square error between two 1-D arrays"""

    a = np.asarray(a)
    b = np.asarray(b)

    return np.sqrt(np.mean((a - b) ** 2))


def build_market_data(S_0, strikes, T, r, c, base_vol = .2):
    """Synthetic market data to use in model calibration - (strike, market_price, implied_vol)"""

    iv, prices = generate_market_smile(S_0, strikes, T, r, c, base_vol)

    return [
        {"strike": k, "market_price": cp, "market_iv": ivs}
        for k, cp, ivs in zip(strikes, prices, iv)
    ]

