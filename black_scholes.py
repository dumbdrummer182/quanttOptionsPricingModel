"""Black-scholes - closed-form option pricing under constant volatility"""

import numpy as np
from scipy.stats import norm

def bs_d1_d2(S_0, K, T, r, c, sigma):
    """Return the (d1, d2) pair from the Black-Scholes formula
    
    Variables:
        S_0:     initial stock price
        K:       option strike price
        T:       time to maturity (years)
        r:       risk-free rate
        c:       dividend yield
        sigma:   static volatility"""
    
    d1 = (np.log(S_0 / K) + (r - c + (sigma ** 2) / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    return (d1, d2)


def bs_call_price(S_0, K, T, r, c, sigma):
    """Black-Scholes European call price"""

    d1, d2 = bs_d1_d2(S_0, K, T, r, c, sigma)
    N_d1 = norm.cdf(d1)
    N_d2 = norm.cdf(d2)

    return S_0 * np.exp(-c * T) * N_d1 - K * np.exp(-r * T) * N_d2


def bs_put_price(S_0, K, T, r, c, sigma):
    """Black-Scholes European put price"""

    d1, d2 = bs_d1_d2(S_0, K, T, r, c, sigma)
    N_d1 = norm.cdf(-d1)
    N_d2 = norm.cdf(-d2)

    return K * np.exp(-r * T) * N_d2 - S_0 * np.exp(-c * T) * N_d1


def put_call_parity_gap(S_0, K, T, r, c, sigma):
    """The absolute value of violation of put-call parity, should be ~0

    put-call parity:   C - P = S_0 * exp(-c * T) - K * exp(-r * T)"""

    call_price = bs_call_price(S_0, K, T, r, c, sigma)
    put_price = bs_put_price(S_0, K, T, r, c, sigma)

    return abs(call_price - put_price - (S_0 * np.exp(-c * T)) + (K * np.exp(-r * T)))


def implied_volatility(market_price, S_0, K, T, r, c, option_type = "call", low = 1e-4, high = 5, max_iter = 60):
    """Bisection solver - solve for sigma that makes Black-Scholes price equal `market_price`"""

    pricer = bs_call_price if option_type == "call" else bs_put_price

    for _ in range(max_iter):
        mid = (low + high) / 2
        if pricer(S_0, K, T, r, c, mid) > market_price:
            high = mid
        else:
            low = mid
    
    return (low + high) / 2


def generate_market_smile(S_0, strikes, T, r, c, base_vol = .2):
    """Return synthetic 'observed' implied volatilities

    Formula, used to generate synthetic options volatilities:
        moneyness:     m = (K - S_0) / S_0
        implied vol:   iv(k) = base_vol - .25 * m + .3 * m ** 2"""
    
    strikes = np.asarray(strikes, dtype = float)
    m = (strikes - S_0) / S_0
    iv = base_vol - .25 * m + .3 * m ** 2

    prices = np.array([
        bs_call_price(S_0, k, T, r, c, sig) for k, sig in zip(strikes, iv)
    ])

    return iv, prices