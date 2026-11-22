"""Greeks - first- and second-order sensitivities via central differences

We differentiate the Black-Scholes price for speed and stability. The Heston Monte Carlo
has too much noise to differentiate directly with small bumps. This is what desks call 
`Black-Scholes implied Greeks under stochastic-volatility model`: use Heston to price,
Black-Scholes to compute risk"""

from black_scholes import bs_call_price, bs_put_price


def _pricer(option_type):
    return bs_call_price if option_type == "call" else bs_put_price


def delta(S_0, K, T, r, c, sigma, option_type = "call", dS = .5):
    """Change in option price for change in the price of the underlying asset"""

    p = _pricer(option_type)

    return (p(S_0 + dS, K, T, r, c, sigma) - p(S_0 - dS, K, T, r, c, sigma)) / (2 * dS)


def gamma(S_0, K, T, r, c, sigma, option_type = "call", dS = .5):
    """Change in option Delta for change in the price of the underlying asset"""

    return (delta(S_0 + dS, K, T, r, c, sigma, option_type, dS) -
            delta(S_0 - dS, K, T, r, c, sigma, option_type, dS)) / (2 * dS)


def vega(S_0, K, T, r, c, sigma, option_type = "call", dSigma = .005):
    """Change in option price for change in the volatility of the underlying asset
    
    Reported per 1 vol point (sigma moving by .01). The raw derivative is multiplied by .01.
    Drop that factor if you want it per 1.0 of sigma"""

    p = _pricer(option_type)

    return (p(S_0, K, T, r, c, sigma + dSigma) - p(S_0, K, T, r, c, sigma - dSigma)) / (2 * dSigma) * .01


def theta_greek(S_0, K, T, r, c, sigma, option_type = "call", dT = 1 / 365):
    """Change in option price for a change in the time to maturity of the option"""

    p = _pricer(option_type)

    return p(S_0, K, T - dT, r, c, sigma) - p(S_0, K, T, r, c, sigma)


def all_greeks(S_0, K, T, r, c, sigma, option_type = "call"):
    """Return a dict {"delta", "gamma", "vega", "theta"} for one option"""

    return {
        "delta": delta(S_0, K, T, r, c, sigma, option_type),
        "gamma": gamma(S_0, K, T, r, c, sigma, option_type),
        "vega": vega(S_0, K, T, r, c, sigma, option_type),
        "theta": theta_greek(S_0, K, T, r, c, sigma, option_type)
    }