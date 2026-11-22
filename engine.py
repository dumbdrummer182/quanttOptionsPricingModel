"""PricingEngine - the client-facing pricer that ties everything together"""

import math

from black_scholes import bs_call_price, bs_put_price, implied_volatility, put_call_parity_gap
from greeks import all_greeks
from heston import heston_call_price, heston_put_price


class PricingEngine:
    
    def __init__(self, model_type = "bs"):
        self.model_type = model_type
        self.calibrated_params = None
        self.atm_vol = None
        self.cache = {}
    
    def calibrate_bs(self, atm_vol):
        self.atm_vol = atm_vol
    
    def calibrate_heston(self, params):
        self.calibrated_params = params
    
    def price(self, S_0, K, T, r, c, option_type = "call"):
        """Price a single option, caching the result"""

        key = (S_0, K, T, r, c, option_type, self.model_type)
        if key in self.cache:
            return self.cache[key]
        
        if self.model_type == "bs":
            pricer = bs_call_price if option_type == "call" else bs_put_price
            p = pricer(S_0, K, T, r, c, self.atm_vol)
        else:
            pricer = heston_call_price if option_type == "call" else heston_put_price
            cp = self.calibrated_params
            p = pricer(S_0, K, T, r, cp["v_0"], cp["kappa"], cp["theta"], cp["xi"], cp["rho"],
                        n_paths = 3000, n_steps = 80, seed = 42)
        self.cache[key] = p

        return p

    def price_chain(self, S_0, strikes, T, r, c):
        """Return a list of dicts, one per strike, with everything a desk wants"""

        rows = []
        for k in strikes:
            price_c = self.price(S_0, k, T, r, c, "call")
            price_p = self.price(S_0, k, T, r, c, "put")
            sigma_c = implied_volatility(price_c, S_0, k, T, r, c, "call")
            sigma_p = implied_volatility(price_p, S_0, k, T, r, c, "put")
            greeks_c = all_greeks(S_0, k, T, r, c, sigma_c, "call")
            rows.append({
                "strike": k, "call_price": price_c, "put_price": price_p,
                "call_iv": sigma_c, "put_iv": sigma_p,
                "delta": greeks_c["delta"],
                "parity_gap": (price_c - price_p) - (S_0 * math.exp(-c * T) - k * math.exp(-r * T))
            })
        
        return rows
    
    def print_chain(self, chain):
        """Nicely formatted print of a price_chain result"""

        header = "  K        Call     Put      C-IV      P-IV    Delta   Parity"
        print(header)
        for row in chain:
            print(
                f"  {row['strike']:<7.1f}  {row['call_price']:>6.2f}   {row['put_price']:>6.2f}"
                f"   {row['call_iv']*100:>5.2f}%   {row['put_iv']*100:>5.2f}%   "
                f"{row['delta']:>5.3f}   {row['parity_gap']:>+.3f}"
            )