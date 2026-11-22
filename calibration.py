"""Calibration - find the Heston (v_0, xi, rho) that best fit the market prices

We fix kappa = 2 and theta = v_0 to keep this a 3-D search. Real desks calibrate all
five (or eight, including a time-dependent vol structure) using analytic Heston pricing
via characteristic functions - much faster than Monte Carlo - but the optimization pattern
is identical"""

import numpy as np
from heston import heston_call_price


# calibration bounds - [(v0_low, v0_high), (xi_low, xi_high), (rho_low, rho_high)]
CALIB_BOUNDS = [(.005, .4), (.05, 1.5), (-.99, 0.0)]

def calibration_objective(params, market_data, S_0, T, r, n_paths = 1000, n_steps = 40, seed = 42):
    """Sum of squared pricve errors across every option in `market_data`

    If any parameter is outside its CALIB_BOUNDS range, return the penalty 1e6 without simulating
    this keeps the optimizer out of nonsense regions"""

    for value, (lo, hi) in zip(params, CALIB_BOUNDS):
        if not lo <= value <= hi:
            return 1e6
    
    v_0, xi, rho = params
    kappa = 2
    theta = v_0
    total = 0

    for i, row in enumerate(market_data):
        price = heston_call_price(S_0, row["strike"], T, r, v_0, kappa, theta, xi, rho,
                                n_paths = n_paths, n_steps = n_steps, seed = seed + i)
        total += (price - row["market_price"]) ** 2
    
    return total


def calibrate(market_data, S_0, T, r, x_0 = (.04, .5, -.5), n_paths = 1000, n_steps = 40, max_iter = 40, seed = 42):
    """Run Nelder-Mead algorithm on the objective to minimize errors

    Real desks avoid Monte Carlo in the calibration loop but using analytic Heston pricing
    via the Fourier transform (Carr-Madan)"""

    from scipy.optimize import minimize

    result = minimize(
        calibration_objective,
        x0 = list(x_0),
        args = (market_data, S_0, T, r, n_paths, n_steps, seed),
        method = "Nelder-Mead",
        options = {"maxiter": max_iter, "xatol": 1e-3, "fatol": 1e-3}
    )

    v_0, xi, rho = result.x
    rmse_ = (result.fun / len(market_data)) ** .5

    return {
        "v_0": v_0,
        "xi": xi,
        "rho": rho,
        "kappa": 2,
        "theta": v_0,
        "rmse": rmse_,
        "evals": int(result.nfev)
    }


def split_by_moneyness(market_data, calibration_strikes):
    """Split `market_data` into a calibration_set and validation_set"""

    cs = set(calibration_strikes)   # creates a set of only unique values
    calib = []
    valid = []

    for i, row in enumerate(market_data):
        if row["strike"] in cs:
            calib.append(row)
        else:
            valid.append(row)

    return calib, valid


def validate(market_data, calibrated, S_0, T, r, n_paths = 2000, n_steps = 80, seed = 42):
    """Reprice each option under the calibrated params and return RMSE"""

    errors = []
    for i, row in enumerate(market_data):
        price = heston_call_price(S_0, row["strike"], T, r,
                                calibrated["v_0"], calibrated["kappa"],
                                calibrated["theta"], calibrated["xi"],
                                calibrated["rho"],
                                n_paths = n_paths, n_steps = n_steps, seed = seed + i)
        errors.append(price - row["market_price"])
    errors = np.array(errors)

    return np.sqrt(np.mean(errors ** 2))