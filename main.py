from calibration import calibrate
from comparison import build_market_data
from engine import PricingEngine
from greeks import delta, gamma, vega


def main():

    S_0, T, r, c = 100, .5, .05, 0
    strikes = list(range(85, 116, 3))

    market = build_market_data(S_0, strikes, T, r, c, base_vol = .2)
    calib = calibrate(market, S_0, T, r, n_paths = 1000, n_steps = 40, max_iter = 30)

    print(
        f"v_0={calib['v_0']:.4f}  xi={calib['xi']:.3f}  rho={calib['rho']:+.3f}   "
        f"pricing RMSE ${calib['rmse']:.3f}"
    )

    engine = PricingEngine(model_type = "heston")
    engine.calibrate_heston(calib)
    chain = engine.price_chain(S_0, strikes, T, r, c)
    engine.print_chain(chain)


if __name__ == "__main__":
    main()