import os
from datetime import date
from functools import partial
from pathlib import Path

import silverfund.data_access_layer as dal
from silverfund.alphas import grindold_kahn
from silverfund.backtester import Backtester
from silverfund.constraints import full_investment, long_only, no_buying_on_margin, unit_beta
from silverfund.enums import Interval
from silverfund.portfolios import mean_variance_efficient
from silverfund.scores import z_score
from silverfund.signals import momentum
from silverfund.strategies import Strategy

if __name__ == "__main__":
    # Date range
    start_date = date(1995, 7, 31)
    end_date = date(2024, 12, 31)
    interval = Interval.MONTHLY

    # Define strategy
    strategy = Strategy(
        signal_constructor=momentum,
        score_constructor=partial(z_score, signal_col="mom"),
        alpha_constructor=partial(grindold_kahn, interval=interval),
        portfolio_constructor=mean_variance_efficient,
        constraints=[
            full_investment,
            no_buying_on_margin,
            long_only,
            partial(unit_beta, interval=interval),
        ],
    )

    # Create training data
    universe = dal.load_universe(interval=interval, start_date=start_date, end_date=end_date)
    training_data = universe.join(
        dal.load_barra_returns(interval=interval, start_date=start_date, end_date=end_date),
        on=["date", "barrid"],
        how="left",
    ).sort(["barrid", "date"])

    # Instantiate backtester
    bt = Backtester(interval=interval, start_date=start_date, end_date=end_date, data=training_data)

    # Run in parallel
    asset_returns = bt.run_parallel(strategy)
    print("-" * 20 + " Asset Returns " + "-" * 20)
    print(asset_returns)

    # Save results
    folder = Path("research/classic_momentum/results")
    os.makedirs(folder, exist_ok=True)
    asset_returns.write_parquet(folder / "monthly_momentum_bt.parquet")
