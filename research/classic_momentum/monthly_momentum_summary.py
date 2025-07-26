from datetime import date

import polars as pl

from silverfund.enums import Compounding, Interval
from silverfund.performance import Performance
from silverfund.records import AssetReturns

if __name__ == "__main__":
    folder = "research/classic_momentum/results/"

    # Asset returns (result of backtester)
    data_file_path = folder + "monthly_momentum_bt.parquet"
    asset_returns = AssetReturns(returns=pl.read_parquet(data_file_path))

    # Performance instance
    performance = Performance(
        start_date=date(1995, 7, 31),
        end_date=date(2024, 12, 31),
        interval=Interval.MONTHLY,
        asset_returns=asset_returns,
        annualize=True,
    )

    # Chart
    title = "Monthly Momentum Backtest"
    decomposed_plot_file_path = folder + "monthly_momentum.png"
    performance.plot_returns(
        compounding=Compounding.SUM,
        title=title,
        decompose=True,
        save_file_path=decomposed_plot_file_path,
    )

    # Table
    summary_file_path = folder + "monthly_momentum.txt"
    performance.summary(summary_file_path)

    # Print summary
    print(performance.summary())
