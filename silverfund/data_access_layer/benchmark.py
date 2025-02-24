from datetime import date

import polars as pl

from silverfund.data_access_layer.barra_returns import load_barra_returns
from silverfund.data_access_layer.universe import load_universe
from silverfund.enums import Interval


def load_benchmark(
    interval: Interval,
    start_date: date | None = None,
    end_date: date | None = None,
) -> pl.DataFrame:
    """Loads benchmark weights based on market capitalization.

    This function retrieves the universe of assets and their respective Barra
    returns for a given time interval and date range. It computes the total
    market capitalization for each day and calculates individual asset weights
    relative to the total market cap.

    Args:
        interval (Interval): The time interval for the data (e.g., daily, monthly).
        start_date (date | None, optional): The start date for filtering the data. Defaults to `None`, meaning earliest available data is used.
        end_date (date | None, optional): The end date for filtering the data. Defaults to `None`, meaning the latest available data is used.

    Returns:
        pl.DataFrame: A Polars DataFrame containing the benchmark weights for each asset.

    Example:
        >>> from datetime import date
        >>> from silverfund.enums import Interval
        >>> df = load_benchmark(Interval.DAILY, start_date=date(2020, 1, 1), end_date=date(2021, 12, 31))
        >>> print(df.head())
    """

    # Load universe
    universe = load_universe(interval=interval, start_date=start_date, end_date=end_date)

    # Load barra returns
    barra = load_barra_returns(interval=interval, start_date=start_date, end_date=end_date)

    # Merge universe and returns
    benchmark = universe.join(barra, on=["date", "barrid"], how="left").sort(["date", "barrid"])

    # Get total market cap by day
    total_market_cap = (
        benchmark.group_by("date").agg(pl.col("mktcap").sum().alias("total_mktcap")).sort("date")
    )

    # Create benchmark weights
    benchmark = (
        benchmark.join(total_market_cap, on="date", how="left")
        .with_columns(
            (pl.col("mktcap") / pl.col("total_mktcap")).alias("weight"),
        )
        .select(["date", "barrid", "weight"])
    )

    return benchmark
