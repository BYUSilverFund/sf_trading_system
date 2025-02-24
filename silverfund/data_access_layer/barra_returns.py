import os
from datetime import date
from pathlib import Path

import polars as pl
from dotenv import load_dotenv
from tqdm import tqdm

from silverfund.data_access_layer.trading_days import load_trading_days
from silverfund.enums import Interval


def load_barra_returns(
    interval: Interval,
    start_date: date | None = None,
    end_date: date | None = None,
) -> pl.DataFrame:
    """Loads Barra returns data for a specified time interval.

    This function reads daily return data from Parquet files stored in a specific
    directory, cleans the data, and optionally aggregates it to a monthly frequency.
    The data is filtered to the given date range and sorted before returning.

    Args:
        interval (Interval): The time interval for the data. If `Interval.MONTHLY`,
            the data is aggregated to monthly returns.
        start_date (date, optional): The start date for filtering the data. Defaults
            to July 31, 1995, if not provided.
        end_date (date, optional): The end date for filtering the data. Defaults to
            the current date if not provided.

    Returns:
        pl.DataFrame: A Polars DataFrame containing the filtered and processed
        Barra returns data.

    Example:
        >>> from datetime import date
        >>> from silverfund.enums import Interval
        >>> df = load_barra_returns(Interval.DAILY, start_date=date(2020, 1, 1), end_date=date(2021, 12, 31))
        >>> print(df.head())
    """

    # Parameters
    start_date = start_date or date(1995, 7, 31)
    end_date = end_date or date.today()

    # File paths
    load_dotenv()
    parts = os.getenv("ROOT").split("/")
    home = parts[1]
    user = parts[2]
    root_dir = Path(f"/{home}/{user}")
    folder = root_dir / "groups" / "grp_quant" / "data" / "barra_usslow_ret"

    # Load daily data
    years = range(start_date.year, end_date.year + 1)

    dfs = []
    for year in tqdm(years, desc="Loading Barra Returns"):
        file = f"ret_{year}.parquet"

        # Load
        df = pl.read_parquet(folder / file)

        # Clean
        df = clean(df)

        # Append
        dfs.append(df)

    # Concat
    df = pl.concat(dfs)

    # Aggregate to monthly
    if interval == Interval.MONTHLY:
        df = aggregate_to_monthly(df, start_date, end_date)

    # Filter
    df = df.filter(pl.col("date").is_between(start_date, end_date))

    # Sort
    df = df.sort(by=["barrid", "date"])

    return df


def clean(df: pl.DataFrame):
    # Drop index column
    df = df.drop("__index_level_0__")

    # Cast and rename date
    df = df.with_columns(pl.col("DataDate").dt.date().alias("date")).drop("DataDate")

    # Lowercase columns
    df = df.rename({col: col.lower() for col in df.columns})

    # Reorder columns
    df = df.select(
        ["date", "barrid"] + [col for col in sorted(df.columns) if col not in ["date", "barrid"]]
    )

    return df


def aggregate_to_monthly(df: pl.DataFrame, start_date: date, end_date: date) -> pl.DataFrame:
    # Add month column to trading days
    monthly_trading_days = load_trading_days(Interval.MONTHLY, start_date, end_date)
    monthly_trading_days = monthly_trading_days.with_columns(
        pl.col("date").dt.truncate("1mo").alias("month")
    )

    # Add month column to df
    df = df.with_columns(pl.col("date").dt.truncate("1mo").alias("month")).drop("date")

    # Merge on month end trading days
    df = df.join(monthly_trading_days, on="month", how="left").drop("month")

    # Add logret column
    df = df.with_columns(pl.col("ret").log1p().alias("logret"))

    # Aggregate
    df = df.group_by(["date", "barrid"]).agg(
        pl.col("currency").last(),
        pl.col("mktcap").last(),
        pl.col("price").last(),
        pl.col("logret").sum(),
    )

    # Compound up log returns
    df = df.with_columns((pl.col("logret").exp() - 1).alias("ret"))

    return df
