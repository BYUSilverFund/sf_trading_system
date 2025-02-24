import os
from datetime import date
from pathlib import Path

import polars as pl
from dotenv import load_dotenv
from tqdm import tqdm

from silverfund.data_access_layer.trading_days import load_trading_days
from silverfund.enums import Interval


def load_total_risk(
    interval: Interval,
    start_date: date | None = None,
    end_date: date | None = None,
    quiet: bool = True,
) -> pl.DataFrame:
    """Loads Barra total risk data for a specified time interval.

    This function reads daily total risk data from Parquet files stored in a
    predefined directory, cleans the data, and optionally aggregates it to a monthly frequency.
    The data is filtered to the given date range and sorted before returning.

    Args:
        interval (Interval): The time interval for the data. If `Interval.MONTHLY`,
            the data is aggregated to monthly values.
        start_date (date | None, optional): The start date for filtering the data. Defaults to July 31, 1995.
        end_date (date | None, optional): The end date for filtering the data. Defaults to today.
        quiet (bool, optional): If `True`, disables the progress bar during data loading. Defaults to `True`.

    Returns:
        pl.DataFrame: A Polars DataFrame containing the filtered and processed
        Barra total risk data.

    Example:
        >>> from datetime import date
        >>> from silverfund.enums import Interval
        >>> df = load_total_risk(Interval.DAILY, start_date=date(2020, 1, 1), end_date=date(2021, 12, 31))
        >>> print(df.head())
    """

    # Parameters
    start_date = start_date or date(1995, 7, 31)
    end_date = end_date or date.today()

    # Paths
    load_dotenv()
    parts = os.getenv("ROOT").split("/")
    home = parts[1]
    user = parts[2]
    root_dir = Path(f"/{home}/{user}")
    folder = root_dir / "groups" / "grp_quant" / "data" / "barra_usslow_asset"

    # Load daily data
    years = range(start_date.year, end_date.year + 1)

    dfs = []
    for year in tqdm(years, desc="Loading Barra Total Risk", disable=quiet):
        file = f"asset_{year}.parquet"

        # Load
        df = pl.read_parquet(folder / file)

        # Clean
        df = clean(df)

        # Append
        dfs.append(df)

    # Concat
    df: pl.DataFrame = pl.concat(dfs)

    # Aggregate
    if interval == Interval.MONTHLY:
        df = aggregate_to_monthly(df, start_date, end_date)

    # Reorder columns
    df = df.select(
        ["date", "barrid"] + [col for col in sorted(df.columns) if col not in ["date", "barrid"]]
    )

    # Filter
    df = df.filter(pl.col("date").is_between(start_date, end_date))

    # Sort
    df = df.sort(by=["date", "barrid"])

    return df


def clean(df: pl.DataFrame) -> pl.DataFrame:
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

    # Aggregate to monthly level on date and barrid
    df = df.group_by(["date", "barrid"]).agg(
        pl.col("div_yield").last(),
        pl.col("total_risk").last(),
        pl.col("spec_risk").last(),
        pl.col("histbeta").last(),
        pl.col("predbeta").last(),
    )

    return df
