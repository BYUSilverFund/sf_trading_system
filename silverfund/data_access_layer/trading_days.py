import os
from datetime import date
from pathlib import Path

import polars as pl
from dotenv import load_dotenv
from tqdm import tqdm

from silverfund.enums import Interval


def load_trading_days(
    interval: Interval,
    start_date: date | None = None,
    end_date: date | None = None,
    quiet: bool = True,
) -> pl.DataFrame:
    """Loads trading days for a given interval within a specified date range.

    This function retrieves trading days from CRSP daily or monthly datasets,
    filters the data by the given date range, and ensures uniqueness.

    Args:
        interval (Interval): The time interval, either DAILY or MONTHLY.
        start_date (date, optional): The start date for filtering (default: 1995-07-31).
        end_date (date, optional): The end date for filtering (default: today).
        quiet (bool, optional): If True, disables the tqdm loading bar for daily data.

    Returns:
        pl.DataFrame: A DataFrame containing unique trading days sorted by date.

    Example:
        >>> df = load_trading_days(Interval.DAILY, date(2020, 1, 1), date(2023, 1, 1))
        >>> print(df)
    """

    # Parameters
    start_date = start_date or date(1995, 7, 31)
    end_date = end_date or date.today()

    # Get file paths
    load_dotenv()
    parts = os.getenv("ROOT").split("/")
    home = parts[1]
    user = parts[2]
    root_dir = Path(f"/{home}/{user}")

    daily_files_folder = root_dir / "groups" / "grp_quant" / "data" / "dsf"
    monthly_file = root_dir / "groups" / "grp_quant" / "data" / "msf.parquet"

    # Load
    if interval == Interval.DAILY:
        years = range(start_date.year, end_date.year + 1)

        dfs = []
        for year in tqdm(years, desc="Loading Trading Days", disable=quiet):

            file = f"dsf_{year}.parquet"
            dfs.append(pl.read_parquet(daily_files_folder / file, columns=["date"]))

        df = pl.concat(dfs)

    if interval == Interval.MONTHLY:
        df = pl.read_parquet(monthly_file, columns=["date"])

    # Clean
    df = clean(df)

    # Filter
    df = filter(df, start_date, end_date)

    return df


def clean(df: pl.DataFrame) -> pl.DataFrame:
    # Cast date type
    df = df.with_columns(pl.col("date").dt.date())

    # Keep unique and sort
    df = df.unique().sort("date")

    return df


def filter(df: pl.DataFrame, start_date: date, end_date: date) -> pl.DataFrame:
    return df.filter(pl.col("date").is_between(start_date, end_date))
