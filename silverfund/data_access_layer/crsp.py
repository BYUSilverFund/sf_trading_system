import os
from datetime import date
from pathlib import Path

import polars as pl
from dotenv import load_dotenv
from tqdm import tqdm

from silverfund.enums import Interval

SCHEMA = {
    "permno": pl.Int64,
    "permco": pl.Int64,
    "date": pl.Date,
    "ncusip": pl.String,
    "ticker": pl.String,
    "shrcd": pl.Int64,
    "exchcd": pl.Int64,
    "siccd": pl.Int64,
    "prc": pl.Float64,
    "ret": pl.Float64,
    "retx": pl.Float64,
    "vol": pl.Float64,
    "shrout": pl.Float64,
    "cfacshr": pl.Float64,
}


def load_crsp(
    interval: Interval,
    start_date: date | None = None,
    end_date: date | None = None,
) -> pl.DataFrame:
    """Loads CRSP (Center for Research in Security Prices) data.

    This function retrieves either daily or monthly CRSP data, cleans it according
    to a predefined schema, and filters it based on the given date range.

    Args:
        interval (Interval): The time interval for the data (e.g., daily or monthly).
        start_date (date | None, optional): The start date for filtering the data. Defaults to `1925-12-31`.
        end_date (date | None, optional): The end date for filtering the data. Defaults to today.

    Returns:
        pl.DataFrame: A Polars DataFrame containing CRSP data for the given interval and date range.

    Example:
        >>> from datetime import date
        >>> from silverfund.enums import Interval
        >>> df = load_crsp(Interval.DAILY, start_date=date(2000, 1, 1), end_date=date(2020, 12, 31))
        >>> print(df.head())
    """

    # Parameters
    start_date = start_date or date(1925, 12, 31)
    end_date = end_date or date.today()

    # File paths
    load_dotenv()
    parts = os.getenv("ROOT").split("/")
    home = parts[1]
    user = parts[2]
    root_dir = Path(f"/{home}/{user}")
    daily_files_folder = root_dir / "groups" / "grp_quant" / "data" / "dsf"
    monthly_file = root_dir / "groups" / "grp_quant" / "data" / "msf.parquet"

    if interval == Interval.DAILY:
        # Load daily data
        years = range(start_date.year, end_date.year + 1)

        dfs = []
        for year in tqdm(years, desc="Loading CRSP Daily"):
            file = f"dsf_{year}.parquet"

            # Load
            df = pl.read_parquet(daily_files_folder / file)

            # Clean
            df = clean(df)

            # Append
            dfs.append(df)

        # Concat
        df = pl.concat(dfs)

    if interval == Interval.MONTHLY:
        # Load
        df = pl.read_parquet(monthly_file)

        # Clean
        df = clean(df)

    # Filter
    df = df.filter(pl.col("date").is_between(start_date, end_date))

    # Sort
    df = df.sort(by=["permno", "date"])

    return df


def clean(df: pl.DataFrame) -> pl.DataFrame:
    # Cast schema
    df = df.cast(SCHEMA)

    return df
