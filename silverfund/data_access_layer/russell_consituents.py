import os
from pathlib import Path

import polars as pl
from dotenv import load_dotenv


def load_russell_constituents() -> pl.DataFrame:
    """Loads historical Russell index constituent data.

    This function retrieves Russell index constituent data from a Parquet file,
    ensures that date columns are properly formatted, and returns the cleaned data.

    Returns:
        pl.DataFrame: A Polars DataFrame containing Russell index constituent data.

    Example:
        >>> df = load_russell_constituents()
        >>> print(df.head())
    """

    # File path
    load_dotenv()
    parts = os.getenv("ROOT").split("/")
    home = parts[1]
    user = parts[2]
    root_dir = Path(f"/{home}/{user}")
    file_path = root_dir / "groups" / "grp_quant" / "data" / "russell_history.parquet"

    # Load
    df = pl.read_parquet(file_path)

    # Clean
    df = df.with_columns(
        pl.col("date").dt.date(),
        pl.col("obsdate").dt.date(),
        pl.col("enddate").dt.date(),
    )

    return df
