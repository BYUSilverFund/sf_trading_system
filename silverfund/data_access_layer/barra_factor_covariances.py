import os
from datetime import date
from pathlib import Path

import polars as pl
from dotenv import load_dotenv


def load_factor_covariances(date_: date) -> pl.DataFrame:
    """Loads factor covariance data for a given date.

    This function retrieves factor covariances from the Barra dataset, extracting relevant information
    and splitting the combined identifier column into `factor_1` and `factor_2`.

    Args:
        date_ (date): The date for which factor covariance data is needed.

    Returns:
        pl.DataFrame: A DataFrame containing factor covariances with columns `factor_1`, `factor_2`, and `covariance`.

    Example:
        >>> df = load_factor_covariances(date(2023, 5, 15))
        >>> print(df)
    """

    # Paths
    load_dotenv()
    parts = os.getenv("ROOT").split("/")
    home = parts[1]
    user = parts[2]
    root_dir = Path(f"/{home}/{user}")
    folder = root_dir / "groups" / "grp_quant" / "data" / "barra_usslow"

    # Load
    file = f"factor_covariance_{date_.year}.parquet"
    date_column = date_.strftime("%Y-%m-%d 00:00:00") if date else None
    columns = ["Combined", date_column]
    df = pl.read_parquet(folder / file, columns=columns)

    # Rename date column
    df = df.rename({date_column: "covariance"})

    # Split Combined column into factor_1 and factor_2
    df = (
        df.with_columns(pl.col("Combined").str.split("/").alias("parts"))
        .with_columns(
            pl.col("parts").list.first().alias("factor_1"),
            pl.col("parts").list.last().alias("factor_2"),
        )
        .drop(["Combined", "parts"])
    )

    # Reorder columns
    df = df.select(["factor_1", "factor_2", "covariance"])

    return df
