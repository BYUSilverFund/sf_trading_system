import os
from datetime import date
from pathlib import Path

import polars as pl
from dotenv import load_dotenv


def load_factor_exposures(
    date_: date,
) -> pl.DataFrame:
    """Loads factor exposure data for a given date.

    This function retrieves factor exposures from the Barra dataset, extracting relevant information
    and splitting the combined identifier column into `barrid` and `factor`.

    Args:
        date_ (date): The date for which factor exposure data is needed.

    Returns:
        pl.DataFrame: A DataFrame containing factor exposures with columns `barrid`, `factor`, and `exposure`.

    Example:
        >>> df = load_factor_exposures(date(2023, 5, 15))
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
    file = f"exposures_{date_.year}.parquet"
    date_column = date_.strftime("%Y-%m-%d 00:00:00") if date else None
    columns = ["Combined", date_column]
    df = pl.read_parquet(folder / file, columns=columns)

    # Rename date column
    df = df.rename({date_column: "exposure"})

    # Split Combined colum into barrid and factor
    df = (
        df.with_columns(pl.col("Combined").str.split("/").alias("parts"))
        .with_columns(
            pl.col("parts").list.first().alias("barrid"),
            pl.col("parts").list.last().alias("factor"),
        )
        .drop(["Combined", "parts"])
    )

    # Reorder columns
    df = df.select(["barrid", "factor", "exposure"])

    return df
