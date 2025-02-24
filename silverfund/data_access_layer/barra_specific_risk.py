import os
from datetime import date
from pathlib import Path

import polars as pl
from dotenv import load_dotenv


def load_specific_risk(date_: date) -> pl.DataFrame:
    """Loads specific risk data for a given date.

    This function retrieves the specific risk values from the Barra dataset for a given year,
    selecting the relevant date column and formatting the output.

    Args:
        date_ (date): The date for which specific risk data is needed.

    Returns:
        pl.DataFrame: A DataFrame containing specific risk values indexed by `barrid`.

    Example:
        >>> df = load_specific_risk(date(2023, 5, 15))
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
    file = f"spec_risk_{date_.year}.parquet"
    date_column = date_.strftime("%Y-%m-%d 00:00:00") if date else None
    columns = ["Barrid", date_column]
    df = pl.read_parquet(folder / file, columns=columns)

    # Rename columns
    df = df.rename({date_column: "specific_risk", "Barrid": "barrid"})

    # Reorder columns
    df = df.select(["barrid", "specific_risk"])

    return df
