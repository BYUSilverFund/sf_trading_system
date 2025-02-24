from typing import Protocol

import polars as pl

import silverfund.data_access_layer as dal
from silverfund.enums import Interval
from silverfund.records import Alpha, Score


class AlphaConstructor(Protocol):
    """Protocol for functions that construct Alpha values from Score objects."""

    def __call__(self, score: Score) -> Alpha: ...


def grindold_kahn(scores: Score, interval: Interval, ic: float = 0.05) -> Alpha:
    """Computes alpha values using the Grindold-Kahn methodology.

    This method adjusts scores using the total risk data from Barra and
    an information coefficient (IC) to compute alpha estimates.

    Args:
        scores (Score): A Polars DataFrame containing score data with 'date' and 'barrid' columns.
        interval (Interval): The time interval for loading total risk data.
        ic (float, optional): The information coefficient (default is 0.05).

    Returns:
        Alpha: A Polars DataFrame with computed alpha values, containing 'date', 'barrid', and 'alpha' columns.
    """

    start_date = scores["date"].min()
    end_date = scores["date"].max()

    vols = dal.load_total_risk(interval, start_date, end_date).with_columns(pl.col("spec_risk"))

    return Alpha(
        scores.join(other=vols, on=["date", "barrid"], how="left")
        .with_columns(((ic * pl.col("total_risk") * pl.col("score")).alias("alpha")))
        .fill_null(0)
        .select(["date", "barrid", "alpha"])
        .sort(["barrid", "date"])
    )


def static_alpha(scores: Score, value: float) -> Alpha:
    """Assigns a constant alpha value to all records in the score data.

    This method replaces the 'score' column with a fixed alpha value.

    Args:
        scores (Score): A Polars DataFrame containing score data.
        value (float): The fixed alpha value to be assigned.

    Returns:
        Alpha: A Polars DataFrame with 'date', 'barrid', and a constant 'alpha' column.
    """
    return Alpha(scores.with_columns(pl.lit(value).cast(pl.Float64).alias("alpha")).drop("score"))
