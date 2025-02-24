from typing import Protocol

import polars as pl

from silverfund.records import Signal


class SignalConstructor(Protocol):
    """Protocol for functions that construct a Signal from a Polars DataFrame."""

    def __call__(self, data: pl.DataFrame) -> Signal: ...


def momentum(data: pl.DataFrame) -> Signal:
    """Computes the momentum signal based on log returns.

    The momentum signal is calculated by first computing the log returns (`logret`)
    of the `ret` column, then calculating the 11-period rolling sum of the log returns
    within each `barrid` group. The resulting momentum is shifted by one period to ensure
    it represents the previous period's momentum for each asset.

    Args:
        data (pl.DataFrame): A Polars DataFrame containing asset return data,
                              where 'ret' represents the returns of assets.

    Returns:
        Signal: A Signal object containing the 'date', 'barrid', and computed 'mom' (momentum) columns.
    """
    signals = (
        data.with_columns(pl.col("ret").log1p().alias("logret"))
        .with_columns(pl.col("logret").rolling_sum(11, min_periods=11).over("barrid").alias("mom"))
        .with_columns(pl.col("mom").shift(1).over("barrid"))
        .select(["date", "barrid", "mom"])
        .sort(["barrid", "date"])
    )
    return Signal(signals, "mom")
