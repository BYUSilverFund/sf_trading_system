from typing import Protocol

import polars as pl

from silverfund.records import Score, Signal


class ScoreConstructor(Protocol):
    """Protocol for functions that construct a Score from signals."""

    def __call__(self, signals: Signal, col: str, over: str) -> Score: ...


def z_score(signals: Signal, signal_col: str) -> Score:
    """Computes the z-score normalization of a given signal.

    The z-score is computed by subtracting the mean and dividing by the
    standard deviation of the signal within each date group. This standardization
    allows signals to be compared across different time periods.

    Args:
        signals (Signal): A dataset containing asset signals.
        signal_col (str): The column in `signals` to normalize.

    Returns:
        Score: A Polars DataFrame wrapped in the Score class,
               containing 'date', 'barrid', and the standardized 'score' column.
    """
    return Score(
        signals.with_columns(
            (
                (pl.col(signal_col) - pl.col(signal_col).mean().over("date"))
                / pl.col(signal_col).std().over("date")
            ).alias("score")
        )
        .select(["date", "barrid", "score"])
        .sort(["barrid", "date"])
    )
