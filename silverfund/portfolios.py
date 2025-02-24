from datetime import date
from functools import partial
from typing import Protocol

import polars as pl

from silverfund.alphas import Alpha
from silverfund.constraints import ConstraintConstructor
from silverfund.covariance_matrix import covariance_matrix_constructor
from silverfund.optimizers import quadratic_program
from silverfund.records import Portfolio


class PortfolioConstructor(Protocol):
    """Protocol for functions that construct a Portfolio from alphas and constraints."""

    def __call__(
        self,
        period: date,
        barrids: list[str],
        alphas: Alpha,
        constraints: list[ConstraintConstructor],
    ) -> Portfolio: ...


def mean_variance_efficient(
    period: date,
    barrids: list[str],
    alphas: Alpha,
    constraints: list[ConstraintConstructor],
    gamma: float = 2.0,
) -> Portfolio:
    """Constructs a mean-variance efficient portfolio using quadratic optimization.

    This function builds an optimal portfolio by solving a quadratic programming
    problem that balances expected returns (alphas) and risk (covariance matrix)
    subject to given constraints.

    Args:
        period (date): The date for which the portfolio is constructed.
        barrids (list[str]): List of asset identifiers (barrids) included in the portfolio.
        alphas (Alpha): Expected returns for the assets.
        constraints (list[ConstraintConstructor]): List of constraints applied to the optimization.
        gamma (float, optional): Risk aversion parameter (default is 2.0).
                                 Higher values penalize risk more heavily.

    Returns:
        Portfolio: A Polars DataFrame wrapped in the Portfolio class,
                   containing 'date', 'barrid', and 'weight' columns.
    """

    # Get covariance matrix
    cov_mat = covariance_matrix_constructor(period, barrids)

    # Cast to numpy arrays
    alphas = alphas.to_vector()
    cov_mat = cov_mat.to_matrix()

    # Construct constraints
    constraints = [partial(constraint, date_=period, barrids=barrids) for constraint in constraints]

    # Find optimal weights
    weights = quadratic_program(alphas, cov_mat, constraints, gamma)

    portfolio = pl.DataFrame({"date": period, "barrid": barrids, "weight": weights})
    portfolio = portfolio.sort(["barrid", "date"])

    return Portfolio(portfolio)
