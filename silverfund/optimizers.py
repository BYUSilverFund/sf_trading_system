from typing import Protocol

import cvxpy as cp
import numpy as np

from silverfund.constraints import ConstraintConstructor


class Optimizer(Protocol):
    """
    Protocol for optimization functions used in portfolio optimization.

    Optimizer functions should implement this protocol by accepting alpha values,
    covariance matrix, constraints, and an optional gamma value, and returning portfolio weights.
    """

    def __call__(
        self,
        alphas: np.array,
        cov_mat: np.array,
        constraints: list[ConstraintConstructor],
        gamma: float = 2.0,
    ) -> np.array: ...


def quadratic_program(
    alphas: np.array, cov_mat: np.array, constraints: list[ConstraintConstructor], gamma: float
) -> np.array:
    """
    Solve a quadratic programming problem for portfolio optimization.

    Args:
        alphas (np.ndarray): Array of asset returns (alphas).
        cov_mat (np.ndarray): Covariance matrix of asset returns.
        constraints (List[ConstraintConstructor]): List of constraints for the optimization.
        gamma (float): Risk-aversion parameter, defaults to 2.0.

    Returns:
        np.ndarray: Array of optimized portfolio weights.
    """

    # Declare variables
    n_assets = len(alphas)
    weights = cp.Variable(n_assets)

    constraints = [constraint(weights) for constraint in constraints]

    # Objective function
    portfolio_alpha = weights.T @ alphas
    portfolio_variance = weights.T @ cov_mat @ weights
    objective = cp.Maximize(portfolio_alpha - 0.5 * gamma * portfolio_variance)

    # Formulate problem
    problem = cp.Problem(objective=objective, constraints=constraints)

    # Solve
    problem.solve(solver=cp.OSQP)

    return weights.value
