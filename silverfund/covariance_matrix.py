from datetime import date

import numpy as np
import polars as pl

import silverfund.data_access_layer as dal
from silverfund.records import CovarianceMatrix


def covariance_matrix_constructor(date_: date, barrids: list[str]) -> CovarianceMatrix:
    """
    Constructs the covariance matrix based on exposures, factor covariances, and specific risks.

    Args:
        date_ (date): The date for which the covariance matrix is computed.
        barrids (List[str]): List of Barrid identifiers for the assets.

    Returns:
        CovarianceMatrix: The computed covariance matrix wrapped in a CovarianceMatrix object.
    """
    # Load
    exposures_matrix = factor_exposure_matrix_constructor(date_, barrids).drop("barrid").to_numpy()
    covariance_matrix = factor_covariance_matrix_constructor(date_).drop("factor_1").to_numpy()
    idio_risk_matrix = specific_risk_matrix(date_, barrids).drop("barrid").to_numpy()

    # Compute covariance matrix
    covariance_matrix = exposures_matrix @ covariance_matrix @ exposures_matrix.T + idio_risk_matrix

    # Put in decimal space
    covariance_matrix = covariance_matrix / (100**2)

    # Package
    covariance_matrix = pl.DataFrame(
        {
            "barrid": barrids,
            **{id: covariance_matrix[:, i] for i, id in enumerate(barrids)},
        }
    )

    return CovarianceMatrix(covariance_matrix, barrids=barrids)


def factor_exposure_matrix_constructor(date_: date, barrids: list[str]) -> pl.DataFrame:
    """
    Constructs the factor exposure matrix for the given date and Barrids.

    Args:
        date_ (date): The date for which the factor exposure matrix is computed.
        barrids (List[str]): List of Barrid identifiers for the assets.

    Returns:
        pl.DataFrame: The factor exposure matrix.
    """
    # Barrids
    barrids_df = pl.DataFrame({"barrid": barrids})

    # Load
    bfe = dal.load_factor_exposures(date_)

    # Factors
    factors = bfe["factor"].unique().sort().to_list()

    # Filter
    bfe = barrids_df.join(bfe, how="left", on="barrid").fill_null(0)

    # Pivot
    exp_mat = bfe.pivot(on="factor", index="barrid", values="exposure")

    # Fill null
    exp_mat = exp_mat.fill_null(0)

    # Reorder columns
    exp_mat = exp_mat.select(["barrid"] + factors)

    # Sort
    exp_mat = exp_mat.sort("barrid")

    return exp_mat


def factor_covariance_matrix_constructor(date_: date) -> pl.DataFrame:
    """
    Constructs the factor covariance matrix for the given date.

    Args:
        date_ (date): The date for which the factor covariance matrix is computed.

    Returns:
        pl.DataFrame: The factor covariance matrix.
    """
    # Load
    fc_df = dal.load_factor_covariances(date_)

    # Pivot
    fc_df = fc_df.pivot(on="factor_2", index="factor_1", values="covariance")

    # Sort headers and columns
    fc_df = fc_df.select(["factor_1"] + sorted([col for col in fc_df.columns if col != "factor_1"]))
    fc_df = fc_df.sort("factor_1")

    # Record factor ids
    factors = fc_df.select("factor_1").to_numpy().flatten()

    # Convert from upper triangular to symetric
    utm = fc_df.drop("factor_1").to_numpy()
    cov_mat = np.where(np.isnan(utm), utm.T, utm)

    # Package
    cov_mat = pl.DataFrame(
        {
            "factor_1": factors,
            **{col: cov_mat[:, idx] for idx, col in enumerate(factors)},
        }
    )

    # Fill NaN (from Barra)
    cov_mat = cov_mat.fill_nan(0)

    return cov_mat


def specific_risk_matrix(date_: date, barrids: list[str]) -> pl.DataFrame:
    """
    Constructs the specific risk matrix for the given date and Barrids.

    Args:
        date_ (date): The date for which the specific risk matrix is computed.
        barrids (List[str]): List of Barrid identifiers for the assets.

    Returns:
        pl.DataFrame: The specific risk matrix.
    """
    # Barrids
    barrids_df = pl.DataFrame({"barrid": barrids})

    # Load
    sr_df = dal.load_specific_risk(date_)

    # Filter
    sr_df = barrids_df.join(sr_df, on=["barrid"], how="left").fill_null(
        0
    )  # ask Brandon about this.

    # Convert vector to diagonal matrix
    diagonal = np.power(np.diag(sr_df["specific_risk"]), 2)

    # Package
    risk_matrix = pl.DataFrame(
        {
            "barrid": barrids,
            **{id: diagonal[:, i] for i, id in enumerate(barrids)},
        }
    )

    return risk_matrix
