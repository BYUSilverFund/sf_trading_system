import polars as pl


def check_columns(expected: list[str], actual: list[str]) -> None:
    left_unique = list(set(expected) - set(actual))
    right_unique = list(set(actual) - set(expected))

    if len(left_unique) > 0:
        raise ValueError(f"Columns missing: {left_unique}")

    if len(right_unique) > 0:
        raise ValueError(f"Extra columns found: {right_unique}")


def check_schema(expected: dict[str, pl.DataType], actual: pl.Schema) -> None:
    for col, dtype in expected.items():
        if actual[col] != dtype:
            raise ValueError(f"Column {col} has incorrect type: {actual[col]}, expected: {dtype}")


class Signal(pl.DataFrame):
    """Represents a financial signal DataFrame with a specific structure.

    Ensures that the DataFrame contains the expected columns, schema, and order,
    and provides sorting and initialization for further use in financial analysis.

    Args:
        signals (pl.DataFrame): DataFrame containing the signal data.
        signal_name (str): The name of the signal column.

    Raises:
        ValueError: If the columns or schema do not match the expected structure.
    """

    def __init__(self, signals: pl.DataFrame, signal_name: str) -> None:
        expected_order = ["date", "barrid", signal_name]

        valid_schema = {
            "date": pl.Date,
            "barrid": pl.String,
            signal_name: pl.Float64,
        }

        # Check columns
        check_columns(expected_order, signals.columns)

        # Check schema
        check_schema(valid_schema, signals.schema)

        # Reorder columns
        signals = signals.select(expected_order)

        # Sort
        signals = signals.sort(["barrid", "date"])

        # Initialize
        super().__init__(signals)


class Score(pl.DataFrame):
    """Represents a financial score DataFrame with a specific structure.

    Ensures that the DataFrame contains the expected columns, schema, and order,
    and provides sorting and initialization for further use in financial analysis.

    Args:
        scores (pl.DataFrame): DataFrame containing the score data.

    Raises:
        ValueError: If the columns or schema do not match the expected structure.
    """

    def __init__(self, scores: pl.DataFrame) -> None:
        expected_order = ["date", "barrid", "score"]

        valid_schema = {
            "date": pl.Date,
            "barrid": pl.String,
            "score": pl.Float64,
        }

        # Check columns
        check_columns(expected_order, scores.columns)

        # Check schema
        check_schema(valid_schema, scores.schema)

        # Reorder columns
        scores = scores.select(expected_order)

        # Sort
        scores = scores.sort(["barrid", "date"])

        # Initialize
        super().__init__(scores)


class Alpha(pl.DataFrame):
    """Represents a financial alpha DataFrame with a specific structure.

    Ensures that the DataFrame contains the expected columns, schema, and order,
    and provides sorting and initialization for further use in financial analysis.

    Args:
        alphas (pl.DataFrame): DataFrame containing the alpha data.

    Raises:
        ValueError: If the columns or schema do not match the expected structure.
    """

    def __init__(self, alphas: pl.DataFrame) -> None:
        expected_order = ["date", "barrid", "alpha"]

        valid_schema = {
            "date": pl.Date,
            "barrid": pl.String,
            "alpha": pl.Float64,
        }

        # Check columns
        check_columns(expected_order, alphas.columns)

        # Check schema
        check_schema(valid_schema, alphas.schema)

        # Reorder columns
        alphas = alphas.select(expected_order)

        # Sort
        alphas = alphas.sort(["barrid", "date"])

        # Initialize
        super().__init__(alphas)

    def to_vector(self):
        """Converts the 'alpha' column to a numpy vector.

        Returns:
            np.ndarray: The 'alpha' column as a numpy array.
        """
        return self.select("alpha").to_numpy()


class CovarianceMatrix(pl.DataFrame):
    """Represents a covariance matrix DataFrame with a specific structure.

    Ensures that the DataFrame contains the expected columns, schema, and order,
    and provides sorting and initialization for further use in financial analysis.

    Args:
        cov_mat (pl.DataFrame): DataFrame containing the covariance matrix data.
        barrids (list[str]): List of 'barrid' values to be included in the covariance matrix.

    Raises:
        ValueError: If the columns or schema do not match the expected structure.
    """

    def __init__(self, cov_mat: pl.DataFrame, barrids: list[str]) -> None:
        expected_order = ["barrid"] + sorted(barrids)

        valid_schema = {barrid: pl.Float64 for barrid in barrids}

        # Check columns
        check_columns(expected_order, cov_mat.columns)

        # Check schema
        check_schema(valid_schema, cov_mat.schema)

        # Reorder columns
        cov_mat = cov_mat.select(expected_order)

        # Sort
        cov_mat = cov_mat.sort("barrid")

        # Initialize
        super().__init__(cov_mat)

    def to_matrix(self):
        """Converts the covariance matrix to a numpy matrix, excluding 'barrid'.

        Returns:
            np.ndarray: The covariance matrix as a numpy array.
        """
        return self.drop("barrid").to_numpy()


class Portfolio(pl.DataFrame):
    """Represents a portfolio DataFrame with a specific structure.

    Ensures that the DataFrame contains the expected columns, schema, and order,
    and provides sorting and initialization for further use in financial analysis.

    Args:
        portfolios (pl.DataFrame): DataFrame containing the portfolio data.

    Raises:
        ValueError: If the columns or schema do not match the expected structure.
    """

    def __init__(self, portfolios: pl.DataFrame) -> None:
        expected_order = ["date", "barrid", "weight"]

        valid_schema = {
            "date": pl.Date,
            "barrid": pl.String,
            "weight": pl.Float64,
        }

        # Check columns
        check_columns(expected_order, portfolios.columns)

        # Check schema
        check_schema(valid_schema, portfolios.schema)

        # Reorder columns
        portfolios = portfolios.select(expected_order)

        # Sort
        portfolios = portfolios.sort(["barrid", "date"])

        # Initialize
        super().__init__(portfolios)


class AssetReturns(pl.DataFrame):
    """Represents asset returns DataFrame with a specific structure.

    Ensures that the DataFrame contains the expected columns, schema, and order,
    and provides sorting and initialization for further use in financial analysis.

    Args:
        returns (pl.DataFrame): DataFrame containing the asset returns data.

    Raises:
        ValueError: If the columns or schema do not match the expected structure.
    """

    def __init__(self, returns: pl.DataFrame) -> None:
        expected_order = ["date", "barrid", "weight", "fwd_ret"]

        valid_schema = {
            "date": pl.Date,
            "barrid": pl.String,
            "weight": pl.Float64,
            "fwd_ret": pl.Float64,
        }

        # Check columns
        check_columns(expected_order, returns.columns)

        # Check schema
        check_schema(valid_schema, returns.schema)

        # Reorder columns
        returns = returns.select(expected_order)

        # Sort
        returns = returns.sort(["barrid", "date"])

        # Initialize
        super().__init__(returns)
