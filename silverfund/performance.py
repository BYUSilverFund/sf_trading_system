from datetime import date

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
import statsmodels.formula.api as smf
from tabulate import tabulate

import silverfund.data_access_layer as dal
from silverfund.enums import Compounding, Interval
from silverfund.records import AssetReturns


class Performance:
    """
    A class to evaluate and visualize portfolio performance.

    This class provides methods to calculate and summarize key performance metrics,
    including return, risk (standard deviation), Sharpe ratio, information ratio,
    beta, and alpha for a portfolio, benchmark, and active returns.

    Attributes:
        interval (Interval): The frequency of data (e.g., daily, monthly).
        start_date (date): The start date of the performance evaluation period.
        end_date (date): The end date of the performance evaluation period.
        asset_returns (AssetReturns): Asset returns data for the evaluation period.
        annualize (bool): Whether to annualize the results (default is True).

    Methods:
        plot_returns(compounding: Compounding, title: str, decompose: bool = False,
                      save_file_path: str | None = None):
            Plots the cumulative returns of the portfolio, benchmark, and active returns.
        summary(save_file_path: str | None = None):
            Generates a summary table with performance metrics and returns it or saves to a file.
    """

    def __init__(
        self,
        interval: Interval,
        start_date: date,
        end_date: date,
        asset_returns: AssetReturns,
        annualize: bool = True,
    ) -> None:
        """
        Initializes the Performance object.

        Args:
            interval (Interval): The data frequency (e.g., daily, monthly).
            start_date (date): The start date for the performance evaluation period.
            end_date (date): The end date for the performance evaluation period.
            asset_returns (AssetReturns): Asset returns data.
            annualize (bool, optional): Whether to annualize the performance metrics. Defaults to True.
        """
        self._start_date = start_date
        self._end_date = end_date
        self._interval = interval

        # Set annualizing variables
        annual_scales = {Interval.DAILY: 252, Interval.MONTHLY: 12}

        self._annual_scale = annual_scales[interval]
        self._annualize = annualize

        # Load benchmark
        bmk = dal.load_benchmark(interval=interval, start_date=start_date, end_date=end_date)

        # Create total_ret, bmk_ret, and active_ret columns.
        self._asset_returns = (
            asset_returns.join(bmk, on=["date", "barrid"], how="left", suffix="_bmk")
            .with_columns((pl.col("weight") - pl.col("weight_bmk")).alias("weight_active"))
            .with_columns(
                (pl.col("weight") * pl.col("fwd_ret")).alias("total_ret"),
                (pl.col("weight_bmk") * pl.col("fwd_ret")).alias("bmk_ret"),
                (pl.col("weight_active") * pl.col("fwd_ret")).alias("active_ret"),
            )
        )

        # Create portfolio returns dataframe
        self._portfolio_returns = (
            self._asset_returns.group_by("date")
            .agg(pl.col("total_ret").sum(), pl.col("bmk_ret").sum(), pl.col("active_ret").sum())
            .sort("date")
        )

        self._periods = self._portfolio_returns["date"].unique().count()

    def plot_returns(
        self,
        compounding: Compounding,
        title: str,
        decompose: bool = False,
        save_file_path: str | None = None,
    ) -> None:
        """
        Plots the cumulative returns for portfolio, benchmark, and active returns.

        Args:
            compounding (Compounding): The compounding method (sum or product).
            title (str): The title of the plot.
            decompose (bool, optional): Whether to decompose the total return into benchmark and active returns.
            save_file_path (str | None, optional): If provided, saves the plot to the given path.
        """
        df = self._portfolio_returns

        # Create cummulative returns dataframe
        # Sum of log returns
        if compounding == compounding.SUM:
            df = (
                df.with_columns(
                    pl.col("total_ret").log1p(),
                    pl.col("bmk_ret").log1p(),
                    pl.col("active_ret").log1p(),
                )
                .sort("date")
                .with_columns(
                    pl.col("total_ret").cum_sum(),
                    pl.col("bmk_ret").cum_sum(),
                    pl.col("active_ret").cum_sum(),
                )
            )

        # Product of gross returns
        if compounding == compounding.PRODUCT:
            df = (
                df.with_columns(
                    pl.col("total_ret") + 1,
                    pl.col("bmk_ret") + 1,
                    pl.col("active_ret") + 1,
                )
                .sort("date")
                .with_columns(
                    pl.col("total_ret").cum_prod(),
                    pl.col("bmk_ret").cum_prod(),
                    pl.col("active_ret").cum_prod(),
                )
                .with_columns(
                    pl.col("total_ret") - 1,
                    pl.col("bmk_ret") - 1,
                    pl.col("active_ret") - 1,
                )
            )

        # Put in percent space
        df = df.with_columns(
            pl.col("total_ret") * 100,
            pl.col("bmk_ret") * 100,
            pl.col("active_ret") * 100,
        )

        # Plot
        plt.figure(figsize=(10, 6))

        sns.lineplot(df, x="date", y="total_ret", label="Total", legend=None)

        if decompose:
            sns.lineplot(df, x="date", y="bmk_ret", label="Benchmark")
            sns.lineplot(df, x="date", y="active_ret", label="Active")
            plt.legend()

        plt.title(title)
        plt.xlabel(None)
        plt.ylabel(f"Cummulative {compounding.value.title()} Returns (%)")
        plt.grid()

        # Save/show
        if save_file_path is None:
            plt.show()
        else:
            plt.savefig(save_file_path)

    def _mean(self, col: str) -> float:
        result = self._portfolio_returns[col].mean()

        if self._annualize:
            result *= self._annual_scale

        return result

    def _std(self, col: str) -> float:
        result = self._portfolio_returns[col].std()

        if self._annualize:
            result *= np.sqrt(self._annual_scale)

        return result

    def _ratio(self, col: str) -> float:
        return self._mean(col) / self._std(col)

    def _coef(self, col: str) -> float:
        formula = f"{col} ~ bmk_ret"
        result = smf.ols(formula, self._portfolio_returns).fit()
        return result.params["bmk_ret"]

    def _intercept(self, col: str) -> float:
        formula = f"{col} ~ bmk_ret"
        result = smf.ols(formula, self._portfolio_returns).fit()
        result = result.params["Intercept"]

        if self._annualize:
            result *= self._annual_scale

        return result

    @property
    def portfolio_return(self) -> float:
        return self._mean("total_ret")

    @property
    def benchmark_return(self) -> float:
        return self._mean("bmk_ret")

    @property
    def active_return(self) -> float:
        return self._mean("active_ret")

    @property
    def portfolio_risk(self) -> float:
        return self._std("total_ret")

    @property
    def benchmark_risk(self) -> float:
        return self._std("bmk_ret")

    @property
    def active_risk(self) -> float:
        return self._std("active_ret")

    @property
    def portfolio_sharpe(self) -> float:
        return self._ratio("total_ret")

    @property
    def benchmark_sharpe(self) -> float:
        return self._ratio("bmk_ret")

    @property
    def information_ratio(self) -> float:
        return self._ratio("active_ret")

    @property
    def portfolio_beta(self) -> float:
        return self._coef("total_ret")

    @property
    def benchmark_beta(self) -> float:
        return self._coef("bmk_ret")

    @property
    def active_beta(self) -> float:
        return self._coef("active_ret")

    @property
    def portfolio_alpha(self) -> float:
        return self._intercept("total_ret")

    @property
    def benchmark_alpha(self) -> float:
        return self._intercept("bmk_ret")

    @property
    def active_alpha(self) -> float:
        return self._intercept("active_ret")

    def summary(self, save_file_path: str | None = None) -> str | None:
        """
        Generates a summary of performance metrics as a table.

        Args:
            save_file_path (str | None, optional): If provided, saves the summary to the file path.

        Returns:
            str | None: The summary as a string, or None if saved to a file.
        """
        # Create data rows for the table
        data = [
            [
                "Return (Mean)",
                f"{self.portfolio_return:.2%}",
                f"{self.benchmark_return:.2%}",
                f"{self.active_return:.2%}",
            ],
            [
                "Risk",
                f"{self.portfolio_risk:.2%}",
                f"{self.benchmark_risk:.2%}",
                f"{self.active_risk:.2%}",
            ],
            [
                "Sharpe Ratio",
                f"{self.portfolio_sharpe:.2f}",
                f"{self.benchmark_sharpe:.2f}",
                "",
            ],
            ["Information Ratio", f"{self.information_ratio:.2f}", "", ""],
            [
                "Beta",
                f"{self.portfolio_beta:.2f}",
                f"{self.benchmark_beta:.2f}",
                f"{self.active_beta:.2f}",
            ],
            [
                "Alpha",
                f"{self.portfolio_alpha:.2%}",
                f"{self.benchmark_alpha:.2%}",
                f"{self.active_alpha:.2%}",
            ],
        ]

        # Define headers
        headers = ["Metric", "Portfolio", "Benchmark", "Active"]

        # Generate the table
        table = tabulate(data, headers=headers, tablefmt="rounded_grid")
        title = "-" * 20 + " Performance Summary " + "-" * 20
        description = "\n".join(
            [
                f"Start Date: {self._start_date}",
                f"End Date: {self._end_date}",
                f"Interval: {self._interval.value.title()}",
                f"Periods: {self._periods}",
                f"Annualized: {self._annualize}",
            ]
        )

        result = "\n\n".join([title, description, table])

        if save_file_path is None:
            return result
        else:
            with open(save_file_path, "w") as file:
                file.write(result)

    def __str__(self) -> str:
        return self.summary()
