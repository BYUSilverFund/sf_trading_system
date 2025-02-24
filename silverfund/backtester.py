import os
from datetime import date

import polars as pl
import ray
from ray.experimental import tqdm_ray
from tqdm import tqdm

import silverfund.data_access_layer as dal
from silverfund.enums import Interval
from silverfund.logging.slack import SlackLogConfig, send_message_to_slack
from silverfund.records import Alpha, AssetReturns, Portfolio
from silverfund.strategies import Strategy


class Backtester:
    """
    A backtesting class that calculates alpha signals, constructs portfolios, and computes forward returns
    for a given strategy within a specified time interval.

    Attributes:
        _interval (Interval): The time interval for the backtest.
        _start_date (date): The start date of the backtest.
        _end_date (date): The end date of the backtest.
        _data (pl.DataFrame): The data to be used for the backtest, in the form of a Polars DataFrame.
    """

    def __init__(
        self,
        interval: Interval,
        start_date: date,
        end_date: date,
        data: pl.DataFrame,
        slack_log_config: SlackLogConfig,
    ):
        """
        Initializes a Backtester instance.

        Args:
            interval (Interval): The time interval for the backtest.
            start_date (date): The start date of the backtest.
            end_date (date): The end date of the backtest.
            data (pl.DataFrame): The data for the backtest.
        """
        self._interval = interval
        self._start_date = start_date
        self._end_date = end_date
        self._data = data
        self._slack_log_config = slack_log_config

    def _compute_alphas(self, strategy: Strategy) -> Alpha:
        """
        Computes the alphas for a given strategy based on signals and scores.

        Args:
            strategy (Strategy): The strategy object used to generate signals, scores, and alphas.

        Returns:
            Alpha: A record containing the computed alpha values.
        """
        # Calculate signals, scores, and alphas
        signals = strategy.signal_constructor(self._data)
        scores = strategy.score_constructor(signals)
        alphas = strategy.alpha_constructor(scores)

        return Alpha(alphas)

    def _compute_forward_returns(self, portfolios: list[Portfolio]) -> AssetReturns:
        """
        Computes forward returns for a list of portfolios.

        Args:
            portfolios (list[Portfolio]): A list of portfolios to compute forward returns for.

        Returns:
            AssetReturns: A record containing the computed asset returns.
        """
        # Getting forward returns
        testing_data = (
            self._data.with_columns(pl.col("ret").shift(-1).over("barrid").alias("fwd_ret"))
            .select(["date", "barrid", "fwd_ret"])
            .sort(["barrid", "date"])
        )

        # Concatenate portfolios
        portfolios = pl.concat(portfolios)

        # Join forward returns on portfolios
        asset_returns = portfolios.join(testing_data, on=["barrid", "date"], how="left")
        asset_returns = asset_returns.sort(["barrid", "date"])

        return AssetReturns(asset_returns)

    def run_sequential(self, strategy: Strategy) -> AssetReturns:
        """
        Runs the backtest sequentially by computing alphas, constructing portfolios,
        and calculating forward returns.

        Args:
            strategy (Strategy): The strategy object used for portfolio construction and signal generation.

        Returns:
            AssetReturns: A record containing the computed asset returns.
        """
        # Get universe
        universe = dal.load_universe(
            interval=self._interval,
            start_date=self._start_date,
            end_date=self._end_date,
        )

        # Compute alphas
        alphas = self._compute_alphas(strategy)

        # Get periods
        periods = universe["date"].unique().sort().to_list()

        # Construct portfolios
        portfolios = [
            self.construct_portfolio(period, universe, alphas, strategy)
            for period in tqdm(periods, desc="Computing portfolios")
        ]

        return self._compute_forward_returns(portfolios)

    @staticmethod
    @ray.remote
    def construct_portfolio(
        period, universe, alphas, strategy, progress_bar: tqdm_ray.tqdm | None = None
    ):
        """
        Constructs a portfolio for a specific period using a given strategy.

        Args:
            period (str): The date period for which the portfolio is being constructed.
            universe (pl.DataFrame): The universe of available assets for portfolio construction.
            alphas (Alpha): The computed alphas used for portfolio construction.
            strategy (Strategy): The strategy used to construct the portfolio.
            progress_bar (tqdm_ray.tqdm, optional): A Ray-based progress bar for tracking progress.

        Returns:
            Portfolio: A constructed portfolio for the given period.
        """
        # Get portfolio constructor parameters
        period_barrids = universe.filter(pl.col("date") == period)["barrid"].sort().to_list()
        period_alphas = Alpha(alphas.filter(pl.col("date") == period).sort(["barrid"]))

        # Construct period portfolio
        portfolio = strategy.portfolio_constructor(
            period=period,
            barrids=period_barrids,
            alphas=period_alphas,
            constraints=strategy.constraints,
        )

        # Update progress bar
        if progress_bar is not None:
            progress_bar.update.remote(1)

        return portfolio

    def run_parallel(self, strategy: Strategy, n_cpus: int | None = None) -> AssetReturns:
        """
        Runs the backtest in parallel by computing alphas, constructing portfolios,
        and calculating forward returns using multiple CPU cores.

        Args:
            strategy (Strategy): The strategy object used for portfolio construction and signal generation.
            n_cpus (int, optional): The number of CPU cores to use for parallel execution.

        Returns:
            AssetReturns: A record containing the computed asset returns.
        """
        # Get universe
        universe = dal.load_universe(
            interval=self._interval,
            start_date=self._start_date,
            end_date=self._end_date,
        )

        # Compute alphas
        alphas = self._compute_alphas(strategy)

        # Get periods
        periods = universe["date"].unique().sort().to_list()

        # Set up ray
        n_cpus = n_cpus or os.cpu_count()
        n_cpus = min(len(periods), n_cpus)
        context = ray.init(ignore_reinit_error=True, num_cpus=n_cpus)

        # Send initial slack message
        if self._slack_log_config is not None:
            self._slack_log_config.ray_url = context.dashboard_url
            send_message_to_slack(self._slack_log_config.to_initial_message())

        # Set up ray progress bar
        remote_tqdm = ray.remote(tqdm_ray.tqdm)
        progress_bar = remote_tqdm.remote(
            total=len(periods), desc=f"Computing portfolios with {n_cpus} cpus"
        )

        # Dispatch parallel tasks
        portfolio_futures = [
            self.construct_portfolio.remote(period, universe, alphas, strategy, progress_bar)
            for period in periods
        ]

        # Retrieve results
        portfolios = ray.get(portfolio_futures)

        # Shutdown ray and progress bar
        progress_bar.close.remote()
        ray.shutdown()

        # Send terminal slack message
        if self._slack_log_config is not None:
            send_message_to_slack(self._slack_log_config.to_terminal_message())

        return self._compute_forward_returns(portfolios)
