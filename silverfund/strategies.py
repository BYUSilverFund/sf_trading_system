from dataclasses import dataclass

from silverfund.alphas import AlphaConstructor
from silverfund.constraints import ConstraintConstructor
from silverfund.portfolios import PortfolioConstructor
from silverfund.scores import ScoreConstructor
from silverfund.signals import SignalConstructor


@dataclass
class Strategy:
    """Represents a financial strategy that constructs signals, scores, alphas, and portfolios.

    This class aggregates the components required to implement a quantitative finance strategy,
    including signal generation, scoring, alpha creation, portfolio construction, and constraints.

    Attributes:
        signal_constructor (SignalConstructor): A callable that constructs signals for assets.
        score_constructor (ScoreConstructor): A callable that calculates scores based on signals.
        alpha_constructor (AlphaConstructor): A callable that constructs alpha values from scores.
        portfolio_constructor (PortfolioConstructor): A callable that constructs portfolios using alphas.
        constraints (list[ConstraintConstructor]): A list of constraint constructors for portfolio optimization.
    """

    signal_constructor: SignalConstructor
    score_constructor: ScoreConstructor
    alpha_constructor: AlphaConstructor
    portfolio_constructor: PortfolioConstructor
    constraints: list[ConstraintConstructor]
