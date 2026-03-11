from __future__ import annotations

import pandas as pd

from .backtest import Backtester
from .config import FundConfig
from .data import MarketDataEngine
from .features import FeatureFactory
from .models import AlphaEnsemble
from .portfolio import PortfolioConstructor
from .risk import RiskEngine


class AINativeHedgeFund:
    """End-to-end orchestration layer for an AI-native hedge fund research stack."""

    def __init__(self, config: FundConfig | None = None) -> None:
        self.config = config or FundConfig()
        self.data_engine = MarketDataEngine(seed=self.config.random_seed)
        self.feature_factory = FeatureFactory()
        self.alpha_model = AlphaEnsemble()
        self.risk_engine = RiskEngine()
        self.allocator = PortfolioConstructor(self.config, self.risk_engine)
        self.backtester = Backtester(self.config, self.risk_engine, self.allocator)

    def run_research_cycle(self, tickers: list[str], periods: int = 756) -> dict[str, float | pd.Series]:
        market = self.data_engine.generate(tickers=tickers, periods=periods)
        features = self.feature_factory.transform(market)
        signal = self.alpha_model.predict(features)
        returns = market.xs("ret", axis=1, level=1)
        result = self.backtester.run(signal=signal, returns=returns)

        scenario = self.risk_engine.scenario_loss(
            weights=signal.iloc[-1] / (signal.iloc[-1].abs().sum() + 1e-12) * self.config.max_gross_leverage,
            shock=self.config.scenario_shock,
        )
        result["stress_scenario_loss"] = scenario
        return result
