from __future__ import annotations

import pandas as pd

from .agents import ResearchSwarm
from .backtest import Backtester
from .config import FundConfig
from .controls import ProductionControlGate
from .data import MarketDataEngine
from .execution import ExecutionEngine, ExecutionReport
from .features import FeatureFactory
from .filings import FilingStream
from .ingestion import EdgarIngestionClient, TranscriptFeedClient
from .models import AlphaEnsemble
from .monitoring import DriftMonitor
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
        self.filing_stream = FilingStream(seed=self.config.random_seed + 97)
        self.research_swarm = ResearchSwarm()

        self.edgar_client = EdgarIngestionClient()
        self.transcript_client = TranscriptFeedClient()
        self.execution_engine = ExecutionEngine()
        self.drift_monitor = DriftMonitor()
        self.control_gate = ProductionControlGate(
            max_gross=self.config.max_gross_leverage,
            max_single_weight=self.config.max_single_weight,
            min_names=max(10, self.config.universe_size // 6),
        )

    def run_research_cycle(self, tickers: list[str], periods: int = 756) -> dict[str, float | pd.Series]:
        market = self.data_engine.generate(tickers=tickers, periods=periods)
        features = self.feature_factory.transform(market)
        signal = self.alpha_model.predict(features)

        if self.config.agent_swarm_enabled:
            filing_docs = self.filing_stream.generate(tickers)
            filing_alpha = self.research_swarm.aggregate(filing_docs)
            signal = self._apply_filing_overlay(signal, filing_alpha)

        returns = market.xs("ret", axis=1, level=1)
        result = self.backtester.run(signal=signal, returns=returns)

        scenario = self.risk_engine.scenario_loss(
            weights=signal.iloc[-1] / (signal.iloc[-1].abs().sum() + 1e-12) * self.config.max_gross_leverage,
            shock=self.config.scenario_shock,
        )
        result["stress_scenario_loss"] = scenario
        return result

    def run_production_cycle(self, tickers: list[str], target_notional: float = 1_000_000.0) -> dict[str, object]:
        market = self.data_engine.generate(tickers=tickers, periods=max(260, self.config.lookback_days + 5))
        features = self.feature_factory.transform(market)
        signal = self.alpha_model.predict(features)

        filing_docs = self.filing_stream.generate(tickers)
        filing_alpha = self.research_swarm.aggregate(filing_docs)
        signal = self._apply_filing_overlay(signal, filing_alpha)

        returns = market.xs("ret", axis=1, level=1)
        cov = self.risk_engine.covariance(returns, lookback=63)
        target_weights = self.allocator.optimize(signal.iloc[-1], cov)

        control = self.control_gate.check(target_weights)
        reports: list[ExecutionReport] = []
        if control.passed:
            reports = self.execution_engine.rebalance(target_weights, notional=target_notional)

        feature_drift = self.drift_monitor.feature_drift_score(
            baseline=features["momentum_21"].tail(120),
            current=features["momentum_21"].tail(30),
        )

        return {
            "control_passed": control.passed,
            "control_reasons": control.reasons,
            "execution_reports": reports,
            "feature_drift_score": feature_drift,
            "target_weights": target_weights,
        }

    def _apply_filing_overlay(self, signal: pd.DataFrame, filing_alpha: pd.Series) -> pd.DataFrame:
        overlay = filing_alpha.reindex(signal.columns).fillna(0.0)
        return signal + self.config.filing_alpha_weight * overlay
