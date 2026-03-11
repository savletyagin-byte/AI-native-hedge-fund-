from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FundConfig
from .risk import RiskEngine


class PortfolioConstructor:
    """Constructs a market-neutral, constraint-aware portfolio from alpha forecasts."""

    def __init__(self, config: FundConfig, risk_engine: RiskEngine) -> None:
        self.config = config
        self.risk_engine = risk_engine

    def optimize(self, alpha: pd.Series, cov: pd.DataFrame) -> pd.Series:
        raw = alpha - alpha.mean()
        if raw.abs().sum() == 0:
            return raw

        raw = raw / raw.abs().sum() * self.config.max_gross_leverage
        clipped = raw.clip(-self.config.max_single_weight, self.config.max_single_weight)
        centered = clipped - clipped.mean()

        gross = centered.abs().sum()
        if gross > self.config.max_gross_leverage:
            centered *= self.config.max_gross_leverage / gross

        scaled = self.risk_engine.volatility_target_scale(centered, cov, self.config.target_volatility)
        return self._enforce_leverage_cap(scaled)

    def _enforce_leverage_cap(self, w: pd.Series) -> pd.Series:
        gross = w.abs().sum()
        if gross <= self.config.max_gross_leverage:
            return w
        return w * (self.config.max_gross_leverage / gross)

    @staticmethod
    def turnover(prev_w: pd.Series, new_w: pd.Series) -> float:
        prev = prev_w.reindex(new_w.index).fillna(0.0)
        return float((new_w - prev).abs().sum())

    @staticmethod
    def transaction_cost(turnover: float, bps: float) -> float:
        return turnover * bps / 10000
