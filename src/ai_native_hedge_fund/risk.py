from __future__ import annotations

import numpy as np
import pandas as pd


class RiskEngine:
    """Risk model for covariance estimation and constraint-aware scaling."""

    def covariance(self, returns: pd.DataFrame, lookback: int = 63) -> pd.DataFrame:
        sample = returns.tail(lookback)
        return sample.cov()

    def volatility_target_scale(self, weights: pd.Series, cov: pd.DataFrame, target_vol: float) -> pd.Series:
        vol = np.sqrt(float(weights.T @ cov.values @ weights)) * np.sqrt(252)
        if vol <= 1e-10:
            return weights
        return weights * (target_vol / vol)

    def expected_shortfall(self, pnl: pd.Series, alpha: float = 0.975) -> float:
        cutoff = pnl.quantile(1 - alpha)
        tail = pnl[pnl <= cutoff]
        if tail.empty:
            return 0.0
        return float(-tail.mean())

    def scenario_loss(self, weights: pd.Series, shock: float) -> float:
        long_book = weights.clip(lower=0).sum()
        short_book = -weights.clip(upper=0).sum()
        return float(shock * (long_book + short_book))
