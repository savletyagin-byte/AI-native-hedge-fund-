from __future__ import annotations

import pandas as pd


class DriftMonitor:
    """Online drift monitoring for feature and performance degradation."""

    def feature_drift_score(self, baseline: pd.DataFrame, current: pd.DataFrame) -> float:
        baseline_mean = baseline.mean().fillna(0.0)
        current_mean = current.mean().fillna(0.0)
        denom = baseline.std().replace(0.0, 1.0).fillna(1.0)
        score = ((current_mean - baseline_mean).abs() / denom).mean()
        return float(score)

    def performance_drift_score(self, baseline_pnl: pd.Series, current_pnl: pd.Series) -> float:
        base_sharpe = baseline_pnl.mean() / (baseline_pnl.std() + 1e-12)
        curr_sharpe = current_pnl.mean() / (current_pnl.std() + 1e-12)
        return float(base_sharpe - curr_sharpe)
