from __future__ import annotations

import pandas as pd


class PerformanceReporter:
    """Builds compact KPI and attribution summaries for research and production."""

    def summarize(self, pnl: pd.Series) -> dict[str, float]:
        if pnl.empty:
            return {"cagr": 0.0, "win_rate": 0.0, "volatility": 0.0, "calmar": 0.0}

        nav = (1 + pnl).cumprod()
        years = max(len(pnl) / 252, 1 / 252)
        cagr = float(nav.iloc[-1] ** (1 / years) - 1)
        volatility = float(pnl.std() * (252**0.5))
        win_rate = float((pnl > 0).mean())
        max_dd = float((nav / nav.cummax() - 1).min())
        calmar = cagr / (abs(max_dd) + 1e-12)
        return {"cagr": cagr, "win_rate": win_rate, "volatility": volatility, "calmar": float(calmar)}

    def exposure_breakdown(self, weights: pd.Series) -> dict[str, float]:
        long_exposure = float(weights.clip(lower=0).sum())
        short_exposure = float(-weights.clip(upper=0).sum())
        net_exposure = float(weights.sum())
        return {
            "long_exposure": long_exposure,
            "short_exposure": short_exposure,
            "net_exposure": net_exposure,
        }
