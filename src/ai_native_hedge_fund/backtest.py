from __future__ import annotations

import pandas as pd

from .config import FundConfig
from .portfolio import PortfolioConstructor
from .risk import RiskEngine


class Backtester:
    """Event-driven backtester with slippage proxy and risk analytics."""

    def __init__(self, config: FundConfig, risk_engine: RiskEngine, allocator: PortfolioConstructor) -> None:
        self.config = config
        self.risk_engine = risk_engine
        self.allocator = allocator

    def run(self, signal: pd.DataFrame, returns: pd.DataFrame) -> dict[str, float | pd.Series]:
        dates = signal.index
        holdings = pd.Series(0.0, index=signal.columns)
        pnl = []

        for idx, dt in enumerate(dates[:-1]):
            if idx < self.config.lookback_days or idx % self.config.rebalance_days != 0:
                next_ret = returns.iloc[idx + 1]
                pnl.append(float((holdings * next_ret).sum()))
                continue

            hist = returns.iloc[max(0, idx - self.config.lookback_days):idx]
            cov = self.risk_engine.covariance(hist, lookback=min(63, len(hist)))
            target = self.allocator.optimize(signal.loc[dt], cov)

            turn = self.allocator.turnover(holdings, target)
            tc = self.allocator.transaction_cost(turn, self.config.transaction_cost_bps)
            next_ret = returns.iloc[idx + 1]
            day_pnl = float((target * next_ret).sum() - tc)

            holdings = target
            pnl.append(day_pnl)

        pnl_series = pd.Series(pnl, index=dates[:-1], name="pnl")
        nav = (1 + pnl_series).cumprod()
        sharpe = (pnl_series.mean() / (pnl_series.std() + 1e-12)) * (252**0.5)
        max_dd = (nav / nav.cummax() - 1).min()
        es = self.risk_engine.expected_shortfall(pnl_series)

        return {
            "pnl": pnl_series,
            "nav": nav,
            "sharpe": float(sharpe),
            "max_drawdown": float(max_dd),
            "expected_shortfall": float(es),
        }
