from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Order:
    ticker: str
    quantity: float
    side: str


@dataclass(frozen=True)
class ExecutionReport:
    order: Order
    status: str
    fill_price: float
    slippage_bps: float


class BrokerAdapter:
    """Broker routing interface for production execution."""

    def submit_order(self, order: Order) -> ExecutionReport:
        return ExecutionReport(order=order, status="simulated_fill", fill_price=0.0, slippage_bps=0.0)


class ExecutionEngine:
    """Converts target weights into routed orders via broker adapters."""

    def __init__(self, broker: BrokerAdapter | None = None, base_slippage_bps: float = 2.0) -> None:
        self.broker = broker or BrokerAdapter()
        self.base_slippage_bps = base_slippage_bps

    def rebalance(self, target_weights: pd.Series, notional: float = 1_000_000.0) -> list[ExecutionReport]:
        reports: list[ExecutionReport] = []
        for ticker, w in target_weights.items():
            qty = abs(float(w)) * notional
            side = "BUY" if w >= 0 else "SELL"
            slip = self._estimate_slippage_bps(abs(float(w)))
            report = self.broker.submit_order(Order(ticker=ticker, quantity=qty, side=side))
            reports.append(
                ExecutionReport(
                    order=report.order,
                    status=report.status,
                    fill_price=report.fill_price,
                    slippage_bps=slip,
                )
            )
        return reports

    def _estimate_slippage_bps(self, weight_abs: float) -> float:
        return float(self.base_slippage_bps + 12.0 * min(1.0, weight_abs))
