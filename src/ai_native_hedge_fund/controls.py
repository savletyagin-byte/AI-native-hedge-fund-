from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ControlResult:
    passed: bool
    reasons: list[str]


class ProductionControlGate:
    """Pre-trade production risk/compliance gate checks."""

    def __init__(self, max_gross: float = 2.5, max_single_weight: float = 0.05, min_names: int = 20) -> None:
        self.max_gross = max_gross
        self.max_single_weight = max_single_weight
        self.min_names = min_names

    def check(self, target_weights: pd.Series) -> ControlResult:
        reasons: list[str] = []
        gross = float(target_weights.abs().sum())
        if gross > self.max_gross:
            reasons.append(f"gross leverage {gross:.2f} exceeds {self.max_gross:.2f}")

        if float(target_weights.abs().max()) > self.max_single_weight:
            reasons.append("single-name limit exceeded")

        active_names = int((target_weights.abs() > 1e-8).sum())
        if active_names < self.min_names:
            reasons.append(f"insufficient diversification: {active_names} names")

        return ControlResult(passed=len(reasons) == 0, reasons=reasons)
