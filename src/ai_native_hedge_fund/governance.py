from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ComplianceDecision:
    approved: bool
    reasons: list[str]


class ComplianceEngine:
    """Post-risk compliance controls (restricted list, hard bans, and short-sale policies)."""

    def __init__(self, restricted_tickers: set[str] | None = None, allow_shorts: bool = True) -> None:
        self.restricted_tickers = restricted_tickers or set()
        self.allow_shorts = allow_shorts

    def review_orders(self, target_weights: pd.Series) -> ComplianceDecision:
        reasons: list[str] = []
        restricted_hits = [t for t in target_weights.index if t in self.restricted_tickers and abs(target_weights[t]) > 1e-12]
        if restricted_hits:
            reasons.append(f"restricted tickers present: {', '.join(sorted(restricted_hits)[:5])}")

        if not self.allow_shorts and float(target_weights.min()) < -1e-12:
            reasons.append("short positions are disabled by policy")

        return ComplianceDecision(approved=len(reasons) == 0, reasons=reasons)
