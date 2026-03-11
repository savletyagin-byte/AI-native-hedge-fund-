from __future__ import annotations

import pandas as pd


class AlphaEnsemble:
    """Model stack that blends medium-term momentum, short-term reversal and liquidity effects."""

    def __init__(self) -> None:
        self.weights = {
            "momentum_21": 0.20,
            "momentum_63": 0.35,
            "mean_reversion_5": 0.15,
            "volatility_21": -0.20,
            "volume_zscore": 0.10,
            "macro_regime": 0.05,
        }

    def predict(self, features: dict[str, pd.DataFrame]) -> pd.DataFrame:
        base = None
        for name, w in self.weights.items():
            f = features[name]
            z = (f - f.mean(axis=1).values.reshape(-1, 1)) / (f.std(axis=1).values.reshape(-1, 1) + 1e-9)
            contribution = w * z
            base = contribution if base is None else base + contribution

        confidence = 1 / (1 + features["volatility_21"].clip(lower=1e-4) * 100)
        signal = base * confidence
        return signal.fillna(0.0)
