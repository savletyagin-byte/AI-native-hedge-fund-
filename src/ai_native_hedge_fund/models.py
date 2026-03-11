from __future__ import annotations

import pandas as pd


class AlphaEnsemble:
    """Model stack that blends medium-term momentum, short-term reversal and liquidity effects."""

    def __init__(self) -> None:
        self.base_weights = {
            "momentum_21": 0.20,
            "momentum_63": 0.35,
            "mean_reversion_5": 0.15,
            "volatility_21": -0.20,
            "volume_zscore": 0.10,
            "macro_regime": 0.05,
        }

    def predict(self, features: dict[str, pd.DataFrame]) -> pd.DataFrame:
        base = None
        macro_regime = features["macro_regime"].fillna(1.0)

        for name, base_w in self.base_weights.items():
            f = features[name]
            z = (f - f.mean(axis=1).values.reshape(-1, 1)) / (f.std(axis=1).values.reshape(-1, 1) + 1e-9)
            dynamic_w = self._dynamic_weight(name, base_w, macro_regime)
            contribution = dynamic_w * z
            base = contribution if base is None else base + contribution

        confidence = 1 / (1 + features["volatility_21"].clip(lower=1e-4) * 100)
        signal = base * confidence
        return signal.fillna(0.0)

    def _dynamic_weight(self, feature_name: str, base_weight: float, macro_regime: pd.DataFrame) -> pd.DataFrame:
        if feature_name in {"momentum_21", "momentum_63"}:
            multiplier = pd.DataFrame(0.0, index=macro_regime.index, columns=macro_regime.columns)
            multiplier[macro_regime > 0] = 1.3
            multiplier[macro_regime <= 0] = 0.6
            return base_weight * multiplier
        if feature_name == "mean_reversion_5":
            multiplier = pd.DataFrame(0.0, index=macro_regime.index, columns=macro_regime.columns)
            multiplier[macro_regime < 0] = 1.25
            multiplier[macro_regime >= 0] = 0.8
            return base_weight * multiplier
        return pd.DataFrame(base_weight, index=macro_regime.index, columns=macro_regime.columns)
