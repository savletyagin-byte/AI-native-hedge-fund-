from __future__ import annotations

import numpy as np
import pandas as pd


class FeatureFactory:
    """Creates multi-horizon alpha and regime features."""

    def transform(self, market_data: pd.DataFrame) -> dict[str, pd.DataFrame]:
        rets = market_data.xs("ret", axis=1, level=1)
        close = market_data.xs("close", axis=1, level=1)
        volume = market_data.xs("volume", axis=1, level=1)

        momentum_21 = close.pct_change(21)
        momentum_63 = close.pct_change(63)
        mean_reversion_5 = -close.pct_change(5)
        volatility_21 = rets.rolling(21).std()
        volume_zscore = (volume - volume.rolling(30).mean()) / volume.rolling(30).std()

        cross_sectional_trend = rets.mean(axis=1).rolling(20).mean()
        macro_regime = np.sign(cross_sectional_trend).replace(0, 1).reindex(rets.index)
        macro_regime_df = pd.DataFrame(
            np.repeat(macro_regime.values.reshape(-1, 1), rets.shape[1], axis=1),
            index=rets.index,
            columns=rets.columns,
        )

        return {
            "momentum_21": momentum_21,
            "momentum_63": momentum_63,
            "mean_reversion_5": mean_reversion_5,
            "volatility_21": volatility_21,
            "volume_zscore": volume_zscore,
            "macro_regime": macro_regime_df,
        }
