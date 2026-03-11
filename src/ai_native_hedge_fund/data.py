from __future__ import annotations

import numpy as np
import pandas as pd


class MarketDataEngine:
    """Synthetic market engine with latent regime dynamics for research workflows."""

    def __init__(self, seed: int = 7) -> None:
        self.rng = np.random.default_rng(seed)

    def generate(self, tickers: list[str], periods: int = 756) -> pd.DataFrame:
        dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=periods)
        n_assets = len(tickers)

        regime = self._latent_regime(periods)
        base_mu = np.where(regime > 0, 0.00045, -0.0002)
        base_vol = np.where(regime > 0, 0.008, 0.015)

        factor = self.rng.normal(0, 1, size=(periods, 4))
        betas = self.rng.normal(0.2, 0.4, size=(n_assets, 4))
        idio = self.rng.normal(0, 1, size=(periods, n_assets))

        rets = np.zeros((periods, n_assets))
        for t in range(periods):
            drift = base_mu[t]
            vol = base_vol[t]
            cross = factor[t] @ betas.T
            rets[t] = drift + vol * (0.5 * cross + 0.7 * idio[t])

        prices = 100 * np.exp(np.cumsum(rets, axis=0))
        volume = self.rng.lognormal(mean=12, sigma=0.6, size=(periods, n_assets))

        cols = pd.MultiIndex.from_product([tickers, ["close", "ret", "volume"]])
        data = np.zeros((periods, n_assets * 3))
        for idx in range(n_assets):
            data[:, idx * 3] = prices[:, idx]
            data[:, idx * 3 + 1] = rets[:, idx]
            data[:, idx * 3 + 2] = volume[:, idx]

        return pd.DataFrame(data=data, index=dates, columns=cols)

    def _latent_regime(self, periods: int) -> np.ndarray:
        states = np.ones(periods)
        for t in range(1, periods):
            if states[t - 1] > 0:
                states[t] = -1 if self.rng.random() < 0.04 else 1
            else:
                states[t] = 1 if self.rng.random() < 0.07 else -1
        return states
