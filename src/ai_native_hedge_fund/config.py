from dataclasses import dataclass, field


@dataclass(frozen=True)
class FundConfig:
    universe_size: int = 150
    lookback_days: int = 252
    rebalance_days: int = 5
    max_gross_leverage: float = 2.5
    max_single_weight: float = 0.03
    target_volatility: float = 0.12
    risk_aversion: float = 4.0
    transaction_cost_bps: float = 3.0
    scenario_shock: float = 0.08
    random_seed: int = 7
    feature_set: tuple[str, ...] = field(
        default_factory=lambda: (
            "momentum_21",
            "momentum_63",
            "mean_reversion_5",
            "volatility_21",
            "volume_zscore",
            "macro_regime",
        )
    )
