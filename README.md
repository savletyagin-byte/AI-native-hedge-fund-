# AI-Native Hedge Fund

An advanced, modular hedge fund research stack that simulates an institutional quant workflow:

- synthetic market data with latent bull/bear regimes
- multi-horizon feature engineering
- ensemble alpha model with confidence scaling
- constraint-aware market-neutral portfolio construction
- volatility targeting and scenario risk checks
- event-driven backtesting with turnover-based transaction costs

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
python run_fund.py
```

## Architecture

- `data.py`: synthetic market generator with latent regime switching
- `features.py`: alpha and regime features
- `models.py`: ensemble alpha forecaster
- `risk.py`: covariance, expected shortfall, scenario loss
- `portfolio.py`: constrained optimizer and execution cost model
- `backtest.py`: event-driven backtesting engine
- `orchestrator.py`: end-to-end strategy pipeline

## Notes

This framework is designed for research and can be extended with:
- real-time data feeds
- broker/exchange adapters
- deep learning signal models
- online learning and drift monitoring
- production risk controls and compliance gates
