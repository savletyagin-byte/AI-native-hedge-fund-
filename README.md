# AI-Native Hedge Fund

An advanced, modular hedge fund research stack that simulates an institutional quant workflow and extends it with agent swarms that read filing-style text.

- synthetic market data with latent bull/bear regimes
- multi-horizon feature engineering
- ensemble alpha model with confidence scaling
- **multi-agent research swarm for SEC filings and earnings-call style language**
- constraint-aware market-neutral portfolio construction
- volatility targeting and scenario risk checks
- event-driven backtesting with turnover-based transaction costs

This is aligned with the emerging AI-native fund thesis (including YC's view): autonomous agent teams can continuously parse filings, synthesize analyst-like views, and feed live alpha overlays.

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
- `filings.py`: synthetic filing/earnings-call text stream
- `agents.py`: autonomous research agents + swarm aggregation
- `risk.py`: covariance, expected shortfall, scenario loss
- `portfolio.py`: constrained optimizer and execution cost model
- `backtest.py`: event-driven backtesting engine
- `orchestrator.py`: end-to-end strategy pipeline

## Notes

This framework is designed for research and can be extended with:
- real SEC EDGAR ingestion and transcript feeds
- LLM-based analyst agents with tool use
- execution adapters and broker routing
- online learning and drift monitoring
- production risk controls and compliance gates
