# AI-Native Hedge Fund

An advanced, modular hedge fund research stack that simulates an institutional quant workflow and extends it with agent swarms that read filing-style text.

- synthetic market data with latent bull/bear regimes
- multi-horizon feature engineering
- regime-aware ensemble alpha model with confidence scaling
- multi-agent research swarm for SEC filings and earnings-call style language
- execution adapters, slippage estimation, and broker routing
- online drift monitoring, production controls, and compliance gating

## Added production-oriented capabilities

- **real SEC EDGAR ingestion and transcript feed adapters** (`ingestion.py`)
- **LLM-based analyst agent interface with tool-use planning** (`llm_agents.py`)
- **execution adapters and broker routing engine** (`execution.py`)
- **online learning/drift monitoring primitives** (`monitoring.py`)
- **production risk controls and compliance gates** (`controls.py`, `governance.py`)
- **performance and exposure reporting** (`reporting.py`)

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
- `models.py`: regime-aware ensemble alpha forecaster
- `filings.py`: synthetic filing/earnings-call text stream
- `agents.py`: autonomous research agents + swarm aggregation
- `ingestion.py`: EDGAR/transcript ingestion adapters
- `llm_agents.py`: LLM analyst agent with tool contracts
- `execution.py`: broker adapters, order routing, slippage estimator
- `monitoring.py`: drift monitoring
- `controls.py`: pre-trade risk controls
- `governance.py`: compliance policy gates
- `reporting.py`: KPI and exposure reporting
- `risk.py`: covariance, expected shortfall, scenario loss
- `portfolio.py`: constrained optimizer and execution cost model
- `backtest.py`: event-driven backtesting engine
- `orchestrator.py`: end-to-end research + production cycle orchestration
