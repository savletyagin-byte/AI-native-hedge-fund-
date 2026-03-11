# AI-Native Hedge Fund

An advanced, modular hedge fund research stack that simulates an institutional quant workflow and extends it with agent swarms that read filing-style text.

- synthetic market data with latent bull/bear regimes
- multi-horizon feature engineering
- ensemble alpha model with confidence scaling
- multi-agent research swarm for SEC filings and earnings-call style language
- constraint-aware market-neutral portfolio construction
- volatility targeting and scenario risk checks
- event-driven backtesting with turnover-based transaction costs

## Added production-oriented capabilities

- **real SEC EDGAR ingestion and transcript feed adapters** (`ingestion.py`)
- **LLM-based analyst agent interface with tool-use planning** (`llm_agents.py`)
- **execution adapters and broker routing engine** (`execution.py`)
- **online learning/drift monitoring primitives** (`monitoring.py`)
- **production risk controls and compliance gates** (`controls.py`)

This is aligned with the AI-native fund thesis: autonomous agent teams parse filings/transcripts, synthesize analyst-like views, and route governed trades.

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
- `ingestion.py`: EDGAR/transcript ingestion adapters
- `llm_agents.py`: LLM analyst agent with tool contracts
- `execution.py`: broker adapters and order routing
- `monitoring.py`: drift monitoring
- `controls.py`: pre-trade risk/compliance gates
- `risk.py`: covariance, expected shortfall, scenario loss
- `portfolio.py`: constrained optimizer and execution cost model
- `backtest.py`: event-driven backtesting engine
- `orchestrator.py`: end-to-end research + production cycle orchestration
