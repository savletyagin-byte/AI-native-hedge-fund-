import pandas as pd

from ai_native_hedge_fund import AINativeHedgeFund, FundConfig, ResearchSwarm
from ai_native_hedge_fund.controls import ProductionControlGate
from ai_native_hedge_fund.execution import ExecutionEngine
from ai_native_hedge_fund.governance import ComplianceEngine
from ai_native_hedge_fund.monitoring import DriftMonitor
from ai_native_hedge_fund.reporting import PerformanceReporter


def test_research_cycle_outputs_metrics():
    config = FundConfig(universe_size=40, lookback_days=80, rebalance_days=5)
    fund = AINativeHedgeFund(config=config)
    tickers = [f"T{i:03d}" for i in range(40)]

    result = fund.run_research_cycle(tickers=tickers, periods=260)

    assert "sharpe" in result
    assert "cagr" in result
    assert "win_rate" in result
    assert "max_drawdown" in result
    assert "nav" in result
    assert isinstance(result["nav"], pd.Series)
    assert result["nav"].iloc[-1] > 0


def test_pipeline_generates_aligned_shapes():
    fund = AINativeHedgeFund(FundConfig(universe_size=25, lookback_days=60))
    tickers = [f"A{i}" for i in range(25)]

    market = fund.data_engine.generate(tickers=tickers, periods=220)
    features = fund.feature_factory.transform(market)
    signal = fund.alpha_model.predict(features)

    returns = market.xs("ret", axis=1, level=1)
    assert signal.shape == returns.shape
    assert set(signal.columns) == set(tickers)


def test_research_swarm_scores_filing_documents():
    filing_texts = {
        "AAA": "guidance raised with margin expansion and share repurchase",
        "BBB": "material weakness and lawsuit with guidance cut",
    }
    swarm = ResearchSwarm()
    scores = swarm.aggregate(filing_texts)

    assert scores["AAA"] > scores["BBB"]
    assert scores.index.tolist() == ["AAA", "BBB"]


def test_production_cycle_controls_and_execution_path():
    fund = AINativeHedgeFund(FundConfig(universe_size=30, lookback_days=80))
    tickers = [f"P{i:03d}" for i in range(30)]

    prod = fund.run_production_cycle(tickers, target_notional=1_000_000)

    assert "control_passed" in prod
    assert "compliance_passed" in prod
    assert "execution_reports" in prod
    assert "feature_drift_score" in prod
    assert "exposure" in prod
    assert isinstance(prod["feature_drift_score"], float)


def test_control_gate_rejects_overweight_portfolio():
    gate = ProductionControlGate(max_gross=1.5, max_single_weight=0.2, min_names=3)
    w = pd.Series({"A": 0.8, "B": -0.8, "C": 0.1})
    result = gate.check(w)
    assert not result.passed
    assert len(result.reasons) >= 1


def test_execution_drift_compliance_and_reporting_utilities():
    engine = ExecutionEngine()
    reports = engine.rebalance(pd.Series({"A": 0.2, "B": -0.1}), notional=1000)
    assert len(reports) == 2
    assert reports[0].slippage_bps > 0

    monitor = DriftMonitor()
    baseline = pd.DataFrame({"f": [0.1, 0.2, 0.3]})
    current = pd.DataFrame({"f": [1.0, 1.2, 1.1]})
    assert monitor.feature_drift_score(baseline, current) > 0

    compliance = ComplianceEngine(restricted_tickers={"A"}, allow_shorts=False)
    decision = compliance.review_orders(pd.Series({"A": 0.1, "B": -0.1}))
    assert not decision.approved

    reporter = PerformanceReporter()
    summary = reporter.summarize(pd.Series([0.01, -0.005, 0.002]))
    assert "cagr" in summary
