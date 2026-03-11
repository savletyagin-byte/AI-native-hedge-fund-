import pandas as pd

from ai_native_hedge_fund import AINativeHedgeFund, FundConfig


def test_research_cycle_outputs_metrics():
    config = FundConfig(universe_size=40, lookback_days=80, rebalance_days=5)
    fund = AINativeHedgeFund(config=config)
    tickers = [f"T{i:03d}" for i in range(40)]

    result = fund.run_research_cycle(tickers=tickers, periods=260)

    assert "sharpe" in result
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
