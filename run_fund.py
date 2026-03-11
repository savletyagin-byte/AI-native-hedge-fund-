from ai_native_hedge_fund import AINativeHedgeFund


if __name__ == "__main__":
    universe = [f"Asset_{i:03d}" for i in range(1, 151)]
    fund = AINativeHedgeFund()
    result = fund.run_research_cycle(universe, periods=900)

    print("=== AI-Native Hedge Fund Report ===")
    print("Mode: Quant + Filing Research Swarm")
    print(f"Sharpe: {result['sharpe']:.2f}")
    print(f"Max Drawdown: {result['max_drawdown']:.2%}")
    print(f"Expected Shortfall (97.5%): {result['expected_shortfall']:.4f}")
    print(f"Stress Scenario Loss: {result['stress_scenario_loss']:.2%}")
    print(f"Final NAV: {result['nav'].iloc[-1]:.3f}")

    prod = fund.run_production_cycle(universe[:40], target_notional=2_000_000)
    print("\n=== Production Cycle Preview ===")
    print(f"Control Passed: {prod['control_passed']}")
    print(f"Control Reasons: {prod['control_reasons']}")
    print(f"Feature Drift Score: {prod['feature_drift_score']:.3f}")
    print(f"Orders Routed: {len(prod['execution_reports'])}")
