from __future__ import annotations

from dataclasses import dataclass

from .agents import AgentInsight


@dataclass(frozen=True)
class ToolCall:
    tool_name: str
    arguments: dict[str, str]


class ToolEnabledAnalystAgent:
    """LLM-style analyst agent interface with explicit tool-use contracts."""

    agent_name: str = "llm_analyst"

    def plan_tools(self, ticker: str, text: str) -> list[ToolCall]:
        _ = text
        return [
            ToolCall("valuation_model", {"ticker": ticker}),
            ToolCall("estimate_revision_tracker", {"ticker": ticker}),
            ToolCall("risk_news_scan", {"ticker": ticker}),
        ]

    def evaluate(self, ticker: str, text: str, tool_outputs: dict[str, str]) -> AgentInsight:
        sentiment = 0.0
        if "beat" in text.lower() or "raised" in text.lower():
            sentiment += 0.3
        if "weakness" in text.lower() or "lawsuit" in text.lower():
            sentiment -= 0.4
        if "valuation_model" in tool_outputs and "undervalued" in tool_outputs["valuation_model"].lower():
            sentiment += 0.2
        confidence = min(1.0, 0.4 + 0.2 * len(tool_outputs))
        return AgentInsight(ticker=ticker, sentiment=max(-1.0, min(1.0, sentiment)), confidence=confidence, rationale="llm+tools")
