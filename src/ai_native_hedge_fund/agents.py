from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class AgentInsight:
    ticker: str
    sentiment: float
    confidence: float
    rationale: str


class BaseResearchAgent:
    """Common interface for autonomous research agents."""

    agent_name: str = "base"

    def evaluate(self, ticker: str, filing_text: str) -> AgentInsight:
        raise NotImplementedError


class FundamentalAgent(BaseResearchAgent):
    agent_name = "fundamental"

    POSITIVE = ("growth", "margin expansion", "guidance raised", "free cash flow", "backlog")
    NEGATIVE = ("impairment", "guidance cut", "liquidity concern", "covenant", "restructuring")

    def evaluate(self, ticker: str, filing_text: str) -> AgentInsight:
        text = filing_text.lower()
        pos_hits = sum(keyword in text for keyword in self.POSITIVE)
        neg_hits = sum(keyword in text for keyword in self.NEGATIVE)
        raw = pos_hits - neg_hits
        sentiment = max(-1.0, min(1.0, raw / 4))
        confidence = min(1.0, (pos_hits + neg_hits + 1) / 6)
        rationale = f"fundamental hits: +{pos_hits}/-{neg_hits}"
        return AgentInsight(ticker=ticker, sentiment=sentiment, confidence=confidence, rationale=rationale)


class RiskAgent(BaseResearchAgent):
    agent_name = "risk"

    RISK_TERMS = ("material weakness", "lawsuit", "downgrade", "churn", "supply constraint", "recall")

    def evaluate(self, ticker: str, filing_text: str) -> AgentInsight:
        text = filing_text.lower()
        hits = sum(term in text for term in self.RISK_TERMS)
        sentiment = -min(1.0, hits / 4)
        confidence = min(1.0, (hits + 1) / 4)
        rationale = f"risk terms: {hits}"
        return AgentInsight(ticker=ticker, sentiment=sentiment, confidence=confidence, rationale=rationale)


class EventAgent(BaseResearchAgent):
    agent_name = "event"

    POSITIVE_EVENTS = ("acquisition synergies", "product launch", "share repurchase", "strategic partnership")
    NEGATIVE_EVENTS = ("investigation", "management departure", "data breach", "production delay")

    def evaluate(self, ticker: str, filing_text: str) -> AgentInsight:
        text = filing_text.lower()
        pos_hits = sum(term in text for term in self.POSITIVE_EVENTS)
        neg_hits = sum(term in text for term in self.NEGATIVE_EVENTS)
        sentiment = max(-1.0, min(1.0, (pos_hits - neg_hits) / 3))
        confidence = min(1.0, (pos_hits + neg_hits + 1) / 5)
        rationale = f"event hits: +{pos_hits}/-{neg_hits}"
        return AgentInsight(ticker=ticker, sentiment=sentiment, confidence=confidence, rationale=rationale)


class ResearchSwarm:
    """Coordinates multiple autonomous agents and converts unstructured filings into alpha overlays."""

    def __init__(self, agents: list[BaseResearchAgent] | None = None) -> None:
        self.agents = agents or [FundamentalAgent(), RiskAgent(), EventAgent()]

    def aggregate(self, filing_texts: dict[str, str]) -> pd.Series:
        scores: dict[str, float] = {}
        for ticker, text in filing_texts.items():
            weighted_sum = 0.0
            total_conf = 0.0
            for agent in self.agents:
                insight = agent.evaluate(ticker=ticker, filing_text=text)
                weighted_sum += insight.sentiment * insight.confidence
                total_conf += insight.confidence
            scores[ticker] = weighted_sum / (total_conf + 1e-12)
        return pd.Series(scores, dtype=float)
