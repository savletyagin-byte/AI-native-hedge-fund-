from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FilingDocument:
    ticker: str
    source: str
    form_type: str
    text: str


class EdgarIngestionClient:
    """Adapter for real SEC EDGAR ingestion.

    This default implementation is offline-safe and acts as an interface/stub.
    Replace `fetch_company_filings` with SEC API integration in production.
    """

    def fetch_company_filings(self, cik: str, forms: tuple[str, ...] = ("10-K", "10-Q", "8-K")) -> list[FilingDocument]:
        _ = (cik, forms)
        return []


class TranscriptFeedClient:
    """Adapter interface for earnings call transcript ingestion."""

    def fetch_latest_transcript(self, ticker: str) -> FilingDocument | None:
        _ = ticker
        return None
