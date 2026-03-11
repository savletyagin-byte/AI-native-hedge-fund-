from __future__ import annotations

import json
from dataclasses import dataclass
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class FilingDocument:
    ticker: str
    source: str
    form_type: str
    text: str


class EdgarIngestionClient:
    """Adapter for SEC EDGAR ingestion with an optional live-submissions helper."""

    def __init__(self, user_agent: str = "ai-native-hedge-fund research@example.com") -> None:
        self.user_agent = user_agent

    def fetch_company_filings(self, cik: str, forms: tuple[str, ...] = ("10-K", "10-Q", "8-K")) -> list[FilingDocument]:
        try:
            submissions = self.fetch_submissions_json(cik)
        except Exception:
            return []

        recent = submissions.get("filings", {}).get("recent", {})
        form_list = recent.get("form", [])
        accession = recent.get("accessionNumber", [])
        primary_doc = recent.get("primaryDocument", [])

        docs: list[FilingDocument] = []
        for form, acc, doc in zip(form_list, accession, primary_doc):
            if form not in forms:
                continue
            docs.append(
                FilingDocument(
                    ticker=cik,
                    source="edgar",
                    form_type=form,
                    text=f"accession={acc}; primaryDocument={doc}",
                )
            )
        return docs

    def fetch_submissions_json(self, cik: str) -> dict:
        cik_padded = cik.zfill(10)
        url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
        req = Request(url, headers={"User-Agent": self.user_agent})
        with urlopen(req, timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))


class TranscriptFeedClient:
    """Adapter interface for earnings call transcript ingestion."""

    def fetch_latest_transcript(self, ticker: str) -> FilingDocument | None:
        _ = ticker
        return None
