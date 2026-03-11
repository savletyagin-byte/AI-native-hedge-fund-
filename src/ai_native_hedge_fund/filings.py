from __future__ import annotations

import numpy as np


class FilingStream:
    """Synthetic SEC/earnings-call style document stream for agentic research."""

    POSITIVE = [
        "revenue growth accelerated with margin expansion",
        "guidance raised and free cash flow inflected",
        "new product launch drove backlog expansion",
        "share repurchase and strategic partnership announced",
    ]
    NEGATIVE = [
        "management disclosed a material weakness",
        "guidance cut after production delay",
        "company faces investigation and lawsuit",
        "liquidity concern increased after covenant pressure",
    ]

    def __init__(self, seed: int = 7) -> None:
        self.rng = np.random.default_rng(seed)

    def generate(self, tickers: list[str]) -> dict[str, str]:
        docs: dict[str, str] = {}
        for ticker in tickers:
            pos = self.rng.choice(self.POSITIVE, size=2, replace=False)
            neg = self.rng.choice(self.NEGATIVE, size=1, replace=False)
            mix = list(pos) + list(neg)
            self.rng.shuffle(mix)
            docs[ticker] = " ".join(mix)
        return docs
