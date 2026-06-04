from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from rumor_system.data.preprocess import simple_tokenize


@dataclass
class EvidenceItem:
    tweet_id: str
    text: str
    label: int
    event: int
    score: float
    source: str = "lexical"
    metadata: dict[str, Any] | None = None


class LexicalRetriever:
    def __init__(self, train_df: pd.DataFrame) -> None:
        self.train_df = train_df.copy()
        self.train_df["tokens"] = self.train_df["normalized_text"].map(simple_tokenize)
        self.train_df["token_set"] = self.train_df["tokens"].map(set)

    @staticmethod
    def _jaccard(query_tokens: set[str], doc_tokens: set[str]) -> float:
        union = query_tokens | doc_tokens
        if not union:
            return 0.0
        return len(query_tokens & doc_tokens) / len(union)

    def search(self, text: str, top_k: int = 5) -> list[EvidenceItem]:
        query_tokens = set(simple_tokenize(text))
        scored: list[EvidenceItem] = []
        for row in self.train_df.itertuples():
            score = self._jaccard(query_tokens, row.token_set)
            scored.append(
                EvidenceItem(
                    tweet_id=str(row.tweet_id),
                    text=row.text,
                    label=int(row.label),
                    event=int(row.event),
                    score=score,
                    source="lexical",
                    metadata={"lexical_score": score},
                )
            )

        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]
