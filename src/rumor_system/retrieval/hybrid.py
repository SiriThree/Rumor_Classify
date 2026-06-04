from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from rumor_system.retrieval.dense import DenseRetriever
from rumor_system.retrieval.lexical import EvidenceItem, LexicalRetriever


@dataclass
class HybridRetriever:
    train_df: pd.DataFrame
    retrieval_cfg: dict

    def __post_init__(self) -> None:
        self.lexical = LexicalRetriever(self.train_df)
        self.dense = None
        if self.retrieval_cfg.get("dense_enabled", False):
            self.dense = DenseRetriever(
                train_df=self.train_df,
                model_name=self.retrieval_cfg["dense_model_name"],
                batch_size=int(self.retrieval_cfg["dense_batch_size"]),
            )

    def search(self, text: str, top_k: int = 5) -> list[EvidenceItem]:
        mode = self.retrieval_cfg.get("mode", "lexical")
        if mode == "lexical" or self.dense is None:
            return self.lexical.search(text, top_k=top_k)
        if mode == "dense":
            return self.dense.search(text, top_k=top_k)
        return self._hybrid_search(text, top_k=top_k)

    def _hybrid_search(self, text: str, top_k: int) -> list[EvidenceItem]:
        candidate_k = max(int(self.retrieval_cfg.get("dense_candidate_k", top_k)), top_k)
        lexical_results = self.lexical.search(text, top_k=candidate_k)
        dense_results = self.dense.search(text, top_k=candidate_k) if self.dense is not None else []

        merged: dict[str, EvidenceItem] = {}
        rrf_k = float(self.retrieval_cfg.get("rrf_k", 60))
        lexical_weight = float(self.retrieval_cfg.get("lexical_weight", 0.5))
        dense_weight = float(self.retrieval_cfg.get("dense_weight", 0.5))

        for rank, item in enumerate(lexical_results, start=1):
            rrf_score = lexical_weight / (rrf_k + rank)
            current = merged.get(item.tweet_id)
            if current is None:
                merged[item.tweet_id] = EvidenceItem(
                    tweet_id=item.tweet_id,
                    text=item.text,
                    label=item.label,
                    event=item.event,
                    score=rrf_score,
                    source="hybrid",
                    metadata={
                        "lexical_score": item.score,
                        "dense_score": None,
                        "rrf_score": rrf_score,
                    },
                )
            else:
                current.score += rrf_score
                current.metadata["lexical_score"] = item.score
                current.metadata["rrf_score"] = current.score

        for rank, item in enumerate(dense_results, start=1):
            rrf_score = dense_weight / (rrf_k + rank)
            current = merged.get(item.tweet_id)
            if current is None:
                merged[item.tweet_id] = EvidenceItem(
                    tweet_id=item.tweet_id,
                    text=item.text,
                    label=item.label,
                    event=item.event,
                    score=rrf_score,
                    source="hybrid",
                    metadata={
                        "lexical_score": None,
                        "dense_score": item.score,
                        "rrf_score": rrf_score,
                    },
                )
            else:
                current.score += rrf_score
                current.metadata["dense_score"] = item.score
                current.metadata["rrf_score"] = current.score

        ranked = sorted(merged.values(), key=lambda item: item.score, reverse=True)
        return ranked[:top_k]
