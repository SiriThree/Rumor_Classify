from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from rumor_system.models.base import BaseClassifier, Prediction
from rumor_system.retrieval.lexical import EvidenceItem


@dataclass
class RetrievalVoteClassifier(BaseClassifier):
    retriever: object
    top_k: int = 5

    def fit(self, texts: list[str], labels: list[int]) -> None:
        return None

    def predict_one(self, text: str) -> Prediction:
        evidence = self.retriever.search(text=text, top_k=self.top_k)
        if not evidence:
            return Prediction(label=0, confidence=0.5, source="retrieval_vote")

        weighted_votes = self._weighted_votes(evidence)
        label, weighted_score = max(weighted_votes.items(), key=lambda pair: pair[1])
        total_weight = max(1e-8, sum(weighted_votes.values()))
        confidence = weighted_score / total_weight
        return Prediction(label=label, confidence=confidence, source="retrieval_vote")

    @staticmethod
    def _weighted_votes(evidence: list[EvidenceItem]) -> dict[int, float]:
        votes = Counter()
        for item in evidence:
            votes[item.label] += max(1e-8, item.score)
        return dict(votes)
