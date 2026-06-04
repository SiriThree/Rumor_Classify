from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Prediction:
    label: int
    confidence: float
    source: str


class BaseClassifier:
    def fit(self, texts: list[str], labels: list[int]) -> None:
        raise NotImplementedError

    def predict_one(self, text: str) -> Prediction:
        raise NotImplementedError

