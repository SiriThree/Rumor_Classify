from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd


@dataclass
class Metrics:
    accuracy: float
    total: int
    correct: int

    def to_dict(self) -> dict:
        return asdict(self)


def compute_metrics(df: pd.DataFrame) -> Metrics:
    correct = int((df["label"] == df["pred_label"]).sum())
    total = int(len(df))
    accuracy = correct / total if total else 0.0
    return Metrics(accuracy=accuracy, total=total, correct=correct)

