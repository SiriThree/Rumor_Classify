from __future__ import annotations

from rumor_system.models.base import Prediction
from rumor_system.retrieval.lexical import EvidenceItem


def build_explanation_prompt(
    text: str,
    prediction: Prediction,
    evidence: list[EvidenceItem],
) -> str:
    evidence_lines = []
    for idx, item in enumerate(evidence, start=1):
        evidence_lines.append(
            f"{idx}. label={item.label}, score={item.score:.3f}, text={item.text}"
        )

    joined_evidence = "\n".join(evidence_lines) if evidence_lines else "None"
    return (
        "You are generating a grounded rumor-detection explanation.\n"
        f"Tweet: {text}\n"
        f"Prediction label: {prediction.label}\n"
        f"Prediction confidence: {prediction.confidence:.3f}\n"
        "Retrieved evidence:\n"
        f"{joined_evidence}\n"
        "Write a concise explanation that only uses the tweet wording and the retrieved evidence. "
        "Do not invent facts outside the input."
    )

