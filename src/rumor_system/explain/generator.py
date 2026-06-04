from __future__ import annotations

import os
from dataclasses import dataclass

import requests

from rumor_system.explain.prompting import build_explanation_prompt
from rumor_system.models.base import Prediction
from rumor_system.retrieval.lexical import EvidenceItem


@dataclass
class ExplanationGenerator:
    mode: str = "offline_template"
    model_name: str = ""

    def generate(
        self,
        text: str,
        prediction: Prediction,
        evidence: list[EvidenceItem],
    ) -> str:
        if self.mode == "sjtu_api":
            return self._generate_online(text, prediction, evidence)
        return self._generate_offline(text, prediction, evidence)

    def _generate_offline(
        self,
        text: str,
        prediction: Prediction,
        evidence: list[EvidenceItem],
    ) -> str:
        if not evidence:
            return (
                f"The tweet is predicted as {'rumor' if prediction.label == 1 else 'non-rumor'} "
                "based on the classifier output, but no strong retrieved evidence is available yet."
            )

        label_counts = {0: 0, 1: 0}
        for item in evidence:
            label_counts[item.label] += 1

        dominant = "rumor" if prediction.label == 1 else "non-rumor"
        return (
            f"The tweet is predicted as {dominant} with confidence {prediction.confidence:.2f}. "
            f"Retrieved evidence shows {label_counts[1]} rumor-labeled and {label_counts[0]} non-rumor-labeled "
            f"similar tweets. The most similar examples share overlapping wording patterns, which supports the final decision."
        )

    def _generate_online(
        self,
        text: str,
        prediction: Prediction,
        evidence: list[EvidenceItem],
    ) -> str:
        api_base = os.getenv("SJTU_API_BASE_URL", "").rstrip("/")
        api_key = os.getenv("SJTU_API_KEY", "")
        model_name = self.model_name or os.getenv("SJTU_API_MODEL", "")
        if not api_base or not api_key or not model_name:
            raise RuntimeError("SJTU API configuration is incomplete. Check your .env file.")

        prompt = build_explanation_prompt(text, prediction, evidence)
        response = requests.post(
            f"{api_base}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
            },
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        return payload["choices"][0]["message"]["content"].strip()

