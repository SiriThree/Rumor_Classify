from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re

import pandas as pd

from rumor_system.data.dataset import add_normalized_text, load_split
from rumor_system.explain.generator import ExplanationGenerator
from rumor_system.models.base import Prediction
from rumor_system.models.factory import build_classifier
from rumor_system.models.retrieval_vote import RetrievalVoteClassifier
from rumor_system.retrieval.hybrid import HybridRetriever
from rumor_system.retrieval.lexical import EvidenceItem


@dataclass
class PipelineArtifacts:
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    retriever: object
    classifier: object
    fallback_classifier: RetrievalVoteClassifier
    explainer: ExplanationGenerator


OFFICIAL_UPDATE_PATTERNS = [
    re.compile(r"\bnews conference\b", re.IGNORECASE),
    re.compile(r"\blive on\b", re.IGNORECASE),
    re.compile(r"\braw video\b", re.IGNORECASE),
    re.compile(r"\bstatement\b", re.IGNORECASE),
    re.compile(r"\baddress the nation\b", re.IGNORECASE),
    re.compile(r"\bofficially cancelled\b", re.IGNORECASE),
    re.compile(r"\bto address\b", re.IGNORECASE),
    re.compile(r"\bpolice storm\b", re.IGNORECASE),
    re.compile(r"\bwar memorial shooting\b", re.IGNORECASE),
]


def is_official_update_like(text: str) -> bool:
    return any(pattern.search(text) for pattern in OFFICIAL_UPDATE_PATTERNS)


def build_artifacts(config) -> PipelineArtifacts:
    train_df = add_normalized_text(load_split(config.data["train_path"], config.data), config.retrieval)
    val_df = add_normalized_text(load_split(config.data["val_path"], config.data), config.retrieval)
    retriever = HybridRetriever(train_df, config.retrieval)
    classifier = build_classifier(config, retriever)
    fallback_classifier = RetrievalVoteClassifier(
        retriever=retriever,
        top_k=config.retrieval["top_k"],
    )
    explainer = ExplanationGenerator(
        mode=config.explanation["mode"],
        model_name=config.explanation["model_name"],
    )
    return PipelineArtifacts(
        train_df=train_df,
        val_df=val_df,
        retriever=retriever,
        classifier=classifier,
        fallback_classifier=fallback_classifier,
        explainer=explainer,
    )


def fuse_prediction(
    base_prediction: Prediction,
    evidence: list[EvidenceItem],
    retrieval_cfg: dict,
    text: str,
) -> Prediction:
    if not evidence:
        return base_prediction

    top_score = evidence[0].score
    weighted_votes = Counter()
    raw_votes = Counter()
    for item in evidence:
        raw_votes[item.label] += 1
        weighted_votes[item.label] += max(1e-8, item.score)

    vote_label, weighted_vote_score = weighted_votes.most_common(1)[0]
    weighted_total = max(1e-8, sum(weighted_votes.values()))
    vote_ratio = weighted_vote_score / weighted_total
    raw_vote_ratio = raw_votes[vote_label] / len(evidence)
    confidence_gap = vote_ratio - base_prediction.confidence
    top_label = evidence[0].label

    if top_score >= retrieval_cfg["near_duplicate_threshold"] and raw_vote_ratio >= retrieval_cfg["vote_margin_threshold"]:
        return Prediction(label=vote_label, confidence=max(base_prediction.confidence, top_score), source="fusion_override")

    if (
        base_prediction.confidence < retrieval_cfg["vote_override_threshold"]
        and vote_ratio >= retrieval_cfg["vote_margin_threshold"]
        and confidence_gap >= retrieval_cfg["confidence_gap_override"]
    ):
        return Prediction(label=vote_label, confidence=vote_ratio, source="fusion_vote")

    if (
        retrieval_cfg.get("strong_agreement_override_enabled", False)
        and top_label == vote_label
        and top_label != base_prediction.label
        and vote_ratio >= retrieval_cfg.get("strong_agreement_vote_threshold", 0.60)
        and raw_vote_ratio >= retrieval_cfg.get("strong_agreement_raw_ratio_threshold", 0.60)
        and top_score >= retrieval_cfg.get("strong_agreement_top_score_threshold", 0.0160)
        and base_prediction.confidence <= retrieval_cfg.get("strong_agreement_max_base_confidence", 0.995)
    ):
        return Prediction(label=top_label, confidence=max(vote_ratio, top_score), source="fusion_strong_agreement")

    top_dense_score = 0.0
    if evidence[0].metadata is not None and evidence[0].metadata.get("dense_score") is not None:
        top_dense_score = float(evidence[0].metadata["dense_score"])

    if (
        retrieval_cfg.get("evidence_conflict_recheck_enabled", False)
        and top_label == vote_label
        and top_label != base_prediction.label
        and base_prediction.confidence <= retrieval_cfg.get("evidence_conflict_max_base_confidence", 0.90)
        and vote_ratio >= retrieval_cfg.get("evidence_conflict_vote_threshold", 0.55)
        and raw_vote_ratio >= retrieval_cfg.get("evidence_conflict_raw_ratio_threshold", 0.60)
        and top_score >= retrieval_cfg.get("evidence_conflict_top_score_threshold", 0.0140)
        and top_dense_score >= retrieval_cfg.get("evidence_conflict_top_dense_threshold", 0.75)
    ):
        return Prediction(label=top_label, confidence=max(vote_ratio, top_dense_score), source="fusion_evidence_conflict")

    if (
        retrieval_cfg.get("official_update_resolver_enabled", False)
        and base_prediction.label == 1
        and top_label == 0
        and base_prediction.confidence <= retrieval_cfg.get("official_update_max_confidence", 0.95)
        and is_official_update_like(text)
    ):
        return Prediction(label=0, confidence=max(base_prediction.confidence, top_score), source="fusion_official_update")

    return base_prediction


def run_split(config, split: str = "val") -> pd.DataFrame:
    artifacts = build_artifacts(config)
    df = artifacts.val_df if split == "val" else artifacts.train_df
    rows = []
    for row in df.itertuples():
        evidence = artifacts.retriever.search(row.text, top_k=config.retrieval["top_k"])
        if hasattr(artifacts.classifier, "is_trained") and not artifacts.classifier.is_trained():
            base_prediction = artifacts.fallback_classifier.predict_one(row.text)
        else:
            base_prediction = artifacts.classifier.predict_one(row.text)
        final_prediction = fuse_prediction(base_prediction, evidence, config.retrieval, row.text)
        explanation = artifacts.explainer.generate(row.text, final_prediction, evidence[: config.explanation["max_evidence_items"]])
        top_evidence_text = evidence[0].text if evidence else ""
        top_evidence_label = evidence[0].label if evidence else -1
        top_evidence_source = evidence[0].source if evidence else ""
        top_evidence_dense_score = ""
        top_evidence_lexical_score = ""
        if evidence and evidence[0].metadata:
            top_evidence_dense_score = evidence[0].metadata.get("dense_score", "")
            top_evidence_lexical_score = evidence[0].metadata.get("lexical_score", "")
        rows.append(
            {
                "tweet_id": row.tweet_id,
                "text": row.text,
                "label": row.label,
                "pred_label": final_prediction.label,
                "confidence": round(final_prediction.confidence, 4),
                "prediction_source": final_prediction.source,
                "top_score": round(evidence[0].score, 4) if evidence else 0.0,
                "top_evidence_label": top_evidence_label,
                "top_evidence_source": top_evidence_source,
                "top_evidence_lexical_score": top_evidence_lexical_score,
                "top_evidence_dense_score": top_evidence_dense_score,
                "top_evidence_text": top_evidence_text,
                "explanation": explanation,
            }
        )
    return pd.DataFrame(rows)
