from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def classify_transition(base_correct: bool, sem_correct: bool) -> str:
    if (not base_correct) and sem_correct:
        return "helped"
    if base_correct and (not sem_correct):
        return "hurt"
    if (not base_correct) and (not sem_correct):
        return "both_wrong"
    return "both_correct"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="outputs/val_predictions.csv")
    parser.add_argument("--semantic", default="outputs/val_predictions_semantic.csv")
    parser.add_argument("--output-summary", default="outputs/semantic_experiment_analysis.txt")
    parser.add_argument("--output-csv", default="outputs/semantic_comparison.csv")
    args = parser.parse_args()

    base = pd.read_csv(args.baseline).copy()
    sem = pd.read_csv(args.semantic).copy()

    compare_cols = [
        "tweet_id",
        "text",
        "label",
        "pred_label",
        "confidence",
        "prediction_source",
        "top_evidence_label",
        "top_evidence_text",
    ]
    sem_cols = compare_cols + [
        "semantic_claim_type",
        "semantic_verification_status",
        "semantic_propagation_style",
        "semantic_tone",
        "semantic_rumor_risk",
        "semantic_label_bias",
        "semantic_override_strength",
        "semantic_retrieval_reliability",
        "semantic_trust_retrieval_more",
        "semantic_source",
        "semantic_reason",
    ]
    compare_cols = [c for c in compare_cols if c in base.columns]
    sem_cols = [c for c in sem_cols if c in sem.columns]

    base = base[compare_cols].rename(
        columns={
            "pred_label": "base_pred_label",
            "confidence": "base_confidence",
            "prediction_source": "base_prediction_source",
            "top_evidence_label": "base_top_evidence_label",
            "top_evidence_text": "base_top_evidence_text",
        }
    )
    sem = sem[sem_cols].rename(
        columns={
            "pred_label": "semantic_pred_label",
            "confidence": "semantic_confidence",
            "prediction_source": "semantic_prediction_source",
            "top_evidence_label": "semantic_top_evidence_label",
            "top_evidence_text": "semantic_top_evidence_text",
        }
    )

    merged = base.merge(sem, on=["tweet_id", "text", "label"], how="inner")
    merged["base_correct"] = merged["label"] == merged["base_pred_label"]
    merged["semantic_correct"] = merged["label"] == merged["semantic_pred_label"]
    merged["transition"] = merged.apply(
        lambda row: classify_transition(bool(row["base_correct"]), bool(row["semantic_correct"])),
        axis=1,
    )

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)

    lines: list[str] = []
    lines.append(f"baseline_accuracy: {(merged['base_correct']).mean():.6f}")
    lines.append(f"semantic_accuracy: {(merged['semantic_correct']).mean():.6f}")
    lines.append("")
    lines.append("transition_counts")
    lines.append(merged["transition"].value_counts().to_string())
    lines.append("")
    if "semantic_source" in merged.columns:
        lines.append("semantic_source_counts")
        lines.append(merged["semantic_source"].fillna("NA").value_counts().to_string())
        lines.append("")

    helped = merged[merged["transition"] == "helped"].copy()
    hurt = merged[merged["transition"] == "hurt"].copy()
    both_wrong = merged[merged["transition"] == "both_wrong"].copy()

    if not helped.empty:
        lines.append("helped_by_semantic")
        cols = [
            "tweet_id",
            "label",
            "base_pred_label",
            "semantic_pred_label",
            "base_confidence",
            "semantic_confidence",
            "semantic_prediction_source",
            "semantic_source",
            "semantic_claim_type",
            "semantic_verification_status",
            "semantic_propagation_style",
            "semantic_tone",
            "semantic_rumor_risk",
            "semantic_label_bias",
            "semantic_override_strength",
            "semantic_retrieval_reliability",
            "semantic_reason",
            "text",
        ]
        cols = [c for c in cols if c in helped.columns]
        lines.append(helped[cols].to_string(index=False, max_colwidth=100))
        lines.append("")

    if not hurt.empty:
        lines.append("hurt_by_semantic")
        cols = [
            "tweet_id",
            "label",
            "base_pred_label",
            "semantic_pred_label",
            "base_confidence",
            "semantic_confidence",
            "base_prediction_source",
            "semantic_prediction_source",
            "semantic_source",
            "semantic_claim_type",
            "semantic_verification_status",
            "semantic_propagation_style",
            "semantic_tone",
            "semantic_rumor_risk",
            "semantic_label_bias",
            "semantic_override_strength",
            "semantic_retrieval_reliability",
            "semantic_reason",
            "text",
        ]
        cols = [c for c in cols if c in hurt.columns]
        lines.append(hurt[cols].to_string(index=False, max_colwidth=100))
        lines.append("")

    if not both_wrong.empty:
        lines.append("both_wrong_semantic_summary")
        for col in [
            "semantic_claim_type",
            "semantic_verification_status",
            "semantic_propagation_style",
            "semantic_tone",
            "semantic_rumor_risk",
            "semantic_label_bias",
            "semantic_override_strength",
            "semantic_retrieval_reliability",
            "semantic_source",
        ]:
            if col in both_wrong.columns:
                lines.append(col)
                lines.append(both_wrong[col].fillna("NA").value_counts().to_string())
                lines.append("")

    output_summary = Path(args.output_summary)
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_summary.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved semantic comparison to {output_csv}")
    print(f"Saved semantic experiment analysis to {output_summary}")


if __name__ == "__main__":
    main()
