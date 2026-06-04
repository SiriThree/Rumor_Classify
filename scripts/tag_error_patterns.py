from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


HEADLINE_PATTERNS = [
    re.compile(r"\bbreaking\b", re.IGNORECASE),
    re.compile(r"\bupdate\b", re.IGNORECASE),
    re.compile(r"\blatest\b", re.IGNORECASE),
    re.compile(r"\breported|reportedly|reports|understood\b", re.IGNORECASE),
]

OPINION_PATTERNS = [
    re.compile(r"\bi think\b", re.IGNORECASE),
    re.compile(r"\bi'm\b", re.IGNORECASE),
    re.compile(r"\bi am\b", re.IGNORECASE),
    re.compile(r"\bdisgusting\b", re.IGNORECASE),
    re.compile(r"\bhate\b", re.IGNORECASE),
    re.compile(r"\bclearly\b", re.IGNORECASE),
    re.compile(r"\bplease\b", re.IGNORECASE),
    re.compile(r"\bthoughts and prayers\b", re.IGNORECASE),
]

SOURCE_PATTERNS = [
    re.compile(r"http", re.IGNORECASE),
    re.compile(r"\bvia\b", re.IGNORECASE),
    re.compile(r"\bimage\b", re.IGNORECASE),
    re.compile(r"\brt\b", re.IGNORECASE),
]


def contains_any(text: str, patterns: list[re.Pattern[str]]) -> bool:
    return any(pattern.search(text) for pattern in patterns)


def classify_row(row: pd.Series) -> tuple[str, list[str]]:
    text = str(row["text"])
    evidence = str(row.get("top_evidence_text", ""))
    tags: list[str] = []

    is_false_alarm = int(row["label"]) == 0 and int(row["pred_label"]) == 1
    is_miss = int(row["label"]) == 1 and int(row["pred_label"]) == 0

    headline_like = contains_any(text, HEADLINE_PATTERNS)
    opinion_like = contains_any(text, OPINION_PATTERNS)
    source_heavy = contains_any(text, SOURCE_PATTERNS)

    evidence_matches_gold = int(row.get("top_evidence_label", -1)) == int(row["label"])
    evidence_matches_pred = int(row.get("top_evidence_label", -1)) == int(row["pred_label"])

    if evidence_matches_gold and not evidence_matches_pred:
        tags.append("evidence_conflict")
    if evidence_matches_pred:
        tags.append("retrieval_supports_error")

    if is_false_alarm and headline_like:
        tags.append("headline_false_alarm")
    if is_false_alarm and source_heavy:
        tags.append("source_cue_false_alarm")
    if is_miss and opinion_like:
        tags.append("opinion_like_rumor_miss")
    if is_miss and not headline_like and not source_heavy:
        tags.append("implicit_claim_miss")

    if "?" in text:
        tags.append("question_or_speculation")
    if text.isupper() or re.search(r"\b[A-Z]{4,}\b", text):
        tags.append("emphasis_format")
    if len(text) < 90:
        tags.append("short_text")
    if len(text) > 130:
        tags.append("long_text")

    if evidence and text[:40].lower() == evidence[:40].lower():
        tags.append("near_duplicate_confusion")

    primary = (
        "headline_false_alarm"
        if "headline_false_alarm" in tags
        else "opinion_like_rumor_miss"
        if "opinion_like_rumor_miss" in tags
        else "evidence_conflict"
        if "evidence_conflict" in tags
        else "retrieval_supports_error"
        if "retrieval_supports_error" in tags
        else "implicit_claim_miss"
        if "implicit_claim_miss" in tags
        else "other"
    )
    return primary, tags


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="outputs/val_predictions.csv")
    parser.add_argument("--output-csv", default="outputs/error_cases_tagged.csv")
    parser.add_argument("--output-summary", default="outputs/error_pattern_summary.txt")
    args = parser.parse_args()

    df = pd.read_csv(args.predictions)
    err = df[df["label"] != df["pred_label"]].copy()

    primaries = []
    all_tags = []
    for _, row in err.iterrows():
        primary, tags = classify_row(row)
        primaries.append(primary)
        all_tags.append("|".join(tags))

    err["primary_error_pattern"] = primaries
    err["error_tags"] = all_tags

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    err.to_csv(output_csv, index=False)

    lines: list[str] = []
    lines.append(f"total_errors: {len(err)}")
    lines.append("")
    lines.append("primary_error_pattern_counts")
    lines.append(err["primary_error_pattern"].value_counts().to_string())
    lines.append("")
    lines.append("prediction_source_by_pattern")
    lines.append(err.groupby(["primary_error_pattern", "prediction_source"]).size().to_string())
    lines.append("")
    lines.append("top_10_examples")
    cols = [
        "tweet_id",
        "label",
        "pred_label",
        "confidence",
        "prediction_source",
        "top_evidence_label",
        "primary_error_pattern",
        "error_tags",
        "text",
    ]
    lines.append(
        err.sort_values("confidence", ascending=False)[cols]
        .head(10)
        .to_string(index=False, max_colwidth=100)
    )

    output_summary = Path(args.output_summary)
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_summary.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved tagged errors to {output_csv}")
    print(f"Saved summary to {output_summary}")


if __name__ == "__main__":
    main()
