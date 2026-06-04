from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rumor_system.config import load_config
from rumor_system.pipeline import run_split
from rumor_system.utils.training import set_global_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/bertweet.yaml")
    parser.add_argument("--output", default="outputs/error_analysis.txt")
    args = parser.parse_args()

    config = load_config(args.config)
    set_global_seed(int(config.raw["project"]["seed"]))
    df = run_split(config, split="val")
    err = df[df["label"] != df["pred_label"]].copy()

    lines: list[str] = []
    lines.append(f"total_samples: {len(df)}")
    lines.append(f"errors: {len(err)}")
    lines.append(f"accuracy: {(df['label'] == df['pred_label']).mean():.6f}")
    lines.append("")
    lines.append("by_prediction_source")
    lines.append(err["prediction_source"].value_counts().to_string())
    lines.append("")
    lines.append("confusion")
    lines.append(err.groupby(["label", "pred_label"]).size().to_string())
    lines.append("")

    if "top_evidence_label" in err.columns:
        evidence_matches_gold = (
            (err["top_evidence_label"] == err["label"])
            & (err["top_evidence_label"] != err["pred_label"])
        ).sum()
        evidence_matches_wrong = (err["top_evidence_label"] == err["pred_label"]).sum()
        lines.append(f"evidence_matches_gold_not_prediction: {int(evidence_matches_gold)}")
        lines.append(f"evidence_matches_wrong_prediction: {int(evidence_matches_wrong)}")
        lines.append("")

    err["conf_bin"] = pd.cut(
        err["confidence"],
        bins=[0, 0.6, 0.7, 0.8, 0.9, 1.0],
        include_lowest=True,
    )
    lines.append("confidence_bins")
    lines.append(err["conf_bin"].value_counts().sort_index().to_string())
    lines.append("")

    focus_cols = [
        "tweet_id",
        "label",
        "pred_label",
        "confidence",
        "prediction_source",
        "top_score",
        "top_evidence_label",
        "top_evidence_source",
        "top_evidence_text",
        "text",
    ]
    focus_cols = [col for col in focus_cols if col in err.columns]
    lines.append("top_15_high_confidence_errors")
    lines.append(
        err.sort_values("confidence", ascending=False)[focus_cols]
        .head(15)
        .to_string(index=False, max_colwidth=100)
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved error analysis to {output_path}")


if __name__ == "__main__":
    main()
