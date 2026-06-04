from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rumor_system.config import load_config
from rumor_system.pipeline import run_split
from rumor_system.utils.training import set_global_seed


def rumor_probability(df: pd.DataFrame) -> pd.Series:
    return df.apply(
        lambda row: row["confidence"] if int(row["pred_label"]) == 1 else 1.0 - row["confidence"],
        axis=1,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-a", default="configs/bertweet.yaml")
    parser.add_argument("--config-b", default="configs/bertweet_seed7.yaml")
    parser.add_argument("--output", default="outputs/ensemble_predictions.csv")
    args = parser.parse_args()

    config_a = load_config(args.config_a)
    config_b = load_config(args.config_b)

    set_global_seed(int(config_a.raw["project"]["seed"]))
    df_a = run_split(config_a, split="val").copy()
    set_global_seed(int(config_b.raw["project"]["seed"]))
    df_b = run_split(config_b, split="val").copy()

    if list(df_a["tweet_id"]) != list(df_b["tweet_id"]):
        raise RuntimeError("Prediction files are not aligned by tweet_id.")

    out = df_a[["tweet_id", "text", "label"]].copy()
    out["rumor_prob_a"] = rumor_probability(df_a)
    out["rumor_prob_b"] = rumor_probability(df_b)
    out["ensemble_rumor_prob"] = (out["rumor_prob_a"] + out["rumor_prob_b"]) / 2.0
    out["pred_label"] = (out["ensemble_rumor_prob"] >= 0.5).astype(int)
    out["confidence"] = out["ensemble_rumor_prob"].apply(lambda x: x if x >= 0.5 else 1.0 - x)
    out["source_a"] = df_a["prediction_source"]
    out["source_b"] = df_b["prediction_source"]
    out["pred_a"] = df_a["pred_label"]
    out["pred_b"] = df_b["pred_label"]

    accuracy = float((out["label"] == out["pred_label"]).mean())
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(
        {
            "accuracy": accuracy,
            "total": int(len(out)),
            "correct": int((out["label"] == out["pred_label"]).sum()),
            "output_path": str(output_path),
        }
    )


if __name__ == "__main__":
    main()
