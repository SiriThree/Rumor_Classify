from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rumor_system.config import load_config
from rumor_system.pipeline import build_artifacts
from rumor_system.utils.io import write_json
from rumor_system.utils.training import set_global_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--force-retrain", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    set_global_seed(int(config.raw["project"]["seed"]))
    artifacts = build_artifacts(config)
    summary = {
        "train_size": int(len(artifacts.train_df)),
        "val_size": int(len(artifacts.val_df)),
        "classifier_type": config.model["classifier_type"],
        "fallback_classifier_type": config.model["fallback_classifier_type"],
        "retrieval_enabled": config.retrieval["enabled"],
    }

    classifier = artifacts.classifier
    if config.model["classifier_type"] == "transformer":
        force_retrain = bool(config.model.get("force_retrain", False)) or args.force_retrain
        if force_retrain or not classifier.is_trained():
            history = classifier.fit(
                train_texts=artifacts.train_df["model_input_text"].tolist(),
                train_labels=artifacts.train_df["label"].astype(int).tolist(),
                val_texts=artifacts.val_df["model_input_text"].tolist(),
                val_labels=artifacts.val_df["label"].astype(int).tolist(),
            )
            summary["training_history"] = history
            summary["checkpoint_dir"] = config.model["checkpoint_dir"]
        else:
            summary["checkpoint_dir"] = config.model["checkpoint_dir"]
            summary["status"] = "Skipped training because a checkpoint already exists."
    else:
        summary["status"] = "No transformer training requested."

    write_json(config.output["train_summary_path"], summary)
    print(f"Saved training summary to {config.output['train_summary_path']}")


if __name__ == "__main__":
    main()
