from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rumor_system.config import load_config
from rumor_system.pipeline import run_split
from rumor_system.utils.io import write_csv
from rumor_system.utils.training import set_global_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--output-path", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    set_global_seed(int(config.raw["project"]["seed"]))
    df = run_split(config, split=args.split)
    output_path = args.output_path or config.output["val_predictions_path"]
    write_csv(output_path, df)
    print(f"Saved predictions to {output_path}")


if __name__ == "__main__":
    main()
