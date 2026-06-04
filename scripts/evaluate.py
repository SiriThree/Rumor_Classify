from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rumor_system.config import load_config
from rumor_system.metrics import compute_metrics
from rumor_system.pipeline import run_split
from rumor_system.utils.io import write_json
from rumor_system.utils.training import set_global_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    set_global_seed(int(config.raw["project"]["seed"]))
    df = run_split(config, split="val")
    metrics = compute_metrics(df)
    write_json(config.output["val_metrics_path"], metrics.to_dict())
    print(metrics.to_dict())


if __name__ == "__main__":
    main()
