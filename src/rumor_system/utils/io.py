from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def ensure_parent(path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def write_json(path: str | Path, payload: dict) -> None:
    out = ensure_parent(path)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: str | Path, df: pd.DataFrame) -> None:
    out = ensure_parent(path)
    df.to_csv(out, index=False)

