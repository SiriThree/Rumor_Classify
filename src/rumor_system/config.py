from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class AppConfig:
    raw: dict[str, Any]

    @property
    def data(self) -> dict[str, Any]:
        return self.raw["data"]

    @property
    def model(self) -> dict[str, Any]:
        return self.raw["model"]

    @property
    def retrieval(self) -> dict[str, Any]:
        return self.raw["retrieval"]

    @property
    def explanation(self) -> dict[str, Any]:
        return self.raw["explanation"]

    @property
    def output(self) -> dict[str, Any]:
        return self.raw["output"]


def load_config(path: str | Path) -> AppConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return AppConfig(raw=data)

