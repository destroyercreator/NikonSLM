from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Config:
    raw: dict[str, Any]

    @property
    def output_excel(self) -> Path:
        return Path(self.raw["project"]["output_excel"])


def load_config(path: Path) -> Config:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return Config(raw=data)
