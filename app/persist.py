from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def atomic_write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=str(path.parent), encoding="utf-8") as tmp:
        json.dump(payload, tmp, indent=2, sort_keys=False, default=str)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_name = tmp.name
    os.replace(tmp_name, path)


def read_json(path: str | Path, default: Any) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return default
