from __future__ import annotations

import json
from pathlib import Path


def write_json_report(path: Path | None, payload: dict) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def summarize_check(name: str, passed: bool, details: dict) -> dict:
    return {
        "name": name,
        "passed": bool(passed),
        "details": details,
    }
