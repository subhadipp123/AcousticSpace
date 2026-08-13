from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


HISTORY_DIR = Path("data/history")
HISTORY_FILE = HISTORY_DIR / "analysis_history.json"


def _ensure_history_file() -> None:
    HISTORY_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not HISTORY_FILE.exists():
        HISTORY_FILE.write_text(
            "[]",
            encoding="utf-8",
        )


def load_history() -> list[dict]:
    _ensure_history_file()

    try:
        data = json.loads(
            HISTORY_FILE.read_text(
                encoding="utf-8"
            )
        )

        if isinstance(data, list):
            return data

        return []

    except (json.JSONDecodeError, OSError):
        return []


def save_analysis(
    filename: str,
    prediction: dict,
) -> dict:
    history = load_history()

    item = {
        "id": datetime.now().isoformat(
            timespec="milliseconds"
        ),
        "filename": filename,
        "timestamp": datetime.now().isoformat(
            timespec="seconds"
        ),
        "prediction": prediction["label"],
        "confidence": prediction["confidence"],
        "model": prediction["model"],
    }

    history.insert(0, item)

    # Keep only the latest 50 analyses.
    history = history[:50]

    _ensure_history_file()

    HISTORY_FILE.write_text(
        json.dumps(
            history,
            indent=2,
        ),
        encoding="utf-8",
    )

    return item