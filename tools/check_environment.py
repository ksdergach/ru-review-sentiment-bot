"""Check pinned versions and real imports without downloading models or reading data."""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import platform
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULES = {
    "scikit-learn": "sklearn",
    "faster-whisper": "faster_whisper",
}


def main() -> int:
    expected_python = (ROOT / ".python-version").read_text().strip()
    errors: list[str] = []
    if platform.python_version() != expected_python:
        errors.append(f"Expected Python {expected_python}, found {platform.python_version()}")

    packages: dict[str, dict[str, str]] = {}
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        line = line.partition("#")[0].strip()
        if not line:
            continue
        name, expected = line.split("==", 1)
        entry = {"expected": expected}
        packages[name] = entry
        try:
            actual = importlib.metadata.version(name)
            entry["installed"] = actual
            # torch 2.10.0+cpu / +cu126 are platform variants of the pinned release.
            if actual.split("+", 1)[0] != expected:
                errors.append(f"{name}: expected {expected}, found {actual}")
            importlib.import_module(MODULES.get(name, name.replace("-", "_")))
            entry["import"] = "ok"
        except Exception as exc:
            entry["import"] = "failed"
            errors.append(f"{name}: {type(exc).__name__}: {exc}")

    # transformers uses lazy imports; importing the top-level package is insufficient.
    try:
        from transformers import AutoTokenizer, BertForSequenceClassification
        from faster_whisper import WhisperModel
        from aiogram import Bot, Dispatcher

        assert all((AutoTokenizer, BertForSequenceClassification, WhisperModel, Bot, Dispatcher))
    except Exception as exc:
        errors.append(f"Model/bot class imports: {type(exc).__name__}: {exc}")

    print(json.dumps({
        "status": "PASS" if not errors else "FAIL",
        "python": platform.python_version(),
        "expected_python": expected_python,
        "system": platform.system(),
        "machine": platform.machine(),
        "packages": packages,
        "errors": errors,
        "scope": "Versions and imports only; no training, model downloads, GPU or Telegram calls.",
    }, ensure_ascii=False, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
