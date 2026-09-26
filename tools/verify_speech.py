"""Run real T08 checks on an explicitly separate debug recording, never final audio."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import sys
import wave

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.speech import SpeechConfig, SpeechRecognizer  # noqa: E402


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


async def verify(audio: Path, *, allow_download: bool) -> dict:
    config = SpeechConfig.load()
    recognizer = SpeechRecognizer(config, allow_download=allow_download)
    local = ROOT / "artifacts/speech"
    local.mkdir(parents=True, exist_ok=True)
    silence, corrupt = local / "silence.wav", local / "corrupt.ogg"
    # Do not let debug setup overwrite the user's input under these reserved names.
    if audio.resolve() in {silence.resolve(), corrupt.resolve()}:
        raise ValueError("Choose a debug recording outside the reserved silence/corrupt paths")
    with wave.open(str(silence), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(b"\0\0" * 16000 * 3)
    corrupt.write_bytes(b"This is not a valid OGG stream.\n")

    results, local_results = {}, {}
    model_after_first = None
    for name, path in (("voice_cold", audio), ("voice_warm", audio), ("silence", silence), ("corrupt", corrupt)):
        print(f"Checking {name}...", file=sys.stderr, flush=True)
        result = await recognizer.transcribe_async(path)
        print(f"{name}: {result.status}, {result.elapsed_seconds:.3f}s", file=sys.stderr, flush=True)
        data = asdict(result)
        local_results[name] = data.copy()
        text = data.pop("text")
        data.update(audio_sha256=sha256(path), audio_bytes=path.stat().st_size,
                    text_characters=len(text), text_sha256=hashlib.sha256(text.encode()).hexdigest())
        results[name] = data
        if name == "voice_cold":
            model_after_first = recognizer._model
    model_reused = model_after_first is not None and recognizer._model is model_after_first
    passed = (
        results["voice_cold"]["status"] == results["voice_warm"]["status"] == "ok"
        and results["silence"]["status"] == "empty"
        and results["corrupt"]["error_code"] == "decode_failed"
        and model_reused and results["voice_warm"]["model_load_seconds"] == 0
    )
    write_json(local / "transcripts.json", local_results)
    backend = getattr(recognizer._model, "model", None)
    return {
        "status": "PASS" if passed else "FAIL", "config": asdict(config),
        "config_sha256": sha256(ROOT / "configs/speech.json"),
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "machine": platform.machine(), "packages": {
                            name: importlib.metadata.version(name)
                            for name in ("faster-whisper", "ctranslate2", "av", "onnxruntime")}},
        "debug_audio_file": audio.name, "declared_debug_only": True,
        "final_audio_used": False, "model_reused": model_reused,
        "backend": {"device": getattr(backend, "device", None),
                    "compute_type": getattr(backend, "compute_type", None)},
        "results": results,
        "transcripts": "artifacts/speech/transcripts.json (not committed)",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--confirm-debug-audio", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()
    if not args.confirm_debug_audio:
        parser.error("Confirm this recording is separate from the final test with --confirm-debug-audio")
    path = Path(args.audio)
    if not path.is_absolute():
        path = ROOT / path
    if not path.is_file():
        parser.error("Debug audio does not exist")
    report = asyncio.run(verify(path, allow_download=args.allow_download))
    write_json(ROOT / "reports/speech/verification.json", report)
    print(json.dumps({"status": report["status"], "report": "reports/speech/verification.json"}))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
