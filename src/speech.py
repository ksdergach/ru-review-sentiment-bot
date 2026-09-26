"""Reusable, synchronous Whisper worker and an event-loop-safe async adapter."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import threading
from time import perf_counter
from typing import Literal

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class SpeechConfig:
    model: str
    revision: str
    device: str
    compute_type: str
    cpu_threads: int
    beam_size: int
    language: str
    vad_filter: bool
    temperature: float
    condition_on_previous_text: bool
    cache_dir: str

    def __post_init__(self) -> None:
        if not self.model or not re.fullmatch(r"[0-9a-f]{40}", self.revision):
            raise ValueError("A model ID and pinned 40-character revision are required")
        if self.language != "ru":
            raise ValueError("The project ASR language must be ru")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")
        if self.compute_type not in {"int8", "float32", "float16", "int8_float16"}:
            raise ValueError("Unsupported compute_type")
        for value in (self.cpu_threads, self.beam_size):
            if type(value) is not int or value < 1:
                raise ValueError("cpu_threads and beam_size must be positive integers")
        if type(self.vad_filter) is not bool or type(self.condition_on_previous_text) is not bool:
            raise ValueError("VAD and conditioning settings must be booleans")
        if not self.vad_filter or self.temperature != 0.0:
            raise ValueError("The selected profile requires VAD and temperature=0")
        path = Path(self.cache_dir)
        if not self.cache_dir or path.is_absolute() or ".." in path.parts:
            raise ValueError("cache_dir must be relative to the repository")

    @classmethod
    def load(cls, path: str | Path = "configs/speech.json") -> SpeechConfig:
        path = Path(path)
        if not path.is_absolute():
            path = ROOT / path
        return cls(**json.loads(path.read_text(encoding="utf-8")))


@dataclass(frozen=True)
class SpeechResult:
    status: Literal["ok", "empty", "error"]
    text: str = ""
    error_code: str | None = None
    duration_seconds: float | None = None
    elapsed_seconds: float = 0.0
    model_load_seconds: float = 0.0
    segment_count: int = 0


class SpeechRecognizer:
    """Create once per application. Owns the model, never the caller's audio file.

    All decoding, loading and lazy segment iteration happen inside transcribe().
    The lock serializes use and protects initialization if several threads arrive.
    Request limits, admission/busy policy and hard timeouts belong to T10.
    """

    def __init__(self, config: SpeechConfig | None = None, *, allow_download: bool = False):
        self.config = config or SpeechConfig.load()
        self.allow_download = allow_download
        self._model = None
        self._lock = threading.Lock()

    def _load_model(self):
        from faster_whisper import WhisperModel

        c = self.config
        return WhisperModel(
            c.model, revision=c.revision, device=c.device, compute_type=c.compute_type,
            cpu_threads=c.cpu_threads, num_workers=1,
            download_root=str(ROOT / c.cache_dir), local_files_only=not self.allow_download,
        )

    @staticmethod
    def _decode(path: Path):
        from faster_whisper.audio import decode_audio

        return decode_audio(str(path), sampling_rate=16000)

    def transcribe(self, audio_path: str | Path) -> SpeechResult:
        """Blocking; returns full text or a controlled empty/error result, never partial text."""
        with self._lock:
            start = perf_counter()
            duration = None
            load_seconds = 0.0

            def result(status, *, text="", error_code=None, segment_count=0):
                return SpeechResult(
                    status=status, text=text, error_code=error_code,
                    duration_seconds=duration, elapsed_seconds=perf_counter() - start,
                    model_load_seconds=load_seconds, segment_count=segment_count,
                )

            try:
                path = Path(audio_path)
                if not path.is_file():
                    return result("error", error_code="file_not_found")
                audio = self._decode(path)
                duration = len(audio) / 16000
            except Exception:
                return result("error", error_code="decode_failed")
            if not len(audio):
                return result("empty")

            if self._model is None:
                load_start = perf_counter()
                try:
                    self._model = self._load_model()
                except Exception:
                    load_seconds = perf_counter() - load_start
                    return result("error", error_code="model_load_failed")
                load_seconds = perf_counter() - load_start
            try:
                c = self.config
                segments, _info = self._model.transcribe(
                    audio, language=c.language, task="transcribe", beam_size=c.beam_size,
                    vad_filter=c.vad_filter, temperature=c.temperature,
                    condition_on_previous_text=c.condition_on_previous_text,
                )
                # This loop performs the actual heavy inference. Never return the generator.
                parts, segment_count = [], 0
                for segment in segments:
                    segment_count += 1
                    part = segment.text.strip()
                    if part:
                        parts.append(part)
                text = " ".join(parts)
            except Exception:
                return result("error", error_code="inference_failed")
            return result("ok" if text else "empty", text=text, segment_count=segment_count)

    async def transcribe_async(self, audio_path: str | Path) -> SpeechResult:
        """Run the whole operation in a worker, keeping audio alive through cancellation.

        Cancellation waits for the worker before propagating: the caller may then
        safely delete its temporary audio in finally. This is NOT a hard timeout.
        """
        worker = asyncio.create_task(asyncio.to_thread(self.transcribe, audio_path))
        try:
            return await asyncio.shield(worker)
        except asyncio.CancelledError:
            while not worker.done():
                try:
                    await asyncio.shield(worker)
                except asyncio.CancelledError:
                    continue
            raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", help="Local audio file, including Telegram OGG/Opus")
    parser.add_argument("--config", default="configs/speech.json")
    parser.add_argument("--allow-download", action="store_true", help="Allow downloading the pinned model on first use")
    parser.add_argument("--show-text", action="store_true", help="Print the transcript locally; avoid sharing output with review text")
    args = parser.parse_args()
    try:
        recognizer = SpeechRecognizer(SpeechConfig.load(args.config), allow_download=args.allow_download)
    except (ValueError, TypeError, OSError) as exc:
        parser.error(f"Invalid speech configuration: {exc}")
    result = recognizer.transcribe(args.audio)
    output = asdict(result)
    if not args.show_text:
        output["text_characters"] = len(output.pop("text"))
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 1 if result.status == "error" else 0


if __name__ == "__main__":
    raise SystemExit(main())
