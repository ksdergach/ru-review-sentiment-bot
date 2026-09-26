import asyncio
from dataclasses import replace
from pathlib import Path
import threading
from types import SimpleNamespace
import wave

import numpy as np
import pytest

from src.speech import SpeechConfig, SpeechRecognizer


@pytest.fixture
def audio_path(tmp_path):
    path = tmp_path / "voice.ogg"
    path.write_bytes(b"fake audio for injected decoder")
    return path


def setup_fake(monkeypatch, transcribe):
    recognizer = SpeechRecognizer()
    loads = []
    model = SimpleNamespace(transcribe=transcribe)
    monkeypatch.setattr(recognizer, "_decode", lambda path: np.zeros(16000, dtype=np.float32))

    def load():
        loads.append(threading.get_ident())
        return model

    monkeypatch.setattr(recognizer, "_load_model", load)
    return recognizer, loads


def test_lazy_segments_fully_consumed_off_event_loop_and_model_reused(monkeypatch, audio_path):
    workers = []
    options = []

    def transcribe(audio, **kwargs):
        workers.append(threading.get_ident())
        options.append(kwargs)

        def segments():
            for text in (" Первый сегмент. ", "  ", " Второй сегмент! "):
                workers.append(threading.get_ident())
                yield SimpleNamespace(text=text)

        return segments(), None

    recognizer, loads = setup_fake(monkeypatch, transcribe)

    async def run():
        loop_thread = threading.get_ident()
        first = await recognizer.transcribe_async(audio_path)
        second = await recognizer.transcribe_async(audio_path)
        assert first.status == second.status == "ok"
        assert first.text == second.text == "Первый сегмент. Второй сегмент!"
        assert first.segment_count == 3
        assert first.duration_seconds == 1
        assert second.model_load_seconds == 0
        assert len(loads) == 1
        assert all(worker != loop_thread for worker in workers + loads)
        assert all(x["language"] == "ru" and x["task"] == "transcribe" and x["vad_filter"] for x in options)

    asyncio.run(run())
    assert audio_path.exists()


@pytest.mark.parametrize("texts", [[], [" ", "\n"]])
def test_empty_speech_is_not_error(monkeypatch, audio_path, texts):
    recognizer, _ = setup_fake(monkeypatch, lambda *a, **kw: (iter(SimpleNamespace(text=t) for t in texts), None))
    result = recognizer.transcribe(audio_path)
    assert result.status == "empty"
    assert result.text == ""
    assert result.error_code is None


def test_exception_mid_generator_discards_partial_text(monkeypatch, audio_path):
    def segments():
        yield SimpleNamespace(text="Incomplete transcript")
        raise RuntimeError("private path / token / model internals")

    recognizer, _ = setup_fake(monkeypatch, lambda *a, **kw: (segments(), None))
    result = recognizer.transcribe(audio_path)
    assert result.status == "error"
    assert result.error_code == "inference_failed"
    assert result.text == ""
    assert "private" not in repr(result)
    assert audio_path.exists()


def test_corrupt_file_and_missing_file_are_controlled(audio_path):
    recognizer = SpeechRecognizer()
    assert recognizer.transcribe(audio_path).error_code == "decode_failed"
    assert recognizer.transcribe(audio_path.with_name("missing.ogg")).error_code == "file_not_found"
    assert recognizer._model is None
    assert audio_path.exists()


def test_actual_decoder_reads_wav_without_system_ffmpeg(tmp_path, monkeypatch):
    path = tmp_path / "silence.wav"
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(2)
        stream.setsampwidth(2)
        stream.setframerate(48000)
        stream.writeframes(b"\x00\x00" * 48000 * 2)
    recognizer = SpeechRecognizer()

    def transcribe(audio, **kwargs):
        assert audio.ndim == 1 and len(audio) == 16000
        assert audio.dtype == np.float32
        return iter(()), None

    monkeypatch.setattr(recognizer, "_load_model", lambda: SimpleNamespace(transcribe=transcribe))
    result = recognizer.transcribe(path)
    assert result.status == "empty" and result.duration_seconds == 1


def test_model_load_failure_is_distinct_and_recoverable(monkeypatch, audio_path):
    recognizer, _ = setup_fake(monkeypatch, lambda *a, **kw: (iter(()), None))

    def fail():
        raise RuntimeError("missing local weights")

    monkeypatch.setattr(recognizer, "_load_model", fail)
    assert recognizer.transcribe(audio_path).error_code == "model_load_failed"
    assert recognizer._model is None
    monkeypatch.setattr(recognizer, "_load_model", lambda: SimpleNamespace(transcribe=lambda *a, **kw: (iter(()), None)))
    assert recognizer.transcribe(audio_path).status == "empty"


def test_cancellation_waits_for_worker_before_caller_deletes_file(monkeypatch, audio_path):
    started, release, finished = threading.Event(), threading.Event(), threading.Event()

    def transcribe(*args, **kwargs):
        def segments():
            started.set()
            assert release.wait(5)
            assert audio_path.exists()
            finished.set()
            yield SimpleNamespace(text="Готово")
        return segments(), None

    recognizer, loads = setup_fake(monkeypatch, transcribe)

    async def run():
        async def caller():
            try:
                await recognizer.transcribe_async(audio_path)
            finally:
                assert finished.is_set()
                audio_path.unlink()

        task = asyncio.create_task(caller())
        try:
            assert await asyncio.to_thread(started.wait, 5)
            task.cancel()
            await asyncio.sleep(0.01)
            task.cancel()  # Repeated cancellation must also keep the file alive.
            await asyncio.sleep(0.01)
            assert not task.done() and audio_path.exists()
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())
    assert not audio_path.exists() and len(loads) == 1


def test_concurrent_requests_share_one_model(monkeypatch, audio_path):
    recognizer, loads = setup_fake(monkeypatch, lambda *a, **kw: (iter([SimpleNamespace(text="Привет")]), None))

    async def run():
        results = await asyncio.gather(*(recognizer.transcribe_async(audio_path) for _ in range(4)))
        assert all(r.text == "Привет" for r in results)

    asyncio.run(run())
    assert len(loads) == 1


@pytest.mark.parametrize("values", [
    {"language": "en"}, {"revision": "main"}, {"beam_size": 0},
    {"cpu_threads": True}, {"cache_dir": "/tmp/models"}, {"cache_dir": "../models"},
    {"vad_filter": "true"}, {"vad_filter": False}, {"temperature": 0.5},
])
def test_invalid_config_fails_before_loading(values):
    with pytest.raises(ValueError):
        replace(SpeechConfig.load(), **values)


def test_model_loader_uses_pinned_offline_config(monkeypatch):
    import faster_whisper

    calls = []
    monkeypatch.setattr(faster_whisper, "WhisperModel", lambda *a, **kw: calls.append((a, kw)))
    SpeechRecognizer()._load_model()
    args, options = calls[0]
    config = SpeechConfig.load()
    assert args == (config.model,)
    assert options["revision"] == config.revision
    assert options["local_files_only"] is True
    assert options["num_workers"] == 1
    assert options["device"] == "cpu" and options["compute_type"] == "int8"
