from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import pytest

import src.bot as bot
import src.classifier as classifier
from test_classifier import baseline, make_real_bundle


class FakeMessage:
    def __init__(self, text: str | None = None) -> None:
        self.text = text
        self.answers: list[str] = []

    async def answer(self, text: str) -> None:
        self.answers.append(text)


def positive_prediction() -> dict[str, Any]:
    return {
        "label": "positive",
        "label_id": 2,
        "confidence": 0.7,
        "probabilities": {
            "negative": 0.1,
            "neutral": 0.2,
            "positive": 0.7,
        },
    }


def test_bot_imports_common_predict() -> None:
    assert bot.predict is classifier.predict


def test_format_prediction_contains_label_and_confidence() -> None:
    answer = bot.format_prediction(positive_prediction())

    # T09: the Russian label is shown; the machine label stays in brackets.
    assert "Тональность: положительная (positive)" in answer
    assert "70.0%" in answer
    assert "не является гарантией" in answer


def test_bot_uses_common_predict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: list[str] = []

    def fake_predict(text: str) -> dict[str, Any]:
        received.append(text)
        return positive_prediction()

    monkeypatch.setattr(bot, "predict", fake_predict)

    answer = bot.classify_text_for_bot("Отличный фильм")

    assert received == ["Отличный фильм"]
    assert "positive" in answer


def test_empty_text_returns_message() -> None:
    assert (
        bot.classify_text_for_bot("   ")
        == bot.EMPTY_TEXT_MESSAGE
    )


def test_long_text_returns_message() -> None:
    text = "а" * (bot.MAX_TEXT_LENGTH + 1)

    assert (
        bot.classify_text_for_bot(text)
        == bot.TOO_LONG_MESSAGE
    )


def test_whitespace_over_2000_returns_empty_message() -> None:
    text = " " * (bot.MAX_TEXT_LENGTH + 1)

    assert (
        bot.classify_text_for_bot(text)
        == bot.EMPTY_TEXT_MESSAGE
    )


def test_model_error_returns_message_and_is_logged(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def fake_predict(text: str) -> dict[str, Any]:
        raise classifier.ClassifierError("prediction failed")

    monkeypatch.setattr(bot, "predict", fake_predict)

    with caplog.at_level(logging.ERROR):
        answer = bot.classify_text_for_bot("Обычный отзыв")

    assert answer == bot.MODEL_ERROR_MESSAGE
    assert "Text classification failed" in caplog.text


def test_start_handler() -> None:
    message = FakeMessage()

    asyncio.run(
        bot.start_handler(message)  # type: ignore[arg-type]
    )

    assert message.answers == [bot.START_TEXT]


def test_help_handler() -> None:
    message = FakeMessage()

    asyncio.run(
        bot.help_handler(message)  # type: ignore[arg-type]
    )

    assert message.answers == [bot.HELP_TEXT]


def test_unknown_command_handler() -> None:
    message = FakeMessage("/settings")

    asyncio.run(
        bot.unknown_command_handler(message)  # type: ignore[arg-type]
    )

    assert message.answers == [
        bot.UNKNOWN_COMMAND_MESSAGE
    ]


def test_text_handler_uses_same_classification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        bot,
        "classify_text_for_bot",
        lambda text: "classification result",
    )

    message = FakeMessage("Обычный фильм")

    asyncio.run(
        bot.text_handler(message)  # type: ignore[arg-type]
    )

    assert message.answers == ["classification result"]


def test_unsupported_handler() -> None:
    message = FakeMessage()

    asyncio.run(
        bot.unsupported_handler(message)  # type: ignore[arg-type]
    )

    assert message.answers == [
        bot.UNSUPPORTED_MESSAGE
    ]


def test_handler_order() -> None:
    callbacks = [
        handler.callback
        for handler in bot.router.message.handlers
    ]

    assert callbacks == [
        bot.start_handler,
        bot.help_handler,
        bot.unknown_command_handler,
        bot.text_handler,
        bot.voice_handler,
        bot.unsupported_handler,
    ]


def test_token_is_read_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "TELEGRAM_BOT_TOKEN",
        "123456:LOCAL_TEST",
    )

    assert bot.get_bot_token() == "123456:LOCAL_TEST"


def test_token_is_stripped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "TELEGRAM_BOT_TOKEN",
        "  123456:LOCAL_TEST  ",
    )

    assert bot.get_bot_token() == "123456:LOCAL_TEST"


def test_missing_token_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(
        "TELEGRAM_BOT_TOKEN",
        raising=False,
    )

    with pytest.raises(RuntimeError):
        bot.get_bot_token()


def test_real_bundle_direct_and_bot_paths_match(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = make_real_bundle(tmp_path)
    model = baseline(model_path)

    monkeypatch.setattr(
        classifier,
        "_default_classifier",
        model,
    )

    text = "отличный фильм"

    direct_result = classifier.predict(text)
    bot_answer = bot.classify_text_for_bot(text)

    assert bot_answer == bot.format_prediction(
        direct_result
    )


def test_exactly_2000_characters_reach_classifier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = make_real_bundle(tmp_path)
    model = baseline(model_path)

    monkeypatch.setattr(
        classifier,
        "_default_classifier",
        model,
    )

    answer = bot.classify_text_for_bot(
        "а" * bot.MAX_TEXT_LENGTH
    )

    assert "Тональность:" in answer
    assert "Уверенность модели:" in answer


def test_2001_characters_are_rejected_before_model() -> None:
    answer = bot.classify_text_for_bot(
        "а" * (bot.MAX_TEXT_LENGTH + 1)
    )

    assert answer == bot.TOO_LONG_MESSAGE

@pytest.mark.parametrize("text", [None, 42, ["отзыв"]])
def test_invalid_input_returns_message(text):
    assert bot.classify_text_for_bot(text) == bot.EMPTY_TEXT_MESSAGE


@pytest.mark.parametrize("token", ["not-a-token", "123:", "abc:secret", "123:has space"])
def test_bad_token_is_reported_without_exposing_secret(monkeypatch, token):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", token)
    with pytest.raises(RuntimeError, match="invalid format") as error:
        bot.get_bot_token()
    assert token not in str(error.value)
    assert error.value.__suppress_context__


@pytest.mark.parametrize("text,command", [("/10, отличный фильм", False), ("/5 скучный фильм", False), ("/settings", True)])
def test_real_router_distinguishes_commands_from_slash_reviews(monkeypatch, text, command):
    from datetime import datetime, timezone
    from aiogram.types import Message, Chat, MessageEntity
    answers = []
    async def answer(self, method, **kwargs):
        answers.append(method.text)
    monkeypatch.setattr(bot.Bot, "__call__", answer)
    monkeypatch.setattr(bot, "classify_text_for_bot", lambda text: "classified: " + text)
    async def run():
        client = bot.Bot("123456:LOCAL_TEST")
        message = Message(message_id=1, date=datetime.now(timezone.utc), chat=Chat(id=1, type="private"), text=text,
                          entities=[MessageEntity(type="bot_command", offset=0, length=len(text))] if command else []).as_(client)
        try:
            await bot.router.propagate_event(update_type="message", event=message, bot=client)
        finally:
            await client.session.close()
    asyncio.run(run())
    assert answers == [bot.UNKNOWN_COMMAND_MESSAGE if command else "classified: " + text]


@pytest.mark.parametrize("cancel", [False, True])
def test_inference_does_not_block_help_or_start_second_job(monkeypatch, cancel):
    from threading import Event
    started, release = Event(), Event()
    calls = []
    def slow_predict(text):
        calls.append(text)
        started.set()
        assert release.wait(5), "worker was not released"
        return "classification result"
    monkeypatch.setattr(bot, "classify_text_for_bot", slow_predict)
    async def run():
        monkeypatch.setattr(bot, "_classification_lock", asyncio.Lock())
        first = asyncio.create_task(bot.text_handler(FakeMessage("первый")))
        try:
            # Event-based synchronisation: fails if inference runs on this loop.
            for _ in range(500):
                if started.is_set(): break
                await asyncio.sleep(.001)
            assert started.is_set()
            if cancel:
                first.cancel()
                await asyncio.sleep(0)
            help_message = FakeMessage()
            await asyncio.wait_for(bot.help_handler(help_message), .5)
            assert help_message.answers == [bot.HELP_TEXT]
            second = FakeMessage("второй")
            await asyncio.wait_for(bot.text_handler(second), .5)
            assert second.answers == [bot.BUSY_MESSAGE]
            assert calls == ["первый"]
        finally:
            release.set()
            if cancel:
                with pytest.raises(asyncio.CancelledError):
                    await first
            else:
                await first
        assert not bot._classification_lock.locked()
    asyncio.run(run())


# --- T09: voice messages -------------------------------------------------------------

import threading
from types import SimpleNamespace

import numpy as np

from src.speech import SpeechRecognizer
from test_classifier import write_report
from src import train_baseline as tb


class FakeVoiceMessage(FakeMessage):
    def __init__(self, duration: int = 5, file_size: int | None = 20_000, file_id: str = "voice-file-id") -> None:
        super().__init__(None)
        self.voice = SimpleNamespace(file_id=file_id, duration=duration, file_size=file_size)


def write_destination(destination, data: bytes) -> Path:
    """Write like Bot.download: into an open binary file or to a path; return the path."""
    if hasattr(destination, "write"):
        destination.write(data)
        return Path(destination.name)
    Path(destination).write_bytes(data)
    return Path(destination)


def destination_path(destination) -> Path:
    return Path(destination.name if hasattr(destination, "write") else destination)


class FakeBot:
    """Telegram download stub: writes synthetic bytes, never touches the network."""

    def __init__(self, error: BaseException | None = None, payload: bytes = b"synthetic voice") -> None:
        self.error = error
        self.payload = payload
        self.downloads: list[tuple[str, Path]] = []

    async def download(self, file_id: str, destination) -> None:
        self.downloads.append((file_id, destination_path(destination)))
        if self.error is not None:
            raise self.error
        write_destination(destination, self.payload)


def fake_speech(monkeypatch, segments=("Отличный фильм,", "понравился сюжет."), *, fail=False, started=None, release=None):
    """Real SpeechRecognizer with injected decoder and Whisper model."""
    speech = SpeechRecognizer()
    calls = SimpleNamespace(loads=0, decoded=[], threads=[])

    def decode(path):
        calls.decoded.append((Path(path), Path(path).exists()))
        return np.zeros(16000, dtype=np.float32)

    def transcribe(audio, **kwargs):
        def generate():
            calls.threads.append(threading.get_ident())
            if started is not None:
                started.set()
                assert release.wait(5), "worker was not released"
            for text in segments:
                if fail:
                    raise RuntimeError("decoder internals")
                yield SimpleNamespace(text=text)
        return generate(), None

    def load():
        calls.loads += 1
        return SimpleNamespace(transcribe=transcribe)

    monkeypatch.setattr(speech, "_decode", decode)
    monkeypatch.setattr(speech, "_load_model", load)
    return speech, calls


def recording_predict(monkeypatch, result=None, error=None):
    calls = []

    def fake_predict(text):
        calls.append((text, threading.get_ident()))
        if error is not None:
            raise error
        return result or positive_prediction()

    monkeypatch.setattr(bot, "predict", fake_predict)
    return calls


def run_voice(message, client, speech):
    async def run():
        await bot.voice_handler(message, client, speech)  # type: ignore[arg-type]
        return threading.get_ident()
    return asyncio.run(run())


@pytest.fixture(autouse=True)
def fresh_lock(monkeypatch):
    monkeypatch.setattr(bot, "_classification_lock", asyncio.Lock())


def test_voice_success_downloads_recognizes_and_classifies(monkeypatch, caplog):
    speech, calls = fake_speech(monkeypatch)
    predicted = recording_predict(monkeypatch)
    client, message = FakeBot(), FakeVoiceMessage()

    with caplog.at_level(logging.DEBUG):
        loop_thread = run_voice(message, client, speech)

    transcript = "Отличный фильм, понравился сюжет."
    assert client.downloads[0][0] == "voice-file-id"
    audio_path = client.downloads[0][1]
    assert calls.decoded == [(audio_path, True)]
    assert [text for text, _ in predicted] == [transcript]
    assert message.answers == [f"Распознанный текст:\n{transcript}\n\n"
                               + bot.format_prediction(positive_prediction())]
    assert "Тональность: положительная" in message.answers[0]
    # Whisper segments and predict run in worker threads, not in the event loop.
    assert calls.threads and all(t != loop_thread for t in calls.threads)
    assert all(t != loop_thread for _, t in predicted)
    assert not audio_path.exists() and not audio_path.parent.exists()
    assert transcript not in caplog.text and "Отличный" not in caplog.text


def test_voice_and_text_modes_give_same_label_for_same_text(tmp_path, monkeypatch):
    model = baseline(make_real_bundle(tmp_path))
    monkeypatch.setattr(classifier, "_default_classifier", model)
    text = "отличный фильм"
    speech, _ = fake_speech(monkeypatch, segments=(text,))
    voice_message, text_message = FakeVoiceMessage(), FakeMessage(text)

    run_voice(voice_message, FakeBot(), speech)
    asyncio.run(bot.text_handler(text_message))  # type: ignore[arg-type]

    expected = bot.format_prediction(classifier.predict(text))
    assert text_message.answers == [expected]
    assert voice_message.answers == [f"Распознанный текст:\n{text}\n\n{expected}"]


@pytest.mark.parametrize("label_id,label,russian", [(0, "negative", "отрицательная"),
                                                    (1, "neutral", "нейтральная"),
                                                    (2, "positive", "положительная")])
def test_russian_names_for_three_classes(label_id, label, russian):
    assert classifier.EXPECTED_LABEL_MAPPING[label] == label_id
    probabilities = {name: 0.1 for name in classifier.EXPECTED_CLASS_NAMES}
    probabilities[label] = 0.8
    answer = bot.format_prediction({"label": label, "label_id": label_id, "confidence": 0.8,
                                    "probabilities": probabilities})
    assert answer.startswith(f"Тональность: {russian} ({label})\n")


def test_russian_labels_cover_exactly_the_three_classes():
    assert list(bot.RUSSIAN_LABELS) == classifier.EXPECTED_CLASS_NAMES == ["negative", "neutral", "positive"]
    assert classifier.EXPECTED_CLASS_IDS == [0, 1, 2]


@pytest.mark.parametrize("segments", [(), (" ", "\n")])
def test_empty_transcript_does_not_reach_classifier(monkeypatch, segments):
    speech, _ = fake_speech(monkeypatch, segments=segments)
    predicted = recording_predict(monkeypatch)
    client, message = FakeBot(), FakeVoiceMessage()
    run_voice(message, client, speech)
    assert message.answers == [bot.EMPTY_SPEECH_MESSAGE]
    assert predicted == []
    assert not client.downloads[0][1].parent.exists()


def test_recognition_error_does_not_reach_classifier_or_send_partial_text(monkeypatch, caplog):
    speech, _ = fake_speech(monkeypatch, segments=("Частичный текст",), fail=True)
    predicted = recording_predict(monkeypatch)
    client, message = FakeBot(), FakeVoiceMessage()
    with caplog.at_level(logging.WARNING):
        run_voice(message, client, speech)
    assert message.answers == [bot.SPEECH_ERROR_MESSAGE]
    assert predicted == []
    assert "inference_failed" in caplog.text and "Частичный" not in caplog.text
    assert not client.downloads[0][1].parent.exists()


def test_corrupt_audio_uses_real_decoder_and_is_controlled(monkeypatch):
    speech = SpeechRecognizer()  # Real decoder; the model must not be needed.
    predicted = recording_predict(monkeypatch)
    client, message = FakeBot(payload=b"this is not an ogg stream"), FakeVoiceMessage()
    run_voice(message, client, speech)
    assert message.answers == [bot.SPEECH_ERROR_MESSAGE]
    assert predicted == [] and speech._model is None
    assert not client.downloads[0][1].parent.exists()


def test_download_error_is_controlled_and_cleaned(monkeypatch):
    speech, calls = fake_speech(monkeypatch)
    predicted = recording_predict(monkeypatch)
    client, message = FakeBot(error=ConnectionError("network")), FakeVoiceMessage()
    run_voice(message, client, speech)
    assert message.answers == [bot.VOICE_DOWNLOAD_ERROR_MESSAGE]
    assert calls.decoded == [] and predicted == []
    assert not client.downloads[0][1].parent.exists()


@pytest.mark.parametrize("error", [classifier.ClassifierError("prediction failed"),
                                   classifier.ModelUnavailableError("bundle missing")])
def test_classifier_error_on_voice_is_safe(monkeypatch, caplog, error):
    speech, _ = fake_speech(monkeypatch)
    recording_predict(monkeypatch, error=error)
    client, message = FakeBot(), FakeVoiceMessage()
    with caplog.at_level(logging.ERROR):
        run_voice(message, client, speech)
    assert message.answers == [bot.MODEL_ERROR_MESSAGE]
    assert "Voice transcript classification failed" in caplog.text
    assert "Отличный" not in caplog.text
    assert not client.downloads[0][1].parent.exists()


@pytest.mark.parametrize("duration,file_size,expected", [
    (bot.MAX_VOICE_SECONDS, bot.MAX_VOICE_BYTES, None),
    (bot.MAX_VOICE_SECONDS + 1, 1000, bot.VOICE_TOO_LONG_MESSAGE),
    (1, bot.MAX_VOICE_BYTES + 1, bot.VOICE_TOO_LARGE_MESSAGE),
    (1, None, None),
])
def test_voice_metadata_limit_boundaries(monkeypatch, duration, file_size, expected):
    assert bot.MAX_VOICE_SECONDS == 30 and bot.MAX_VOICE_BYTES == 5_242_880
    speech, _ = fake_speech(monkeypatch)
    recording_predict(monkeypatch)
    client, message = FakeBot(), FakeVoiceMessage(duration=duration, file_size=file_size)
    run_voice(message, client, speech)
    if expected is None:
        assert len(client.downloads) == 1 and "Распознанный текст" in message.answers[0]
    else:
        assert message.answers == [expected] and client.downloads == []


def test_transcript_length_boundary_uses_classifier_contract(tmp_path, monkeypatch):
    model = baseline(make_real_bundle(tmp_path))
    monkeypatch.setattr(classifier, "_default_classifier", model)
    assert "Тональность:" in bot.classify_transcript_for_bot("а" * bot.MAX_TEXT_LENGTH)

    def must_not_run(*args, **kwargs):
        pytest.fail("Over-long transcript must be rejected before the model")
    monkeypatch.setattr(model, "predict", must_not_run)
    long_text = "а" * (bot.MAX_TEXT_LENGTH + 1)
    answer = bot.classify_transcript_for_bot(long_text)
    assert answer == bot.TRANSCRIPT_TOO_LONG_MESSAGE
    assert long_text not in answer


@pytest.mark.parametrize("cancel", [False, True])
def test_voice_does_not_block_commands_and_keeps_audio_until_worker_ends(monkeypatch, cancel):
    started, release = threading.Event(), threading.Event()
    speech, calls = fake_speech(monkeypatch, started=started, release=release)
    recording_predict(monkeypatch)
    client = FakeBot()

    async def run():
        bot._classification_lock = asyncio.Lock()
        first = FakeVoiceMessage()
        task = asyncio.create_task(bot.voice_handler(first, client, speech))  # type: ignore[arg-type]
        try:
            for _ in range(500):
                if started.is_set():
                    break
                await asyncio.sleep(.001)
            assert started.is_set()
            audio_path = client.downloads[0][1]
            if cancel:
                task.cancel()
                await asyncio.sleep(0)
            for handler, text in ((bot.start_handler, bot.START_TEXT), (bot.help_handler, bot.HELP_TEXT)):
                command = FakeMessage()
                await asyncio.wait_for(handler(command), .5)  # type: ignore[arg-type]
                assert command.answers == [text]
            second_voice, second_text = FakeVoiceMessage(), FakeMessage("второй отзыв")
            await asyncio.wait_for(bot.voice_handler(second_voice, client, speech), .5)  # type: ignore[arg-type]
            await asyncio.wait_for(bot.text_handler(second_text), .5)  # type: ignore[arg-type]
            assert second_voice.answers == second_text.answers == [bot.BUSY_MESSAGE]
            assert len(client.downloads) == 1
            assert audio_path.exists(), "audio must outlive the running ASR worker"
        finally:
            release.set()
            if cancel:
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                await task
        assert not audio_path.parent.exists()
        assert not bot._classification_lock.locked()
        if cancel:
            assert first.answers == []
        else:
            assert len(first.answers) == 1 and first.answers[0].startswith("Распознанный текст:")

    asyncio.run(run())


def test_models_load_once_and_are_reused_by_all_messages(tmp_path, monkeypatch):
    model = baseline(make_real_bundle(tmp_path))
    monkeypatch.setattr(classifier, "_default_classifier", model)
    loads = []
    real_load = classifier.load_bundle
    monkeypatch.setattr(classifier, "load_bundle", lambda path: loads.append(path) or real_load(path))
    speech, calls = fake_speech(monkeypatch, segments=("отличный фильм",))

    assert bot.load_models(speech) is speech
    assert calls.loads == 1 and len(loads) == 1
    for _ in range(3):
        message = FakeVoiceMessage()
        run_voice(message, FakeBot(), speech)
        assert "Тональность:" in message.answers[0]
    text_message = FakeMessage("прекрасный фильм")
    asyncio.run(bot.text_handler(text_message))  # type: ignore[arg-type]
    assert "Тональность:" in text_message.answers[0]
    assert calls.loads == 1 and len(loads) == 1
    assert classifier.get_default_classifier() is model


def test_missing_classifier_bundle_fails_startup_with_reason(tmp_path, monkeypatch):
    monkeypatch.setattr(classifier, "_default_classifier", baseline(tmp_path / "missing.joblib"))
    speech, calls = fake_speech(monkeypatch)
    with pytest.raises(bot.StartupError, match="baseline bundle file not found") as error:
        bot.load_models(speech)
    assert "CLASSIFIER_BACKEND" in str(error.value)
    assert calls.loads == 0, "Whisper must not be loaded after a classifier failure"


def test_incompatible_classifier_bundle_fails_startup(tmp_path, monkeypatch):
    model_path = make_real_bundle(tmp_path)
    write_report(model_path, tb.load_bundle(model_path)["training"], bundle_sha256="0" * 64)
    monkeypatch.setattr(classifier, "_default_classifier", baseline(model_path))
    with pytest.raises(bot.StartupError, match="checksum differs from run report"):
        bot.load_models(fake_speech(monkeypatch)[0])


def test_invalid_backend_fails_startup_without_printing_variable_value(monkeypatch):
    monkeypatch.setattr(classifier, "_default_classifier", None)
    monkeypatch.setenv("CLASSIFIER_BACKEND", "private-value")
    with pytest.raises(bot.StartupError, match="must be baseline or rubert") as error:
        bot.load_models(fake_speech(monkeypatch)[0])
    assert "private-value" not in str(error.value)


def test_missing_whisper_model_fails_startup_without_private_details(tmp_path, monkeypatch):
    model = baseline(make_real_bundle(tmp_path))
    monkeypatch.setattr(classifier, "_default_classifier", model)
    speech = SpeechRecognizer()

    def missing():
        raise OSError(f"{Path.home()}/private/cache: snapshot not found")
    monkeypatch.setattr(speech, "_load_model", missing)
    with pytest.raises(bot.StartupError, match="Whisper model .* unavailable offline") as error:
        bot.load_models(speech)
    assert "private" not in str(error.value) and str(Path.home()) not in str(error.value)
    assert "--allow-download" in str(error.value)
    assert speech._model is None


def test_public_reason_hides_repository_and_home_paths():
    error = classifier.ModelUnavailableError("wrapper")
    error.__cause__ = FileNotFoundError(f"No such file: '{bot.ROOT}/reports/x.json' or '{Path.home()}/y'")
    reason = bot._public_reason(error)
    assert reason == "FileNotFoundError: No such file: './reports/x.json' or '<path>/y'"


def test_public_reason_hides_windows_paths():
    error = classifier.ModelUnavailableError("wrapper")
    error.__cause__ = FileNotFoundError(2, "No such file or directory",
                                        r"C:\Users\Private User\models\run_metadata.json")
    reason = bot._public_reason(error)
    assert reason == "FileNotFoundError: [Errno 2] No such file or directory: '<path>/run_metadata.json'"
    unquoted = bot._public_reason(ValueError(r"bad D:\data\model.joblib and \\server\share\x.bin"))
    assert unquoted == "ValueError: bad <path>/model.joblib and <path>/x.bin"


def test_public_reason_hides_absolute_paths_outside_project():
    reason = bot._public_reason(ValueError("checksum of /srv/private/models/x.joblib differs; see ./reports/a.json"))
    assert reason == "ValueError: checksum of <path>/x.joblib differs; see ./reports/a.json"
    assert bot._public_reason(ValueError("CLASSIFIER_BACKEND must be baseline or rubert")).endswith("baseline or rubert")


def test_missing_classifier_report_fails_startup_without_absolute_path(tmp_path, monkeypatch):
    model_path = tmp_path / "model.joblib"
    model_path.write_bytes(b"not used: the report is read first")
    report_path = tmp_path / "private dir" / "run_metadata.json"
    monkeypatch.setattr(classifier, "_default_classifier", None)
    monkeypatch.setenv("CLASSIFIER_BACKEND", "baseline")
    monkeypatch.setenv("CLASSIFIER_MODEL_PATH", str(model_path))
    monkeypatch.setenv("CLASSIFIER_REPORT_PATH", str(report_path))
    speech, calls = fake_speech(monkeypatch)
    with pytest.raises(bot.StartupError) as error:
        bot.load_models(speech)
    message = str(error.value)
    assert "FileNotFoundError" in message and "<path>/run_metadata.json" in message
    assert "CLASSIFIER_REPORT_PATH" in message
    assert str(tmp_path) not in message and "private dir" not in message
    assert calls.loads == 0


@pytest.mark.parametrize("token,reason", [(None, "is not set"), ("123:has space", "invalid format")])
def test_main_exits_with_reason_before_loading_models(monkeypatch, caplog, token, reason):
    if token is None:
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    else:
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", token)
    monkeypatch.setattr(bot, "load_models", lambda: pytest.fail("models must not load without a token"))
    monkeypatch.setattr(bot, "run_bot", lambda *a: pytest.fail("polling must not start"))
    with caplog.at_level(logging.ERROR), pytest.raises(SystemExit) as exit_info:
        bot.main()
    assert exit_info.value.code == 1
    assert "Startup failed" in caplog.text and reason in caplog.text
    if token:
        assert token not in caplog.text


def test_main_exits_when_model_is_unavailable(monkeypatch, caplog):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123456:LOCAL_TEST")

    def unavailable():
        raise bot.StartupError("Classifier bundle is unavailable (FileNotFoundError: x)")
    monkeypatch.setattr(bot, "load_models", unavailable)
    monkeypatch.setattr(bot, "run_bot", lambda *a: pytest.fail("polling must not start"))
    with caplog.at_level(logging.ERROR), pytest.raises(SystemExit) as exit_info:
        bot.main()
    assert exit_info.value.code == 1
    assert "Classifier bundle is unavailable" in caplog.text
    assert "123456:LOCAL_TEST" not in caplog.text


def test_run_bot_passes_single_speech_instance_to_dispatcher(monkeypatch):
    created = {}

    class FakeDispatcher:
        def __init__(self, **kwargs):
            created["kwargs"] = kwargs

        def include_router(self, router):
            created["router"] = router

        async def start_polling(self, client):
            created["bot"] = client

    speech = object()
    monkeypatch.setattr(bot, "Dispatcher", FakeDispatcher)
    monkeypatch.setattr(bot, "load_models", lambda: pytest.fail("given models must be reused"))

    async def run():
        await bot.run_bot("123456:LOCAL_TEST", speech)  # type: ignore[arg-type]
        await created["bot"].session.close()
    asyncio.run(run())
    assert created["kwargs"] == {"speech": speech}
    assert created["router"] is bot.router


def test_help_and_start_describe_purpose_and_implemented_limits():
    for text in (bot.START_TEXT, bot.HELP_TEXT):
        assert "киноотзыв" in text or "отзывов о фильмах" in text
        assert "30 секунд" in text and "5 МиБ" in text and "2000 символов" in text
        assert "голосов" in text
    for russian in bot.RUSSIAN_LABELS.values():
        assert russian in bot.HELP_TEXT
    assert "не является гарантией" in bot.HELP_TEXT


def _telegram_message(client, **fields):
    from datetime import datetime, timezone
    from aiogram.types import Chat, Message
    return Message(message_id=1, date=datetime.now(timezone.utc), chat=Chat(id=1, type="private"),
                   **fields).as_(client)


def _propagate(monkeypatch, build, speech):
    answers, downloads = [], []

    async def answer(self, method, **kwargs):
        answers.append(method.text)

    async def download(self, file_id, destination=None, **kwargs):
        downloads.append(file_id)
        write_destination(destination, b"synthetic voice")

    monkeypatch.setattr(bot.Bot, "__call__", answer)
    monkeypatch.setattr(bot.Bot, "download", download)

    async def run():
        client = bot.Bot("123456:LOCAL_TEST")
        try:
            await bot.router.propagate_event(update_type="message", event=build(client), bot=client, speech=speech)
        finally:
            await client.session.close()
    asyncio.run(run())
    return answers, downloads


def test_real_router_routes_voice_with_injected_speech(monkeypatch):
    from aiogram.types import Voice
    speech, calls = fake_speech(monkeypatch, segments=("Хороший фильм",))
    recording_predict(monkeypatch)
    voice = Voice(file_id="voice-id", file_unique_id="u", duration=3, mime_type="audio/ogg", file_size=1000)
    answers, downloads = _propagate(monkeypatch, lambda c: _telegram_message(c, voice=voice), speech)
    assert downloads == ["voice-id"] and calls.loads == 1
    assert answers and answers[0].startswith("Распознанный текст:\nХороший фильм\n\nТональность: положительная")


@pytest.mark.parametrize("text", ["/start", "/help", "/settings"])
def test_real_router_never_classifies_commands(monkeypatch, text):
    from aiogram.types import MessageEntity
    predicted = recording_predict(monkeypatch)
    entity = MessageEntity(type="bot_command", offset=0, length=len(text))
    answers, _ = _propagate(monkeypatch, lambda c: _telegram_message(c, text=text, entities=[entity]), None)
    expected = {"/start": bot.START_TEXT, "/help": bot.HELP_TEXT}.get(text, bot.UNKNOWN_COMMAND_MESSAGE)
    assert answers == [expected] and predicted == []


def _unsupported_fields():
    from aiogram.types import Audio, Document, Sticker, VideoNote
    return {
        "audio": {"audio": Audio(file_id="a", file_unique_id="a", duration=3)},
        "document": {"document": Document(file_id="d", file_unique_id="d")},
        "video_note": {"video_note": VideoNote(file_id="v", file_unique_id="v", length=1, duration=3)},
        "sticker": {"sticker": Sticker(file_id="s", file_unique_id="s", type="regular", width=1, height=1,
                                       is_animated=False, is_video=False)},
    }


@pytest.mark.parametrize("kind", ["audio", "document", "video_note", "sticker"])
def test_real_router_rejects_unsupported_formats(monkeypatch, kind):
    speech, calls = fake_speech(monkeypatch)
    predicted = recording_predict(monkeypatch)
    fields = _unsupported_fields()[kind]
    answers, downloads = _propagate(monkeypatch, lambda c: _telegram_message(c, **fields), speech)
    assert answers == [bot.UNSUPPORTED_MESSAGE]
    assert downloads == [] and predicted == [] and calls.loads == 0


# --- T09 review fixes: token in logs, cancellation -----------------------------------

import contextlib
import io

from aiogram import Bot
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.client.telegram import TelegramAPIServer

FAKE_TOKEN = "123456:FAKE-local-test-secret-not-a-real-token_xx"
FAKE_SECRET = FAKE_TOKEN.partition(":")[2]


@pytest.fixture(autouse=True)
def reset_secret_filter():
    factory = logging.getLogRecordFactory()
    yield
    logging.setLogRecordFactory(factory)
    bot._secret_filter.secrets.clear()


@pytest.fixture
def log_stream():
    """Actual formatted log output, as a console handler would write it."""
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    root = logging.getLogger()
    level = root.level
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    yield stream
    root.removeHandler(handler)
    root.setLevel(level)


@pytest.fixture
def telegram_stand(monkeypatch):
    """Local Telegram API stub; getUpdates (and optionally getMe) loops redirects."""
    from aiohttp import web

    state = {"redirect_get_me": False}

    async def handle(request):
        if request.path.endswith("/getMe") and not state["redirect_get_me"]:
            return web.json_response({"ok": True, "result": {
                "id": 123456, "is_bot": True, "first_name": "Stand", "username": "stand_bot"}})
        raise web.HTTPFound(request.path)

    async def start():
        app = web.Application()
        app.router.add_route("*", "/{tail:.*}", handle)
        runner = web.AppRunner(app, access_log=None)
        await runner.setup()
        await web.TCPSite(runner, "127.0.0.1", 0).start()
        return runner

    loop = asyncio.new_event_loop()
    runner = loop.run_until_complete(start())
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    api = TelegramAPIServer.from_base(f"http://127.0.0.1:{runner.addresses[0][1]}")
    monkeypatch.setattr(bot, "Bot", lambda token: Bot(token, session=AiohttpSession(api=api)))
    # The module router attaches to one Dispatcher per process; detach it after the test.
    monkeypatch.setattr(bot.router, "_parent_router", None)
    try:
        yield state
    finally:
        asyncio.run_coroutine_threadsafe(runner.cleanup(), loop).result(5)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(5)
        loop.close()


def test_token_is_redacted_from_polling_network_error_logs(telegram_stand, log_stream, caplog):
    async def run():
        task = asyncio.create_task(bot.run_bot(FAKE_TOKEN, object()))  # type: ignore[arg-type]
        for _ in range(500):
            if "Failed to fetch updates" in log_stream.getvalue():
                break
            await asyncio.sleep(.01)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    with caplog.at_level(logging.INFO):
        asyncio.run(run())
    text = log_stream.getvalue()
    assert "Run polling for bot @stand_bot" in text, "polling must still start"
    assert "TooManyRedirects" in text and "/bot<redacted>/getUpdates" in text
    assert FAKE_SECRET not in text and FAKE_SECRET not in caplog.text


def test_token_is_redacted_from_startup_traceback(telegram_stand, log_stream, monkeypatch, capsys):
    telegram_stand["redirect_get_me"] = True
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", FAKE_TOKEN)
    monkeypatch.setattr(bot, "load_models", lambda: object())
    with pytest.raises(SystemExit) as exit_info:
        bot.main()
    assert exit_info.value.code == 1
    text = log_stream.getvalue()
    assert "Bot stopped because of an error" in text and "Traceback" in text
    # The chained aiohttp error keeps its diagnostics, only the token is hidden.
    assert "TooManyRedirects" in text and "/bot<redacted>/getMe" in text
    captured = capsys.readouterr()
    assert FAKE_SECRET not in text + captured.out + captured.err


@contextlib.contextmanager
def own_logger_stream(name):
    """Handler set directly on a logger that does not propagate to root."""
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    target = logging.getLogger(name)
    level, propagate = target.level, target.propagate
    target.addHandler(handler)
    target.setLevel(logging.INFO)
    target.propagate = False
    try:
        yield stream
    finally:
        target.removeHandler(handler)
        target.setLevel(level)
        target.propagate = propagate


def test_token_is_redacted_for_handler_on_aiogram_dispatcher_logger(telegram_stand):
    async def run(stream):
        task = asyncio.create_task(bot.run_bot(FAKE_TOKEN, object()))  # type: ignore[arg-type]
        for _ in range(500):
            if "Failed to fetch updates" in stream.getvalue():
                break
            await asyncio.sleep(.01)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    with own_logger_stream("aiogram.dispatcher") as stream:
        asyncio.run(run(stream))
    text = stream.getvalue()
    assert "TooManyRedirects" in text and "/bot<redacted>/getUpdates" in text
    assert FAKE_SECRET not in text


def test_token_is_redacted_from_traceback_for_handler_on_bot_logger(telegram_stand, monkeypatch):
    telegram_stand["redirect_get_me"] = True
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", FAKE_TOKEN)
    monkeypatch.setattr(bot, "load_models", lambda: object())
    with own_logger_stream("src.bot") as stream, pytest.raises(SystemExit):
        bot.main()
    text = stream.getvalue()
    assert "Traceback" in text and "/bot<redacted>/getMe" in text
    assert FAKE_SECRET not in text


def test_redaction_covers_arguments_tracebacks_and_bad_records(log_stream):
    bot.protect_token_in_logs(FAKE_TOKEN)
    log = logging.getLogger("tests.redaction")
    try:
        raise RuntimeError(f"url=https://api.telegram.org/bot{FAKE_TOKEN}/getMe")
    except RuntimeError:
        log.exception("request %s failed", f"/bot{FAKE_TOKEN}/x")
    log.error("secret part alone: %s", FAKE_SECRET)
    log.error("unformattable %s %s " + FAKE_SECRET, "one")
    text = log_stream.getvalue()
    assert FAKE_SECRET not in text
    assert text.count("<redacted>") >= 4 and "RuntimeError: url=" in text


def test_repeated_cancellation_keeps_slot_until_classification_finishes(monkeypatch):
    started, release = threading.Event(), threading.Event()

    def slow_classification(text):
        started.set()
        assert release.wait(5)
        return "done"
    monkeypatch.setattr(bot, "classify_text_for_bot", slow_classification)

    async def run():
        message = FakeMessage("отзыв")
        task = asyncio.create_task(bot.text_handler(message))  # type: ignore[arg-type]
        try:
            assert await asyncio.to_thread(started.wait, 5)
            for _ in range(3):
                task.cancel()
                await asyncio.sleep(.01)
                assert not task.done() and bot._classification_lock.locked()
            busy = FakeMessage("второй отзыв")
            await bot.text_handler(busy)  # type: ignore[arg-type]
            assert busy.answers == [bot.BUSY_MESSAGE]
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not bot._classification_lock.locked() and message.answers == []

    asyncio.run(run())


def test_event_loop_shutdown_keeps_voice_audio_until_asr_finishes(monkeypatch):
    started, release = threading.Event(), threading.Event()
    speech, calls = fake_speech(monkeypatch, started=started, release=release)
    predicted = recording_predict(monkeypatch)
    client, message = FakeBot(), FakeVoiceMessage()
    audio_present_when_worker_ended = []
    transcribe = speech.transcribe

    def observed_transcribe(path):
        result = transcribe(path)
        audio_present_when_worker_ended.append(Path(path).exists())
        return result
    monkeypatch.setattr(speech, "transcribe", observed_transcribe)

    async def main():
        asyncio.create_task(bot.voice_handler(message, client, speech))  # type: ignore[arg-type]
        assert await asyncio.to_thread(started.wait, 5)
        # Return with the handler pending: asyncio.run cancels it while ASR still runs.
        threading.Timer(.3, release.set).start()

    asyncio.run(main())
    assert audio_present_when_worker_ended == [True]
    assert not client.downloads[0][1].parent.exists()
    assert message.answers == [] and predicted == [] and calls.loads == 1


# --- T10: actual size, busy slot, response timeout, cleanup ---------------------------

import gc

MAX = 5_242_880  # 5 MiB, not 5_000_000


def notifying(message, fail_on=None, on_answer=None):
    """Lets a test await the first answer instead of sleeping; optionally fails a send."""
    message.answered = asyncio.Event()

    async def answer(text):
        message.answers.append(text)
        message.answered.set()
        if on_answer is not None:
            on_answer(text)
        if text == fail_on:
            raise ConnectionError("telegram unavailable")
    message.answer = answer
    return message


def run_checked(coro):
    """asyncio.run that fails on unretrieved task exceptions and other loop errors."""
    errors = []

    async def main():
        asyncio.get_running_loop().set_exception_handler(lambda loop, context: errors.append(context))
        try:
            return await coro
        finally:
            gc.collect()
    result = asyncio.run(main())
    gc.collect()  # Task.__del__ of finished tasks reports to the same handler
    assert errors == [], errors
    return result


def observe_audio_at_worker_end(monkeypatch, speech):
    seen = []
    transcribe = speech.transcribe

    def observed(path):
        result = transcribe(path)
        seen.append(Path(path).exists())
        return result
    monkeypatch.setattr(speech, "transcribe", observed)
    return seen


@pytest.mark.parametrize("file_size,payload_size,rejected", [
    (None, MAX, False),
    (None, MAX + 1, True),
    (1000, MAX + 1, True),   # metadata smaller than the real file
    (MAX, MAX, False),
])
def test_downloaded_size_is_checked_before_whisper(monkeypatch, caplog, file_size, payload_size, rejected):
    assert bot.MAX_VOICE_BYTES == MAX
    speech, calls = fake_speech(monkeypatch)
    predicted = recording_predict(monkeypatch)
    client = FakeBot(payload=b"\0" * payload_size)
    message = FakeVoiceMessage(file_size=file_size)
    with caplog.at_level(logging.WARNING):
        run_voice(message, client, speech)
    assert len(client.downloads) == 1
    if rejected:
        assert message.answers == [bot.VOICE_TOO_LARGE_MESSAGE]
        assert calls.decoded == [] and calls.loads == 0 and predicted == []
        assert "downloaded file exceeds the size limit" in caplog.text
    else:
        assert calls.decoded and message.answers[0].startswith("Распознанный текст:")
    assert not client.downloads[0][1].parent.exists()


@pytest.mark.parametrize("duration,accepted", [(30, True), (31, False)])
def test_duration_limit_without_file_size(monkeypatch, duration, accepted):
    speech, _ = fake_speech(monkeypatch)
    recording_predict(monkeypatch)
    client, message = FakeBot(), FakeVoiceMessage(duration=duration, file_size=None)
    run_voice(message, client, speech)
    if accepted:
        assert len(client.downloads) == 1 and "Тональность:" in message.answers[0]
    else:
        assert message.answers == [bot.VOICE_TOO_LONG_MESSAGE] and client.downloads == []


class PartialDownloadBot(FakeBot):
    async def download(self, file_id, destination):
        self.downloads.append((file_id, destination_path(destination)))
        write_destination(destination, b"partial")
        raise ConnectionError("connection reset")


def test_partially_downloaded_audio_is_removed(monkeypatch):
    speech, calls = fake_speech(monkeypatch)
    predicted = recording_predict(monkeypatch)
    client, message = PartialDownloadBot(), FakeVoiceMessage()
    run_voice(message, client, speech)
    assert message.answers == [bot.VOICE_DOWNLOAD_ERROR_MESSAGE]
    assert calls.decoded == [] and predicted == []
    assert not client.downloads[0][1].parent.exists()


def test_text_handler_length_boundary(tmp_path, monkeypatch):
    monkeypatch.setattr(classifier, "_default_classifier", baseline(make_real_bundle(tmp_path)))
    accepted, rejected = FakeMessage("а" * 2000), FakeMessage("а" * 2001)
    asyncio.run(bot.text_handler(accepted))  # type: ignore[arg-type]
    asyncio.run(bot.text_handler(rejected))  # type: ignore[arg-type]
    assert "Тональность:" in accepted.answers[0]
    assert rejected.answers == [bot.TOO_LONG_MESSAGE]


@pytest.mark.parametrize("first,second", [("voice", "voice"), ("voice", "text"),
                                          ("text", "voice"), ("text", "text")])
def test_near_simultaneous_requests_start_one_heavy_operation(monkeypatch, first, second):
    started, release = threading.Event(), threading.Event()
    speech, calls = fake_speech(monkeypatch, started=started, release=release)
    recording_predict(monkeypatch)
    texts = []
    real_classify = bot.classify_text_for_bot

    def slow_classify(text):
        texts.append(text)
        started.set()
        assert release.wait(5)
        return real_classify(text)
    monkeypatch.setattr(bot, "classify_text_for_bot", slow_classify)
    client = FakeBot()

    def request(kind):
        message = FakeVoiceMessage() if kind == "voice" else FakeMessage("отзыв")
        handler = (bot.voice_handler(message, client, speech) if kind == "voice"
                   else bot.text_handler(message))  # type: ignore[arg-type]
        return message, handler

    async def run():
        (m1, h1), (m2, h2) = request(first), request(second)
        # Both scheduled before either runs: admission must not let both in.
        t1, t2 = asyncio.create_task(h1), asyncio.create_task(h2)
        try:
            assert await asyncio.to_thread(started.wait, 5)
            await asyncio.wait_for(t2, .5)
            assert m2.answers == [bot.BUSY_MESSAGE]
            help_message = FakeMessage()
            await asyncio.wait_for(bot.help_handler(help_message), .5)  # type: ignore[arg-type]
            assert help_message.answers == [bot.HELP_TEXT]
            assert len(calls.threads) + len(texts) == 1 and bot._classification_lock.locked()
        finally:
            release.set()
        await t1
        assert len(m1.answers) == 1 and "Тональность:" in m1.answers[0]
        assert not bot._classification_lock.locked()
        # The slot is free again: the next request of the rejected kind is accepted.
        m3, h3 = request(second)
        await h3
        assert "Тональность:" in m3.answers[0]
        assert len(calls.threads) + len(texts) == 2

    asyncio.run(run())


@pytest.mark.parametrize("kind", ["voice", "text"])
def test_result_before_threshold_sends_no_delay_notice(monkeypatch, kind):
    monkeypatch.setattr(bot, "RESPONSE_TIMEOUT_SECONDS", 5)
    speech, _ = fake_speech(monkeypatch)
    recording_predict(monkeypatch)
    if kind == "voice":
        message = FakeVoiceMessage()
        run_voice(message, FakeBot(), speech)
    else:
        message = FakeMessage("отзыв")
        asyncio.run(bot.text_handler(message))  # type: ignore[arg-type]
    assert len(message.answers) == 1 and "Тональность:" in message.answers[0]
    assert bot.RESPONSE_TIMEOUT_MESSAGE not in message.answers


def test_production_threshold_is_60_seconds():
    assert bot.RESPONSE_TIMEOUT_SECONDS == 60
    assert "60 секунд" in bot.RESPONSE_TIMEOUT_MESSAGE and "60 секунд" in bot.HELP_TEXT


@pytest.mark.parametrize("cancellations", [0, 1, 3])
def test_slow_voice_gets_one_notice_and_holds_slot_and_audio_until_asr_ends(monkeypatch, caplog, cancellations):
    monkeypatch.setattr(bot, "RESPONSE_TIMEOUT_SECONDS", .01)
    started, release = threading.Event(), threading.Event()
    speech, calls = fake_speech(monkeypatch, started=started, release=release)
    audio_at_worker_end = observe_audio_at_worker_end(monkeypatch, speech)
    predicted = recording_predict(monkeypatch)
    client, message = FakeBot(), notifying(FakeVoiceMessage())

    async def run():
        task = asyncio.create_task(bot.voice_handler(message, client, speech))  # type: ignore[arg-type]
        try:
            assert await asyncio.to_thread(started.wait, 5)
            await asyncio.wait_for(message.answered.wait(), 5)
            audio_path = client.downloads[0][1]
            for _ in range(cancellations):
                task.cancel()
                await asyncio.sleep(.01)
            assert not task.done() and bot._classification_lock.locked()
            assert audio_path.exists(), "audio must outlive the running ASR worker"
            busy_voice, busy_text = FakeVoiceMessage(), FakeMessage("второй")
            await asyncio.wait_for(bot.voice_handler(busy_voice, client, speech), .5)  # type: ignore[arg-type]
            await asyncio.wait_for(bot.text_handler(busy_text), .5)  # type: ignore[arg-type]
            assert busy_voice.answers == busy_text.answers == [bot.BUSY_MESSAGE]
            assert len(client.downloads) == 1 and len(calls.threads) == 1
        finally:
            release.set()
        if cancellations:
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            await task
        assert audio_at_worker_end == [True]
        assert not audio_path.parent.exists()
        assert not bot._classification_lock.locked()
        # One notice, no late transcript or label; classification after ASR is skipped.
        assert message.answers == [bot.RESPONSE_TIMEOUT_MESSAGE] and predicted == []
        monkeypatch.setattr(bot, "RESPONSE_TIMEOUT_SECONDS", 60)
        next_message = FakeVoiceMessage()
        await bot.voice_handler(next_message, client, speech)  # type: ignore[arg-type]
        assert next_message.answers[0].startswith("Распознанный текст:")

    with caplog.at_level(logging.INFO):
        run_checked(run())
    assert "Response timeout reached" in caplog.text
    assert "Отличный" not in caplog.text


@pytest.mark.parametrize("outcome", ["result", "error"])
def test_slow_text_after_threshold_sends_nothing_late(monkeypatch, caplog, outcome):
    monkeypatch.setattr(bot, "RESPONSE_TIMEOUT_SECONDS", .01)
    started, release = threading.Event(), threading.Event()
    finished = []

    def slow_classify(text):
        started.set()
        assert release.wait(5)
        finished.append(text)
        if outcome == "error":
            raise RuntimeError("worker failed")
        return "late classification result"
    monkeypatch.setattr(bot, "classify_text_for_bot", slow_classify)
    message = notifying(FakeMessage("отзыв"))

    async def run():
        task = asyncio.create_task(bot.text_handler(message))  # type: ignore[arg-type]
        try:
            await asyncio.wait_for(message.answered.wait(), 5)
            assert bot._classification_lock.locked() and not task.done()
        finally:
            release.set()
        await task
        assert finished == ["отзыв"] and not bot._classification_lock.locked()
        monkeypatch.setattr(bot, "RESPONSE_TIMEOUT_SECONDS", 60)
        monkeypatch.setattr(bot, "classify_text_for_bot", lambda text: "next result")
        next_message = FakeMessage("следующий")
        await bot.text_handler(next_message)  # type: ignore[arg-type]
        assert next_message.answers == ["next result"]

    with caplog.at_level(logging.WARNING):
        run_checked(run())
    assert message.answers == [bot.RESPONSE_TIMEOUT_MESSAGE]
    assert "never retrieved" not in caplog.text


def test_cancel_before_threshold_never_sends_notice(monkeypatch):
    monkeypatch.setattr(bot, "RESPONSE_TIMEOUT_SECONDS", .01)
    started, release = threading.Event(), threading.Event()
    speech, _ = fake_speech(monkeypatch, started=started, release=release)
    predicted = recording_predict(monkeypatch)
    client, message = FakeBot(), FakeVoiceMessage()

    async def run():
        task = asyncio.create_task(bot.voice_handler(message, client, speech))  # type: ignore[arg-type]
        try:
            assert await asyncio.to_thread(started.wait, 5)
            task.cancel()
            await asyncio.sleep(.1)  # well past the shortened threshold
            assert not task.done() and bot._classification_lock.locked()
            assert client.downloads[0][1].exists()
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not client.downloads[0][1].parent.exists()
        assert not bot._classification_lock.locked()

    run_checked(run())
    assert message.answers == [] and predicted == []


def test_failed_delay_notice_keeps_lifecycle(monkeypatch, caplog):
    monkeypatch.setattr(bot, "RESPONSE_TIMEOUT_SECONDS", .01)
    started, release = threading.Event(), threading.Event()
    calls = []

    def slow_classify(text):
        calls.append(text)
        started.set()
        assert release.wait(5)
        return "late classification result"
    monkeypatch.setattr(bot, "classify_text_for_bot", slow_classify)
    message = notifying(FakeMessage("отзыв"), fail_on=bot.RESPONSE_TIMEOUT_MESSAGE)

    async def run():
        task = asyncio.create_task(bot.text_handler(message))  # type: ignore[arg-type]
        try:
            await asyncio.wait_for(message.answered.wait(), 5)
            await asyncio.sleep(0)
            assert bot._classification_lock.locked() and not task.done()
            busy = FakeMessage("второй")
            await bot.text_handler(busy)  # type: ignore[arg-type]
            assert busy.answers == [bot.BUSY_MESSAGE]
        finally:
            release.set()
        await task  # the send error does not escape or end the operation early
        assert not bot._classification_lock.locked()

    with caplog.at_level(logging.WARNING):
        run_checked(run())
    assert calls == ["отзыв"]
    assert message.answers == [bot.RESPONSE_TIMEOUT_MESSAGE]
    assert "Could not send the delay notice: ConnectionError" in caplog.text


class SlowDownloadBot(FakeBot):
    """Writes part of the file and hangs; once cancelled, aborts only when allowed."""

    def __init__(self) -> None:
        super().__init__()
        self.in_progress, self.abort = asyncio.Event(), asyncio.Event()
        self.cancellations = 0

    async def download(self, file_id, destination):
        self.downloads.append((file_id, destination_path(destination)))
        write_destination(destination, b"partial")
        self.in_progress.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancellations += 1
            while not self.abort.is_set():  # a slow abort, still owning the file
                try:
                    await self.abort.wait()
                except asyncio.CancelledError:
                    self.cancellations += 1
            raise


@pytest.mark.parametrize("timeout,handler_cancellations", [(.01, 0), (.01, 1), (.01, 3), (60, 1)])
def test_delayed_download_counts_towards_threshold_and_is_cleaned(monkeypatch, timeout, handler_cancellations):
    monkeypatch.setattr(bot, "RESPONSE_TIMEOUT_SECONDS", timeout)
    speech, calls = fake_speech(monkeypatch)
    predicted = recording_predict(monkeypatch)
    client, message = SlowDownloadBot(), notifying(FakeVoiceMessage())
    timed_out = timeout < 1

    async def run():
        task = asyncio.create_task(bot.voice_handler(message, client, speech))  # type: ignore[arg-type]
        try:
            await asyncio.wait_for(client.in_progress.wait(), 5)
            audio_path = client.downloads[0][1]
            if timed_out:
                # The notice comes while bot.download is still pending: the threshold covers it.
                await asyncio.wait_for(message.answered.wait(), 5)
                assert message.answers == [bot.RESPONSE_TIMEOUT_MESSAGE]
            for _ in range(handler_cancellations):
                task.cancel()
                await asyncio.sleep(0)
            await asyncio.sleep(0)
            # The download stage is being stopped, no new stage starts; slot and directory
            # stay (downloaded bytes are kept in memory until the download ends).
            assert client.cancellations >= 1 and not task.done()
            assert bot._classification_lock.locked() and audio_path.parent.exists()
            busy = FakeMessage("второй")
            await bot.text_handler(busy)  # type: ignore[arg-type]
            assert busy.answers == [bot.BUSY_MESSAGE]
        finally:
            client.abort.set()
        if handler_cancellations:
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            await task
        assert not audio_path.parent.exists() and not bot._classification_lock.locked()
        assert calls.decoded == [] and predicted == []
        assert message.answers == ([bot.RESPONSE_TIMEOUT_MESSAGE] if timed_out else [])
        monkeypatch.setattr(bot, "RESPONSE_TIMEOUT_SECONDS", 60)
        next_client, next_message = FakeBot(), FakeVoiceMessage()
        await bot.voice_handler(next_message, next_client, speech)  # type: ignore[arg-type]
        assert next_message.answers[0].startswith("Распознанный текст:")
        assert not next_client.downloads[0][1].parent.exists()

    run_checked(run())


def test_completion_in_the_threshold_iteration_sends_result_only(monkeypatch):
    monkeypatch.setattr(bot, "RESPONSE_TIMEOUT_SECONDS", 0)

    async def finished_work():
        return "result"

    async def run():
        message = notifying(FakeMessage())
        assert await bot.answer_within_timeout(message, finished_work()) == "result"  # type: ignore[arg-type]
        assert message.answers == []
    asyncio.run(run())


def test_completion_right_after_the_threshold_decision_sends_notice_only(monkeypatch):
    monkeypatch.setattr(bot, "RESPONSE_TIMEOUT_SECONDS", .01)
    release = threading.Event()

    def worker():
        assert release.wait(5)
        return "late result"
    # The worker ends as soon as the notice is chosen, before the handler resumes.
    message = notifying(FakeMessage(), on_answer=lambda text: release.set())

    async def run():
        answer = await bot.answer_within_timeout(message, bot.run_in_thread(worker))  # type: ignore[arg-type]
        assert answer is None
    asyncio.run(run())
    assert message.answers == [bot.RESPONSE_TIMEOUT_MESSAGE]


def test_every_interleaving_at_the_threshold_has_exactly_one_outcome(monkeypatch):
    monkeypatch.setattr(bot, "RESPONSE_TIMEOUT_SECONDS", 0)

    async def work(steps):
        for _ in range(steps):
            await asyncio.sleep(0)
        return "result"

    async def run():
        outcomes = set()
        for steps in range(8):
            message = notifying(FakeMessage())
            answer = await bot.answer_within_timeout(message, work(steps))  # type: ignore[arg-type]
            sent = message.answers + ([answer] if answer is not None else [])
            assert sent in (["result"], [bot.RESPONSE_TIMEOUT_MESSAGE]), sent
            outcomes.add(sent[0])
        assert outcomes == {"result", bot.RESPONSE_TIMEOUT_MESSAGE}, "both sides exercised"
    asyncio.run(run())


@pytest.mark.parametrize("kind", ["voice", "text"])
def test_truncation_warning_is_shown_in_both_modes(monkeypatch, kind):
    recording_predict(monkeypatch, result={**positive_prediction(), "truncated": True})
    speech, _ = fake_speech(monkeypatch)
    if kind == "voice":
        message = FakeVoiceMessage()
        run_voice(message, FakeBot(), speech)
    else:
        message = FakeMessage("длинный отзыв")
        asyncio.run(bot.text_handler(message))  # type: ignore[arg-type]
    assert "Тональность: положительная" in message.answers[0]
    assert "текст усечён моделью" in message.answers[0]


def test_unavailable_model_on_text_is_not_a_label(monkeypatch):
    recording_predict(monkeypatch, error=classifier.ModelUnavailableError("bundle missing"))
    message = FakeMessage("отзыв")
    asyncio.run(bot.text_handler(message))  # type: ignore[arg-type]
    assert message.answers == [bot.MODEL_ERROR_MESSAGE]
    assert "Тональность" not in message.answers[0]


# --- T10 review fixes: real aiogram download, file I/O off the event loop, safe logs --

import os
import time

from aiogram.types import File

GATE_TIMEOUT = 5  # seconds; bounds every wait so a broken run fails instead of hanging


class GatedFile(io.BufferedWriter):
    """The real voice file; write() or flush() (close() runs flush) waits for a gate."""

    def _gated(self, operation):
        probe = self.probe
        if operation != probe.stage:
            return
        probe.calls.append(threading.get_ident())
        probe.in_progress += 1
        probe.started.set()
        try:
            if not probe.gate.wait(GATE_TIMEOUT):
                probe.timed_out = True
                probe.gate.set()  # never hang the run, even when blocked inside the event loop
        finally:
            probe.ends.append((Path(self.name).exists(), bot._classification_lock.locked()))
            probe.in_progress -= 1

    def write(self, data):
        self._gated("write")
        probe = self.probe
        probe.bytes_written += memoryview(data).nbytes
        return super().write(data)

    def flush(self):
        self._gated("flush")
        return super().flush()


def _open_fds_under(directory):
    """Open descriptors of this process that point into directory (Linux /proc)."""
    fd_dir = Path("/proc/self/fd")
    if not fd_dir.is_dir():
        return None
    found = 0
    for fd in fd_dir.iterdir():
        try:
            found += os.readlink(fd).startswith(str(directory))
        except OSError:
            pass
    return found


NO_OPEN_FDS = 0 if Path("/proc/self/fd").is_dir() else None


def instrument_voice_files(monkeypatch, stage):
    """Every real open of voice.ogg for writing returns a GatedFile; cleanups are recorded."""
    probe = SimpleNamespace(stage=stage, gate=threading.Event(), started=threading.Event(), calls=[],
                            ends=[], in_progress=0, timed_out=False, bytes_written=0, files=[], cleanups=[])
    real_open = Path.open

    def gated_open(self, mode="r", *args, **kwargs):
        if self.name != "voice.ogg" or "w" not in mode:
            return real_open(self, mode, *args, **kwargs)
        file = GatedFile(io.FileIO(self, mode))
        file.probe = probe
        probe.files.append(file)
        return file
    monkeypatch.setattr(Path, "open", gated_open)

    real_rmtree = bot.shutil.rmtree

    def recording_rmtree(directory, *args, **kwargs):
        if Path(directory).name.startswith("ru-review-voice-"):
            probe.cleanups.append({"directory": Path(directory), "in_progress": probe.in_progress,
                                   "files_closed": all(file.closed for file in probe.files),
                                   "open_fds": _open_fds_under(directory)})
        return real_rmtree(directory, *args, **kwargs)
    monkeypatch.setattr(bot.shutil, "rmtree", recording_rmtree)
    return probe


def local_aiogram_bot(chunks, yielded, hang=False):
    """Real aiogram Bot: download, download_file and its file adapter run unchanged.

    Only get_file and the HTTP stream are local; no Telegram network request is made.
    """
    client = bot.Bot("123456:LOCAL_TEST")

    async def get_file(file_id):
        return File(file_id=file_id, file_unique_id="unique", file_path="voice/file_0.oga")

    async def stream_content(url, timeout, chunk_size, raise_for_status):
        for chunk in chunks:
            yielded.append(len(chunk))
            yield chunk
        if hang:
            await asyncio.Event().wait()  # no more data arrives: the download hangs

    client.get_file = get_file
    client.session.stream_content = stream_content
    return client


async def wait_until(condition, timeout=GATE_TIMEOUT):
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            return False
        await asyncio.sleep(.005)
    return True


@pytest.mark.parametrize("extra_cancellations", [1, 3])
@pytest.mark.parametrize("stage", ["write", "flush"])
def test_delayed_audio_file_operation_keeps_loop_responsive(monkeypatch, stage, extra_cancellations):
    """S1: slow write(); S2: slow flush() on close of the real voice file, after a real download."""
    monkeypatch.setattr(bot, "RESPONSE_TIMEOUT_SECONDS", .5)
    speech, calls = fake_speech(monkeypatch)
    predicted = recording_predict(monkeypatch)
    probe = instrument_voice_files(monkeypatch, stage)
    chunks, yielded = [b"\x4f\x67\x67\x53" * 1024] * 3, []
    message = notifying(FakeVoiceMessage(file_size=None))
    seen = {}

    async def run():
        seen["loop_thread"] = threading.get_ident()
        client = local_aiogram_bot(chunks, yielded)
        beats = [0]

        async def heartbeat():
            while True:
                beats[0] += 1
                await asyncio.sleep(.005)
        pulse = asyncio.create_task(heartbeat())
        task = asyncio.create_task(bot.voice_handler(message, client, speech))  # type: ignore[arg-type]
        try:
            assert await wait_until(probe.started.is_set), "the delayed file operation never started"
            beats_at_start = beats[0]
            help_message, start_message = FakeMessage("/help"), FakeMessage("/start")
            await bot.help_handler(help_message)  # type: ignore[arg-type]
            await bot.start_handler(start_message)  # type: ignore[arg-type]
            seen["commands"] = (help_message.answers, start_message.answers, probe.in_progress)
            await asyncio.wait_for(message.answered.wait(), GATE_TIMEOUT)  # the threshold passes
            for _ in range(extra_cancellations):
                task.cancel()
                await asyncio.sleep(0)
            busy = FakeVoiceMessage()
            await bot.voice_handler(busy, FakeBot(), speech)  # type: ignore[arg-type]
            await asyncio.sleep(.05)
            seen["blocked"] = {
                "in_progress": probe.in_progress, "released": probe.gate.is_set(),
                "answers": list(message.answers), "busy": busy.answers,
                "locked": bot._classification_lock.locked(), "task_done": task.done(),
                "file_exists": Path(probe.files[0].name).exists(),
            }
            seen["beats"] = beats[0] - beats_at_start
        finally:
            probe.gate.set()
        try:
            done, _ = await asyncio.wait((task,), timeout=GATE_TIMEOUT)
            assert done and task.cancelled()
            seen["unlocked"] = not bot._classification_lock.locked()
            seen["stages"] = (len(calls.decoded), len(predicted))
            probe.stage = None
            next_message = FakeVoiceMessage()
            await bot.voice_handler(next_message, FakeBot(), speech)  # type: ignore[arg-type]
            seen["next"] = next_message.answers
        finally:
            pulse.cancel()
            await client.session.close()

    run_checked(run())
    assert not probe.timed_out
    # The gated operation really ran, in a worker thread, never in the event loop thread.
    assert len(probe.calls) == 1 and seen["loop_thread"] not in probe.calls
    assert yielded == [4096] * 3
    # While it was blocked the loop served commands, the threshold and a busy reply.
    assert seen["commands"] == ([bot.HELP_TEXT], [bot.START_TEXT], 1)
    assert seen["beats"] >= 5
    assert seen["blocked"] == {
        "in_progress": 1, "released": False, "answers": [bot.RESPONSE_TIMEOUT_MESSAGE],
        "busy": [bot.BUSY_MESSAGE], "locked": True, "task_done": False, "file_exists": True,
    }
    # It finished with the file and the slot still held; cleanup came after, once.
    assert probe.ends == [(True, True)]
    first_cleanup = probe.cleanups[0]
    assert first_cleanup["directory"] == Path(probe.files[0].name).parent
    assert {key: first_cleanup[key] for key in ("in_progress", "files_closed", "open_fds")} == {
        "in_progress": 0, "files_closed": True, "open_fds": NO_OPEN_FDS}
    assert not first_cleanup["directory"].exists() and seen["unlocked"]
    # Nothing late and no later stage after the timeout; the next review is accepted.
    assert message.answers == [bot.RESPONSE_TIMEOUT_MESSAGE] and seen["stages"] == (0, 0)
    assert seen["next"][0].startswith("Распознанный текст:") and len(probe.cleanups) == 2


@pytest.mark.parametrize("handler_cancellations", [0, 1, 3])
def test_timeout_during_real_aiogram_download_writes_no_file(monkeypatch, handler_cancellations):
    """The threshold passes while the network stream hangs, before any file operation."""
    monkeypatch.setattr(bot, "RESPONSE_TIMEOUT_SECONDS", .01)
    speech, calls = fake_speech(monkeypatch)
    predicted = recording_predict(monkeypatch)
    probe = instrument_voice_files(monkeypatch, stage="write")
    yielded = []
    message = notifying(FakeVoiceMessage(file_size=None))

    async def run():
        client = local_aiogram_bot([b"\x4f\x67\x67\x53" * 1024], yielded, hang=True)
        try:
            task = asyncio.create_task(bot.voice_handler(message, client, speech))  # type: ignore[arg-type]
            try:
                await asyncio.wait_for(message.answered.wait(), GATE_TIMEOUT)
                for _ in range(handler_cancellations):
                    task.cancel()
                    await asyncio.sleep(0)
            finally:
                probe.gate.set()
            done, _ = await asyncio.wait((task,), timeout=GATE_TIMEOUT)
            assert done and task.cancelled() == bool(handler_cancellations)
            assert not bot._classification_lock.locked()
        finally:
            await client.session.close()

    run_checked(run())
    assert message.answers == [bot.RESPONSE_TIMEOUT_MESSAGE]
    assert yielded == [4096] and probe.files == [] and probe.calls == []
    assert calls.decoded == [] and predicted == []
    assert len(probe.cleanups) == 1 and not probe.cleanups[0]["directory"].exists()
    assert probe.cleanups[0]["open_fds"] == NO_OPEN_FDS


def test_audio_save_failure_is_controlled(monkeypatch, caplog):
    speech, calls = fake_speech(monkeypatch)
    predicted = recording_predict(monkeypatch)
    probe = instrument_voice_files(monkeypatch, stage=None)

    def full_disk(path, data):
        raise OSError(28, f"No space left on device: {PRIVATE_POSIX_PATH}")
    monkeypatch.setattr(bot, "_save_audio", full_disk)
    client, message = FakeBot(), FakeVoiceMessage()
    with caplog.at_level(logging.WARNING):
        run_voice(message, client, speech)
    assert message.answers == [bot.VOICE_DOWNLOAD_ERROR_MESSAGE]
    assert "Voice download failed: OSError" in caplog.text and "private-path-marker" not in caplog.text
    assert calls.decoded == [] and predicted == []
    assert len(probe.cleanups) == 1 and not client.downloads[0][1].parent.exists()


@pytest.mark.parametrize("extra_bytes,rejected", [(0, False), (1, True)])
def test_real_aiogram_download_limit_is_enforced_in_memory(monkeypatch, caplog, extra_bytes, rejected):
    """At most MAX_VOICE_BYTES are buffered; a larger download stops and never reaches the disk."""
    speech, calls = fake_speech(monkeypatch)
    predicted = recording_predict(monkeypatch)
    probe = instrument_voice_files(monkeypatch, stage=None)
    chunk = b"\0" * 65536
    chunks = [chunk] * (bot.MAX_VOICE_BYTES // len(chunk)) + [b"\0"] * extra_bytes + [chunk] * 3 * extra_bytes
    yielded = []
    message = FakeVoiceMessage(file_size=None)

    async def run():
        client = local_aiogram_bot(chunks, yielded)
        try:
            await bot.voice_handler(message, client, speech)  # type: ignore[arg-type]
        finally:
            await client.session.close()

    with caplog.at_level(logging.WARNING):
        run_checked(run())
    assert len(probe.cleanups) == 1 and not probe.cleanups[0]["directory"].exists()
    if rejected:
        assert message.answers == [bot.VOICE_TOO_LARGE_MESSAGE]
        assert "downloaded file exceeds the size limit" in caplog.text
        # The download stopped at the first byte over the limit; nothing was written to disk.
        assert sum(yielded) == bot.MAX_VOICE_BYTES + 1 and probe.files == []
        assert calls.decoded == [] and calls.loads == 0 and predicted == []
    else:
        assert message.answers[0].startswith("Распознанный текст:")
        assert probe.bytes_written == bot.MAX_VOICE_BYTES and calls.decoded[0][1]


PRIVATE_POSIX_PATH = "/srv/private-path-marker/models/tfidf_logreg_baseline.joblib"
PRIVATE_WINDOWS_PATH = r"C:\Users\Private Path Marker\models\run_metadata.json"
USER_TEXT_MARKER = "ЛИЧНЫЙ-ТЕКСТ-МАРКЕР"
PROTECTED_MARKERS = ("private-path-marker", "Private Path Marker", USER_TEXT_MARKER, FAKE_SECRET, "Traceback")


def run_mode(monkeypatch, mode, text):
    if mode == "voice":
        speech, _ = fake_speech(monkeypatch, segments=(text,))
        message = FakeVoiceMessage()
        run_voice(message, FakeBot(), speech)
    else:
        message = FakeMessage(text)
        asyncio.run(bot.text_handler(message))  # type: ignore[arg-type]
    return message


def logged_output(log_stream, caplog):
    """Formatted records as handlers write them, plus what caplog captured with tracebacks."""
    assert all(record.exc_info is None and not record.exc_text for record in caplog.records)
    return log_stream.getvalue() + caplog.text


@pytest.mark.parametrize("mode", ["text", "voice"])
def test_classifier_failure_log_has_no_paths_text_or_token(monkeypatch, log_stream, caplog, mode):
    bot.protect_token_in_logs(FAKE_TOKEN)
    cause = FileNotFoundError(2, f"No such file ({USER_TEXT_MARKER}, {FAKE_TOKEN})", PRIVATE_WINDOWS_PATH)
    error = classifier.ModelUnavailableError(
        f"Could not load verified baseline bundle {PRIVATE_POSIX_PATH}: {cause}")
    error.__cause__ = cause
    predicted = recording_predict(monkeypatch, error=error)
    with caplog.at_level(logging.DEBUG):
        message = run_mode(monkeypatch, mode, f"{USER_TEXT_MARKER} отзыв")
    assert predicted and message.answers == [bot.MODEL_ERROR_MESSAGE]
    output = logged_output(log_stream, caplog)
    assert "classification failed: ModelUnavailableError (FileNotFoundError)" in output
    for marker in PROTECTED_MARKERS:
        assert marker not in output, marker


@pytest.mark.parametrize("mode", ["text", "voice"])
def test_unavailable_real_bundle_log_has_no_absolute_path(tmp_path, monkeypatch, log_stream, caplog, mode):
    model_path = tmp_path / "private-path-marker" / "missing.joblib"
    monkeypatch.setattr(classifier, "_default_classifier", baseline(model_path))
    with caplog.at_level(logging.DEBUG):
        message = run_mode(monkeypatch, mode, f"{USER_TEXT_MARKER} отзыв")
    assert message.answers == [bot.MODEL_ERROR_MESSAGE]
    assert "Тональность" not in message.answers[0]
    output = logged_output(log_stream, caplog)
    assert "classification failed: ModelUnavailableError" in output
    for marker in (str(tmp_path), "private-path-marker", USER_TEXT_MARKER, "Traceback"):
        assert marker not in output, marker


def test_worker_failure_after_threshold_is_logged_by_type_only(monkeypatch, caplog):
    monkeypatch.setattr(bot, "RESPONSE_TIMEOUT_SECONDS", .01)
    release = threading.Event()

    def failing_classify(text):
        assert release.wait(5)
        raise RuntimeError(f"internal detail for {text} at {PRIVATE_POSIX_PATH}")
    monkeypatch.setattr(bot, "classify_text_for_bot", failing_classify)
    message = notifying(FakeMessage(USER_TEXT_MARKER), on_answer=lambda text: release.set())
    with caplog.at_level(logging.DEBUG):
        run_checked(bot.text_handler(message))  # type: ignore[arg-type]
    assert message.answers == [bot.RESPONSE_TIMEOUT_MESSAGE]
    assert "Worker failed after cancellation: RuntimeError" in caplog.text
    assert USER_TEXT_MARKER not in caplog.text and "private-path-marker" not in caplog.text
    assert not bot._classification_lock.locked()
