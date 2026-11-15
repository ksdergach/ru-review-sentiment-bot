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


class FakeBot:
    """Telegram download stub: writes synthetic bytes, never touches the network."""

    def __init__(self, error: BaseException | None = None, payload: bytes = b"synthetic voice") -> None:
        self.error = error
        self.payload = payload
        self.downloads: list[tuple[str, Path]] = []

    async def download(self, file_id: str, destination: Path) -> None:
        self.downloads.append((file_id, Path(destination)))
        if self.error is not None:
            raise self.error
        Path(destination).write_bytes(self.payload)


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
    assert reason == "FileNotFoundError: No such file: './reports/x.json' or '~/y'"


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
        Path(destination).write_bytes(b"synthetic voice")

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
