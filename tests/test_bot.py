from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import pytest

import src.bot as bot
import src.classifier as classifier
from test_classifier import make_real_bundle


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

    assert "positive" in answer
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
    assert "Baseline classification failed" in caplog.text


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
        bot.unsupported_handler,
    ]


def test_token_is_read_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "TELEGRAM_BOT_TOKEN",
        "test-token",
    )

    assert bot.get_bot_token() == "test-token"


def test_token_is_stripped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "TELEGRAM_BOT_TOKEN",
        "  test-token  ",
    )

    assert bot.get_bot_token() == "test-token"


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
    model = classifier.BaselineClassifier(model_path)

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
    model = classifier.BaselineClassifier(model_path)

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