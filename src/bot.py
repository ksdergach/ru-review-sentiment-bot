"""Telegram-интерфейс общего классификатора: текстовые и голосовые киноотзывы.

Голос распознаётся общим ``SpeechRecognizer`` (T08), текст обоих режимов
классифицируется одним ``predict`` (T07). Окончательная политика лимитов,
занятости, таймаутов и жизненного цикла аудио относится к T10.
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
import logging
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable, Iterator, TypeVar

from aiogram import Bot, Dispatcher, F, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import Message, Voice
from aiogram.utils.token import TokenValidationError, validate_token

from src.classifier import (
    MAX_TEXT_LENGTH,
    ClassifierError,
    EmptyTextError,
    InvalidTextError,
    TextTooLongError,
    get_default_classifier,
    predict,
)
from src.speech import ROOT, SpeechConfig, SpeechRecognizer


router = Router()
logger = logging.getLogger(__name__)
_classification_lock = asyncio.Lock()
BUSY_MESSAGE = "Сейчас обрабатывается другой отзыв. Повторите запрос чуть позже."

_T = TypeVar("_T")

# Начальные ограничения из docs/team_plan.md, раздел 7.
MAX_VOICE_SECONDS = 30
MAX_VOICE_BYTES = 5 * 1024 * 1024

RUSSIAN_LABELS = {
    "negative": "отрицательная",
    "neutral": "нейтральная",
    "positive": "положительная",
}


START_TEXT = (
    "Привет! Я определяю тональность русскоязычных киноотзывов.\n\n"
    "Отправьте отзыв голосовым или обычным текстовым сообщением. "
    "Результат: отрицательная, нейтральная или положительная тональность.\n\n"
    f"Голосовое сообщение — до {MAX_VOICE_SECONDS} секунд и 5 МиБ, "
    f"текст — до {MAX_TEXT_LENGTH} символов.\n"
    "Подробнее: /help"
)


HELP_TEXT = (
    "Бот предназначен для анализа тональности русскоязычных отзывов о фильмах.\n"
    "Для других текстов результат классификации не имеет "
    "содержательного смысла.\n\n"
    "Как отправить отзыв:\n"
    "• голосовым сообщением — бот распознает речь и покажет распознанный текст;\n"
    "• обычным текстовым сообщением.\n\n"
    "Доступные классы:\n"
    "• отрицательная (negative);\n"
    "• нейтральная (neutral);\n"
    "• положительная (positive).\n\n"
    "Ограничения:\n"
    f"• голосовое сообщение — до {MAX_VOICE_SECONDS} секунд и до 5 МиБ;\n"
    f"• текст или распознанная речь — до {MAX_TEXT_LENGTH} символов;\n"
    "• одновременно обрабатывается один отзыв.\n\n"
    "Распознавание речи может содержать ошибки. "
    "Показанная уверенность модели не является гарантией "
    "правильности классификации."
)


EMPTY_TEXT_MESSAGE = (
    "Отправьте непустой текстовый отзыв о фильме."
)


TOO_LONG_MESSAGE = (
    f"Сообщение слишком длинное. Максимум — {MAX_TEXT_LENGTH} символов."
)


MODEL_ERROR_MESSAGE = (
    "Не удалось выполнить классификацию. "
    "Попробуйте повторить запрос позже."
)


UNKNOWN_COMMAND_MESSAGE = (
    "Неизвестная команда. "
    "Отправьте отзыв голосовым или текстовым сообщением либо используйте /help."
)


UNSUPPORTED_MESSAGE = (
    "Поддерживаются только голосовые и текстовые сообщения "
    "с отзывами о фильмах."
)


VOICE_TOO_LONG_MESSAGE = (
    f"Голосовое сообщение слишком длинное. Максимум — {MAX_VOICE_SECONDS} секунд."
)


VOICE_TOO_LARGE_MESSAGE = (
    "Голосовое сообщение слишком большое. Максимум — 5 МиБ."
)


VOICE_DOWNLOAD_ERROR_MESSAGE = (
    "Не удалось получить голосовое сообщение. Попробуйте отправить его ещё раз."
)


EMPTY_SPEECH_MESSAGE = (
    "В голосовом сообщении не удалось распознать речь. "
    "Запишите отзыв ещё раз или отправьте его текстом."
)


SPEECH_ERROR_MESSAGE = (
    "Не удалось распознать голосовое сообщение. "
    "Попробуйте записать его ещё раз или отправьте отзыв текстом."
)


TRANSCRIPT_TOO_LONG_MESSAGE = (
    f"Распознанный текст длиннее {MAX_TEXT_LENGTH} символов, поэтому он не "
    "классифицирован. Запишите более короткий отзыв."
)


class StartupError(RuntimeError):
    """Приложение нельзя запустить; сообщение безопасно для вывода."""


def format_prediction(result: dict[str, Any]) -> str:
    """Преобразовать результат классификатора в ответ пользователю."""
    label = str(result["label"])
    confidence = float(result["confidence"])

    truncation_note = "\nУчтено только начало отзыва: текст усечён моделью." if result.get("truncated") else ""
    return (
        f"Тональность: {RUSSIAN_LABELS[label]} ({label})\n"
        f"Уверенность модели: {confidence:.1%}\n\n"
        "Уверенность модели не является гарантией правильности ответа."
        + truncation_note
    )


def classify_text_for_bot(text: str) -> str:
    """Выполнить тот же predict, который доступен для прямого вызова."""
    try:
        result = predict(text)
    except EmptyTextError:
        return EMPTY_TEXT_MESSAGE
    except TextTooLongError:
        return TOO_LONG_MESSAGE
    except InvalidTextError:
        return EMPTY_TEXT_MESSAGE
    except ClassifierError:
        logger.exception("Text classification failed")
        return MODEL_ERROR_MESSAGE

    return format_prediction(result)


def classify_transcript_for_bot(transcript: str) -> str:
    """Классифицировать распознанную речь тем же predict, что и текст."""
    try:
        result = predict(transcript)
    except TextTooLongError:
        # The shared contract rejects long input; nothing is truncated silently.
        return TRANSCRIPT_TOO_LONG_MESSAGE
    except InvalidTextError:
        return EMPTY_SPEECH_MESSAGE
    except ClassifierError:
        logger.exception("Voice transcript classification failed")
        return MODEL_ERROR_MESSAGE

    return f"Распознанный текст:\n{transcript}\n\n{format_prediction(result)}"


async def _run_in_worker(func: Callable[[str], _T], argument: str) -> _T:
    """Выполнить тяжёлую операцию вне event loop."""
    task = asyncio.create_task(asyncio.to_thread(func, argument))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        # Cancellation does not stop a running thread. Keep the slot until it exits.
        await task
        raise


def check_voice_limits(voice: Voice) -> str | None:
    """Предварительная проверка по метаданным Telegram; полная защита — T10."""
    if voice.duration > MAX_VOICE_SECONDS:
        return VOICE_TOO_LONG_MESSAGE
    if voice.file_size is not None and voice.file_size > MAX_VOICE_BYTES:
        return VOICE_TOO_LARGE_MESSAGE
    return None


@contextmanager
def _temporary_voice_path() -> Iterator[Path]:
    """Уникальный закрытый каталог для одного аудио; удаляется при любом исходе."""
    directory = Path(tempfile.mkdtemp(prefix="ru-review-voice-"))
    try:
        yield directory / "voice.ogg"
    finally:
        try:
            shutil.rmtree(directory)
        except OSError:
            logger.warning("Could not remove temporary voice directory")


async def recognize_and_classify_voice(bot: Bot, speech: SpeechRecognizer, voice: Voice) -> str:
    """Скачать голос, распознать его и классифицировать полный текст."""
    with _temporary_voice_path() as audio_path:
        try:
            await bot.download(voice.file_id, destination=audio_path)
        except Exception as exc:
            logger.warning("Voice download failed: %s", type(exc).__name__)
            return VOICE_DOWNLOAD_ERROR_MESSAGE
        # Waits for the worker even on cancellation, so the file outlives its use.
        result = await speech.transcribe_async(audio_path)

    if result.status == "empty":
        return EMPTY_SPEECH_MESSAGE
    if result.status != "ok":
        logger.warning("Speech recognition failed: %s", result.error_code)
        return SPEECH_ERROR_MESSAGE
    return await _run_in_worker(classify_transcript_for_bot, result.text)


@router.message(CommandStart())
async def start_handler(message: Message) -> None:
    """Обработать команду /start."""
    await message.answer(START_TEXT)


@router.message(Command("help"))
async def help_handler(message: Message) -> None:
    """Обработать команду /help."""
    await message.answer(HELP_TEXT)


def is_bot_command(message: Message) -> bool:
    """Only Telegram-marked commands, not every slash-prefixed review."""
    return any(entity.type == "bot_command" and entity.offset == 0
               for entity in (message.entities or []))


@router.message(is_bot_command)
async def unknown_command_handler(message: Message) -> None:
    """Не отправлять неизвестные команды в классификатор."""
    await message.answer(UNKNOWN_COMMAND_MESSAGE)


@router.message(F.text)
async def text_handler(message: Message) -> None:
    """Классифицировать обычное текстовое сообщение."""
    text = message.text or ""
    if _classification_lock.locked():
        await message.answer(BUSY_MESSAGE)
        return
    async with _classification_lock:
        answer = await _run_in_worker(classify_text_for_bot, text)
    await message.answer(answer)


@router.message(F.voice)
async def voice_handler(message: Message, bot: Bot, speech: SpeechRecognizer) -> None:
    """Распознать голосовой отзыв и классифицировать его текст."""
    voice = message.voice
    limit_message = check_voice_limits(voice)
    if limit_message is not None:
        await message.answer(limit_message)
        return
    if _classification_lock.locked():
        await message.answer(BUSY_MESSAGE)
        return
    async with _classification_lock:
        answer = await recognize_and_classify_voice(bot, speech, voice)
    await message.answer(answer)


@router.message()
async def unsupported_handler(message: Message) -> None:
    """Обработать аудиофайлы, стикеры и другие неподдерживаемые типы."""
    await message.answer(UNSUPPORTED_MESSAGE)


def get_bot_token() -> str:
    """Получить Telegram-токен только из окружения."""
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()

    if not token:
        raise StartupError(
            "Environment variable TELEGRAM_BOT_TOKEN is not set"
        )

    try:
        validate_token(token)
    except TokenValidationError:
        raise StartupError("Environment variable TELEGRAM_BOT_TOKEN has invalid format") from None
    return token


def _public_reason(exc: BaseException) -> str:
    """Причина отказа без абсолютных путей рабочего каталога и домашней папки."""
    cause = exc.__cause__ or exc
    reason = f"{type(cause).__name__}: {cause}"
    for private, public in ((str(ROOT), "."), (str(Path.home()), "~")):
        reason = reason.replace(private, public)
    return reason


def load_models(speech: SpeechRecognizer | None = None) -> SpeechRecognizer:
    """Загрузить классификатор и Whisper один раз до начала polling."""
    try:
        # Same eager load as tools/evaluate_classifier.py; predict() reuses this instance.
        get_default_classifier()._get_bundle()
    except ClassifierError as exc:
        raise StartupError(
            f"Classifier bundle is unavailable ({_public_reason(exc)}). "
            "Check CLASSIFIER_BACKEND, CLASSIFIER_MODEL_PATH and CLASSIFIER_REPORT_PATH "
            "and install the accepted bundle (README, section T09)."
        ) from None
    try:
        if speech is None:
            speech = SpeechRecognizer(SpeechConfig.load())
        speech.load()
    except Exception as exc:
        raise StartupError(
            f"Whisper model from configs/speech.json is unavailable offline ({type(exc).__name__}). "
            "Download the pinned model once with python -m src.speech <debug audio> --allow-download "
            "(README, section T09)."
        ) from None
    return speech


async def run_bot(token: str | None = None, speech: SpeechRecognizer | None = None) -> None:
    """Запустить Telegram polling с моделями, загруженными один раз."""
    if token is None:
        token = get_bot_token()
    if speech is None:
        speech = load_models()
    bot = Bot(token=token)
    dispatcher = Dispatcher(speech=speech)
    dispatcher.include_router(router)

    await dispatcher.start_polling(bot)


def main() -> None:
    """Точка входа Telegram-бота."""
    logging.basicConfig(level=logging.INFO)
    try:
        token = get_bot_token()
        logger.info("Loading classifier and Whisper models")
        speech = load_models()
    except StartupError as exc:
        logger.error("Startup failed: %s", exc)
        raise SystemExit(1) from None
    asyncio.run(run_bot(token, speech))


if __name__ == "__main__":
    main()
