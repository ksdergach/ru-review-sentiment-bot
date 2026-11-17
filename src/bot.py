"""Telegram-интерфейс общего классификатора: текстовые и голосовые киноотзывы.

Голос распознаётся общим ``SpeechRecognizer`` (T08), текст обоих режимов
классифицируется одним ``predict`` (T07). Лимиты, занятость, порог ожидания
ответа и жизненный цикл аудио — T10.
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
import logging
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Coroutine, Iterator

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
from src.speech import ROOT, SpeechConfig, SpeechRecognizer, run_in_thread


router = Router()
logger = logging.getLogger(__name__)
_classification_lock = asyncio.Lock()
BUSY_MESSAGE = "Сейчас обрабатывается другой отзыв. Повторите запрос чуть позже."

# Начальные ограничения из docs/team_plan.md, раздел 7.
MAX_VOICE_SECONDS = 30
MAX_VOICE_BYTES = 5 * 1024 * 1024
# Порог ожидания ответа пользователем, а не предел длительности вычисления.
RESPONSE_TIMEOUT_SECONDS = 60

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
    "• одновременно обрабатывается один отзыв;\n"
    f"• если ответ не готов за {RESPONSE_TIMEOUT_SECONDS} секунд, бот сообщит "
    "о задержке, а результат этого запроса не отправит.\n\n"
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


RESPONSE_TIMEOUT_MESSAGE = (
    f"Обработка отзыва заняла больше {RESPONSE_TIMEOUT_SECONDS} секунд, "
    "результат этого запроса не будет отправлен. "
    "Повторите попытку позже или отправьте более короткий отзыв."
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


def check_voice_limits(voice: Voice) -> str | None:
    """Проверка по метаданным Telegram до скачивания; размер файла проверяется и после."""
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
            size = audio_path.stat().st_size
        except Exception as exc:
            logger.warning("Voice download failed: %s", type(exc).__name__)
            return VOICE_DOWNLOAD_ERROR_MESSAGE
        # file_size may be absent or wrong; the downloaded bytes decide before Whisper.
        if size > MAX_VOICE_BYTES:
            logger.warning("Voice rejected: downloaded file exceeds the size limit")
            return VOICE_TOO_LARGE_MESSAGE
        # Waits for the worker even on cancellation, so the file outlives its use.
        result = await speech.transcribe_async(audio_path)

    if result.status == "empty":
        return EMPTY_SPEECH_MESSAGE
    if result.status != "ok":
        logger.warning("Speech recognition failed: %s", result.error_code)
        return SPEECH_ERROR_MESSAGE
    return await run_in_thread(classify_transcript_for_bot, result.text)


async def _wait_until_done(task: asyncio.Task[Any]) -> None:
    """Return or re-raise cancellation only after task ended; pass each cancel on to it."""
    cancelled: asyncio.CancelledError | None = None
    while not task.done():
        try:
            # Unlike awaiting the task directly, wait() never cancels it.
            await asyncio.wait((task,))
        except asyncio.CancelledError as exc:
            cancelled = exc
            task.cancel()
    if cancelled is not None:
        raise cancelled


async def answer_within_timeout(message: Message, work: Coroutine[Any, Any, str]) -> str | None:
    """Return the answer of work, or None after sending one delay notice at the threshold.

    Call while holding _classification_lock: it returns only after work and its worker
    thread ended, also after the notice or on (repeated) cancellation, so the slot and
    the temporary audio are released no earlier. Which outcome wins is decided once, in
    the event loop, so a result is never sent after the notice. Not a hard timeout:
    a running thread finishes naturally; only the later stages are cancelled.
    """
    task = asyncio.create_task(work)
    try:
        done, _ = await asyncio.wait((task,), timeout=RESPONSE_TIMEOUT_SECONDS)
        if done:
            return task.result()
        task.cancel()
        logger.warning("Response timeout reached; the result will not be sent")
        try:
            await message.answer(RESPONSE_TIMEOUT_MESSAGE)
        except Exception as exc:
            logger.warning("Could not send the delay notice: %s", type(exc).__name__)
    except asyncio.CancelledError:
        task.cancel()
        raise
    finally:
        await _wait_until_done(task)
    if not task.cancelled() and task.exception() is not None:
        logger.warning("Processing failed after the response timeout: %s",
                       type(task.exception()).__name__)
    return None


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
    # No await between the check and the acquisition: admission is atomic in the loop.
    async with _classification_lock:
        # Returns only after the worker thread ends, so the slot is held until then.
        answer = await answer_within_timeout(message, run_in_thread(classify_text_for_bot, text))
    if answer is not None:
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
        answer = await answer_within_timeout(message, recognize_and_classify_voice(bot, speech, voice))
    if answer is not None:
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


_ABSOLUTE_PATH_START = r"(?:[A-Za-z]:[\\/]|\\\\|/)"
# Quoted paths (repr of OSError) may contain spaces; unquoted ones end at whitespace.
_QUOTED_ABSOLUTE_PATH = re.compile(rf"""(['"])({_ABSOLUTE_PATH_START}[^'"]*)\1""")
_ABSOLUTE_PATH = re.compile(rf"""(?<![\w.~/\\>]){_ABSOLUTE_PATH_START}[^\s'"]*""")


def _public_path(path: str) -> str:
    """Путь внутри проекта — относительный, любой другой абсолютный — только имя файла."""
    normalized = path.replace("\\\\", "\\")
    for root in (str(ROOT), ROOT.as_posix()):
        rest = normalized[len(root):]
        if normalized.startswith(root) and (not rest or rest[0] in "/\\"):
            return "." + rest.replace("\\", "/")
    name = re.split(r"[\\/]", normalized.rstrip("\\/"))[-1]
    return f"<path>/{name}" if name else "<path>"


def _public_reason(exc: BaseException) -> str:
    """Причина отказа без абсолютных путей, в том числе внешних и Windows-путей."""
    cause = exc.__cause__ or exc
    reason = f"{type(cause).__name__}: {cause}"
    reason = _QUOTED_ABSOLUTE_PATH.sub(lambda m: m[1] + _public_path(m[2]) + m[1], reason)
    return _ABSOLUTE_PATH.sub(lambda m: _public_path(m[0]), reason)


class _SecretRedactingFilter(logging.Filter):
    """Убирает токен из текста записи и traceback до передачи любому обработчику.

    aiohttp включает URL запроса с токеном в тексты своих ошибок (например,
    TooManyRedirects), а aiogram пишет их в журнал polling.
    """

    def __init__(self) -> None:
        super().__init__()
        self.secrets: set[str] = set()

    def _redact(self, text: str) -> str:
        for secret in sorted(self.secrets, key=len, reverse=True):
            text = text.replace(secret, "<redacted>")
        return text

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:
            message = str(record.msg)
        record.msg, record.args = self._redact(message), None
        if record.exc_info:
            record.exc_text = logging.Formatter().formatException(record.exc_info)
        if record.exc_text:
            record.exc_text = self._redact(record.exc_text)
        if record.stack_info:
            record.stack_info = self._redact(record.stack_info)
        return True


_secret_filter = _SecretRedactingFilter()


def protect_token_in_logs(token: str) -> None:
    """Скрыть токен в каждой записи журнала при её создании.

    Обработчики могут быть на любом логгере (например, на aiogram.dispatcher без
    передачи записей в root), поэтому токен убирается в фабрике записей, а не в них.
    """
    # The part after ":" is the secret; the numeric bot ID alone is public.
    _secret_filter.secrets.update({token, token.partition(":")[2]} - {""})
    base_factory = logging.getLogRecordFactory()
    if getattr(base_factory, "redacts_secrets", False):
        return

    def redacting_factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
        record = base_factory(*args, **kwargs)
        _secret_filter.filter(record)
        return record

    redacting_factory.redacts_secrets = True  # type: ignore[attr-defined]
    logging.setLogRecordFactory(redacting_factory)


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
    protect_token_in_logs(token)
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
    protect_token_in_logs(token)
    try:
        asyncio.run(run_bot(token, speech))
    except Exception:
        # Through logging, so the traceback is redacted (not the default excepthook).
        logger.exception("Bot stopped because of an error")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
