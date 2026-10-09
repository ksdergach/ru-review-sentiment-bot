"""Текстовый Telegram-интерфейс для baseline-классификатора.

На этом этапе поддерживаются только текстовые киноотзывы.
Голосовой ввод будет подключён в следующих задачах.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from aiogram import Bot, Dispatcher, F, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import Message
from aiogram.utils.token import TokenValidationError, validate_token

from src.classifier import (
    MAX_TEXT_LENGTH,
    ClassifierError,
    EmptyTextError,
    InvalidTextError,
    TextTooLongError,
    predict,
)


router = Router()
logger = logging.getLogger(__name__)
_classification_lock = asyncio.Lock()
BUSY_MESSAGE = "Сейчас обрабатывается другой отзыв. Повторите запрос чуть позже."


START_TEXT = (
    "Привет! Я определяю тональность текстовых киноотзывов.\n\n"
    "Отправьте отзыв обычным текстовым сообщением. "
    "Результат: negative, neutral или positive.\n\n"
    f"Максимальная длина сообщения — {MAX_TEXT_LENGTH} символов.\n"
    "Подробнее: /help"
)


HELP_TEXT = (
    "Бот предназначен для классификации текстовых отзывов о фильмах.\n"
    "Для других текстов результат классификации не имеет "
    "содержательного смысла.\n\n"
    "Доступные классы:\n"
    "• negative — отрицательный отзыв;\n"
    "• neutral — нейтральный отзыв;\n"
    "• positive — положительный отзыв.\n\n"
    f"Максимальная длина текста — {MAX_TEXT_LENGTH} символов.\n"
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
    "Отправьте текст отзыва или используйте /help."
)


UNSUPPORTED_MESSAGE = (
    "Сейчас поддерживаются только текстовые сообщения "
    "с отзывами о фильмах."
)


def format_prediction(result: dict[str, Any]) -> str:
    """Преобразовать результат классификатора в ответ пользователю."""
    label = str(result["label"])
    confidence = float(result["confidence"])

    return (
        f"Тональность: {label}\n"
        f"Уверенность модели: {confidence:.1%}\n\n"
        "Уверенность модели не является гарантией правильности ответа."
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
        logger.exception("Baseline classification failed")
        return MODEL_ERROR_MESSAGE

    return format_prediction(result)


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
        task = asyncio.create_task(asyncio.to_thread(classify_text_for_bot, text))
        try:
            answer = await asyncio.shield(task)
        except asyncio.CancelledError:
            # Cancellation does not stop a running thread. Keep the slot until it exits.
            await task
            raise
    await message.answer(answer)


@router.message()
async def unsupported_handler(message: Message) -> None:
    """Обработать голос, стикеры и другие неподдерживаемые типы."""
    await message.answer(UNSUPPORTED_MESSAGE)


def get_bot_token() -> str:
    """Получить Telegram-токен только из окружения."""
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()

    if not token:
        raise RuntimeError(
            "Environment variable TELEGRAM_BOT_TOKEN is not set"
        )

    try:
        validate_token(token)
    except TokenValidationError:
        raise RuntimeError("Environment variable TELEGRAM_BOT_TOKEN has invalid format") from None
    return token


async def run_bot() -> None:
    """Запустить Telegram polling."""
    bot = Bot(token=get_bot_token())
    dispatcher = Dispatcher()
    dispatcher.include_router(router)

    await dispatcher.start_polling(bot)


def main() -> None:
    """Точка входа текстового Telegram-бота."""
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_bot())


if __name__ == "__main__":
    main()
