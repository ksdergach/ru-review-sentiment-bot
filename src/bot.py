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

from src.classifier import (
    MAX_TEXT_LENGTH,
    ClassifierError,
    EmptyTextError,
    TextTooLongError,
    predict,
)


router = Router()
logger = logging.getLogger(__name__)


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


@router.message(F.text.startswith("/"))
async def unknown_command_handler(message: Message) -> None:
    """Не отправлять неизвестные команды в классификатор."""
    await message.answer(UNKNOWN_COMMAND_MESSAGE)


@router.message(F.text)
async def text_handler(message: Message) -> None:
    """Классифицировать обычное текстовое сообщение."""
    text = message.text or ""
    await message.answer(classify_text_for_bot(text))


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