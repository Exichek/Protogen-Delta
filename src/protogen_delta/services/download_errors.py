"""Ограниченные коды отказов: внешние сообщения и URL не покидают worker."""

import json
from pathlib import Path

ERROR_MESSAGES = {
    "youtube_bot": (
        "YouTube требует подтверждения, что запрос не от бота. "
        "Сейчас Дельта не может пройти эту проверку."
    ),
    "youtube_age": (
        "YouTube требует входа в аккаунт с подтверждённым возрастом. "
        "Бот пока не может получить доступ к этому ролику."
    ),
    "login": "Для этого видео требуется вход в аккаунт. Бот не получил доступ.",
    "unavailable": "Видео удалено, закрыто или недоступно для загрузки.",
    "too_large": "Видео превышает лимит 100 МБ.",
    "too_long": "Видео превышает лимит 10 минут.",
    "formats": "Сайт не отдал подходящий видеоформат для скачивания.",
    "runtime": "Не удалось обработать YouTube-ссылку. Нужна проверка загрузчика.",
    "cookies": "Не удалось прочитать подключённую сессию YouTube.",
    "network": "Сайт не ответил или прервал загрузку. Попробуй позже.",
    "access_denied": "Сайт отказал в доступе к видео (HTTP 403).",
    "rate_limited": "Сайт временно ограничил частоту загрузок. Попробуй позже.",
    "compression": "Не удалось сжать видео для отправки в Telegram.",
    "unknown": "Не удалось скачать видео. Попробуй другой ролик или повтори позже.",
}


class DownloadFailure(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code if code in ERROR_MESSAGES else "unknown"
        super().__init__(self.code)


def classify_failure(error: Exception) -> str:
    if isinstance(error, DownloadFailure):
        return error.code
    message = str(error).casefold()
    if "sign in to confirm" in message:
        return "youtube_age" if "age" in message else "youtube_bot"
    if any(text in message for text in ("login required", "requires authentication")):
        return "login"
    if any(
        text in message for text in ("private video", "video is unavailable", "removed")
    ):
        return "unavailable"
    if any(text in message for text in ("requested format", "no video formats")):
        return "formats"
    if "http error 403" in message:
        return "access_denied"
    if "http error 429" in message:
        return "rate_limited"
    if "http error 404" in message:
        return "unavailable"
    if "javascript runtime" in message or "running deno process" in message:
        return "runtime"
    if any(text in message for text in ("timed out", "connection")):
        return "network"
    return "unknown"


def read_failure(directory: Path) -> str:
    try:
        with (directory / "error.json").open("rb") as stream:
            raw = stream.read(513)
        if len(raw) > 512:
            return "unknown"
        value = json.loads(raw)
        code = value.get("error") if isinstance(value, dict) else None
        return code if isinstance(code, str) and code in ERROR_MESSAGES else "unknown"
    except OSError, ValueError:
        return "unknown"
