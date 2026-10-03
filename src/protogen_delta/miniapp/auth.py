"""Проверка подписи Telegram Mini App initData."""

import hashlib
import hmac
import json
from dataclasses import dataclass
from time import time
from urllib.parse import parse_qsl


class MiniAppAuthError(ValueError):
    """Входные данные Mini App отсутствуют, просрочены или подделаны."""


@dataclass(frozen=True, slots=True)
class MiniAppUser:
    """Минимальные проверенные данные Telegram-пользователя."""

    id: int
    first_name: str
    username: str | None = None


def validate_init_data(
    init_data: str,
    bot_token: str,
    *,
    max_age_seconds: int = 3600,
    now: float | None = None,
) -> MiniAppUser:
    """Проверить HMAC, свежесть и user из Telegram.WebApp.initData."""
    if not init_data:
        raise MiniAppAuthError("initData отсутствует")
    if max_age_seconds <= 0:
        raise ValueError("max_age_seconds должен быть больше нуля")

    values = dict(parse_qsl(init_data, keep_blank_values=True))
    received_hash = values.pop("hash", "")
    if not received_hash:
        raise MiniAppAuthError("hash отсутствует")
    # HMAC включает signature; её исключают только при Ed25519-проверке.
    data_check_string = "\n".join(
        f"{key}={value}" for key, value in sorted(values.items())
    )
    secret_key = hmac.new(
        b"WebAppData",
        bot_token.encode("utf-8"),
        hashlib.sha256,
    ).digest()
    calculated = hmac.new(
        secret_key,
        data_check_string.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(calculated, received_hash):
        raise MiniAppAuthError("подпись initData не совпала")

    try:
        auth_date = int(values["auth_date"])
    except (KeyError, ValueError) as error:
        raise MiniAppAuthError("auth_date отсутствует или некорректен") from error
    current_time = time() if now is None else now
    if auth_date > current_time + 30 or current_time - auth_date > max_age_seconds:
        raise MiniAppAuthError("initData просрочен")

    try:
        raw_user = json.loads(values["user"])
        user_id = raw_user["id"]
        first_name = raw_user["first_name"]
        username = raw_user.get("username")
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise MiniAppAuthError("user отсутствует или некорректен") from error
    if (
        type(user_id) is not int
        or not isinstance(first_name, str)
        or username is not None
        and not isinstance(username, str)
    ):
        raise MiniAppAuthError("user содержит некорректные поля")
    return MiniAppUser(user_id, first_name, username)
