"""Настройка логирования приложения."""

import logging

from protogen_delta.core.log_context import LogContextFilter

LOG_FORMAT = (
    "%(asctime)s [%(levelname)s] "
    "[user_id=%(user_id)s request_id=%(request_id)s] "
    "%(name)s: %(message)s"
)


def setup_logging(level: str = "INFO") -> None:
    """Настроить единое логирование для всего приложения."""
    numeric_level = getattr(logging, level.upper(), None)

    if not isinstance(numeric_level, int):
        raise ValueError(f"Неизвестный уровень логирования: {level}")

    logging.basicConfig(
        level=numeric_level,
        format=LOG_FORMAT,
    )

    context_filter = LogContextFilter()

    for handler in logging.getLogger().handlers:
        handler.addFilter(context_filter)
