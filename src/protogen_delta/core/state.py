"""Глобальное состояние процесса бота."""

from dataclasses import dataclass, field
from time import monotonic


@dataclass(slots=True)
class BotState:
    """Хранить глобальное состояние текущего процесса бота."""

    reply_count: int = 0
    start_time: float = field(default_factory=monotonic)

    def register_reply(self) -> None:
        """Увеличить общий счётчик отправленных ботом ответов."""
        self.reply_count += 1
