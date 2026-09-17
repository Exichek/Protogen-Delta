"""Состояние бота во время работы приложения."""

from dataclasses import dataclass, field
from time import monotonic


@dataclass(slots=True)
class BotState:
    """Хранить изменяемое состояние текущего процесса бота."""

    reply_count: int = 0
    mood: str = "playful"
    start_time: float = field(default_factory=monotonic)

    def register_reply(self) -> None:
        """Увеличить счётчик отправленных ботом ответов."""
        self.reply_count += 1
