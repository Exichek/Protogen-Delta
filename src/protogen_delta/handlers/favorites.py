"""Просмотр и удаление выбранных сообщений собственного личного чата."""

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

from protogen_delta.core.message_utils import split_message
from protogen_delta.core.user_state import UserStateStore
from protogen_delta.services.favorites import FavoritesService

_HELP = (
    "⭐ Избранные сообщения\n/favorites — последние записи\n"
    "/favorites <слова> — поиск\n/favorites 12 — полный текст записи #12\n"
    "/favorites remove 12 — убрать запись #12. Полный сброс: /reset."
)


def create_favorites_router(
    service: FavoritesService, states: UserStateStore
) -> Router:
    router = Router(name=__name__)

    @router.message(Command("favorites"))
    async def favorites(message: Message) -> None:
        if (
            message.from_user is None
            or message.from_user.is_bot
            or message.forward_origin is not None
            or message.chat.type != "private"
            or message.chat.id != message.from_user.id
        ):
            await message.answer(
                "Избранное доступно в собственном личном чате с Дельтой.",
                parse_mode=None,
            )
            return
        owner = message.from_user.id
        if not service.enabled_for(owner):
            await message.answer(
                "Для этого аккаунта избранное пока не подключено.", parse_mode=None
            )
            return
        arguments = (message.text or "").split(maxsplit=1)
        query = arguments[1].strip() if len(arguments) == 2 else ""
        async with states.use(owner):
            try:
                if query.startswith("remove "):
                    identifier = query[7:].strip()
                    if (
                        not identifier.isascii()
                        or not identifier.isdecimal()
                        or len(identifier) > 19
                    ):
                        raise ValueError(
                            "Номер записи должен быть положительным целым числом."
                        )
                    deleted = await service.repository.remove(owner, int(identifier))
                    reply = (
                        "Запись убрана из избранного."
                        if deleted
                        else "Такой записи в твоём избранном нет."
                    )
                elif query.isascii() and query.isdecimal():
                    if len(query) > 19:
                        raise ValueError("Слишком большой номер записи.")
                    entry = await service.repository.get(owner, int(query))
                    reply = (
                        service.describe(entry, limit=8000)
                        if entry
                        else "Такой записи в твоём избранном нет."
                    )
                else:
                    if len(query) > 200:
                        raise ValueError("Поисковый запрос: не более 200 символов.")
                    entries = (
                        await service.repository.search(owner, query, limit=5)
                        if query
                        else await service.repository.recent(owner)
                    )
                    reply = (
                        "⭐ Выбранные сообщения:\n\n"
                        + "\n\n".join(
                            service.describe(entry, query=query) for entry in entries
                        )
                        if entries
                        else (
                            "Пока нет выбранных сообщений."
                            if not query
                            else "Совпадений в избранном не нашёл."
                        )
                    )
                    reply += "\n\n" + _HELP
            except ValueError as error:
                reply = str(error) + "\n\n" + _HELP
            for chunk in split_message(reply):
                await message.answer(chunk, parse_mode=None)

    return router
