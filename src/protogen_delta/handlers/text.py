"""Обработчик обычных текстовых сообщений."""

from aiogram import Bot, F, Router
from aiogram.types import Message

from protogen_delta.core.chat_scope import ChatScopeOptions, chat_scope_options
from protogen_delta.core.rate_limiter import UserRateLimiter
from protogen_delta.handlers.delivery import create_reply_delivery, show_typing
from protogen_delta.handlers.rp import (
    RP_ALREADY_DISABLED_REPLY,
    RP_DISABLED_REPLY,
    is_roleplay_stop_message,
)
from protogen_delta.services.capabilities import is_capability_overview
from protogen_delta.services.response_engine import ResponseBusyError, ResponseEngine
from protogen_delta.services.stickers import ContextualStickerService

RATE_LIMIT_REPLY = "Слишком быстро :D Подожди пару секунд."
BUSY_REPLY = "Я ещё отвечаю на предыдущее сообщение. Подожди немного."


class _InputContext(ChatScopeOptions, total=False):
    trusted_input_context: str
    use_personal_facts: bool


def create_text_router(
    response_engine: ResponseEngine,
    rate_limiter: UserRateLimiter | None = None,
    bot: Bot | None = None,
    sticker_service: ContextualStickerService | None = None,
) -> Router:
    """Создать роутер обычных текстовых сообщений."""
    router = Router(name=__name__)
    limiter = rate_limiter or UserRateLimiter()

    @router.message(F.text)
    async def handle_text(message: Message) -> None:
        """Передать сообщение движку и отправить сформированный ответ."""
        if message.text is None:
            return

        if message.text.startswith("/"):
            return

        if message.from_user is None:
            return

        user_id = message.from_user.id
        scope = chat_scope_options(message.chat.id, message.chat.type)

        if is_roleplay_stop_message(message.text):
            was_active = await response_engine.disable_roleplay(
                user_id,
                **scope,
            )

            if was_active:
                await message.answer(RP_DISABLED_REPLY)
            else:
                await message.answer(RP_ALREADY_DISABLED_REPLY)

            return

        safety_signal = (
            response_engine.is_safety_signal(user_id, message.text, **scope) is True
        )
        if not safety_signal and not limiter.allow(user_id):
            await message.answer(RATE_LIMIT_REPLY)
            return

        if sticker_service is not None and sticker_service.is_request(
            user_id, message.text, **scope
        ):
            if not sticker_service.is_available(user_id):
                await message.answer(
                    "Стикеры сейчас отключены или нет доступных реакций."
                )
            elif not await sticker_service.maybe_send(
                chat_id=message.chat.id,
                user_id=user_id,
                context_text=message.text,
                scope_chat_id=scope.get("chat_id"),
            ):
                await message.answer(
                    "Сейчас не получилось отправить стикер. Подожди немного "
                    "и попробуй ещё раз."
                )
            return

        try:
            input_context: _InputContext = {**scope}
            if (
                message.chat.type in {"group", "supergroup", "channel"}
                or message.forward_origin is not None
            ):
                input_context["use_personal_facts"] = False
            if sticker_service is not None:
                input_context["trusted_input_context"] = (
                    sticker_service.capabilities_context(user_id)
                )
            async with show_typing(message, bot):
                await response_engine.respond_and_deliver(
                    user_id,
                    message.text,
                    create_reply_delivery(
                        message,
                        bot,
                        sticker_service,
                        user_id=user_id,
                        format_capabilities=is_capability_overview(message.text),
                    ),
                    **input_context,
                )
        except ResponseBusyError:
            await message.answer(BUSY_REPLY)

    return router
