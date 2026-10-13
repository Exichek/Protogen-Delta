"""Администраторская команда ограниченного обзора кода."""

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

from protogen_delta.services.code_review import CodeReviewService, answer_review


def create_code_review_router(service: CodeReviewService) -> Router:
    router = Router(name=__name__)

    @router.message(Command("code_review"))
    async def review(message: Message) -> None:
        await answer_review(service, message)

    return router
