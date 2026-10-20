"""Каждая HTTP-попытка модели резервируется отдельно, включая retries SDK."""

import asyncio
import logging
import sqlite3
from typing import Any

from openai import APIConnectionError, APIStatusError, AsyncOpenAI

from protogen_delta.core.async_completion import finish_operation
from protogen_delta.core.llm_budget import current_llm_user
from protogen_delta.repositories.llm_usage import LlmReservation, LlmUsageRepository

logger = logging.getLogger(__name__)


async def record_response_usage(
    repository: LlmUsageRepository | None, receipt: LlmReservation | None, response: Any
) -> None:
    if repository is None or receipt is None:
        return
    usage = getattr(response, "usage", None)

    def number(name: str) -> int | None:
        value = getattr(usage, name, None)
        return value if type(value) is int and 0 <= value < 2**63 else None

    try:
        await finish_operation(
            repository.record_usage(
                receipt, number("prompt_tokens"), number("completion_tokens")
            )
        )
    except sqlite3.Error:
        # The attempt remains charged and its missing usage remains visible.
        logger.warning("LLM token accounting unavailable; attempt retained")


async def accounted_completion(
    client: AsyncOpenAI, repository: LlmUsageRepository | None, **kwargs: Any
) -> tuple[Any, LlmReservation | None]:
    user_id = current_llm_user.get()
    if repository is None:
        return await client.chat.completions.create(**kwargs), None
    retries = client.max_retries
    single_attempt_client = client.with_options(max_retries=0)
    for attempt in range(retries + 1):
        receipt = await repository.reserve(user_id if user_id is not None else 0)
        try:
            return (
                await single_attempt_client.chat.completions.create(**kwargs),
                receipt,
            )
        except (APIConnectionError, APIStatusError) as error:
            retryable = isinstance(error, APIConnectionError) or (
                error.response.headers.get("x-should-retry") == "true"
                or error.response.headers.get("x-should-retry") != "false"
                and (error.status_code in {408, 409, 429} or error.status_code >= 500)
            )
            if attempt == retries or not retryable:
                raise
            delay = min(0.5 * 2**attempt, 8.0)
            if isinstance(error, APIStatusError):
                try:
                    specified = float(error.response.headers.get("retry-after", "0"))
                    if 0 < specified <= 60:
                        delay = specified
                except ValueError:
                    pass
            await asyncio.sleep(delay)
    raise AssertionError("Недостижимая ветка retry")
