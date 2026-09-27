"""Утилиты для работы с сообщениями Telegram."""

import random

TELEGRAM_MESSAGE_LIMIT = 4096
CONVERSATIONAL_PART_MIN_LENGTH = 60
CONVERSATIONAL_TARGET_PART_LENGTH = 550
CONVERSATIONAL_MAX_PARTS = 4


def split_message(
    text: str,
    limit: int = TELEGRAM_MESSAGE_LIMIT,
) -> list[str]:
    """Разбить длинный текст на части, не превышающие заданный лимит."""
    parts: list[str] = []

    while len(text) > limit:
        cut = text.rfind("\n", 0, limit)

        if cut <= 0:
            cut = limit

        parts.append(text[:cut])
        text = text[cut:].lstrip()

    if text:
        parts.append(text)

    return parts


def split_reply(
    text: str,
    limit: int = TELEGRAM_MESSAGE_LIMIT,
) -> list[str]:
    """Разделить уместный многоабзацный ответ на несколько реплик."""
    clean_text = text.strip()
    hard_parts = split_message(clean_text, limit)
    blocks = _split_blocks(clean_text)
    if len(blocks) < 2:
        return hard_parts
    max_parts = min(
        CONVERSATIONAL_MAX_PARTS,
        max(
            1,
            (len(text) + CONVERSATIONAL_TARGET_PART_LENGTH - 1)
            // CONVERSATIONAL_TARGET_PART_LENGTH,
        ),
    )
    if max_parts == 1:
        return hard_parts

    target_length = max(
        CONVERSATIONAL_TARGET_PART_LENGTH,
        (len(clean_text) + max_parts - 1) // max_parts,
    )
    parts: list[str] = []
    current = ""
    for block in blocks:
        candidate = f"{current}\n\n{block}" if current else block
        can_start_new_part = len(parts) < max_parts - 1
        block_is_tiny = len(
            block
        ) < CONVERSATIONAL_PART_MIN_LENGTH and not block.lstrip().startswith("```")
        should_split = (
            current
            and can_start_new_part
            and len(candidate) > target_length
            and len(current) >= CONVERSATIONAL_PART_MIN_LENGTH
            and not block_is_tiny
        )
        if should_split:
            parts.extend(split_message(current, limit))
            current = block
        else:
            current = candidate

    if current:
        parts.extend(split_message(current, limit))

    return parts if len(parts) > 1 else hard_parts


def reply_delay_seconds(next_part: str) -> float:
    """Выбрать естественную паузу перед следующим фрагментом ответа."""
    length = len(next_part)
    if length <= 160:
        return random.uniform(1.0, 1.6)
    if length <= 500:
        return random.uniform(1.7, 2.4)
    return random.uniform(2.5, 3.3)


def _split_blocks(text: str) -> list[str]:
    """Найти абзацы, не разделяя содержимое fenced code block."""
    blocks: list[str] = []
    current: list[str] = []
    in_code = False

    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_code = not in_code
        if not line.strip() and not in_code:
            if current:
                blocks.append("\n".join(current).strip())
                current = []
            continue
        current.append(line)

    if current:
        blocks.append("\n".join(current).strip())

    return [block for block in blocks if block]
