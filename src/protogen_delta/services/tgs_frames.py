"""Безопасный рендер нескольких кадров Telegram TGS-стикера."""

import gzip
import io
import json
from typing import Any

from rlottie_python import LottieAnimation

from protogen_delta.services.deepseek import ImageInput

MAX_TGS_BYTES = 1024 * 1024
MAX_TGS_JSON_BYTES = 5 * 1024 * 1024
MAX_TGS_DIMENSION = 512


def extract_tgs_frames(
    data: bytes,
    *,
    label: str = "TGS-анимация",
    max_frames: int = 4,
) -> tuple[ImageInput, ...]:
    """Распаковать TGS и отрисовать равномерно выбранные PNG-кадры."""
    if max_frames <= 0:
        raise ValueError("max_frames должен быть больше нуля")
    if not data or len(data) > MAX_TGS_BYTES:
        return ()
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(data)) as compressed:
            raw = compressed.read(MAX_TGS_JSON_BYTES + 1)
        if len(raw) > MAX_TGS_JSON_BYTES:
            return ()
        text = raw.decode("utf-8")
        payload = json.loads(text)
        if not _valid_lottie(payload):
            return ()
        with LottieAnimation.from_data(text) as animation:
            total = max(1, int(animation.lottie_animation_get_totalframe()))
            width, height = animation.lottie_animation_get_size()
            width, height = _bounded_size(int(width), int(height))
            frames: list[ImageInput] = []
            for frame_number in _frame_numbers(total, max_frames):
                rendered = animation.render_pillow_frame(
                    frame_number,
                    width=width,
                    height=height,
                )
                output = io.BytesIO()
                rendered.save(output, format="PNG", optimize=True)
                frames.append(
                    ImageInput(
                        data=output.getvalue(),
                        mime_type="image/png",
                        label=(
                            f"{label}, кадр {len(frames) + 1} " "из последовательности"
                        ),
                    )
                )
            return tuple(frames)
    except (
        OSError,
        EOFError,
        UnicodeError,
        ValueError,
        TypeError,
        RuntimeError,
        json.JSONDecodeError,
    ):
        return ()


def _valid_lottie(payload: Any) -> bool:
    """Отсеять повреждённый или явно не-Lottie JSON до вызова нативного рендера."""
    if not isinstance(payload, dict) or not isinstance(payload.get("layers"), list):
        return False
    # Telegram TGS — самостоятельная векторная анимация. Внешние изображения
    # и шрифты не должны открывать локальные пути или URL из загруженного JSON.
    pending = [payload]
    while pending:
        node = pending.pop()
        if isinstance(node, dict):
            if any(
                isinstance(node.get(key), str) and node[key]
                for key in ("p", "u", "fPath")
            ):
                return False
            pending.extend(node.values())
        elif isinstance(node, list):
            pending.extend(node)
    try:
        width = int(payload["w"])
        height = int(payload["h"])
        first_frame = float(payload.get("ip", 0))
        last_frame = float(payload["op"])
    except KeyError, TypeError, ValueError:
        return False
    return width > 0 and height > 0 and last_frame > first_frame


def _bounded_size(width: int, height: int) -> tuple[int, int]:
    """Сохранить пропорции внутри стандартного размера Telegram-стикера."""
    if width <= 0 or height <= 0:
        raise ValueError("Некорректный размер TGS")
    scale = min(MAX_TGS_DIMENSION / width, MAX_TGS_DIMENSION / height, 1.0)
    return max(1, round(width * scale)), max(1, round(height * scale))


def _frame_numbers(total: int, limit: int) -> tuple[int, ...]:
    """Выбрать начало, промежуточные позиции и последний кадр без дублей."""
    count = min(total, limit)
    if count == 1:
        return (0,)
    return tuple(round(index * (total - 1) / (count - 1)) for index in range(count))
