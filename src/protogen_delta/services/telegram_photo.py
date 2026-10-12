"""Подготовить статичное изображение под ограничения sendPhoto Telegram."""

import math
from io import BytesIO

from PIL import Image, ImageOps, UnidentifiedImageError

from protogen_delta.services.e621 import E621Error

_MAX_BYTES = 9 * 1024 * 1024
_MAX_PIXELS = 40_000_000


def prepare_photo(data: bytes) -> bytes:
    """Ограничить размеры и пропорции, сохранить весь кадр и прозрачность на белом."""
    try:
        with Image.open(BytesIO(data)) as source:
            width, height = source.size
            if width * height > _MAX_PIXELS or source.format not in {
                "JPEG",
                "PNG",
                "WEBP",
            }:
                raise E621Error(
                    "Изображение слишком большое или неподдерживаемого формата."
                )
            source.load()
            if (
                source.format == "JPEG"
                and not source.getexif().get(274)
                and len(data) <= _MAX_BYTES
                and width + height <= 9800
                and max(width, height) / min(width, height) <= 20
            ):
                return data
            image = ImageOps.exif_transpose(source)
            scale = min(1.0, 8192 / max(image.size), 9300 / sum(image.size))
            image.thumbnail(
                (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
                Image.Resampling.LANCZOS,
            )
            rgba = image.convert("RGBA")
            target = Image.new(
                "RGB",
                (
                    max(image.width, math.ceil(image.height / 20)),
                    max(image.height, math.ceil(image.width / 20)),
                ),
                "white",
            )
            target.paste(
                rgba,
                (
                    (target.width - image.width) // 2,
                    (target.height - image.height) // 2,
                ),
                rgba,
            )
            for quality in (92, 82, 70):
                output = BytesIO()
                target.save(output, "JPEG", quality=quality)
                if output.tell() <= _MAX_BYTES:
                    return output.getvalue()
            raise E621Error("Не удалось уменьшить изображение для Telegram.")
    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        Image.DecompressionBombError,
    ) as error:
        raise E621Error("Не удалось подготовить изображение для Telegram.") from error
