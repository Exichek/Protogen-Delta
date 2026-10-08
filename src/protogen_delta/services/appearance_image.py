"""Проверка и подготовка загруженного референса внешности."""

import base64
from io import BytesIO

from PIL import Image, ImageOps, UnidentifiedImageError

from protogen_delta.services.deepseek import ImageInput

MAX_APPEARANCE_BYTES = 20 * 1024 * 1024


def prepare_appearance_image(data: bytes) -> ImageInput:
    """Убрать метаданные, сохранить весь кадр и ограничить изображение для vision."""
    if not data or len(data) > MAX_APPEARANCE_BYTES:
        raise ValueError("Нужна картинка размером до 20 МБ.")
    try:
        with Image.open(BytesIO(data)) as source:
            if source.format not in {"JPEG", "PNG", "WEBP"}:
                raise ValueError("Поддерживаются JPEG, PNG и WebP.")
            if source.width * source.height > 40_000_000:
                raise ValueError("Картинка слишком большая: максимум 40 мегапикселей.")
            source.load()
            image = ImageOps.exif_transpose(source)
            image.thumbnail((1536, 1536), Image.Resampling.LANCZOS)
            rgba = image.convert("RGBA")
            result = Image.new("RGB", image.size, "white")
            result.paste(rgba, mask=rgba)
            output = BytesIO()
            result.save(output, "JPEG", quality=90)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise ValueError(
            "Не удалось прочитать картинку. Выбери JPEG, PNG или WebP."
        ) from error
    return ImageInput(output.getvalue(), "image/jpeg", "референс облика из Mini App")


def prepare_appearance_upload(data: bytes) -> tuple[ImageInput, str]:
    """Подготовить vision-референс и маленькую JPEG-миниатюру без метаданных."""
    image = prepare_appearance_image(data)
    with Image.open(BytesIO(image.data)) as thumbnail:
        thumbnail.thumbnail((256, 256), Image.Resampling.LANCZOS)
        output = BytesIO()
        thumbnail.save(output, "JPEG", quality=65)
    data = output.getvalue()
    if len(data) > 64 * 1024:
        raise ValueError("Не удалось подготовить миниатюру картинки.")
    return image, "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii")
