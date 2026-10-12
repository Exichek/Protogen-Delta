"""Локальный OCR целых PDF-страниц, без сетевых запросов."""

import io
import subprocess
from typing import Any, cast

from PIL import Image


def render_pdf_page(document: Any, index: int) -> Image.Image:
    """Рендерить целую страницу, учитывая поворот, слои и несколько изображений."""
    page = document[index]
    try:
        width, height = page.get_size()
        scale = min(2.0, 2048 / max(width, height))
        bitmap = page.render(scale=scale)
        try:
            return cast(Image.Image, bitmap.to_pil().convert("RGB"))
        finally:
            bitmap.close()
    finally:
        page.close()


def recognize_page(image: Image.Image) -> str:
    """Tesseract rus+eng; один процесс и таймаут на страницу."""
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    result = subprocess.run(
        ["tesseract", "stdin", "stdout", "-l", "rus+eng", "--psm", "3"],
        input=buffer.getvalue(),
        capture_output=True,
        timeout=15,
        check=True,
    )
    return result.stdout.decode("utf-8", errors="replace").strip()
