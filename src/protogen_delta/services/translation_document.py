"""Скачать перевод как DOCX без хранения пользовательских текстов."""

import io
import re

from docx import Document

from protogen_delta.services.translation import MAX_TRANSLATION_OUTPUT_CHARS


def make_translated_docx(data: bytes) -> bytes:
    text = data.decode("utf-8")
    if not text.strip() or len(text) > MAX_TRANSLATION_OUTPUT_CHARS:
        raise ValueError("Некорректный размер перевода.")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\ud800-\udfff\ufffe\uffff]", "", text)
    document = Document()
    for paragraph in text.split("\n"):
        document.add_paragraph(paragraph)
    output = io.BytesIO()
    document.save(output)
    return output.getvalue()
