"""Тесты извлечения текста из пользовательских документов."""

import io
from types import SimpleNamespace

import pytest
from docx import Document
from openpyxl import Workbook

import protogen_delta.services.documents as documents_module
from protogen_delta.services.documents import (
    DocumentReadError,
    DocumentTooLargeError,
    UnsupportedDocumentError,
    extract_document,
)


def test_extracts_utf8_and_cp1251_text() -> None:
    """Текстовые файлы должны сохранять кириллицу в обычных кодировках."""
    utf8 = extract_document("Привет".encode(), "note.md", "text/markdown")
    cp1251 = extract_document(
        "Проверка".encode("cp1251"),
        "legacy.txt",
        "text/plain",
    )

    assert utf8.text == "Привет"
    assert utf8.kind == "MD"
    assert cp1251.text == "Проверка"

    utf16 = extract_document(
        "Юникод".encode("utf-16"),
        "unicode.txt",
        "text/plain",
    )
    assert utf16.text == "Юникод"


def test_rejects_binary_and_unsupported_documents() -> None:
    """Произвольный бинарный файл не должен маскироваться под текст."""
    with pytest.raises(DocumentReadError, match="бинарный"):
        extract_document(b"abc\x00def", "fake.txt", "text/plain")
    with pytest.raises(UnsupportedDocumentError):
        extract_document(b"data", "archive.zip", "application/zip")
    with pytest.raises(DocumentReadError, match="не найден"):
        extract_document(b"   ", "empty.txt", "text/plain")


def test_rejects_document_over_size_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Общий лимит должен проверяться до разбора формата."""
    monkeypatch.setattr(documents_module, "MAX_DOCUMENT_BYTES", 3)
    with pytest.raises(DocumentTooLargeError):
        extract_document(b"four", "note.txt", "text/plain")


def test_marks_long_text_as_truncated(monkeypatch: pytest.MonkeyPatch) -> None:
    """Большой извлечённый текст должен обрезаться с явной отметкой."""
    monkeypatch.setattr(documents_module, "MAX_DOCUMENT_CHARS", 5)
    result = extract_document(b"123456789", "note.txt", "text/plain")

    assert result.text == "12345"
    assert result.truncated is True


def test_extracts_docx_paragraphs_and_tables() -> None:
    """DOCX должен отдавать и обычный текст, и содержимое таблиц."""
    document = Document()
    document.add_paragraph("Заголовок")
    table = document.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "Имя"
    table.cell(0, 1).text = "Дельта"
    buffer = io.BytesIO()
    document.save(buffer)

    result = extract_document(
        buffer.getvalue(),
        "example.docx",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )

    assert result.kind == "DOCX"
    assert "Заголовок" in result.text
    assert "Имя\tДельта" in result.text


def test_extracts_xlsx_sheets_and_values() -> None:
    """XLSX должен превращать листы и вычисленные значения в текст."""
    workbook = Workbook()
    sheet = workbook.active
    assert sheet is not None
    sheet.title = "Данные"
    sheet.append(["Имя", "Баллы"])
    sheet.append(["Дельта", 42])
    sheet.append([None, "после пустой ячейки"])
    buffer = io.BytesIO()
    workbook.save(buffer)
    workbook.close()

    result = extract_document(
        buffer.getvalue(),
        "scores.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    assert result.kind == "XLSX"
    assert "[Лист: Данные]" in result.text
    assert "Дельта\t42" in result.text
    assert "\tпосле пустой ячейки" in result.text


def test_extracts_pdf_pages(monkeypatch: pytest.MonkeyPatch) -> None:
    """PDF должен сохранять границы страниц в извлечённом тексте."""
    pages = [
        SimpleNamespace(extract_text=lambda: "Первая"),
        SimpleNamespace(extract_text=lambda: "Вторая"),
    ]
    reader = SimpleNamespace(is_encrypted=False, pages=pages)
    monkeypatch.setattr(documents_module, "PdfReader", lambda stream: reader)

    result = extract_document(b"%PDF", "book.pdf", "application/pdf")

    assert result.kind == "PDF"
    assert "[Страница 1]\nПервая" in result.text
    assert "[Страница 2]\nВторая" in result.text


def test_rejects_password_protected_pdf(monkeypatch: pytest.MonkeyPatch) -> None:
    """PDF с неизвестным паролем должен давать понятную ошибку чтения."""
    reader = SimpleNamespace(is_encrypted=True, decrypt=lambda password: 0, pages=[])
    monkeypatch.setattr(documents_module, "PdfReader", lambda stream: reader)

    with pytest.raises(DocumentReadError, match="паролем"):
        extract_document(b"%PDF", "secret.pdf", "application/pdf")


def test_reports_unreadable_office_documents() -> None:
    """Повреждённые архивные документы должны давать стабильную ошибку."""
    with pytest.raises(DocumentReadError, match="DOCX"):
        extract_document(b"broken", "broken.docx", None)
    with pytest.raises(DocumentReadError, match="XLSX"):
        extract_document(b"broken", "broken.xlsx", None)
