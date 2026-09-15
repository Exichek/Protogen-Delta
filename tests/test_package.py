"""Тесты метаданных пакета."""

from importlib.metadata import version

import protogen_delta


def test_package_version_matches_metadata() -> None:
    """Версия пакета должна браться из установленных метаданных."""
    assert protogen_delta.__version__ == version("protogen-delta")
