"""Protogen Delta package."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("protogen-delta")
except PackageNotFoundError:
    __version__ = "0+unknown"
