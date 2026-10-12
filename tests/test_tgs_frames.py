"""Тесты локального рендера Telegram TGS-стикеров."""

import gzip
import json

import pytest

import protogen_delta.services.tgs_frames as tgs_module
from protogen_delta.services.tgs_frames import (
    _bounded_size,
    _frame_numbers,
    _valid_lottie,
    extract_tgs_frames,
)


def _tgs() -> bytes:
    payload = {
        "v": "5.5.7",
        "fr": 30,
        "ip": 0,
        "op": 12,
        "w": 64,
        "h": 64,
        "nm": "test",
        "ddd": 0,
        "assets": [],
        "layers": [],
    }
    return gzip.compress(json.dumps(payload).encode())


def test_extract_tgs_frames_returns_ordered_png_sequence() -> None:
    frames = extract_tgs_frames(_tgs(), max_frames=4)

    assert len(frames) == 4
    assert all(frame.mime_type == "image/png" for frame in frames)
    assert all(frame.data.startswith(b"\x89PNG\r\n\x1a\n") for frame in frames)
    assert frames[0].label == "TGS-анимация, кадр 1 из последовательности"
    assert frames[-1].label == "TGS-анимация, кадр 4 из последовательности"


def test_extract_tgs_frames_rejects_invalid_and_oversized_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert extract_tgs_frames(b"not-tgs") == ()
    assert extract_tgs_frames(gzip.compress(b"{}")) == ()
    monkeypatch.setattr(tgs_module, "MAX_TGS_BYTES", 2)
    assert extract_tgs_frames(_tgs()) == ()
    with pytest.raises(ValueError, match="больше нуля"):
        extract_tgs_frames(_tgs(), max_frames=0)


def test_extract_tgs_frames_limits_unpacked_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tgs_module, "MAX_TGS_JSON_BYTES", 10)
    assert extract_tgs_frames(_tgs()) == ()


def test_tgs_validation_and_frame_helpers_cover_boundaries() -> None:
    assert _valid_lottie({"layers": [], "w": 1, "h": 1}) is False
    assert _frame_numbers(1, 4) == (0,)
    assert _frame_numbers(3, 4) == (0, 1, 2)
    with pytest.raises(ValueError, match="размер"):
        _bounded_size(0, 512)
