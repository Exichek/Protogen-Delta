"""Распознавание без AudD: фрагмент, ограниченная сеть, отсутствие совпадения."""

import asyncio
import io
import json
import wave
from typing import Any
from unittest.mock import AsyncMock, Mock

import aiohttp
import pytest
from test_audio_understanding import wav_data
from test_music import _Response

import protogen_delta.config.settings as settings_module
import protogen_delta.services.shazam_music as module
from protogen_delta.services.blocking_work import BlockingWorkPool
from protogen_delta.services.music import MusicRecognitionError, MusicRecognitionService
from protogen_delta.services.shazam_music import (
    ShazamMusicRecognitionService,
    _ShazamHTTPClient,
    create_music_recognizer,
    shazam_report,
)


def match() -> dict[str, Any]:
    return {
        "matches": [{"id": "test"}],
        "track": {
            "title": " Track  Name ",
            "subtitle": " Artist ",
            "sections": [
                None,
                {"type": "OTHER"},
                {"type": "SONG", "metadata": "bad"},
                {
                    "type": "SONG",
                    "metadata": [
                        None,
                        {"title": []},
                        {"title": "Album", "text": "Album"},
                        {"title": "Released", "text": "2026"},
                        {"title": "Lyrics", "text": "ignored"},
                    ],
                },
            ],
        },
    }


def test_only_matching_metadata_is_returned() -> None:
    assert json.loads(shazam_report(match()) or "{}") == {
        "provider": "Shazam",
        "title": "Track Name",
        "artist": "Artist",
        "album": "Album",
        "release_date": "2026",
    }
    assert shazam_report({"matches": [], "track": {"title": "guess"}}) is None
    for result in (
        {},
        {"matches": None},
        {"matches": [1]},
        {"matches": [{}], "track": {"title": "x", "subtitle": " "}},
    ):
        with pytest.raises(MusicRecognitionError):
            shazam_report(result)
    payload = match()
    payload["track"]["title"] = "x" * 500
    payload["track"]["sections"] = None
    assert len(json.loads(shazam_report(payload) or "{}")["title"]) == 200


@pytest.mark.parametrize(
    "url,method",
    [
        ("https://example.com/x", "POST"),
        ("http://amp.shazam.com/x", "POST"),
        ("https://amp.shazam.com:444/x", "POST"),
        ("https://x@amp.shazam.com/x", "POST"),
        ("https://amp.shazam.com/x", "GET"),
    ],
)
def test_transport_does_not_follow_arbitrary_addresses(url: str, method: str) -> None:
    session = Mock()
    with pytest.raises(MusicRecognitionError):
        asyncio.run(_ShazamHTTPClient(session).request(method, url))
    session.post.assert_not_called()


@pytest.mark.parametrize(
    "body,status",
    [(b"[]", 200), (b"bad", 200), (b"x" * 131073, 200), (b"{}", 429)],
    ids=["wrong-type", "bad-json", "oversize", "http-error"],
)
def test_transport_rejects_invalid_or_oversized_response(
    body: bytes, status: int
) -> None:
    session = Mock()
    session.post.return_value = _Response(body, status)
    with pytest.raises((MusicRecognitionError, ValueError)):
        asyncio.run(
            _ShazamHTTPClient(session).request(
                "POST", "https://amp.shazam.com/x", json={}
            )
        )
    session.post.assert_called_once()


def test_transport_has_one_request_and_no_redirects() -> None:
    session = Mock()
    session.post.return_value = _Response(b'{"matches":[]}')
    transport = _ShazamHTTPClient(session)
    assert asyncio.run(
        transport.request("POST", "https://amp.shazam.com/x", json={})
    ) == {"matches": []}
    session.post.assert_called_once_with(
        "https://amp.shazam.com/x", allow_redirects=False, json={}
    )
    with pytest.raises(MusicRecognitionError):
        asyncio.run(
            transport.request_text(
                "https://amp.shazam.com/x", content_type="text/plain"
            )
        )


def patched_sdk(monkeypatch: pytest.MonkeyPatch) -> tuple[AsyncMock, Mock, Mock]:
    sdk = AsyncMock()
    sdk.recognize.return_value = match()
    context = Mock()
    context.__aenter__ = AsyncMock(return_value=sdk)
    context.__aexit__ = AsyncMock(return_value=False)
    constructor = Mock(return_value=context)
    monkeypatch.setattr(module, "Shazam", constructor)
    session = Mock()
    session.__aenter__ = AsyncMock(return_value=Mock())
    session.__aexit__ = AsyncMock(return_value=False)
    monkeypatch.setattr(module.aiohttp, "ClientSession", Mock(return_value=session))
    return sdk, constructor, session


def test_service_limits_audio_and_closes_connections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk, constructor, session = patched_sdk(monkeypatch)
    service = ShazamMusicRecognitionService(native_work=BlockingWorkPool(2))
    report = asyncio.run(service.recognize(wav_data(15)))
    assert json.loads(report or "{}")["provider"] == "Shazam"
    call = sdk.recognize.await_args
    assert call.kwargs == {"max_skew": 1e-3}
    with wave.open(io.BytesIO(call.args[0])) as audio:
        assert audio.getnframes() / audio.getframerate() == 12
    assert isinstance(constructor.call_args.kwargs["http_client"], _ShazamHTTPClient)
    session.__aexit__.assert_awaited_once()


@pytest.mark.parametrize(
    "error",
    [
        aiohttp.ClientError("test"),
        TimeoutError(),
        ValueError("test"),
        MusicRecognitionError("test"),
    ],
)
def test_service_failure_is_independent_and_closes_session(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    sdk, _, session = patched_sdk(monkeypatch)
    sdk.recognize.side_effect = error
    service = ShazamMusicRecognitionService(native_work=BlockingWorkPool(2))
    with pytest.raises(MusicRecognitionError):
        asyncio.run(service.recognize(wav_data()))
    session.__aexit__.assert_awaited_once()


def test_bad_audio_never_creates_network_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, constructor, session = patched_sdk(monkeypatch)
    service = ShazamMusicRecognitionService(native_work=BlockingWorkPool(2))
    with pytest.raises(MusicRecognitionError):
        asyncio.run(service.recognize(b"bad"))
    constructor.assert_not_called()
    session.__aenter__.assert_not_awaited()


def test_no_match_and_cancellation_do_not_leak_slots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        sdk, _, session = patched_sdk(monkeypatch)
        service = ShazamMusicRecognitionService(native_work=BlockingWorkPool(2))
        entered = asyncio.Event()

        async def wait(*args: Any, **kwargs: Any) -> None:
            entered.set()
            await asyncio.Event().wait()

        sdk.recognize.side_effect = wait
        task = asyncio.create_task(service.recognize(wav_data()))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        session.__aexit__.assert_awaited_once()
        sdk.recognize.side_effect = None
        sdk.recognize.return_value = {"matches": []}
        assert await service.recognize(wav_data()) is None

    asyncio.run(run())


def test_provider_selection_never_falls_back_to_audd() -> None:
    work = BlockingWorkPool(2)
    assert isinstance(
        create_music_recognizer("shazam", "unused", work), ShazamMusicRecognitionService
    )
    assert isinstance(
        create_music_recognizer("audd", "test", work), MusicRecognitionService
    )
    assert create_music_recognizer("off", "test", work) is None
    with pytest.raises(ValueError):
        create_music_recognizer("audd", None, work)
    with pytest.raises(ValueError):
        create_music_recognizer("invalid", None, work)


@pytest.mark.parametrize("provider", ["shazam", "audd", "off", "invalid"])
def test_provider_configuration(monkeypatch: pytest.MonkeyPatch, provider: str) -> None:
    monkeypatch.setattr(settings_module, "load_dotenv", lambda: False)
    for key, value in {
        "TELEGRAM_TOKEN": "test",
        "DEEPSEEK_API_KEY": "test",
        "ART_CHAT_ID": "-1",
        "MUSIC_RECOGNITION_PROVIDER": provider,
    }.items():
        monkeypatch.setenv(key, value)
    if provider == "invalid":
        with pytest.raises(ValueError, match="MUSIC_RECOGNITION_PROVIDER"):
            settings_module.load_settings()
    else:
        assert settings_module.load_settings().music_recognition_provider == provider
