"""Чтение YouTube: строгие URL, JSON страницы, резервный oEmbed и субтитры."""

import asyncio
import json
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, Mock

import pytest

import protogen_delta.services.tools.http as http
import protogen_delta.services.tools.youtube as youtube
from protogen_delta.services.tools import ToolExecutor, default_registry, fetch_web_page

_ID = "87kiHGcycOs"
_URL = "https://www.youtube.com/watch?v=" + _ID


@pytest.mark.parametrize(
    "url",
    [
        _URL + "&t=25",
        "https://youtu.be/" + _ID,
        "https://m.youtube.com/watch?v=" + _ID,
        "https://music.youtube.com/watch?v=" + _ID,
        "https://www.youtube.com/shorts/" + _ID,
        "http://youtube.com/embed/" + _ID,
        "https://youtube.com/live/" + _ID,
    ],
)
def test_video_url_canonicalization(url: str) -> None:
    assert youtube.youtube_video_id(url) == _ID


@pytest.mark.parametrize(
    "url",
    [
        "https://youtube.com@evil.example/watch?v=" + _ID,
        "https://youtube.com.evil.example/watch?v=" + _ID,
        "https://www.youtube.com/@channel",
        "https://example.org/article",
    ],
)
def test_other_urls_are_not_youtube_video(url: str) -> None:
    assert youtube.youtube_video_id(url) is None


@pytest.mark.parametrize(
    "url",
    [
        "https://user@youtube.com/watch?v=" + _ID,
        "https://youtube.com:8443/watch?v=" + _ID,
        "ftp://youtube.com/watch?v=" + _ID,
        "https://youtube.com/watch?v=short",
        _URL + "&v=abcdefghijk",
        "https://youtu.be/" + _ID + "/extra",
    ],
)
def test_invalid_video_url_rejected(url: str) -> None:
    with pytest.raises(ValueError):
        youtube.youtube_video_id(url)


def _payload(transcript_url: str | None = None) -> dict[str, object]:
    value: dict[str, object] = {
        "videoDetails": {
            "videoId": _ID,
            "title": "Почему Наруто",
            "author": "Кинопоиск",
            "shortDescription": "Описание ролика",
            "lengthSeconds": "1638",
        }
    }
    if transcript_url:
        value["captions"] = {
            "playerCaptionsTracklistRenderer": {
                "captionTracks": [
                    {"baseUrl": transcript_url, "languageCode": "ru", "kind": "asr"}
                ]
            }
        }
    return value


@pytest.mark.parametrize("failure", ["none", "watch", "oembed", "captions"])
def test_reader_preserves_available_evidence(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    payload = _payload("https://www.youtube.com/api/timedtext?v=" + _ID)
    provider = AsyncMock(
        return_value=json.dumps({"title": "Название", "author_name": "Автор"}).encode()
    )

    async def page(url: str, **kwargs: object) -> tuple[bytes, str, str]:
        if "/timedtext" in url:
            if failure == "captions":
                return b"", url, "text/html"
            return (
                json.dumps({"events": [{"segs": [{"utf8": "Моя любимая"}]}]}).encode(),
                url,
                "application/json",
            )
        if failure == "watch":
            raise ValueError("too large")
        return (
            (
                "<script>var ytInitialPlayerResponse = "
                + json.dumps(payload)
                + ";</script>"
            ).encode(),
            url,
            "text/html",
        )

    monkeypatch.setattr(youtube, "fetch_provider", provider)
    monkeypatch.setattr(youtube, "fetch_public_page", page)
    if failure == "oembed":
        provider.side_effect = TimeoutError()
    result = asyncio.run(fetch_web_page({"url": "https://youtu.be/" + _ID}))
    assert result["url"] == _URL
    if failure == "watch":
        assert result["title"] == "Название" and result["author"] == "Автор"
        assert result["content_scope"] == "metadata_only"
    else:
        assert (
            result["title"] == "Почему Наруто" and "Описание ролика" in result["text"]
        )
        assert result["duration_seconds"] == 1638
        assert result["content_scope"] == (
            "metadata_only" if failure == "captions" else "metadata_and_transcript"
        )
        if failure != "captions":
            assert (
                result["transcript"] == "Моя любимая" and result["transcript_automatic"]
            )
    provider.assert_awaited_once()
    assert provider.await_args is not None
    assert provider.await_args.args[0] == "https://www.youtube.com/oembed"


def test_json_player_and_oembed_invalid_data(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(youtube, "fetch_provider", AsyncMock(return_value=b"[]"))
    assert asyncio.run(youtube._oembed(_URL, None)) == {}
    for raw in (
        b"<script>ytInitialPlayerResponse=broken;</script>",
        b'{"ytInitialPlayerResponse":[]}',
        b"<p>no player</p>",
    ):
        monkeypatch.setattr(
            youtube,
            "fetch_public_page",
            AsyncMock(return_value=(raw, _URL, "text/html")),
        )
        assert asyncio.run(youtube._player(_URL, None)) == {}
    player = json.dumps(_payload())
    monkeypatch.setattr(
        youtube,
        "fetch_public_page",
        AsyncMock(
            return_value=(
                ('{"ytInitialPlayerResponse":' + player + "}").encode(),
                _URL,
                "text/html",
            )
        ),
    )
    assert asyncio.run(youtube._player(_URL, None))["videoDetails"]["videoId"] == _ID


def test_unavailable_video_and_invalid_id_do_not_guess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = AsyncMock(
        return_value=({}, {"videoDetails": {"videoId": "other", "title": "wrong"}})
    )
    monkeypatch.setattr(youtube, "_metadata", metadata)
    with pytest.raises(ValueError):
        asyncio.run(youtube.read_youtube_video(_ID))
    with pytest.raises(ValueError):
        asyncio.run(youtube.read_youtube_video("bad"))
    assert metadata.await_count == 1


@pytest.mark.parametrize(
    "url",
    [
        "http://www.youtube.com/api/timedtext",
        "https://localhost/api/timedtext",
        "https://www.youtube.com:8443/api/timedtext",
        "https://u@youtube.com/api/timedtext",
        "https://www.youtube.com/evil",
    ],
)
def test_subtitle_urls_do_not_allow_arbitrary_requests(
    monkeypatch: pytest.MonkeyPatch, url: str
) -> None:
    fetch = AsyncMock()
    monkeypatch.setattr(youtube, "fetch_public_page", fetch)
    with pytest.raises(ValueError):
        asyncio.run(youtube._transcript({"baseUrl": url}, None))
    fetch.assert_not_awaited()


@pytest.mark.parametrize("data", [b"[]", b'{"events":{}}', b"not json"])
def test_invalid_subtitle_data_rejected(data: bytes) -> None:
    with pytest.raises(ValueError):
        youtube._transcript_text(data)


def test_subtitles_skip_control_events_and_repeated_lines() -> None:
    raw = {
        "events": [
            None,
            {"segs": None},
            {"segs": [{"utf8": " hi\n"}, {"utf8": "there"}]},
            {"segs": [{"utf8": "hi there"}]},
            {"segs": [{"utf8": "bye"}, {}]},
        ]
    }
    assert youtube._transcript_text(json.dumps(raw).encode()) == "hi there\nbye"


def test_long_transcript_fits_tool_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _payload("https://www.youtube.com/api/timedtext")
    details = payload["videoDetails"]
    assert isinstance(details, dict)
    details["shortDescription"] = '"' * 6000
    monkeypatch.setattr(youtube, "_metadata", AsyncMock(return_value=({}, payload)))
    monkeypatch.setattr(youtube, "_transcript", AsyncMock(return_value='"' * 16000))
    result = json.loads(
        asyncio.run(
            ToolExecutor(default_registry()).execute(
                "fetch_web_page", json.dumps({"url": _URL})
            )
        )
    )
    assert result["ok"] and result["data"]["truncated"]
    assert len(result["data"]["transcript"]) == 5000
    assert (
        len(
            youtube._transcript_text(
                json.dumps(
                    {
                        "events": [
                            {"segs": [{"utf8": "x" * 9000}]},
                            {"segs": [{"utf8": "not visited"}]},
                        ]
                    }
                ).encode()
            )
        )
        == 9000
    )


@pytest.mark.parametrize("limit", [0, 3 * 1024**2 + 1])
def test_public_page_limit_is_bounded(limit: int) -> None:
    with pytest.raises(ValueError):
        asyncio.run(http.fetch_public_page(_URL, max_bytes=limit))


def test_larger_youtube_page_does_not_raise_generic_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = Mock(status=200, headers={"Content-Type": "text/html"})

    async def chunks(size: int) -> AsyncIterator[bytes]:
        yield b"x" * 1_250_000

    response.content.iter_chunked = chunks
    request = Mock()
    request.__aenter__ = AsyncMock(return_value=response)
    request.__aexit__ = AsyncMock(return_value=False)
    session = Mock()
    session.get.return_value = request
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    monkeypatch.setattr(http, "_session", Mock(return_value=session))
    monkeypatch.setattr(http, "_validate_public_url", AsyncMock())
    assert (
        len(asyncio.run(http.fetch_public_page(_URL, max_bytes=3 * 1024**2))[0])
        == 1_250_000
    )
    with pytest.raises(ValueError):
        asyncio.run(http.fetch_public_page("https://example.org"))
