"""Чтение сведений YouTube и доступных субтитров без загрузки видео."""

import asyncio
import json
import re
from datetime import datetime, timezone
from typing import Any
from urllib.parse import parse_qs, parse_qsl, urlencode, urlsplit, urlunsplit

from protogen_delta.services.tools.http import fetch_provider, fetch_public_page

_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "music.youtube.com",
    "youtu.be",
}
_ID = re.compile(r"[a-zA-Z0-9_-]{11}")
_PLAYER = re.compile(
    r'(?:ytInitialPlayerResponse\s*=|["\']ytInitialPlayerResponse["\']\s*:)\s*'
)
_TRANSCRIPT_CHARS = 8000


def youtube_video_id(url: str) -> str | None:
    """Канонизировать только точные хосты и один ID ролика."""
    parsed = urlsplit(url.strip())
    if parsed.hostname not in _HOSTS:
        return None
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port not in {None, 80 if parsed.scheme == "http" else 443}
    ):
        raise ValueError("Некорректная ссылка YouTube")
    paths = parsed.path.strip("/").split("/")
    if parsed.hostname == "youtu.be":
        value = paths[0] if len(paths) == 1 else ""
    elif paths == ["watch"]:
        values = parse_qs(parsed.query).get("v", [])
        value = values[0] if len(values) == 1 else ""
    elif len(paths) == 2 and paths[0] in {"shorts", "embed", "live"}:
        value = paths[1]
    else:
        return None
    if not _ID.fullmatch(value):
        raise ValueError("Некорректный ID ролика YouTube")
    return value


async def _oembed(url: str, proxy: str | None) -> dict[str, Any]:
    value = json.loads(
        await fetch_provider(
            "https://www.youtube.com/oembed",
            {"url": url, "format": "json"},
            proxy_url=proxy,
        )
    )
    return value if isinstance(value, dict) else {}


async def _player(url: str, proxy: str | None) -> dict[str, Any]:
    raw, _, _ = await fetch_public_page(url, proxy_url=proxy, max_bytes=3 * 1024**2)
    text = raw.decode("utf-8", "replace")
    for match in _PLAYER.finditer(text):
        try:
            offset = match.end()
            value, _ = json.JSONDecoder().raw_decode(text[offset:])
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    return {}


async def _metadata(
    url: str, proxy: str | None
) -> tuple[dict[str, Any], dict[str, Any]]:
    results = await asyncio.gather(
        asyncio.wait_for(_oembed(url, proxy), 6),
        asyncio.wait_for(_player(url, proxy), 6),
        return_exceptions=True,
    )
    return (
        results[0] if isinstance(results[0], dict) else {},
        results[1] if isinstance(results[1], dict) else {},
    )


def _transcript_text(raw: bytes) -> str:
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("Некорректные субтитры")
    parts: list[str] = []
    total = 0
    events = payload.get("events", [])
    if not isinstance(events, list):
        raise ValueError("Некорректные субтитры")
    for event in events:
        if not isinstance(event, dict):
            continue
        segments = event.get("segs", [])
        if not isinstance(segments, list):
            continue
        text = " ".join(
            segment["utf8"]
            for segment in segments
            if isinstance(segment, dict) and isinstance(segment.get("utf8"), str)
        )
        text = " ".join(text.split())
        if text and (not parts or text != parts[-1]):
            parts.append(text)
            total += len(text) + 1
            if total > _TRANSCRIPT_CHARS:
                break
    return "\n".join(parts)


async def _transcript(track: dict[str, Any], proxy: str | None) -> str:
    url = track.get("baseUrl", "")
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname not in {"www.youtube.com", "youtube.com"}
        or parsed.path != "/api/timedtext"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port not in {None, 443}
    ):
        raise ValueError("Некорректный источник субтитров")
    parameters = dict(parse_qsl(parsed.query))
    parameters["fmt"] = "json3"
    url = urlunsplit(
        (parsed.scheme, parsed.netloc, parsed.path, urlencode(parameters), "")
    )
    raw, _, _ = await fetch_public_page(
        url,
        proxy_url=proxy,
        max_redirects=0,
        content_types=frozenset({"application/json", "text/plain", "text/html"}),
    )
    return _transcript_text(raw)


async def read_youtube_video(
    video_id: str, *, proxy_url: str | None = None
) -> dict[str, Any]:
    """Вернуть доказуемые сведения и явно обозначить доступность содержания."""
    if not _ID.fullmatch(video_id):
        raise ValueError("Некорректный ID ролика")
    url = "https://www.youtube.com/watch?v=" + video_id
    oembed, player = await _metadata(url, proxy_url)
    details = player.get("videoDetails", {})
    if not isinstance(details, dict) or details.get("videoId") != video_id:
        details = {}
    title = details.get("title") or oembed.get("title")
    if not isinstance(title, str) or not title.strip():
        raise ValueError("YouTube не отдал сведения о ролике")
    description = str(details.get("shortDescription") or "")
    captions = player.get("captions")
    caption_data = (
        captions.get("playerCaptionsTracklistRenderer")
        if isinstance(captions, dict)
        else None
    )
    raw_tracks = (
        caption_data.get("captionTracks") if isinstance(caption_data, dict) else None
    )
    tracks = (
        [t for t in raw_tracks if isinstance(t, dict)]
        if isinstance(raw_tracks, list)
        else []
    )
    tracks.sort(
        key=lambda t: (
            str(t.get("languageCode")) not in {"ru", "en"},
            t.get("kind") == "asr",
        )
    )
    transcript = ""
    status = "not_available" if not tracks else "unavailable"
    track = tracks[0] if tracks else {}
    if tracks and details:
        try:
            transcript = await asyncio.wait_for(_transcript(track, proxy_url), 3)
            status = "available" if transcript else "unavailable"
        except Exception:
            status = "unavailable"
    author = str(details.get("author") or oembed.get("author_name") or "")[:300]
    text = f"Название: {title[:300]}\nАвтор: {author}"
    if description:
        text += "\nОписание:\n" + description[:3000]
    if transcript:
        text += "\nТекст субтитров находится в поле transcript."
    duration = str(details.get("lengthSeconds") or "")
    result = {
        "url": url,
        "title": title[:300],
        "author": author,
        "duration_seconds": (
            int(duration) if duration.isdigit() and len(duration) <= 9 else None
        ),
        "text": text,
        "transcript": transcript[:_TRANSCRIPT_CHARS],
        "transcript_status": status,
        "transcript_language": (
            str(track.get("languageCode") or "")[:20] if transcript else None
        ),
        "transcript_automatic": track.get("kind") == "asr" if transcript else None,
        "content_scope": "metadata_and_transcript" if transcript else "metadata_only",
        "truncated": len(description) > 3000 or len(transcript) > _TRANSCRIPT_CHARS,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "source": "YouTube",
    }
    if len(json.dumps(result, ensure_ascii=False)) > 15000:
        result["transcript"] = transcript[:5000]
        result["text"] = text[:1500]
        result["truncated"] = True
    return result
