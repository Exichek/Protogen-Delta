"""Присылать проверенное публичное изображение только по текущей просьбе владельца."""

import asyncio
import hashlib
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from html.parser import HTMLParser
from io import BytesIO
from urllib.parse import urljoin, urlsplit

from aiogram import Bot
from aiogram.types import BufferedInputFile
from PIL import Image, ImageOps, UnidentifiedImageError

from protogen_delta.core.async_completion import finish_operation
from protogen_delta.services.delivery_ledger import current_delivery
from protogen_delta.services.tools.http import fetch_public_page

logger = logging.getLogger(__name__)
_IMAGE_TYPES = frozenset({"image/jpeg", "image/png", "image/webp", "image/gif"})
_PAGE_TYPES = frozenset({"text/html", "application/xhtml+xml"})
MAX_IMAGE_BYTES = 3 * 1024 * 1024


@dataclass(slots=True)
class ImageTurn:
    owner: int
    receipts: list[str] = field(default_factory=list)
    sent: dict[str, dict[str, object]] = field(default_factory=dict)
    attempts: int = 0
    delivery_uncertain: bool = False


_active: ContextVar[ImageTurn | None] = ContextVar("web_image_turn", default=None)


def web_image_delivery_available() -> bool:
    return _active.get() is not None


class _Images(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.primary: list[str] = []
        self.inline: list[str] = []
        self.title = ""
        self.in_title = False
        self.ignored = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        data = dict(attrs)
        if tag in {"script", "style", "template", "noscript", "svg"}:
            self.ignored += 1
        if self.ignored:
            return
        if tag == "title":
            self.in_title = True
        if tag == "meta" and (
            data.get("property") or data.get("name") or ""
        ).lower() in {
            "og:image",
            "og:image:secure_url",
            "twitter:image",
            "twitter:image:src",
        }:
            if data.get("content"):
                self.primary.append(data["content"] or "")
        if tag == "img" and len(self.inline) < 30:
            url = data.get("data-src") or data.get("src")
            if url and not any(
                word in (url + " " + (data.get("alt") or "")).lower()
                for word in ("logo", "avatar", "icon", "pixel", "tracking")
            ):
                self.inline.append(url)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "template", "noscript", "svg"}:
            self.ignored = max(0, self.ignored - 1)
        if tag == "title":
            self.in_title = False

    def handle_data(self, data: str) -> None:
        if self.in_title and not self.ignored:
            self.title = (self.title + " " + data).strip()[:200]


def _image(data: bytes) -> tuple[bytes, int, int]:
    if not data or len(data) > MAX_IMAGE_BYTES:
        raise ValueError("Недопустимый размер изображения")
    try:
        with Image.open(BytesIO(data)) as source:
            if (
                source.format not in {"JPEG", "PNG", "WEBP", "GIF"}
                or source.width * source.height > 20_000_000
            ):
                raise ValueError("Неподдерживаемое изображение")
            if min(source.size) < 128 or max(source.size) / min(source.size) > 20:
                raise ValueError("Изображение слишком маленькое или узкое")
            source.seek(0)
            source.load()
            picture = ImageOps.exif_transpose(source)
            picture.thumbnail((1920, 1920), Image.Resampling.LANCZOS)
            rgba = picture.convert("RGBA")
            result = Image.new("RGB", picture.size, "white")
            result.paste(rgba, mask=rgba)
            output = BytesIO()
            result.save(output, "JPEG", quality=88)
            prepared = output.getvalue()
            if len(prepared) > MAX_IMAGE_BYTES:
                raise ValueError("Изображение слишком большое")
            return prepared, result.width, result.height
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise ValueError("Не удалось прочитать изображение") from error


async def load_web_image(
    url: str, *, proxy_url: str | None = None
) -> tuple[bytes, str, str, int, int]:
    """Только ограниченная публичная сеть; HTML не принимается за картинку."""
    async with asyncio.timeout(9):
        raw, source, content_type = await fetch_public_page(
            url,
            proxy_url=proxy_url,
            max_bytes=MAX_IMAGE_BYTES,
            content_types=_PAGE_TYPES | _IMAGE_TYPES,
        )
        if content_type in _IMAGE_TYPES:
            image, width, height = await asyncio.to_thread(_image, raw)
            return image, source, source, width, height
        parser = _Images()
        parser.feed(raw.decode("utf-8", "replace"))
        urls = []
        for value in parser.primary + parser.inline:
            candidate = urljoin(source, value.strip())
            parsed = urlsplit(candidate)
            if (
                parsed.scheme in {"http", "https"}
                and parsed.hostname
                and not parsed.username
                and not parsed.password
                and len(candidate) <= 2048
                and candidate not in urls
            ):
                urls.append(candidate)
            if len(urls) >= 3:
                break
        for candidate in urls:
            try:
                data, final, kind = await fetch_public_page(
                    candidate,
                    proxy_url=proxy_url,
                    max_bytes=MAX_IMAGE_BYTES,
                    content_types=_IMAGE_TYPES,
                )
                image, width, height = await asyncio.to_thread(_image, data)
                return image, source, parser.title, width, height
            except ValueError, OSError, TimeoutError:
                continue
    raise ValueError("На странице нет подходящего изображения")


class WebImages:
    def __init__(self, bot: Bot, *, proxy_url: str | None = None) -> None:
        self._bot, self._proxy = bot, proxy_url
        self._slots = asyncio.Semaphore(2)

    @contextmanager
    def turn(self, owner: int, *, allowed: bool) -> Iterator[ImageTurn]:
        turn = ImageTurn(owner)
        token = _active.set(turn if allowed and owner > 0 else None)
        try:
            yield turn
        finally:
            _active.reset(token)

    async def send_tool(self, args: dict[str, str]) -> dict[str, object]:
        turn = _active.get()
        if turn is None or turn.delivery_uncertain or turn.attempts >= 3:
            return {"status": "not_sent", "reason": "image_delivery_not_available"}
        turn.attempts += 1
        try:
            async with self._slots:
                data, source, title, width, height = await load_web_image(
                    args["url"], proxy_url=self._proxy
                )
        except Exception:
            return {"status": "not_sent", "reason": "public_image_unavailable"}
        fingerprint = hashlib.sha256(data).hexdigest()
        if fingerprint in turn.sent:
            return {**turn.sent[fingerprint], "already_sent": True}
        title = " ".join(title.split())[:180]
        caption = (
            "Изображение со страницы"
            + (": " + title if title and title != source else "")
            + "\nИсточник: "
            + source
        )
        if len(caption) > 1000:
            caption = "Изображение со страницы: " + (
                urlsplit(source).hostname or "источник"
            )
        # После неопределённой доставки повторов в этом ходе не будет.
        turn.delivery_uncertain = True
        try:

            async def deliver() -> None:
                operation = current_delivery.get()
                if operation is not None:
                    operation.answering = operation.reply_started = True
                await self._bot.send_photo(
                    turn.owner,
                    BufferedInputFile(data, filename="web-image.jpg"),
                    caption=caption,
                    parse_mode=None,
                    request_timeout=8,
                )
                turn.receipts.append("Подтверждённая отправка фото: " + caption)

            await finish_operation(deliver())
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("Отправка веб-изображения не подтверждена")
            return {"status": "not_sent", "reason": "telegram_delivery_unconfirmed"}
        turn.delivery_uncertain = False
        result: dict[str, object] = {
            "status": "sent",
            "source_url": source,
            "width": width,
            "height": height,
            "notice": (
                "Реальный файл отправлен в текущий личный чат. "
                "Содержимое изображения модель не анализировала; не выдумывай детали."
            ),
        }
        turn.sent[fingerprint] = result
        return result
