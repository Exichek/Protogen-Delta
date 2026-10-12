"""Небольшой график активности за 365 дней, без сетевых запросов."""

from datetime import datetime, timedelta
from io import BytesIO

from PIL import Image, ImageDraw, ImageFont

from protogen_delta.repositories.user_statistics import STATS_ZONE, UserStatistics


def activity_chart(stats: UserStatistics, now: float) -> bytes:
    end = datetime.fromtimestamp(now, STATS_ZONE).date()
    start = end - timedelta(days=364)
    counts = dict(stats.daily)
    values = [
        counts.get((start + timedelta(days=i)).isoformat(), 0) for i in range(365)
    ]
    image = Image.new("RGB", (1000, 400), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=15)
    top, bottom, left, right = 25, 345, 55, 980
    maximum = max(1, max(values))
    for step in range(5):
        y = bottom - (bottom - top) * step / 4
        draw.line((left, y, right, y), fill="#e5e9ef")
        draw.text(
            (10, y - 8), str(round(maximum * step / 4, 1)), font=font, fill="#64748b"
        )
    for i, value in enumerate(values):
        if value:
            x = left + i * (right - left) / 365
            draw.rectangle(
                (x, bottom - (bottom - top) * value / maximum, x + 2, bottom),
                fill="#84aa43",
            )
    for i in (0, 60, 120, 180, 240, 300, 364):
        x = left + i * (right - left) / 365
        draw.text(
            (min(x - 22, right - 48), bottom + 14),
            (start + timedelta(days=i)).strftime("%d.%m"),
            font=font,
            fill="#64748b",
        )
    output = BytesIO()
    image.save(output, "PNG")
    return output.getvalue()
