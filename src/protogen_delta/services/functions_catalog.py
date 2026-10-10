"""Справка пользовательских функций без запроса к модели."""

from protogen_delta.config.settings import Settings
from protogen_delta.core.telegram_commands import commands_for_mode
from protogen_delta.core.user_state import ContentMode


def functions_text(
    mode: ContentMode = "unselected",
    *,
    settings: Settings | None = None,
) -> str:
    """Показать команды текущего режима и фактически подключённые возможности."""
    descriptions = {
        "start": "познакомиться с Дельтой или получить новое приветствие",
        "funcs": "этот список функций и команд",
        "menu": "панель возможностей и настроек",
        "e6": "поиск артов e621/e926 по тегам, с продолжением без повторов",
        "randomart": "случайный арт из сохранённой коллекции, только в режиме 18+",
        "download": (
            "скачать публичное видео или GIF по ссылке: YouTube, Instagram, "
            "TikTok, Vimeo, VK, X/Twitter; до 10 минут и 100 МБ"
        ),
        "source": "ответом на фото — поиск похожих публикаций через SauceNAO; сервису отправляется уменьшенная копия",
        "rp": "/rp off — закончить сцену, сохранив профиль",
        "adult": "выбрать возрастной режим общения и доступный рейтинг артов",
        "id": "твой Telegram ID; ответом на сообщение — ID отправителя; /id @username — доступной группы или канала",
        "memory": "сохранённый профиль фактов; /memory forget <поле> — удалить поле (в личке)",
        "reset": "полностью сбросить диалог и память (с подтверждением)",
        "help": "помощь и этот же список функций",
    }
    if settings is not None and not settings.saucenao_api_key:
        descriptions["source"] = "поиск источника арта; сейчас сервис не подключён"
    lines = ["Возможности Дельты", "", "Команды:"]
    lines.extend(
        f"/{command.command} — {descriptions.get(command.command, command.description)}"
        for command in commands_for_mode(mode)
    )
    lines.extend(
        (
            "",
            (
                "Поиск /e6: укажи теги через пробел, например /e6 dragon order:favcount. "
                "Без выбранного режима 18+ выдаются только safe-арты."
                if mode != "adult"
                else "Поиск /e6: теги через пробел. Рейтинг зависит от запроса и возрастного режима."
            ),
            "",
            "Без команд:",
            "• Тексты, переводы, код, объяснения и планы — напиши задачу обычным сообщением.",
            "• Фото, арты, альбомы, стикеры и кадры GIF/видео — приложи файл и вопрос.",
            "• PDF, DOCX, XLSX, текст и код до 20 МБ — приложи документ для разбора.",
            "• Голосовые и аудиофайлы — распознавание речи.",
            "• RP — начни сцену или действие в *звёздочках*; /rp off завершает её.",
        )
    )
    if settings is not None:
        if settings.brave_search_api_key:
            lines.append("• Поиск в интернете с источниками и чтение публичных ссылок.")
        else:
            lines.append(
                "• Чтение публичных ссылок; отдельный поисковый сервис не подключён."
            )
        if settings.audio_understanding_enabled:
            lines.append("• Музыка и звуки — анализ отдельной аудиомоделью.")
        if settings.pdf_ocr_enabled:
            lines.append("• Сканированные PDF — распознавание текста через OCR.")
        if settings.mini_app_url:
            lines.append(
                "• «Открыть Дельту» — облик по картинке или тексту, "
                "исправление описания и вида, отдельный профиль персонажа."
            )
    return "\n".join(lines)
