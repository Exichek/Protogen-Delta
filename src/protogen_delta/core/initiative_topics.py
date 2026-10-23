"""Нейтральные темы инициативы; наружу не отправляются личные рассказы пользователя."""

TOPIC_LABELS = {"games": "игры", "ai": "ИИ", "tech": "техника"}
DEFAULT_TOPICS = tuple(TOPIC_LABELS)
DISCUSSION_KINDS = frozenset("discussion_" + topic for topic in DEFAULT_TOPICS)

DISCUSSION_DIRECTIONS = {
    "games": "игровых механиках, одиночных или кооперативных играх",
    "ai": "пользе, ограничениях или повседневном применении ИИ",
    "tech": "удобстве устройств, программ или домашних технологий",
}
