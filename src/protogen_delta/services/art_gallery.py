"""Добавление фактически доставленных вложений в общую коллекцию."""

from aiogram.types import Message

from protogen_delta.repositories.images import ImagesRepository


def remember_art(repository: ImagesRepository, message: Message) -> bool:
    """Сохранить Telegram ID доставленного медиа, включая превью вместо оригинала."""
    for kind in ("video", "animation", "document"):
        item = getattr(message, kind)
        if item is not None:
            return repository.add(
                item.file_id, kind=kind, file_unique_id=item.file_unique_id
            )
    if message.photo:
        photo = message.photo[-1]
        return repository.add(photo.file_id, file_unique_id=photo.file_unique_id)
    return False
