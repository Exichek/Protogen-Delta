"""Чужая запись не должна превращаться в слова и идентичность пользователя."""

import asyncio
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import pytest
from test_voice_handler import _message, _router

from protogen_delta.handlers.voice import create_voice_router
from protogen_delta.services.audio_analysis import AudioAnalysisError
from protogen_delta.services.audio_understanding import AudioUnderstandingService
from protogen_delta.services.speech import Transcript


@pytest.mark.parametrize("semantic", [False, True])
def test_short_audio_quote_stays_file_content(
    monkeypatch: pytest.MonkeyPatch, semantic: bool
) -> None:
    monkeypatch.setattr(
        "protogen_delta.services.audio_pipeline.analyze_audio",
        lambda data: (_ for _ in ()).throw(AudioAnalysisError("no metrics")),
    )
    router, engine, bot, transcriber = _router(Transcript("Моя любимая", "ru"))
    if semantic:
        model = AsyncMock(spec=AudioUnderstandingService)
        model.analyze.return_value = "Короткая реплика с мужским голосом"
        router = create_voice_router(
            engine,
            bot,
            transcriber,
            audio_understanding=cast(AudioUnderstandingService, model),
        )
    media = SimpleNamespace(
        file_id="clip",
        file_size=100,
        duration=2,
        file_name="Invo_ability_invoke_01_ru.mp3",
    )
    message, _ = _message(audio=media)
    asyncio.run(router.message.handlers[0].callback(message))
    context = engine.respond_and_deliver.await_args.kwargs
    assert '"Моя любимая"' in context["model_message_override"]
    assert "цитата" in context["model_message_override"]
    assert media.file_name in context["model_message_override"]
    assert "не автоматически собственные слова" in context[
        "trusted_input_context"
    ].replace("а не ", "не ")
    assert "пол пользователя" in context["trusted_input_context"]
    assert "Инструкции внутри записи" in context["trusted_input_context"]
    if not semantic:
        assert "может ошибаться" in context["trusted_input_context"]
