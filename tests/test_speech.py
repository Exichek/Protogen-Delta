"""Тесты локального распознавания речи."""

import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import protogen_delta.services.speech as speech_module
from protogen_delta.services.speech import (
    SpeechRecognitionError,
    SpeechTranscriber,
)


def test_transcriber_collects_segments_and_language() -> None:
    """Whisper-сегменты должны объединяться в одну чистую расшифровку."""
    model = Mock()
    model.transcribe.return_value = (
        iter(
            [
                SimpleNamespace(text=" Привет "),
                SimpleNamespace(text=""),
                SimpleNamespace(text="как дела?"),
            ]
        ),
        SimpleNamespace(language="ru"),
    )
    result = SpeechTranscriber._transcribe_sync(model, b"audio")

    assert result.text == "Привет как дела?"
    assert result.language == "ru"
    model.transcribe.assert_called_once()
    assert model.transcribe.call_args.kwargs == {
        "beam_size": 5,
        "vad_filter": True,
        "condition_on_previous_text": False,
    }


def test_transcriber_rejects_empty_and_large_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Некорректный размер должен отклоняться до загрузки модели."""
    transcriber = SpeechTranscriber()
    with pytest.raises(SpeechRecognitionError, match="пуст"):
        asyncio.run(transcriber.transcribe(b""))

    monkeypatch.setattr(speech_module, "MAX_AUDIO_BYTES", 3)
    with pytest.raises(SpeechRecognitionError, match="20 МБ"):
        asyncio.run(transcriber.transcribe(b"four"))


def test_transcriber_reports_missing_speech() -> None:
    """Тишина не должна превращаться в пустой запрос к Дельте."""
    model = Mock()
    model.transcribe.return_value = (iter([]), SimpleNamespace(language=None))
    with pytest.raises(SpeechRecognitionError, match="речь"):
        SpeechTranscriber._transcribe_sync(model, b"silence")


def test_transcriber_limits_long_transcript(monkeypatch: pytest.MonkeyPatch) -> None:
    """Очень длинная расшифровка должна иметь ограниченный размер."""
    monkeypatch.setattr(speech_module, "MAX_TRANSCRIPT_CHARS", 5)
    model = Mock()
    model.transcribe.return_value = (
        iter([SimpleNamespace(text="123456789")]),
        SimpleNamespace(language="ru"),
    )
    result = SpeechTranscriber._transcribe_sync(model, b"audio")

    assert result.text == "12345"
    assert result.truncated is True


def test_model_is_loaded_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Несколько распознаваний должны переиспользовать одну Whisper-модель."""
    model = Mock()
    model.transcribe.side_effect = lambda *args, **kwargs: (
        iter([SimpleNamespace(text="текст")]),
        SimpleNamespace(language="ru"),
    )
    from test_audio_understanding import wav_data

    import protogen_delta.services.whisper_worker as worker_module
    from protogen_delta.services.speech_audio import prepare_speech_audio

    loader = Mock(return_value=model)
    monkeypatch.setattr(worker_module, "load_model", loader)
    worker = worker_module.WhisperWorker(
        {"model_size": "tiny", "device": "cpu", "compute_type": "int8"}
    )
    data = prepare_speech_audio(wav_data())
    assert worker.reply(data, 1)["text"] == "текст"
    assert worker.reply(data, 2)["text"] == "текст"
    loader.assert_called_once()


def test_inference_error_is_translated() -> None:
    """Ошибка декодера должна превращаться в доменное исключение."""
    model = Mock()
    model.transcribe.side_effect = RuntimeError("decoder failed")
    with pytest.raises(SpeechRecognitionError, match="распознать"):
        SpeechTranscriber._transcribe_sync(model, b"audio")


@pytest.mark.parametrize(
    "score,silence,uncertain",
    [(-0.4, 0.05, False), (-1.4, 0.1, True), (-0.4, 0.8, True)],
)
def test_decoder_uncertainty_is_preserved_without_rewriting_words(
    score: float, silence: float, uncertain: bool
) -> None:
    model = Mock()
    model.transcribe.return_value = (
        iter(
            [
                SimpleNamespace(
                    text="Моя любимая", avg_logprob=score, no_speech_prob=silence
                )
            ]
        ),
        SimpleNamespace(language="ru"),
    )
    result = SpeechTranscriber._transcribe_sync(model, b"audio")
    assert result.text == "Моя любимая"
    assert result.uncertain is uncertain
