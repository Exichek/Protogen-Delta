"""Локальный PCM -> JSON; одна модель на процесс, без ключей и текста в логах."""

import io
import json
import os
import sys
import wave
from dataclasses import asdict
from pathlib import Path
from typing import Any

from protogen_delta.services.speech import (
    MAX_AUDIO_BYTES,
    SpeechRecognitionError,
    SpeechTranscriber,
)


def apply_limits(device: str) -> None:
    if sys.platform != "win32":
        import resource

        if device == "cpu":
            resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def load_model(config: dict[str, str]) -> Any:
    from faster_whisper import WhisperModel  # type: ignore[import-untyped]

    return WhisperModel(
        config["model_size"],
        device=config["device"],
        compute_type=config["compute_type"],
        cpu_threads=2,
        num_workers=1,
    )


class WhisperWorker:
    def __init__(self, config: dict[str, str]) -> None:
        self._config = config
        self._model: Any | None = None

    def reply(self, data: bytes, sequence: int) -> dict[str, Any]:
        try:
            if not 0 < len(data) <= MAX_AUDIO_BYTES:
                raise ValueError("size")
            with wave.open(io.BytesIO(data), "rb") as wav:
                if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (
                    1,
                    2,
                    16000,
                ) or not 0 < wav.getnframes() <= 600 * 16000:
                    raise ValueError("pcm")
                if len(wav.readframes(wav.getnframes())) != wav.getnframes() * 2:
                    raise ValueError("frames")
        except Exception:
            return {"sequence": sequence, "error": "recognition"}
        if self._model is None:
            try:
                self._model = load_model(self._config)
            except Exception:
                return {"sequence": sequence, "error": "load"}
        try:
            result = SpeechTranscriber._transcribe_sync(self._model, data)
            return {"sequence": sequence, **asdict(result)}
        except SpeechRecognitionError as error:
            return {
                "sequence": sequence,
                "error": (
                    "no_speech"
                    if str(error) == "В аудио не удалось распознать речь"
                    else "recognition"
                ),
            }
        except Exception:
            return {"sequence": sequence, "error": "recognition"}


def main() -> None:
    config_path, directory = map(Path, sys.argv[1:])
    if config_path.stat().st_size > 8192:
        raise ValueError("config")
    config = json.loads(config_path.read_text("utf-8"))
    if (
        not isinstance(config, dict)
        or set(config) != {"model_size", "device", "compute_type"}
        or any(
            not isinstance(v, str) or not 0 < len(v) <= 1024 for v in config.values()
        )
    ):
        raise ValueError("config")
    apply_limits(config["device"])
    protocol_fd = os.dup(sys.stdout.fileno())
    with open(os.devnull, "w") as silent:
        os.dup2(silent.fileno(), sys.stdout.fileno())
    with os.fdopen(protocol_fd, "w", encoding="utf-8", buffering=1) as output:
        worker = WhisperWorker(config)
        while True:
            line = sys.stdin.buffer.readline(4097)
            if not line:
                return
            request = json.loads(line)
            if (
                len(line) > 4096
                or not isinstance(request, dict)
                or set(request) != {"sequence"}
                or type(request["sequence"]) is not int
                or request["sequence"] <= 0
            ):
                raise ValueError("request")
            source = directory / "audio.wav"
            if not 0 < source.stat().st_size <= MAX_AUDIO_BYTES:
                raise ValueError("size")
            data = source.read_bytes()
            reply = worker.reply(data, request["sequence"])
            source.unlink(missing_ok=True)
            payload = json.dumps(reply, ensure_ascii=False)
            if len(payload.encode("utf-8")) >= 128 * 1024:
                payload = json.dumps(
                    {"sequence": request["sequence"], "error": "recognition"}
                )
            output.write(payload + "\n")
            del data, reply


if __name__ == "__main__":
    main()
