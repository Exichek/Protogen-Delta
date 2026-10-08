"""Локальный worker: только фиксированные операции и JSON, без pickle/RPC-кода."""

import base64
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

MAX_OUTPUT = 16 * 1024 * 1024


def apply_limits() -> None:
    if sys.platform != "win32":
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (1024**3, 1024**3))
        resource.setrlimit(resource.RLIMIT_CPU, (60, 60))
        resource.setrlimit(resource.RLIMIT_FSIZE, (20 * 1024 * 1024, 20 * 1024 * 1024))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def perform(data: bytes, request: dict[str, Any]) -> dict[str, Any]:
    from protogen_delta.services.deepseek import ImageInput
    from protogen_delta.services.documents import (
        DocumentReadError,
        DocumentTooLargeError,
        ExtractedImage,
        UnsupportedDocumentError,
        extract_document,
    )

    images: tuple[ExtractedImage, ...] | tuple[ImageInput, ...]
    operation = request["operation"]
    args, kwargs = request["args"], request["kwargs"]
    try:
        if operation == "audio":
            from protogen_delta.services.audio_analysis import analyze_audio

            return {"analysis": asdict(analyze_audio(data))}
        if operation == "tags":
            from protogen_delta.services.music import audio_tags

            return {"tags": audio_tags(data)}
        if operation == "clip":
            from protogen_delta.services.audio_understanding import audio_excerpt

            return {
                "clip": base64.b64encode(audio_excerpt(data, *args, **kwargs)).decode(
                    "ascii"
                )
            }
        if operation == "document":
            result = extract_document(data, *args, **kwargs)
            value: dict[str, Any] = {
                "text": result.text,
                "kind": result.kind,
                "truncated": result.truncated,
            }
            images = result.images
        elif operation == "animation":
            from protogen_delta.services.animation_frames import (
                extract_animation_frames,
            )

            images = extract_animation_frames(data, *args, **kwargs)
            value = {}
        elif operation == "tgs":
            from protogen_delta.services.tgs_frames import extract_tgs_frames

            images = extract_tgs_frames(data, *args, **kwargs)
            value = {}
        elif operation == "appearance":
            from protogen_delta.services.appearance_image import (
                prepare_appearance_image,
            )

            images = (prepare_appearance_image(data),)
            value = {}
        else:
            raise ValueError("operation")
        value["images"] = [
            {
                "data": base64.b64encode(i.data).decode("ascii"),
                "mime_type": i.mime_type,
                "label": i.label,
            }
            for i in images
        ]
        return value
    except DocumentTooLargeError:
        return {"error": "too_large"}
    except UnsupportedDocumentError:
        return {"error": "unsupported"}
    except DocumentReadError:
        return {"error": "read_error"}
    except MemoryError:
        return {"error": "too_large"}
    except Exception:
        return {"error": "failed"}


def main() -> None:
    apply_limits()
    source, target, request = map(Path, sys.argv[1:])
    if (
        not 0 < source.stat().st_size <= 20 * 1024 * 1024
        or request.stat().st_size > 8192
    ):
        raise ValueError("input_size")
    parsed_request = json.loads(request.read_text("utf-8"))
    if parsed_request.get("operation") == "speech":
        from protogen_delta.services.speech_audio import prepare_speech_audio

        try:
            pcm = prepare_speech_audio(source.read_bytes())
            if len(pcm) > 20 * 1024 * 1024:
                raise ValueError("speech_size")
            target.write_bytes(pcm)
        except Exception:
            target.write_bytes(b'{"error":"failed"}')
        return
    payload = json.dumps(
        perform(source.read_bytes(), json.loads(request.read_text("utf-8"))),
        ensure_ascii=False,
    ).encode("utf-8")
    if len(payload) > MAX_OUTPUT:
        payload = b'{"error":"too_large"}'
    target.write_bytes(payload)


if __name__ == "__main__":
    main()
