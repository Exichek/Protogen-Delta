"""Объединение скачанных AV-потоков без внешних процессов или перекодирования."""

from collections.abc import Iterator
from contextlib import ExitStack
from heapq import merge
from pathlib import Path

import av


def remux_streams(inputs: list[Path], output: Path) -> None:
    """Скопировать пакеты локальных потоков в MP4; сетевые протоколы запрещены."""
    with ExitStack() as stack:
        sources = [
            stack.enter_context(
                av.open(
                    str(path), mode="r", options={"protocol_whitelist": "file,crypto"}
                )
            )
            for path in inputs
        ]
        target = stack.enter_context(
            av.open(str(output), mode="w", options={"movflags": "+faststart"})
        )
        iterators: list[Iterator[av.Packet]] = []
        for source in sources:
            mapping = {
                stream.index: target.add_stream_from_template(stream, opaque=True)
                for stream in source.streams
                if stream.type in {"video", "audio"}
            }

            def packets(
                container: av.container.InputContainer,
                streams: dict[int, av.stream.Stream],
            ) -> Iterator[av.Packet]:
                for packet in container.demux():
                    if packet.dts is not None and packet.stream.index in streams:
                        packet.stream = streams[packet.stream.index]
                        yield packet

            iterators.append(packets(source, mapping))
        for packet in merge(
            *iterators, key=lambda item: float((item.dts or 0) * (item.time_base or 1))
        ):
            target.mux(packet)
