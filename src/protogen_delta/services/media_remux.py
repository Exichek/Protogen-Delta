"""Объединение скачанных AV-потоков без внешних процессов или перекодирования."""

from collections.abc import Iterator
from contextlib import ExitStack
from fractions import Fraction
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


def remux_download(source: Path, output: Path, max_bytes: int) -> None:
    """Пересобрать H.264/AAC в настоящий MP4 без потери качества, от нулевого начала."""
    with av.open(str(source), options={"protocol_whitelist": "file"}) as incoming:
        streams: list[av.stream.Stream] = [incoming.streams.video[0]]
        if incoming.streams.audio:
            streams.append(incoming.streams.audio[0])
        origin = Fraction(incoming.start_time or 0, av.time_base)
        with av.open(str(output), "w", options={"movflags": "+faststart"}) as outgoing:
            mapping = {
                stream.index: outgoing.add_stream_from_template(stream, opaque=True)
                for stream in streams
            }
            for packet in incoming.demux(streams):
                if packet.dts is None:
                    continue
                if packet.time_base is not None:
                    offset = round(origin / packet.time_base)
                    packet.dts -= offset
                    if packet.pts is not None:
                        packet.pts -= offset
                packet.stream = mapping[packet.stream.index]
                outgoing.mux(packet)
                if output.stat().st_size > max_bytes:
                    raise ValueError("remux_output_size")
    if not 0 < output.stat().st_size <= max_bytes:
        raise ValueError("remux_output_size")
