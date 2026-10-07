"""Повторная отмена не бросает незавершённую запись или освобождение ресурса."""

import asyncio

import pytest

from protogen_delta.core.async_completion import finish_operation


def test_repeated_cancel_waits_for_operation_to_finish() -> None:
    async def scenario() -> None:
        entered, release, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def write() -> str:
            entered.set()
            await release.wait()
            finished.set()
            return "done"

        task = asyncio.create_task(finish_operation(write()))
        await entered.wait()
        for _ in range(2):
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert finished.is_set()

    asyncio.run(scenario())


@pytest.mark.parametrize("cancel", [False, True])
def test_operation_failure_is_not_left_running(cancel: bool) -> None:
    async def scenario() -> None:
        entered, release = asyncio.Event(), asyncio.Event()

        async def fail() -> None:
            entered.set()
            await release.wait()
            raise RuntimeError("failed")

        inner = asyncio.create_task(fail())
        outer = asyncio.create_task(finish_operation(inner))
        await entered.wait()
        if cancel:
            outer.cancel()
            await asyncio.sleep(0)
        release.set()
        with pytest.raises(asyncio.CancelledError if cancel else RuntimeError):
            await outer
        assert inner.done()

    asyncio.run(scenario())


def test_cancelled_inner_operation_propagates_cancel() -> None:
    async def scenario() -> None:
        async def wait() -> None:
            await asyncio.Event().wait()

        inner = asyncio.create_task(wait())
        inner.cancel()
        with pytest.raises(asyncio.CancelledError):
            await finish_operation(inner)

    asyncio.run(scenario())
