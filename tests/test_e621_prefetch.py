"""Подготовка двух медиа сохраняет порядок и не оставляет задачи после остановки."""

import asyncio
from pathlib import Path
from typing import cast

import pytest
from test_e621 import _JPEG, _call_message, _Client, _message, _post
from test_e621_albums import _button

import protogen_delta.handlers.e621 as handler
from protogen_delta.core.user_state import UserStateStore
from protogen_delta.repositories.e621_history import E621HistoryRepository
from protogen_delta.services.e621 import E621Client, E621Error, E621Post
from protogen_delta.services.e621_media import E621MediaService, PreparedMedia


@pytest.mark.parametrize("failed", [False, True])
def test_prefetch_overlaps_two_jobs_and_preserves_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failed: bool
) -> None:
    async def scenario() -> None:
        entered: list[int] = []
        finished: list[int] = []
        second_started = asyncio.Event()
        active = peak = 0

        async def prepare(
            self: E621MediaService, post: E621Post, **kwargs: object
        ) -> PreparedMedia:
            nonlocal active, peak
            entered.append(post.post_id)
            active += 1
            peak = max(peak, active)
            try:
                if post.post_id == 1:
                    await asyncio.wait_for(second_started.wait(), 1)
                    await asyncio.sleep(0)
                    if failed:
                        raise E621Error("unavailable")
                if post.post_id == 2:
                    second_started.set()
                finished.append(post.post_id)
                return PreparedMedia(post, str(post.post_id), "jpg", _JPEG)
            finally:
                active -= 1

        monkeypatch.setattr(E621MediaService, "prepare", prepare)
        history = E621HistoryRepository(tmp_path)
        router = handler.create_e621_router(
            cast(E621Client, _Client([_post(i) for i in range(1, 6)])),
            history,
            UserStateStore(),
        )
        message, raw = _message("/e6 dragon count:5")
        await _call_message(router, 0, message)
        assert entered == [1, 2, 3, 4, 5]
        assert (
            finished.index(2) < finished.index(1) if not failed else 1 not in finished
        )
        assert peak == 2 and active == 0
        items = raw.answer_media_group.await_args.args[0]
        assert [item.caption.split("#")[1].split(" ")[0] for item in items] == [
            str(i) for i in range(2 if failed else 1, 6)
        ]
        assert await history.seen_ids(7) == set(range(2 if failed else 1, 6))

    asyncio.run(scenario())


@pytest.mark.parametrize("stop", [False, True])
def test_timeout_or_stop_cancels_prefetch_and_keeps_unfinished_posts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stop: bool
) -> None:
    async def scenario() -> None:
        active: set[int] = set()
        ready = asyncio.Event()
        release = asyncio.Event()
        blocked = True

        async def prepare(
            self: E621MediaService, post: E621Post, **kwargs: object
        ) -> PreparedMedia:
            active.add(post.post_id)
            try:
                if post.post_id == 2:
                    ready.set()
                if blocked and (stop or post.post_id == 1):
                    await release.wait()
                return PreparedMedia(post, str(post.post_id), "jpg", _JPEG)
            finally:
                active.remove(post.post_id)

        monkeypatch.setattr(E621MediaService, "prepare", prepare)
        monkeypatch.setattr(handler, "_BATCH_SECONDS", 0.05 if not stop else 10)
        history = E621HistoryRepository(tmp_path)
        router = handler.create_e621_router(
            cast(E621Client, _Client([_post(1), _post(2)])), history, UserStateStore()
        )
        message, raw = _message("/e6 dragon count:2")
        task = asyncio.create_task(_call_message(router, 0, message))
        await asyncio.wait_for(ready.wait(), 1)
        if stop:
            await router.callback_query.handlers[1].callback(_button(raw, "stop"))
        await asyncio.wait_for(task, 1)
        assert not active
        assert await history.seen_ids(7) == (set() if stop else {2})
        if not stop:
            raw.answer_photo.assert_awaited_once()
        blocked = False
        monkeypatch.setattr(handler, "_BATCH_SECONDS", 10)
        await router.callback_query.handlers[1].callback(_button(raw, "next"))
        assert await history.seen_ids(7) == {1, 2}
        assert not active

    asyncio.run(scenario())
