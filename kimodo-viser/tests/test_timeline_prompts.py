# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from concurrent.futures import ThreadPoolExecutor

import pytest

from viser import _messages
from viser._timeline_api import TimelineApi


class DummyWebsockInterface:
    def __init__(self) -> None:
        self.handlers = {}
        self.queued = []

    def register_handler(self, message_type, handler) -> None:
        self.handlers[message_type] = handler

    def queue_message(self, message) -> None:
        self.queued.append(message)


@pytest.mark.asyncio
async def test_prompt_split_clamps_min_duration() -> None:
    loop = asyncio.get_event_loop()
    websock = DummyWebsockInterface()
    with ThreadPoolExecutor(max_workers=1) as executor:
        api = TimelineApi(
            owner=None,
            websock_interface=websock,
            thread_executor=executor,
            event_loop=loop,
        )
        api.set_defaults(default_text="walk", default_duration=10, min_duration=3)

        original = next(iter(api._prompts.values()))
        await api._handle_prompt_split(
            "client",
            _messages.TimelinePromptSplitMessage(
                prompt_id=original.uuid,
                split_frame=original.start_frame + 1,
            ),
        )

        prompts = sorted(api._prompts.values(), key=lambda p: p.start_frame)
        assert len(prompts) == 2
        assert prompts[0].end_frame - prompts[0].start_frame >= 3
        assert prompts[1].end_frame - prompts[1].start_frame >= 3
        assert prompts[0].end_frame == prompts[1].start_frame


@pytest.mark.asyncio
async def test_prompt_delete_keeps_contiguous() -> None:
    loop = asyncio.get_event_loop()
    websock = DummyWebsockInterface()
    with ThreadPoolExecutor(max_workers=1) as executor:
        api = TimelineApi(
            owner=None,
            websock_interface=websock,
            thread_executor=executor,
            event_loop=loop,
        )
        api.set_defaults(default_text="walk", default_duration=5, min_duration=1)
        api.set_defaults(default_text="run", default_duration=5, min_duration=1)
        api.set_defaults(default_text="jump", default_duration=5, min_duration=1)

        prompts = sorted(api._prompts.values(), key=lambda p: p.start_frame)
        middle_id = prompts[1].uuid

        await api._handle_prompt_delete(
            "client",
            _messages.TimelinePromptDeleteMessage(prompt_id=middle_id),
        )

        remaining = sorted(api._prompts.values(), key=lambda p: p.start_frame)
        assert len(remaining) == 2
        assert remaining[0].end_frame == remaining[1].start_frame
