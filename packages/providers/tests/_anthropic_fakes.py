"""Narrow test doubles for the Anthropic provider adapter.

These fakes match only the small surface the adapter actually uses:
``client.messages.create(...)`` and the shape of the returned ``Message``
(content blocks + usage). They never perform network I/O.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import httpx


@dataclass
class FakeTextBlock:
    text: str
    type: str = "text"


@dataclass
class FakeThinkingBlock:
    thinking: str = "hidden"
    type: str = "thinking"


@dataclass
class FakeUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int | None = 0
    cache_creation_input_tokens: int | None = 0


@dataclass
class FakeMessage:
    content: list[Any]
    usage: FakeUsage = field(default_factory=FakeUsage)


def text_message(*texts: str, usage: FakeUsage | None = None) -> FakeMessage:
    blocks: list[Any] = [FakeTextBlock(text=t) for t in texts]
    return FakeMessage(content=blocks, usage=usage or FakeUsage())


def _request() -> httpx.Request:
    return httpx.Request("POST", "https://api.anthropic.com/v1/messages")


def status_response(status_code: int, headers: dict[str, str] | None = None) -> httpx.Response:
    return httpx.Response(status_code, headers=headers or {}, request=_request())


class _Messages:
    def __init__(self, create: Callable[..., Awaitable[Any]]) -> None:
        self._create = create

    def create(self, **kwargs: Any) -> Awaitable[Any]:
        return self._create(**kwargs)


class FakeClient:
    """Fake async Anthropic client recording calls to ``messages.create``."""

    def __init__(self, behavior: Callable[..., Awaitable[Any]]) -> None:
        self.calls: list[dict[str, Any]] = []
        self.messages = _Messages(self._record(behavior))

    def _record(self, behavior: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        async def create(**kwargs: Any) -> Any:
            self.calls.append(kwargs)
            return await behavior(**kwargs)

        return create

    @property
    def call_count(self) -> int:
        return len(self.calls)


def returns(message: Any) -> Callable[..., Awaitable[Any]]:
    async def behavior(**_: Any) -> Any:
        return message

    return behavior


def always_raises(exc: BaseException) -> Callable[..., Awaitable[Any]]:
    async def behavior(**_: Any) -> Any:
        raise exc

    return behavior


def raises_then_returns(
    exc: BaseException,
    message: Any,
    *,
    failures: int = 1,
) -> Callable[..., Awaitable[Any]]:
    state = {"count": 0}

    async def behavior(**_: Any) -> Any:
        if state["count"] < failures:
            state["count"] += 1
            raise exc
        return message

    return behavior


def hangs_until_cancelled(counter: dict[str, int]) -> Callable[..., Awaitable[Any]]:
    async def behavior(**_: Any) -> Any:
        counter["started"] = counter.get("started", 0) + 1
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            counter["cancelled"] = counter.get("cancelled", 0) + 1
            raise

    return behavior
