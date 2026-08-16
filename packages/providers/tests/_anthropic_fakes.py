"""Narrow test doubles for the Anthropic provider adapter.

These fakes match only the small surface the adapter actually uses:
``client.messages.create(...)`` / ``client.messages.parse(...)`` and the shapes
of the returned ``Message`` / ``ParsedMessage`` (content blocks, ``usage``,
``stop_reason``, ``parsed_output``). They never perform network I/O.
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


@dataclass
class FakeParsedMessage:
    """Shape of an Anthropic ``ParsedMessage`` as consumed by the adapter.

    The adapter reads only ``stop_reason``, the ``parsed_output`` property
    result, and ``usage``; those are all this double needs to expose.
    """

    parsed_output: Any
    stop_reason: str = "end_turn"
    usage: FakeUsage = field(default_factory=FakeUsage)


def text_message(*texts: str, usage: FakeUsage | None = None) -> FakeMessage:
    blocks: list[Any] = [FakeTextBlock(text=t) for t in texts]
    return FakeMessage(content=blocks, usage=usage or FakeUsage())


def parsed_message(
    parsed_output: Any,
    *,
    stop_reason: str = "end_turn",
    usage: FakeUsage | None = None,
) -> FakeParsedMessage:
    return FakeParsedMessage(
        parsed_output=parsed_output,
        stop_reason=stop_reason,
        usage=usage or FakeUsage(),
    )


def _request() -> httpx.Request:
    return httpx.Request("POST", "https://api.anthropic.com/v1/messages")


def status_response(status_code: int, headers: dict[str, str] | None = None) -> httpx.Response:
    return httpx.Response(status_code, headers=headers or {}, request=_request())


def _unconfigured(surface: str) -> Callable[..., Awaitable[Any]]:
    async def behavior(**_: Any) -> Any:
        raise AssertionError(f"messages.{surface} was not configured for this fake")

    return behavior


class _Messages:
    def __init__(
        self,
        create: Callable[..., Awaitable[Any]],
        parse: Callable[..., Awaitable[Any]],
    ) -> None:
        self._create = create
        self._parse = parse

    def create(self, **kwargs: Any) -> Awaitable[Any]:
        return self._create(**kwargs)

    def parse(self, **kwargs: Any) -> Awaitable[Any]:
        return self._parse(**kwargs)


class FakeClient:
    """Fake async Anthropic client recording ``messages.create``/``parse``.

    ``calls`` records ``create`` kwargs and ``parse_calls`` records ``parse``
    kwargs. Either surface may be left unconfigured; invoking an unconfigured
    surface fails loudly rather than silently succeeding.
    """

    def __init__(
        self,
        behavior: Callable[..., Awaitable[Any]] | None = None,
        *,
        parse: Callable[..., Awaitable[Any]] | None = None,
    ) -> None:
        self.calls: list[dict[str, Any]] = []
        self.parse_calls: list[dict[str, Any]] = []
        create_behavior = behavior if behavior is not None else _unconfigured("create")
        parse_behavior = parse if parse is not None else _unconfigured("parse")
        self.messages = _Messages(
            self._record(create_behavior),
            self._record_parse(parse_behavior),
        )

    def _record(self, behavior: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        async def create(**kwargs: Any) -> Any:
            self.calls.append(kwargs)
            return await behavior(**kwargs)

        return create

    def _record_parse(
        self, behavior: Callable[..., Awaitable[Any]]
    ) -> Callable[..., Awaitable[Any]]:
        async def parse(**kwargs: Any) -> Any:
            self.parse_calls.append(kwargs)
            return await behavior(**kwargs)

        return parse

    @property
    def call_count(self) -> int:
        return len(self.calls)

    @property
    def parse_call_count(self) -> int:
        return len(self.parse_calls)


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
