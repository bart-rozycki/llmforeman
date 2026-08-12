"""Narrow test doubles for the Ollama runtime adapter.

These fakes match only the small surface the adapter actually uses:
``client.generate(**kwargs)`` returning an ``ollama.GenerateResponse``. They
never perform network I/O and never require a running Ollama server.

Successful responses use the real ``ollama.GenerateResponse`` model so the
tests exercise the same fields (``response``, ``thinking``,
``prompt_eval_count``, ``eval_count``) the adapter normalizes. Failures use the
real ``ollama.ResponseError`` (with a status code) or genuine transport
exception types, so error classification is tested against realistic objects.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from ollama import GenerateResponse

Behavior = Callable[..., Awaitable[Any]]


def generate_response(
    *,
    response: str | None = "ok",
    thinking: str | None = None,
    prompt_eval_count: int | None = 1,
    eval_count: int | None = 1,
) -> GenerateResponse:
    """Build a realistic non-streaming ``GenerateResponse`` for tests."""

    return GenerateResponse(
        response=response,
        thinking=thinking,
        prompt_eval_count=prompt_eval_count,
        eval_count=eval_count,
        done=True,
        done_reason="stop",
        # Timing fields Ollama returns but which must NOT reach ModelUsage.
        total_duration=123_456,
        load_duration=42,
        prompt_eval_duration=10,
        eval_duration=100,
    )


class FakeClient:
    """Fake async Ollama client recording calls to ``generate``."""

    def __init__(self, behavior: Behavior) -> None:
        self.calls: list[dict[str, Any]] = []
        self._behavior = behavior

    async def generate(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return await self._behavior(**kwargs)

    @property
    def call_count(self) -> int:
        return len(self.calls)


def returns(response: Any) -> Behavior:
    async def behavior(**_: Any) -> Any:
        return response

    return behavior


def always_raises(exc: BaseException) -> Behavior:
    async def behavior(**_: Any) -> Any:
        raise exc

    return behavior


def raises_then_returns(
    exc: BaseException,
    response: Any,
    *,
    failures: int = 1,
) -> Behavior:
    state = {"count": 0}

    async def behavior(**_: Any) -> Any:
        if state["count"] < failures:
            state["count"] += 1
            raise exc
        return response

    return behavior


def hangs_until_cancelled(counter: dict[str, int]) -> Behavior:
    async def behavior(**_: Any) -> Any:
        counter["started"] = counter.get("started", 0) + 1
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            counter["cancelled"] = counter.get("cancelled", 0) + 1
            raise

    return behavior
