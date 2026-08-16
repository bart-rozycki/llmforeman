"""Opt-in live smoke test for the Anthropic Foreman planning vertical.

This test exercises the *real* production path end-to-end::

    engineering objective
        -> AnthropicForeman.create_plan()
        -> AnthropicProvider.generate_structured()  (real StructuredModelProvider)
        -> RelPrim reliability boundary
        -> AsyncAnthropic.messages.parse()  (Anthropic Structured Outputs)
        -> private Foreman planning DTO
        -> Foreman semantic validation
        -> TaskPlan

It wires only the genuine production components (``AnthropicProvider`` and
``AnthropicForeman``) with no fakes, mocks, or monkeypatching, so a passing run
proves the actual external integration currently works.

Safety: this test is DISABLED BY DEFAULT. It makes a real, paid Anthropic API
request and therefore runs only when the user explicitly opts in with the
environment variable ``LLMFOREMAN_RUN_LIVE_TESTS=1``. Ordinary ``uv run pytest``
never executes it. See CONTRIBUTING.md for manual invocation instructions.

The API key is read only from ``ANTHROPIC_API_KEY`` in the process environment
(resolved by the Anthropic SDK through the normal production construction path).
The key value is never printed, logged, stored, or placed in assertion output.
"""

from __future__ import annotations

import asyncio
import os
import re
from collections.abc import Awaitable

import pytest

from llmforeman_core import TaskPlan, TaskStatus
from llmforeman_providers import AnthropicForeman, AnthropicProvider

# Environment gate. Only the exact value ``"1"`` enables live execution; no other
# truthy string is interpreted, matching the requested explicit opt-in contract.
_LIVE_FLAG_ENV = "LLMFOREMAN_RUN_LIVE_TESTS"
_API_KEY_ENV = "ANTHROPIC_API_KEY"

# Established Foreman/task-plan id contract: ``TASK-`` followed by three digits.
_TASK_ID_PATTERN = re.compile(r"^TASK-\d{3}$")

# Modest local output budget: enough for a small planning response without an
# unnecessarily large (and more expensive) token allowance. This is local to the
# test and does not touch the production default.
_LIVE_MAX_TOKENS = 4096

# The deliberately simple, stable smoke objective.
_OBJECTIVE = "Add input validation to a Python function and cover it with unit tests."


pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        os.getenv(_LIVE_FLAG_ENV) != "1",
        reason=(
            f"Live Anthropic smoke test disabled. Set {_LIVE_FLAG_ENV}=1 "
            "(and ANTHROPIC_API_KEY) to run it; it makes a real, paid API call."
        ),
    ),
]


def _run[T](coro: Awaitable[T]) -> T:
    """Drive an async coroutine, matching the repository's asyncio.run style."""

    return asyncio.run(coro)  # type: ignore[arg-type]


def _require_api_key() -> None:
    """Fail fast when live tests are enabled but no API key is configured.

    The user explicitly opted in, so a missing/blank credential is a
    configuration failure rather than a silent skip. This runs before any
    provider is constructed, so no API request is attempted. The key value is
    never included in the error.
    """

    key = os.getenv(_API_KEY_ENV)
    if key is None or not key.strip():
        pytest.fail(
            f"Live tests enabled ({_LIVE_FLAG_ENV}=1) but {_API_KEY_ENV} is "
            "missing or blank. Set it in the environment to run the live smoke "
            "test.",
            pytrace=False,
        )


async def _create_live_plan() -> TaskPlan:
    """Construct the real production vertical and produce one plan.

    Uses ``AnthropicProvider`` as an async context manager so its owned HTTP
    client is closed via the existing production lifecycle mechanism.
    """

    async with AnthropicProvider(max_tokens=_LIVE_MAX_TOKENS) as provider:
        foreman = AnthropicForeman(provider)
        return await foreman.create_plan(_OBJECTIVE)


def test_anthropic_foreman_create_plan_live() -> None:
    _require_api_key()

    plan = _run(_create_live_plan())

    # Plan exists and is the expected domain type with at least one task.
    assert isinstance(plan, TaskPlan)
    assert len(plan.tasks) >= 1

    ids = [task.id for task in plan.tasks]
    known_ids = set(ids)

    # Task ids conform to the established contract and are unique.
    assert len(known_ids) == len(ids), "task ids must be unique"
    for task in plan.tasks:
        assert _TASK_ID_PATTERN.fullmatch(task.id), f"unexpected task id shape: {task.id!r}"

    for task in plan.tasks:
        # The live model must not have execution-state authority.
        assert task.status is TaskStatus.TODO

        # Basic domain field quality (already guaranteed non-blank by the model).
        assert task.title.strip()
        assert task.description.strip()

        # Dependencies reference tasks in the same plan and never self-reference.
        for dependency_id in task.dependencies:
            assert dependency_id != task.id, f"task {task.id!r} depends on itself"
            assert dependency_id in known_ids, (
                f"task {task.id!r} depends on unknown task {dependency_id!r}"
            )

    # Optional, non-sensitive human-readable summary (visible under ``-s``).
    print(f"\nLive Anthropic TaskPlan ({len(plan.tasks)} task(s)):")
    for task in plan.tasks:
        deps = f" deps={sorted(task.dependencies)}" if task.dependencies else ""
        print(f"  {task.id} [{task.status.value}] {task.title}{deps}")
