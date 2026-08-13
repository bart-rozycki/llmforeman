"""Contract tests for the core ``Foreman`` planning port.

These prove the architectural intent rather than any runtime behavior: a
test-local async implementation can structurally satisfy ``Foreman`` (checked
statically by mypy via the ``Foreman`` annotation below), typed code can
``await foreman.create_plan(objective)``, and the result is the existing core
``TaskPlan`` — all without referencing any provider or runtime type.
"""

import asyncio
from collections.abc import Awaitable

from llmforeman_core import Foreman, Task, TaskPlan


def run[T](coro: Awaitable[T]) -> T:
    return asyncio.run(coro)  # type: ignore[arg-type]


class FakeForeman:
    """Test-local structural implementation of the ``Foreman`` port."""

    async def create_plan(self, objective: str) -> TaskPlan:
        return TaskPlan(tasks=[Task(id="t1", title=objective, description=objective)])


def test_fake_satisfies_foreman_protocol() -> None:
    # The annotation is the assertion: FakeForeman must structurally satisfy
    # Foreman for mypy to accept this under strict mode.
    foreman: Foreman = FakeForeman()
    assert isinstance(foreman, FakeForeman)


def test_create_plan_accepts_objective_and_returns_task_plan() -> None:
    foreman: Foreman = FakeForeman()

    plan = run(foreman.create_plan("Add retry support"))

    assert isinstance(plan, TaskPlan)
    assert plan.tasks[0].title == "Add retry support"
