"""Anthropic-backed Foreman planning adapter.

Implements the core-owned :class:`~llmforeman_core.Foreman` planning port using
the provider-layer structured-generation capability. The semantic flow is::

    objective -> ModelRequest -> StructuredModelProvider.generate_structured(...)
              -> validated private planning output -> semantic validation
              -> deterministic TaskPlan conversion

Layering is deliberate: this adapter consumes the provider-agnostic
:class:`~llmforeman_providers.contracts.StructuredModelProvider` capability and
never touches the Anthropic SDK, RelPrim, retries, or manual JSON parsing.
Reliability already lives below this adapter in the concrete provider. The
private planning DTOs describe only what the planner is allowed to choose; they
carry no execution state, so the model can never decide a task's status.
"""

from __future__ import annotations

import re
from typing import Final

from pydantic import BaseModel, Field

from llmforeman_core import (
    AgentRole,
    ForemanPlanValidationError,
    Task,
    TaskPlan,
    TaskStatus,
)
from llmforeman_providers.contracts import ModelRequest, StructuredModelProvider

__all__ = ["AnthropicForeman"]


FOREMAN_SYSTEM_PROMPT: Final[str] = (
    "You are LLMForeman's engineering planning manager. Decompose the supplied "
    "engineering objective into small, concrete implementation tasks.\n"
    "\n"
    "Rules:\n"
    "- Each task must be independently understandable and narrowly scoped.\n"
    "- Order tasks in a sensible implementation sequence.\n"
    "- Dependencies must reference actual prerequisite tasks in this plan.\n"
    "- Avoid duplicate or redundant tasks.\n"
    "- Use stable task identifiers in the form TASK-001, TASK-002, and so on.\n"
    "- Assign each task the most appropriate logical role.\n"
    "- Do not implement the solution and do not write code.\n"
    "- Do not report any work as already completed.\n"
    "- Return planning information only through the requested structured output."
)
"""Stable system instruction for the first Foreman planning implementation."""


_TASK_ID_PATTERN: Final[re.Pattern[str]] = re.compile(r"^TASK-\d{3}$")
"""Simple stable task-id shape: ``TASK-`` followed by exactly three digits."""


class _ForemanTaskOutput(BaseModel):
    """Private structured shape for a single planned task.

    Describes only what the planner is permitted to choose. It intentionally
    omits every execution-domain concern (status, timestamps, attempts, result,
    failure reason, model/provider/runtime, cost, usage, review state) so the
    model has no authority over execution state.
    """

    id: str
    title: str
    description: str
    assigned_role: AgentRole
    dependencies: list[str] = Field(default_factory=list)


class _ForemanPlanOutput(BaseModel):
    """Private structured shape for a full plan: an ordered list of tasks."""

    tasks: list[_ForemanTaskOutput]


def _validate_and_convert(output: _ForemanPlanOutput) -> TaskPlan:
    """Validate plan semantics, then deterministically build a ``TaskPlan``.

    Enforces only the minimal cross-task invariants justified now: at least one
    task, unique and well-shaped task ids, and dependencies that reference an
    existing task without self-reference. Full dependency-cycle detection is out
    of scope. On success, every planned task enters the execution domain as
    ``TaskStatus.TODO``; the model never selects a task's status. Order, ids,
    titles, descriptions, roles, and dependency lists are preserved exactly.
    """

    tasks = output.tasks
    if not tasks:
        raise ForemanPlanValidationError("plan must contain at least one task")

    known_ids: set[str] = set()
    for task in tasks:
        if not _TASK_ID_PATTERN.fullmatch(task.id):
            raise ForemanPlanValidationError(
                f"task id {task.id!r} does not match the required TASK-000 shape"
            )
        if task.id in known_ids:
            raise ForemanPlanValidationError(f"duplicate task id {task.id!r}")
        known_ids.add(task.id)

    for task in tasks:
        for dependency in task.dependencies:
            if dependency == task.id:
                raise ForemanPlanValidationError(
                    f"task {task.id!r} must not depend on itself"
                )
            if dependency not in known_ids:
                raise ForemanPlanValidationError(
                    f"task {task.id!r} depends on unknown task {dependency!r}"
                )

    domain_tasks = [
        Task(
            id=task.id,
            title=task.title,
            description=task.description,
            status=TaskStatus.TODO,
            assigned_role=task.assigned_role,
            dependencies=list(task.dependencies),
        )
        for task in tasks
    ]
    return TaskPlan(tasks=domain_tasks)


class AnthropicForeman:
    """Anthropic-backed implementation of the core ``Foreman`` planning port.

    Consumes an injected :class:`StructuredModelProvider` (in production, the
    concrete ``AnthropicProvider``) so it can be unit-tested with a small fake
    provider and never constructs or knows about the Anthropic SDK. It performs
    exactly one structured-generation call per valid objective and adds no
    retry, timeout, rate-limit, fallback, or replanning behavior of its own:
    reliability belongs to the layers below, and provider errors propagate
    unchanged.
    """

    def __init__(self, provider: StructuredModelProvider) -> None:
        self._provider = provider

    async def create_plan(self, objective: str) -> TaskPlan:
        """Turn ``objective`` into a validated core ``TaskPlan``.

        Rejects a blank or whitespace-only objective before any provider call,
        so an invalid objective produces exactly zero structured-generation
        calls. The caller's original objective text is preserved as the request
        prompt (trimming is used only to detect blankness).
        """

        if not objective.strip():
            raise ValueError("objective must not be empty or whitespace-only")

        request = ModelRequest(
            prompt=objective,
            system_prompt=FOREMAN_SYSTEM_PROMPT,
        )
        response = await self._provider.generate_structured(request, _ForemanPlanOutput)
        return _validate_and_convert(response.output)
