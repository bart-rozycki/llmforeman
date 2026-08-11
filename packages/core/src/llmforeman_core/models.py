"""Provider- and runtime-agnostic execution domain models.

These models describe an in-memory engineering run as a task plan of
individual tasks. They contain data, local validation, and the pure task
lifecycle transition policy only: no orchestration, execution, scheduling, or
persistence logic lives here.
"""

from enum import StrEnum
from typing import Final

from pydantic import BaseModel, field_validator

__all__ = [
    "AgentRole",
    "Run",
    "Task",
    "TaskPlan",
    "TaskStatus",
    "can_transition",
]


class TaskStatus(StrEnum):
    """Current lifecycle state of a task.

    Membership only. Legal transitions between states are defined separately by
    the lifecycle policy (see ``_LEGAL_TRANSITIONS`` / ``can_transition``).
    """

    TODO = "TODO"
    IN_PROGRESS = "IN_PROGRESS"
    REVIEW = "REVIEW"
    BLOCKED = "BLOCKED"
    DONE = "DONE"
    FAILED = "FAILED"


_LEGAL_TRANSITIONS: Final[dict[TaskStatus, frozenset[TaskStatus]]] = {
    TaskStatus.TODO: frozenset({TaskStatus.IN_PROGRESS}),
    TaskStatus.IN_PROGRESS: frozenset(
        {TaskStatus.REVIEW, TaskStatus.BLOCKED, TaskStatus.FAILED}
    ),
    TaskStatus.REVIEW: frozenset(
        {TaskStatus.DONE, TaskStatus.IN_PROGRESS, TaskStatus.FAILED}
    ),
    TaskStatus.BLOCKED: frozenset({TaskStatus.IN_PROGRESS}),
    TaskStatus.DONE: frozenset(),
    TaskStatus.FAILED: frozenset(),
}
"""Single source of truth for the v0.1 task lifecycle.

Maps each status to the set of statuses it may legally transition to. Any
transition not listed here (including every same-status transition) is illegal.
``DONE`` and ``FAILED`` are terminal and have no outgoing transitions.
"""


def can_transition(current: TaskStatus, target: TaskStatus) -> bool:
    """Return whether moving from ``current`` to ``target`` is legal in v0.1.

    Pure and deterministic; consults the single lifecycle source of truth.
    Same-status transitions are always illegal.
    """

    return target in _LEGAL_TRANSITIONS[current]


class AgentRole(StrEnum):
    """Logical execution role a task may be assigned to.

    Roles are purely logical labels; they carry no association with models,
    providers, runtimes, prompts, capabilities, or permissions.
    """

    FOREMAN = "FOREMAN"
    DEVELOPER = "DEVELOPER"
    TESTER = "TESTER"
    REVIEWER = "REVIEWER"


class Task(BaseModel):
    """A single unit of engineering work within a task plan."""

    id: str
    title: str
    description: str
    status: TaskStatus = TaskStatus.TODO
    assigned_role: AgentRole | None = None
    dependencies: list[str] = []

    def can_transition_to(self, target: TaskStatus) -> bool:
        """Return whether this task's status may legally move to ``target``."""

        return can_transition(self.status, target)

    @field_validator("id", "title")
    @classmethod
    def _reject_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be empty or whitespace-only")
        return value

    @field_validator("dependencies")
    @classmethod
    def _reject_blank_dependencies(cls, value: list[str]) -> list[str]:
        for dependency in value:
            if not dependency.strip():
                raise ValueError("dependency ids must not be empty or whitespace-only")
        return value


class TaskPlan(BaseModel):
    """An ordered collection of tasks."""

    tasks: list[Task] = []


class Run(BaseModel):
    """An in-memory engineering run wrapping a single task plan."""

    plan: TaskPlan
