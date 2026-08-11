"""Provider- and runtime-agnostic execution domain models.

These models describe an in-memory engineering run as a task plan of
individual tasks. They intentionally contain data and local validation only:
no orchestration, execution, status-transition, scheduling, or persistence
logic lives here.
"""

from enum import StrEnum

from pydantic import BaseModel, field_validator

__all__ = [
    "AgentRole",
    "Run",
    "Task",
    "TaskPlan",
    "TaskStatus",
]


class TaskStatus(StrEnum):
    """Current lifecycle state of a task.

    Membership only; allowed transitions and terminal-state classification are
    intentionally not modeled here.
    """

    TODO = "TODO"
    IN_PROGRESS = "IN_PROGRESS"
    REVIEW = "REVIEW"
    BLOCKED = "BLOCKED"
    DONE = "DONE"
    FAILED = "FAILED"


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
