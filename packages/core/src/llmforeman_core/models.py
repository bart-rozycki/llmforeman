"""Provider- and runtime-agnostic execution domain models.

These models describe an in-memory engineering run as a task plan of
individual tasks. They contain data, local validation, and the pure task
lifecycle transition policy only: no orchestration, execution, scheduling, or
persistence logic lives here.
"""

from enum import StrEnum
from pathlib import PurePosixPath, PureWindowsPath
from typing import Final

from pydantic import BaseModel, Field, field_validator

__all__ = [
    "AgentRole",
    "ModelUsage",
    "RepositoryContext",
    "RepositoryFile",
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


def _validate_repository_relative_path(value: str) -> str:
    """Validate that ``value`` is a safe repository-relative logical path.

    Core-internal, filesystem-free privacy invariant shared by all domain
    models that carry a repository-relative path. Rejects blank, NUL-bearing,
    absolute (POSIX or Windows), and parent-traversal paths without touching
    the filesystem, resolving against a working directory, or checking
    existence. Valid paths are returned unchanged.
    """

    if not value.strip():
        raise ValueError("path must not be empty or whitespace-only")
    if "\x00" in value:
        raise ValueError("path must not contain NUL characters")
    # Cross-platform, filesystem-free absolute-path detection. Parse the
    # value as both POSIX and Windows pure paths so an absolute path is
    # rejected regardless of which OS runs this code.
    if PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute():
        raise ValueError("path must be repository-relative, not absolute")
    # Reject parent traversal in either separator convention without
    # resolving the path against the filesystem or the working directory.
    parts = PurePosixPath(value).parts + PureWindowsPath(value).parts
    if ".." in parts:
        raise ValueError("path must not contain parent traversal segments ('..')")
    return value


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


class ModelUsage(BaseModel):
    """Provider- and runtime-agnostic token usage measurement.

    A stable, normalized representation of token counts that future provider
    and runtime adapters map their own usage formats into. This model is
    deliberately descriptive, not interpretive: it stores the counters an
    adapter supplies and makes no assumption about additive, billing, or
    caching relationships between them. It performs no summation and derives
    no totals; core does not know how any specific system accounts for usage.
    """

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cache_read_input_tokens: int = Field(default=0, ge=0)
    cache_creation_input_tokens: int = Field(default=0, ge=0)


class RepositoryFile(BaseModel):
    """One selected repository file identified by a repository-relative path.

    Represents already-prepared context: the ``content`` is supplied as-is by
    whatever future component selected and read the file. This model performs no
    filesystem access, encoding detection, or content interpretation of any
    kind; it only guarantees that ``path`` is a repository-relative logical
    path suitable for serialization and model context.
    """

    path: str
    content: str

    @field_validator("path")
    @classmethod
    def _validate_path(cls, value: str) -> str:
        return _validate_repository_relative_path(value)


class RepositoryContext(BaseModel):
    """Normalized, provider-independent repository context.

    Holds a lightweight structural overview (``file_tree``) together with the
    full contents of only the files deliberately selected for context
    (``files``). It does not model the repository root, describe how the tree
    or selection were produced, or imply that every repository file is
    included.
    """

    file_tree: str
    files: list[RepositoryFile] = []
