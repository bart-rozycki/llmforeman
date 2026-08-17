"""Typed v0.1 worker-action vocabulary.

Defines the closed set of actions a local coding worker may request LLMForeman
to perform next. These are provider-, runtime-, and workspace-agnostic control
messages: they describe *what the worker wants done*, not how a capability is
executed and not what was observed as a result. Execution, observation/result
vocabulary, dispatch, and any worker/orchestration loop are deliberately out of
scope and live elsewhere.

The vocabulary is intentionally closed and statically typed (no generic
tool-call/dict payloads) so that future orchestration can dispatch through
explicit structural pattern matching on the concrete action variant.

Every action model rejects unknown fields (``extra="forbid"``) because these
messages are machine-generated control instructions: a stray or misspelled
field indicates a schema/hallucination problem that must fail loudly rather
than be silently discarded.

``WorkerAction`` wraps the discriminated union in a Pydantic ``RootModel`` so
it is a concrete ``BaseModel`` subtype that can be supplied directly as the
``output_type`` of the structured runtime contract (``type[T]`` where
``T: BaseModel``) while keeping the on-the-wire JSON flat (no ``root``
wrapper).
"""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, RootModel, field_validator

from llmforeman_core.models import _validate_repository_relative_path

__all__ = [
    "FinishAction",
    "ReadFileAction",
    "RunCommandAction",
    "SearchAction",
    "WorkerAction",
    "WriteFileAction",
]

# Shared strict configuration for every action model. Unknown fields on a
# machine-generated control instruction are an error, never something to
# silently ignore.
_STRICT_ACTION_CONFIG = ConfigDict(extra="forbid")


class SearchAction(BaseModel):
    """Request a repository text search for ``query``.

    Describes only the search intent. Result limits, regex/case flags, globs,
    and file filters are execution semantics owned by the concrete searcher,
    not by this control message.
    """

    model_config = _STRICT_ACTION_CONFIG

    action: Literal["search"]
    query: str

    @field_validator("query")
    @classmethod
    def _validate_query(cls, value: str) -> str:
        # Trimming is used only to detect a blank query; the original value is
        # preserved because leading/trailing whitespace may be significant in a
        # literal text search.
        if not value.strip():
            raise ValueError("query must not be empty or whitespace-only")
        if "\x00" in value:
            raise ValueError("query must not contain NUL characters")
        return value


class ReadFileAction(BaseModel):
    """Request reading a repository-relative file at ``path``.

    ``path`` must satisfy the core repository-relative privacy invariant. This
    is domain/control validation only: it performs no filesystem access and is
    defense in depth, not a substitute for the reader's own security boundary.
    """

    model_config = _STRICT_ACTION_CONFIG

    action: Literal["read"]
    path: str

    @field_validator("path")
    @classmethod
    def _validate_path(cls, value: str) -> str:
        return _validate_repository_relative_path(value)


class WriteFileAction(BaseModel):
    """Request writing ``content`` to a repository-relative file at ``path``.

    ``path`` uses the same repository-relative privacy invariant as
    :class:`ReadFileAction`, rejecting traversal/absolute/NUL paths before any
    workspace capability could receive them. ``content`` is preserved exactly
    and may be empty; no size limit, syntax check, or normalization is applied
    here (those remain the concrete writer's concern).
    """

    model_config = _STRICT_ACTION_CONFIG

    action: Literal["write"]
    path: str
    content: str

    @field_validator("path")
    @classmethod
    def _validate_path(cls, value: str) -> str:
        return _validate_repository_relative_path(value)


class RunCommandAction(BaseModel):
    """Request running an argv command.

    ``command`` is argv (``list[str]``), never a shell string; the model
    performs no shell interpretation and grants no authorization to execute.
    Shell metacharacters are preserved verbatim as inert data. The concrete
    subprocess runner still owns runtime execution validation and policy.
    """

    model_config = _STRICT_ACTION_CONFIG

    action: Literal["run"]
    command: list[str]

    @field_validator("command")
    @classmethod
    def _validate_command(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("command must contain at least one argv element")
        for element in value:
            if "\x00" in element:
                raise ValueError(
                    "command argv elements must not contain NUL characters"
                )
            if element == "":
                raise ValueError("command argv elements must not be empty")
        # Only the executable (argv[0]) must be non-whitespace; later arguments
        # may legitimately be padded and are preserved exactly.
        if not value[0].strip():
            raise ValueError(
                "command executable (argv[0]) must not be whitespace-only"
            )
        return value


class FinishAction(BaseModel):
    """Signal that the worker believes its implementation loop is complete.

    ``summary`` describes what the worker did. This is *not* a task lifecycle
    transition: it does not imply ``TaskStatus.DONE`` and must not mutate a
    ``Task``. Future orchestration may still validate, test, or review before
    accepting the work.
    """

    model_config = _STRICT_ACTION_CONFIG

    action: Literal["finish"]
    summary: str

    @field_validator("summary")
    @classmethod
    def _validate_summary(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("summary must not be empty or whitespace-only")
        if "\x00" in value:
            raise ValueError("summary must not contain NUL characters")
        return value


# Internal discriminated union over the closed action vocabulary. Pydantic owns
# variant selection via the required ``action`` discriminator; unknown or
# missing discriminators fail validation. Kept private: callers use the public
# ``WorkerAction`` root model.
_WorkerActionValue = Annotated[
    SearchAction
    | ReadFileAction
    | WriteFileAction
    | RunCommandAction
    | FinishAction,
    Field(discriminator="action"),
]


class WorkerAction(RootModel[_WorkerActionValue]):
    """The closed, typed v0.1 worker-action vocabulary.

    A concrete ``BaseModel`` (``RootModel``) subtype whose ``root`` is exactly
    one of the five action variants, selected by the required ``action``
    discriminator. Being a real ``BaseModel`` it can be passed directly as the
    structured runtime ``output_type`` without weakening that contract, while
    its JSON representation stays flat (e.g. ``{"action": "read", "path":
    ...}``) with no externally visible ``root`` wrapper.
    """
