"""Typed v0.1 worker-observation vocabulary.

Defines the closed set of observations LLMForeman exposes back to a local
coding worker after a single requested worker action has been executed. These
are the product/orchestration-facing information messages that sit between a
:class:`~llmforeman_core.worker_actions.WorkerAction` and the worker's next
structured decision:

    WorkerAction
        -> orchestrator executes a workspace capability
        -> WorkerObservation
        -> next structured worker decision

This module defines the *destination* vocabulary only. It does not execute a
workspace capability, invoke a model, or implement the worker loop. It performs
no filesystem, Git, subprocess, network, or model access.

Semantic boundary: a worker observation is deliberately distinct from any
workspace infrastructure result even where fields currently resemble each
other. Core intentionally does not import or reuse workspace DTOs
(``RepositorySearchMatch``, ``RepositoryFile``, ``CommandResult``, etc.); a
future executor will perform the explicit mapping and sanitization. This keeps
the ``core -/-> workspace`` invariant intact.

The vocabulary is closed and statically typed. It contains exactly five
observation variants -- ``search``, ``read``, ``write``, ``run``, ``error`` --
and no ``FinishObservation``: ``FinishAction`` terminates the worker loop and
executes no workspace capability, so it produces no post-action observation.

A tool/workspace failure is not automatically a worker-run failure. A future
executor may convert selected expected workspace failures into an
:class:`ActionErrorObservation` so the worker can react (e.g. recover by
searching), while other infrastructure failures may still propagate as
exceptions. Task #30 defines the error *shape* only; it does not decide that
classification, add retryability metadata, or carry raw exception data.

Every observation model rejects unknown fields (``extra="forbid"``) because
these messages are serialized directly into model context: a stray or
misspelled field indicates a schema/hallucination problem that must fail
loudly rather than be silently discarded.

``WorkerObservation`` wraps the discriminated union in a Pydantic ``RootModel``
so it is a concrete ``BaseModel`` subtype (analogous to ``WorkerAction``) while
keeping the on-the-wire JSON flat (no ``root`` wrapper).
"""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, RootModel, field_validator

from llmforeman_core.models import (
    _validate_command_argv,
    _validate_non_blank_text,
    _validate_repository_relative_path,
)

__all__ = [
    "ActionErrorObservation",
    "ReadObservation",
    "RunObservation",
    "SearchObservation",
    "WorkerObservation",
    "WorkerSearchMatch",
    "WriteObservation",
]

# Shared strict configuration for every observation model. Unknown fields on a
# message that will be serialized into model context are an error, never
# something to silently ignore.
_STRICT_OBSERVATION_CONFIG = ConfigDict(extra="forbid")


class WorkerSearchMatch(BaseModel):
    """One search result exposed to the worker.

    A core-owned semantic representation of a single match, intentionally
    distinct from any workspace search DTO. ``path`` satisfies the core
    repository-relative privacy invariant; ``line_number`` is 1-based; ``line``
    is the exact matching textual line, preserved verbatim (no stripping,
    context lines, or column data).
    """

    model_config = _STRICT_OBSERVATION_CONFIG

    path: str
    line_number: int = Field(ge=1)
    line: str

    @field_validator("path")
    @classmethod
    def _validate_path(cls, value: str) -> str:
        return _validate_repository_relative_path(value)


class SearchObservation(BaseModel):
    """Result of executing a worker search request.

    Carries the originating ``query`` (validated identically to
    ``SearchAction.query``) and the ``matches`` exposed to the worker. The
    match list is preserved exactly as supplied: an empty list is a valid
    no-match result (never an error), and the model neither sorts, deduplicates,
    nor ranks matches -- ordering is owned by the executor/searcher.
    """

    model_config = _STRICT_OBSERVATION_CONFIG

    observation: Literal["search"]
    query: str
    matches: list[WorkerSearchMatch]

    @field_validator("query")
    @classmethod
    def _validate_query(cls, value: str) -> str:
        return _validate_non_blank_text(value)


class ReadObservation(BaseModel):
    """Result of a successful worker read request.

    ``path`` satisfies the core repository-relative privacy invariant.
    ``content`` is the exact text exposed to the worker: it may be empty and is
    preserved verbatim with no normalization, size/hash/token metadata, or
    truncation. The executor constructs this only when it has a complete, safe
    read according to its own semantics.
    """

    model_config = _STRICT_OBSERVATION_CONFIG

    observation: Literal["read"]
    path: str
    content: str

    @field_validator("path")
    @classmethod
    def _validate_path(cls, value: str) -> str:
        return _validate_repository_relative_path(value)


class WriteObservation(BaseModel):
    """Acknowledgement that a worker write completed for ``path``.

    Intentionally carries only ``observation`` and ``path``: the written
    content is deliberately not echoed, because the worker just produced it in
    the preceding action and repeating it would needlessly duplicate context
    tokens. It records no Git status, diff, hash, byte count, or created/
    overwritten flags. ``path`` satisfies the core repository-relative privacy
    invariant.
    """

    model_config = _STRICT_OBSERVATION_CONFIG

    observation: Literal["write"]
    path: str

    @field_validator("path")
    @classmethod
    def _validate_path(cls, value: str) -> str:
        return _validate_repository_relative_path(value)


class RunObservation(BaseModel):
    """Result of executing a worker command.

    ``command`` is explicit argv (``list[str]``), validated identically to
    ``RunCommandAction.command`` and never converted to shell text.
    ``exit_code`` is an unrestricted ``int``: zero, positive, and negative
    codes are all valid, and a non-zero exit is normal information for the
    worker -- it is never inferred into a success flag nor converted into an
    error observation. ``stdout``/``stderr`` are exact diagnostic strings that
    may be empty and are preserved verbatim (no merging, stripping, line-ending
    normalization, or ANSI removal).
    """

    model_config = _STRICT_OBSERVATION_CONFIG

    observation: Literal["run"]
    command: list[str]
    exit_code: int
    stdout: str
    stderr: str

    @field_validator("command")
    @classmethod
    def _validate_command(cls, value: list[str]) -> list[str]:
        return _validate_command_argv(value)


class ActionErrorObservation(BaseModel):
    """Safe, agent-facing failure information for one workspace-backed action.

    Reports that a single executable action (``search``/``read``/``write``/
    ``run``) could not be completed. ``finish`` is intentionally excluded: it
    executes no workspace capability and produces no observation. ``message``
    is a safe, agent-facing explanation (non-blank, NUL-free, preserved
    exactly); it carries no exception class, repr, traceback, errno, absolute
    path, PID, or other raw infrastructure data, and no retryable/fatal
    metadata. Sanitizing raw failures into such a message is future executor
    logic; this model defines the shape only and makes no recovery promise.
    """

    model_config = _STRICT_OBSERVATION_CONFIG

    observation: Literal["error"]
    action: Literal["search", "read", "write", "run"]
    message: str

    @field_validator("message")
    @classmethod
    def _validate_message(cls, value: str) -> str:
        return _validate_non_blank_text(value)


# Internal discriminated union over the closed observation vocabulary. Pydantic
# owns variant selection via the required ``observation`` discriminator;
# unknown or missing discriminators fail validation. Kept private: callers use
# the public ``WorkerObservation`` root model.
_WorkerObservationValue = Annotated[
    SearchObservation
    | ReadObservation
    | WriteObservation
    | RunObservation
    | ActionErrorObservation,
    Field(discriminator="observation"),
]


class WorkerObservation(RootModel[_WorkerObservationValue]):
    """The closed, typed v0.1 worker-observation vocabulary.

    A concrete ``BaseModel`` (``RootModel``) subtype whose ``root`` is exactly
    one of the five observation variants, selected by the required
    ``observation`` discriminator. Being a real ``BaseModel`` it can be used
    directly wherever a structured model type is expected, while its JSON
    representation stays flat (e.g. ``{"observation": "write", "path": ...}``)
    with no externally visible ``root`` wrapper.
    """
