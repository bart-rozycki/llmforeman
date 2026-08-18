"""Focused tests for the :class:`CommandApprovalAuthorizer` policy.

These tests exercise the first concrete authorization policy only. They use
pure, test-local async callbacks and construct in-memory ``WorkerAction``
values; they perform no workspace capability execution and no filesystem, Git,
subprocess, network, or model access. No ``WorkspaceActionExecutor`` or
``ActionErrorObservation`` is created.

Async methods are driven with ``asyncio.run`` (matching the other orchestration
tests); the repository intentionally has no ``pytest-asyncio`` dependency.
"""

import asyncio
from typing import Any

import pytest

from llmforeman_core import (
    FinishAction,
    ReadFileAction,
    RunCommandAction,
    SearchAction,
    WorkerAction,
    WriteFileAction,
)
from llmforeman_orchestration import (
    CommandApprovalAuthorizer,
    WorkerActionAuthorizer,
    WorkerActionDeniedError,
)


class _RecordingApprover:
    """Async approval callback recording each received ``RunCommandAction``."""

    def __init__(self, result: object) -> None:
        self._result = result
        self.calls: list[RunCommandAction] = []

    async def __call__(self, action: RunCommandAction) -> Any:
        self.calls.append(action)
        return self._result


async def _never_called(action: RunCommandAction) -> bool:
    raise AssertionError("approval callback must not be invoked for this action")


def test_structurally_satisfies_authorizer_protocol() -> None:
    # Binding through the Protocol type proves structural compatibility under
    # mypy without runtime registration or explicit inheritance.
    authorizer: WorkerActionAuthorizer = CommandApprovalAuthorizer(_never_called)

    assert isinstance(authorizer, CommandApprovalAuthorizer)


def test_search_auto_allows_without_callback() -> None:
    authorizer = CommandApprovalAuthorizer(_never_called)
    action = WorkerAction(SearchAction(action="search", query="needle"))

    result = asyncio.run(authorizer.authorize(action))

    assert result is None


def test_read_auto_allows_without_callback() -> None:
    authorizer = CommandApprovalAuthorizer(_never_called)
    action = WorkerAction(ReadFileAction(action="read", path="pkg/module.py"))

    result = asyncio.run(authorizer.authorize(action))

    assert result is None


def test_write_auto_allows_without_callback() -> None:
    authorizer = CommandApprovalAuthorizer(_never_called)
    action = WorkerAction(
        WriteFileAction(action="write", path="pkg/module.py", content="body")
    )

    result = asyncio.run(authorizer.authorize(action))

    assert result is None


def test_finish_auto_allows_without_callback() -> None:
    authorizer = CommandApprovalAuthorizer(_never_called)
    action = WorkerAction(FinishAction(action="finish", summary="done"))

    result = asyncio.run(authorizer.authorize(action))

    assert result is None


def test_run_invokes_callback_exactly_once_with_exact_action() -> None:
    approver = _RecordingApprover(result=True)
    authorizer = CommandApprovalAuthorizer(approver)
    action = WorkerAction(RunCommandAction(action="run", command=["uv", "run", "pytest"]))

    result = asyncio.run(authorizer.authorize(action))

    assert result is None
    assert len(approver.calls) == 1
    # Identity, not equality: the exact RunCommandAction from WorkerAction.root.
    assert approver.calls[0] is action.root


def test_run_true_authorizes() -> None:
    approver = _RecordingApprover(result=True)
    authorizer = CommandApprovalAuthorizer(approver)
    action = WorkerAction(RunCommandAction(action="run", command=["pytest"]))

    result = asyncio.run(authorizer.authorize(action))

    assert result is None


def test_run_false_denies() -> None:
    approver = _RecordingApprover(result=False)
    authorizer = CommandApprovalAuthorizer(approver)
    action = WorkerAction(RunCommandAction(action="run", command=["pytest"]))

    with pytest.raises(WorkerActionDeniedError):
        asyncio.run(authorizer.authorize(action))

    # Denial does not re-attempt approval.
    assert len(approver.calls) == 1


def test_every_run_is_reapproved() -> None:
    approver = _RecordingApprover(result=True)
    authorizer = CommandApprovalAuthorizer(approver)
    first = WorkerAction(RunCommandAction(action="run", command=["pytest"]))
    second = WorkerAction(RunCommandAction(action="run", command=["ruff", "check"]))

    asyncio.run(authorizer.authorize(first))
    asyncio.run(authorizer.authorize(second))

    assert len(approver.calls) == 2
    assert approver.calls[0] is first.root
    assert approver.calls[1] is second.root


def test_no_callback_for_non_run_even_after_run() -> None:
    # Protects against accidental mutable/session policy state.
    approver = _RecordingApprover(result=True)
    authorizer = CommandApprovalAuthorizer(approver)
    run_action = WorkerAction(RunCommandAction(action="run", command=["pytest"]))
    search_action = WorkerAction(SearchAction(action="search", query="needle"))

    asyncio.run(authorizer.authorize(run_action))
    asyncio.run(authorizer.authorize(search_action))

    assert len(approver.calls) == 1


def test_callback_exception_propagates_unchanged() -> None:
    sentinel = RuntimeError("approval backend failed")

    async def failing(action: RunCommandAction) -> bool:
        raise sentinel

    authorizer = CommandApprovalAuthorizer(failing)
    action = WorkerAction(RunCommandAction(action="run", command=["pytest"]))

    with pytest.raises(RuntimeError) as excinfo:
        asyncio.run(authorizer.authorize(action))

    assert excinfo.value is sentinel


def test_callback_cancellation_propagates() -> None:
    async def cancelling(action: RunCommandAction) -> bool:
        raise asyncio.CancelledError

    authorizer = CommandApprovalAuthorizer(cancelling)
    action = WorkerAction(RunCommandAction(action="run", command=["pytest"]))

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(authorizer.authorize(action))


@pytest.mark.parametrize("bad_result", [1, "yes", None, [], object()])
def test_non_bool_result_fails_closed_with_type_error(bad_result: object) -> None:
    approver = _RecordingApprover(result=bad_result)
    authorizer = CommandApprovalAuthorizer(approver)
    action = WorkerAction(RunCommandAction(action="run", command=["pytest"]))

    with pytest.raises(TypeError):
        asyncio.run(authorizer.authorize(action))


def test_exact_argv_is_preserved_for_callback() -> None:
    approver = _RecordingApprover(result=True)
    authorizer = CommandApprovalAuthorizer(approver)
    argv = ["bash", "-c", "echo a b", "*", "&&", "$HOME", ";"]
    action = WorkerAction(RunCommandAction(action="run", command=list(argv)))

    asyncio.run(authorizer.authorize(action))

    received = approver.calls[0]
    assert isinstance(received, RunCommandAction)
    # No parsing/rewriting: argv is preserved verbatim.
    assert received.command == argv
