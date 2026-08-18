"""Contract/type tests for the worker-action authorization seam.

These tests exercise the :class:`WorkerActionAuthorizer` Protocol and the
:class:`WorkerActionDeniedError` denial signal only. They use pure, test-local
structural fakes and construct in-memory ``WorkerAction`` values; they perform
no workspace capability execution and no filesystem, Git, subprocess, network,
or model access. No concrete authorizer or ``ActionErrorObservation`` is created.

Async methods are driven with ``asyncio.run`` (matching the executor tests);
the repository intentionally has no ``pytest-asyncio`` dependency.
"""

import asyncio
from typing import assert_type, get_type_hints

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
    WorkerActionAuthorizer,
    WorkerActionDeniedError,
)


class _RecordingAuthorizer:
    """Structural :class:`WorkerActionAuthorizer` that records and authorizes."""

    def __init__(self) -> None:
        self.received_action: WorkerAction | None = None

    async def authorize(self, action: WorkerAction) -> None:
        self.received_action = action


class _DenyingAuthorizer:
    """Structural :class:`WorkerActionAuthorizer` that denies every action."""

    async def authorize(self, action: WorkerAction) -> None:
        raise WorkerActionDeniedError("denied")


def _all_actions() -> list[WorkerAction]:
    return [
        WorkerAction(SearchAction(action="search", query="needle")),
        WorkerAction(ReadFileAction(action="read", path="pkg/module.py")),
        WorkerAction(
            WriteFileAction(action="write", path="pkg/module.py", content="body")
        ),
        WorkerAction(RunCommandAction(action="run", command=["pytest"])),
        WorkerAction(FinishAction(action="finish", summary="done")),
    ]


def test_recording_fake_satisfies_protocol() -> None:
    # Binding through the Protocol type proves the fake structurally satisfies
    # WorkerActionAuthorizer under mypy without runtime registration.
    authorizer: WorkerActionAuthorizer = _RecordingAuthorizer()

    assert isinstance(authorizer, _RecordingAuthorizer)


def test_normal_authorization_returns_none() -> None:
    authorizer: WorkerActionAuthorizer = _RecordingAuthorizer()
    action = WorkerAction(SearchAction(action="search", query="needle"))

    result = asyncio.run(authorizer.authorize(action))

    assert result is None


def test_exact_worker_action_instance_is_received() -> None:
    fake = _RecordingAuthorizer()
    authorizer: WorkerActionAuthorizer = fake
    action = WorkerAction(ReadFileAction(action="read", path="pkg/module.py"))

    asyncio.run(authorizer.authorize(action))

    # Identity, not equality: the contract forbids serialization/reconstruction.
    assert fake.received_action is action


def test_denial_propagates_to_caller() -> None:
    authorizer: WorkerActionAuthorizer = _DenyingAuthorizer()
    action = WorkerAction(RunCommandAction(action="run", command=["rm", "-rf", "/"]))

    with pytest.raises(WorkerActionDeniedError):
        asyncio.run(authorizer.authorize(action))


def test_all_five_action_variants_pass_through() -> None:
    fake = _RecordingAuthorizer()
    authorizer: WorkerActionAuthorizer = fake

    for action in _all_actions():
        asyncio.run(authorizer.authorize(action))
        assert fake.received_action is action


def test_finish_action_is_not_bypassed() -> None:
    fake = _RecordingAuthorizer()
    authorizer: WorkerActionAuthorizer = fake
    action = WorkerAction(FinishAction(action="finish", summary="done"))

    asyncio.run(authorizer.authorize(action))

    assert fake.received_action is action
    assert isinstance(action.root, FinishAction)


def test_finish_action_may_be_denied() -> None:
    # The seam does not silently auto-approve finish; a policy MAY deny it.
    authorizer: WorkerActionAuthorizer = _DenyingAuthorizer()
    action = WorkerAction(FinishAction(action="finish", summary="done"))

    with pytest.raises(WorkerActionDeniedError):
        asyncio.run(authorizer.authorize(action))


def test_denied_error_is_not_a_workspace_error() -> None:
    # Package-ownership / inheritance protection without coupling to workspace:
    # a direct Exception subclass, never part of a workspace error hierarchy.
    assert issubclass(WorkerActionDeniedError, Exception)
    assert WorkerActionDeniedError.__mro__[1] is Exception


def test_authorize_return_annotation_is_none() -> None:
    hints = get_type_hints(WorkerActionAuthorizer.authorize)

    assert hints["return"] is type(None)


def test_authorize_result_is_typed_none() -> None:
    authorizer: WorkerActionAuthorizer = _RecordingAuthorizer()
    action = WorkerAction(SearchAction(action="search", query="needle"))

    result = asyncio.run(authorizer.authorize(action))

    assert_type(result, None)
