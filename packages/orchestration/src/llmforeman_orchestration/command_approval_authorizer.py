"""First concrete worker-action authorization policy.

This module implements :class:`CommandApprovalAuthorizer`, the initial v0.1
product trust decision for model-generated worker actions. It structurally
satisfies the :class:`~llmforeman_orchestration.WorkerActionAuthorizer` contract
and lives in the orchestration (application/composition) layer because it
encodes a product/security policy, not intrinsic task-domain semantics.

The policy is intentionally simple::

    SearchAction    -> authorized automatically
    ReadFileAction  -> authorized automatically
    WriteFileAction -> authorized automatically
    FinishAction    -> authorized automatically
    RunCommandAction -> requires explicit per-action approval

Repository-scoped semantic capabilities (search/read/write) and the finish
control message are auto-authorized under the current workspace boundaries;
host process execution (``run``) always requires explicit approval, obtained by
invoking an injected async callback with the exact ``RunCommandAction``.

This is authorization, not a sandbox: approval grants no process, filesystem,
network, or OS-level isolation. Once approved, the eventual command runner will
execute with its existing process permissions. The policy does not claim that
repository operations are universally harmless; it encodes the current v0.1
trust decision only.
"""

from collections.abc import Awaitable, Callable

from llmforeman_core import RunCommandAction, WorkerAction
from llmforeman_orchestration.worker_action_authorization import WorkerActionDeniedError

__all__ = [
    "CommandApprovalAuthorizer",
]

# Private, non-exported alias for the injected approval callback. It receives
# the exact ``RunCommandAction`` being authorized and returns an awaitable
# ``bool`` (``True`` allows, ``False`` denies). Kept private on purpose: the
# task requires a plain typed async callable, not a public Protocol/abstraction.
_ApproveCommand = Callable[[RunCommandAction], Awaitable[bool]]


class CommandApprovalAuthorizer:
    """Authorize repository-scoped actions automatically; approve every ``run``.

    Concrete :class:`~llmforeman_orchestration.WorkerActionAuthorizer` policy.
    ``SearchAction``, ``ReadFileAction``, ``WriteFileAction``, and
    ``FinishAction`` are authorized automatically without consulting the
    callback. Every :class:`~llmforeman_core.RunCommandAction` requires explicit
    approval: the injected async callback is invoked with the exact action, and
    only a literal ``True`` authorizes it.

    Approval applies to one concrete ``RunCommandAction`` only. No decision is
    cached and no permission survives to another action, so each ``run`` is
    reapproved independently; the authorizer holds no mutable authorization
    state beyond the injected callback.

    This is an authorization policy, not a command sandbox: it provides no
    process, filesystem, network, or OS-level isolation. It also does not
    execute any action or workspace capability -- it makes a decision only.
    """

    def __init__(self, approve_command: _ApproveCommand) -> None:
        self._approve_command = approve_command

    async def authorize(self, action: WorkerAction) -> None:
        """Authorize ``action`` or raise :class:`WorkerActionDeniedError`.

        Non-``run`` actions return ``None`` immediately. For a
        ``RunCommandAction`` the injected callback is awaited with the exact
        action instance; ``True`` authorizes, ``False`` raises
        :class:`WorkerActionDeniedError`. A non-``bool`` result is a callback
        contract violation and fails closed with :class:`TypeError`. Callback
        exceptions and cancellation propagate unchanged.
        """
        value = action.root

        if not isinstance(value, RunCommandAction):
            return

        approved = await self._approve_command(value)

        if approved is True:
            return

        if approved is False:
            raise WorkerActionDeniedError("Command execution was denied.")

        raise TypeError("Command approval callback must return a bool.")
