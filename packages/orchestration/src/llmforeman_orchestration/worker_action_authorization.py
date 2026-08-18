"""Authorization seam for model-generated worker actions.

This module defines the application/security boundary that decides whether a
single, already-validated :class:`~llmforeman_core.WorkerAction` may proceed. It
sits in the orchestration (application/composition) layer -- not in ``core`` --
because authorization is a product/security concern, not intrinsic task-domain
semantics. It depends only on the public core ``WorkerAction`` vocabulary.

The intended (future, not implemented here) worker-loop ordering is::

    1. generate WorkerAction
    2. authorize WorkerAction
    3. unwrap WorkerAction.root
    4. if FinishAction:
           end worker loop
       else:
           execute via WorkspaceActionExecutor

The key invariant is that authorization happens *before* both execution and
finish control-flow interpretation, for every model-generated action.

Two boundaries are deliberately kept distinct:

* ``WorkerAction`` *validation* is protocol/schema correctness (owned by the
  core action models); it guarantees a well-formed control message.
* ``WorkerAction`` *authorization* is permission to act; a perfectly valid
  action -- for example a ``RunCommandAction`` requesting ``["rm", "-rf",
  ...]`` -- may still be denied. Valid does not mean safe, and authorization is
  not a sandbox: it grants no filesystem, process, network, or OS-level
  containment. Those remain lower-level execution/environment concerns.
"""

from typing import Protocol

from llmforeman_core import WorkerAction

__all__ = [
    "WorkerActionAuthorizer",
    "WorkerActionDeniedError",
]


class WorkerActionDeniedError(Exception):
    """Signal that a :class:`~llmforeman_core.WorkerAction` was *denied*.

    Raised by a :class:`WorkerActionAuthorizer` to communicate an authorization
    decision: the caller has *not* authorized the action, so it must not
    proceed. It is not a workspace execution failure and must not be mapped into
    an ``ActionErrorObservation``: denial means "you were not permitted to
    attempt this", never "the workspace tried and failed". Keeping it distinct
    from execution errors ensures a denied action is not silently fed back to
    the model as an ordinary failure it might route around.

    It is intentionally a direct ``Exception`` subclass: authorization failure
    is neither a workspace capability failure nor part of any workspace error
    hierarchy. It carries only the standard Python exception message; structured
    reason/severity/retryability, approval identifiers, and diagnostics are
    deliberately out of scope.
    """


class WorkerActionAuthorizer(Protocol):
    """Typed, async contract deciding whether a ``WorkerAction`` may proceed.

    A structural interface that a concrete authorizer satisfies. It receives the
    complete public :class:`~llmforeman_core.WorkerAction` root model -- every
    variant, including ``FinishAction`` -- so that all model-generated control
    messages cross the same authorization boundary before the caller unwraps or
    interprets them. No variant is special-cased here; whether reads are auto
    allowed, writes require approval, or finish is trivial is a future policy
    decision this contract intentionally does not make.

    The decision is communicated by control flow, not a return value:

    * returning normally means the action is authorized;
    * raising :class:`WorkerActionDeniedError` means it is denied.

    This is deliberate so the decision cannot be casually ignored: a caller must
    write ``await authorizer.authorize(action)`` and only then use
    ``action.root``, rather than checking a boolean it might forget. There is no
    boolean/enum result and no authorization-result object.

    Authorization does not execute the action, does not invoke any workspace
    capability, and does not re-validate or mutate the action; the exact
    ``WorkerAction`` instance is authorized as supplied. Implementations may be
    asynchronous and interactive (for example UI or CLI approval), which is why
    the contract is async from day one.
    """

    async def authorize(self, action: WorkerAction) -> None:
        """Authorize ``action`` or raise :class:`WorkerActionDeniedError`.

        ``action`` is a complete, already-validated ``WorkerAction`` root model
        supplied by the caller, independent of which model backend produced it.
        It is authorized exactly as given: this method performs no
        serialization, reconstruction, dict conversion, shape re-validation, or
        mutation, and it does not execute the action.

        Returns ``None`` on success (the action is authorized). Raises
        :class:`WorkerActionDeniedError` to deny it. This declaration defines the
        contract only and implements no authorization policy; a concrete
        implementation owns the actual decision (and any of its own async
        dependencies).
        """
        ...
