"""LLMForeman orchestration package.

Application/composition layer that intentionally composes core worker semantics
(``WorkerAction``/``WorkerObservation``) with workspace capabilities
(search/read/write/run). It depends only on ``llmforeman-core`` and
``llmforeman-workspace``; nothing in ``core`` or ``workspace`` depends back on
it, and it must not depend on providers, runtimes, the CLI, or the desktop app.

It exposes the concrete application service :class:`WorkspaceActionExecutor`
and the authorization seam (:class:`WorkerActionAuthorizer` plus its denial
signal :class:`WorkerActionDeniedError`) that decides whether a model-generated
``WorkerAction`` may proceed before it is executed or interpreted.
"""

from importlib.metadata import version

from llmforeman_orchestration.worker_action_authorization import (
    WorkerActionAuthorizer,
    WorkerActionDeniedError,
)
from llmforeman_orchestration.workspace_action_executor import WorkspaceActionExecutor

__all__ = [
    "WorkerActionAuthorizer",
    "WorkerActionDeniedError",
    "WorkspaceActionExecutor",
    "__version__",
]

__version__: str = version("llmforeman-orchestration")
