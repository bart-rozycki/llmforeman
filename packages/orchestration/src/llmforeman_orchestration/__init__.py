"""LLMForeman orchestration package.

Application/composition layer that intentionally composes core worker semantics
(``WorkerAction``/``WorkerObservation``) with workspace capabilities
(search/read/write/run) and, for the local coding-agent loop, a local
structured runtime. It depends only on ``llmforeman-core``,
``llmforeman-workspace``, and ``llmforeman-runtimes``; nothing in ``core`` or
``workspace`` depends back on it, and it must not depend on providers, the CLI,
or the desktop app.

It exposes the concrete application service :class:`WorkspaceActionExecutor`,
the authorization seam (:class:`WorkerActionAuthorizer` plus its denial signal
:class:`WorkerActionDeniedError`) that decides whether a model-generated
``WorkerAction`` may proceed before it is executed or interpreted, and the first
bounded local coding-agent loop :class:`LocalCodingWorker` (with its
:class:`LocalWorkerResult` and :class:`WorkerStepLimitError`) that composes a
local structured runtime with those capabilities. Composing the runtime here is
why orchestration now also depends on ``llmforeman-runtimes``.
"""

from importlib.metadata import version

from llmforeman_orchestration.local_coding_worker import (
    LocalCodingWorker,
    LocalWorkerResult,
    WorkerStepLimitError,
)
from llmforeman_orchestration.worker_action_authorization import (
    WorkerActionAuthorizer,
    WorkerActionDeniedError,
)
from llmforeman_orchestration.workspace_action_executor import WorkspaceActionExecutor

__all__ = [
    "LocalCodingWorker",
    "LocalWorkerResult",
    "WorkerActionAuthorizer",
    "WorkerActionDeniedError",
    "WorkerStepLimitError",
    "WorkspaceActionExecutor",
    "__version__",
]

__version__: str = version("llmforeman-orchestration")
