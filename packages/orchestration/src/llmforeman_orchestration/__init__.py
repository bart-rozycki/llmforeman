"""LLMForeman orchestration package.

Application/composition layer that intentionally composes core worker semantics
(``WorkerAction``/``WorkerObservation``) with workspace capabilities
(search/read/write/run). It depends only on ``llmforeman-core`` and
``llmforeman-workspace``; nothing in ``core`` or ``workspace`` depends back on
it, and it must not depend on providers, runtimes, the CLI, or the desktop app.

At this stage it exposes exactly one concrete application service,
:class:`WorkspaceActionExecutor`.
"""

from importlib.metadata import version

from llmforeman_orchestration.workspace_action_executor import WorkspaceActionExecutor

__all__ = [
    "WorkspaceActionExecutor",
    "__version__",
]

__version__: str = version("llmforeman-orchestration")
