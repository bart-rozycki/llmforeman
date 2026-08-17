"""LLMForeman core package.

Home of the provider- and runtime-agnostic product/domain/orchestration
runtime. This module is intentionally empty of product logic at this stage;
it only establishes the package boundary.
"""

from importlib.metadata import version

from llmforeman_core.foreman import Foreman, ForemanPlanValidationError
from llmforeman_core.models import (
    AgentRole,
    ModelUsage,
    RepositoryContext,
    RepositoryFile,
    Run,
    Task,
    TaskPlan,
    TaskStatus,
    can_transition,
)
from llmforeman_core.worker_actions import (
    FinishAction,
    ReadFileAction,
    RunCommandAction,
    SearchAction,
    WorkerAction,
    WriteFileAction,
)
from llmforeman_core.worker_observations import (
    ActionErrorObservation,
    ReadObservation,
    RunObservation,
    SearchObservation,
    WorkerObservation,
    WorkerSearchMatch,
    WriteObservation,
)

__all__ = [
    "ActionErrorObservation",
    "AgentRole",
    "FinishAction",
    "Foreman",
    "ForemanPlanValidationError",
    "ModelUsage",
    "ReadFileAction",
    "ReadObservation",
    "RepositoryContext",
    "RepositoryFile",
    "Run",
    "RunCommandAction",
    "RunObservation",
    "SearchAction",
    "SearchObservation",
    "Task",
    "TaskPlan",
    "TaskStatus",
    "WorkerAction",
    "WorkerObservation",
    "WorkerSearchMatch",
    "WriteFileAction",
    "WriteObservation",
    "__version__",
    "can_transition",
]

__version__: str = version("llmforeman-core")
