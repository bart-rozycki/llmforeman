"""LLMForeman core package.

Home of the provider- and runtime-agnostic product/domain/orchestration
runtime. This module is intentionally empty of product logic at this stage;
it only establishes the package boundary.
"""

from importlib.metadata import version

from llmforeman_core.foreman import Foreman
from llmforeman_core.models import (
    AgentRole,
    ModelUsage,
    Run,
    Task,
    TaskPlan,
    TaskStatus,
    can_transition,
)

__all__ = [
    "AgentRole",
    "Foreman",
    "ModelUsage",
    "Run",
    "Task",
    "TaskPlan",
    "TaskStatus",
    "__version__",
    "can_transition",
]

__version__: str = version("llmforeman-core")
