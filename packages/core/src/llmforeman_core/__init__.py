"""LLMForeman core package.

Home of the provider- and runtime-agnostic product/domain/orchestration
runtime. This module is intentionally empty of product logic at this stage;
it only establishes the package boundary.
"""

from importlib.metadata import version

__all__ = ["__version__"]

__version__: str = version("llmforeman-core")
