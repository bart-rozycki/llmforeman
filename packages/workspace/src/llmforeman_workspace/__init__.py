"""LLMForeman local coding workspace infrastructure package.

Home of infrastructure concerned with the local coding workspace (repository
and filesystem access; future Git operations, diffs, and workspace isolation).
This boundary is deliberately distinct from cloud *providers*
(``llmforeman_providers``) and local inference *runtimes*
(``llmforeman_runtimes``); it depends only on ``llmforeman_core``.

Exposes the typed async ``RepositoryContextLoader`` contract for loading a core
``RepositoryContext`` from a local repository root. No concrete loader is
implemented yet.
"""

from importlib.metadata import version

from llmforeman_workspace.contracts import RepositoryContextLoader

__all__ = [
    "RepositoryContextLoader",
    "__version__",
]

__version__: str = version("llmforeman-workspace")
