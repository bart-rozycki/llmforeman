"""LLMForeman local coding workspace infrastructure package.

Home of infrastructure concerned with the local coding workspace (repository
and filesystem access; future Git operations, diffs, and workspace isolation).
This boundary is deliberately distinct from cloud *providers*
(``llmforeman_providers``) and local inference *runtimes*
(``llmforeman_runtimes``); it depends only on ``llmforeman_core``.

Exposes the typed async ``RepositoryContextLoader`` contract, its concrete
Git-backed implementation ``GitRepositoryContextLoader``, the typed async
``RepositoryFileReader`` contract for explicit on-demand file retrieval, its
concrete Git-backed implementation ``GitRepositoryFileReader``, and the small
workspace error hierarchy those implementations raise.
"""

from importlib.metadata import version

from llmforeman_workspace.contracts import (
    RepositoryContextLoader,
    RepositoryFileReader,
)
from llmforeman_workspace.errors import (
    InvalidRepositoryError,
    RepositoryFileAccessError,
    RepositoryInspectionError,
    WorkspaceError,
)
from llmforeman_workspace.file_reader import GitRepositoryFileReader
from llmforeman_workspace.git_loader import GitRepositoryContextLoader

__all__ = [
    "GitRepositoryContextLoader",
    "GitRepositoryFileReader",
    "InvalidRepositoryError",
    "RepositoryContextLoader",
    "RepositoryFileAccessError",
    "RepositoryFileReader",
    "RepositoryInspectionError",
    "WorkspaceError",
    "__version__",
]

__version__: str = version("llmforeman-workspace")
