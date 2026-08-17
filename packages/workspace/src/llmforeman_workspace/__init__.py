"""LLMForeman local coding workspace infrastructure package.

Home of infrastructure concerned with the local coding workspace (repository
and filesystem access; future Git operations, diffs, and workspace isolation).
This boundary is deliberately distinct from cloud *providers*
(``llmforeman_providers``) and local inference *runtimes*
(``llmforeman_runtimes``); it depends only on ``llmforeman_core``.

Exposes the typed async ``RepositoryContextLoader`` contract, its concrete
Git-backed implementation ``GitRepositoryContextLoader``, the typed async
``RepositoryFileReader`` contract for explicit on-demand file retrieval, its
concrete Git-backed implementation ``GitRepositoryFileReader``, the typed async
``RepositoryTextSearcher`` contract for plain-text repository search together
with its workspace-owned ``RepositorySearchMatch``/``RepositorySearchResult``
result models and the concrete ripgrep-backed, Git-tracked-only
``RipgrepRepositoryTextSearcher``, the typed async ``RepositoryFileWriter``
contract for writing one repository file's requested state together with its
concrete Git-bounded implementation ``GitRepositoryFileWriter`` (a
Git-independent mutation capability that overwrites tracked/untracked files and
creates new untracked files within the effective working tree), and the small
workspace error hierarchy those implementations raise.
"""

from importlib.metadata import version

from llmforeman_workspace.contracts import (
    RepositoryContextLoader,
    RepositoryFileReader,
    RepositoryFileWriter,
    RepositoryTextSearcher,
)
from llmforeman_workspace.errors import (
    InvalidRepositoryError,
    RepositoryFileAccessError,
    RepositoryFileWriteError,
    RepositoryInspectionError,
    RepositorySearchError,
    WorkspaceError,
)
from llmforeman_workspace.file_reader import GitRepositoryFileReader
from llmforeman_workspace.file_writer import GitRepositoryFileWriter
from llmforeman_workspace.git_loader import GitRepositoryContextLoader
from llmforeman_workspace.ripgrep_searcher import RipgrepRepositoryTextSearcher
from llmforeman_workspace.search import (
    RepositorySearchMatch,
    RepositorySearchResult,
)

__all__ = [
    "GitRepositoryContextLoader",
    "GitRepositoryFileReader",
    "GitRepositoryFileWriter",
    "InvalidRepositoryError",
    "RepositoryContextLoader",
    "RepositoryFileAccessError",
    "RepositoryFileReader",
    "RepositoryFileWriteError",
    "RepositoryFileWriter",
    "RepositoryInspectionError",
    "RepositorySearchError",
    "RepositorySearchMatch",
    "RepositorySearchResult",
    "RepositoryTextSearcher",
    "RipgrepRepositoryTextSearcher",
    "WorkspaceError",
    "__version__",
]

__version__: str = version("llmforeman-workspace")
