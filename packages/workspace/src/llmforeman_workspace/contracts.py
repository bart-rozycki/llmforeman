"""Local coding workspace repository-access contracts.

Defines the minimal typed interfaces that future concrete implementations
satisfy so that typed application code can depend on::

    context = await loader.load(repository_root)
    file = await reader.read(repository_root, path)
    result = await searcher.search(repository_root, query)

without knowing how the local repository is inspected, how an individual file
is retrieved, or how repository text is searched. This module declares interface
semantics only: it contains no implementation, no filesystem access, no Git or
subprocess behavior, and no repository scanning.

The ``repository_root`` is a local-machine ``pathlib.Path``. It belongs at this
infrastructure boundary precisely because ``core`` (and its
``RepositoryContext``/``RepositoryFile``) is deliberately unaware of local
checkout locations, absolute paths, or the current working directory. The
context/file results remain the core models, while text-search results are the
workspace-owned ``RepositorySearchResult``: this boundary translates a local
repository into normalized, provider- and runtime-agnostic data.
"""

from pathlib import Path
from typing import Protocol

from llmforeman_core import RepositoryContext, RepositoryFile
from llmforeman_workspace.search import RepositorySearchResult

__all__ = [
    "RepositoryContextLoader",
    "RepositoryFileReader",
    "RepositoryTextSearcher",
]


class RepositoryContextLoader(Protocol):
    """Typed, async contract for loading a ``RepositoryContext``.

    A structural interface that a concrete local workspace loader satisfies. It
    is stateless and holds no configuration, selection policy, ignore rules,
    token budget, caching, or lifecycle hooks. It exists only so typed
    application code can depend on ``await loader.load(repository_root)``.
    """

    async def load(self, repository_root: Path) -> RepositoryContext:
        """Load a ``RepositoryContext`` from a local ``repository_root``.

        Asynchronous from day one because real implementations will involve
        local filesystem I/O, Git/subprocess operations, and orchestration
        concurrency. This declaration defines the contract only and implements
        no loading behavior; it does not access the filesystem or validate
        ``repository_root``.
        """
        ...


class RepositoryFileReader(Protocol):
    """Typed, async contract for reading one repository file on demand.

    A structural interface that a concrete local workspace reader satisfies. It
    is the narrower, on-demand complement to ``RepositoryContextLoader``:
    where the loader produces the initial repository snapshot, this reader
    retrieves a single, explicitly named file. It is stateless and holds no
    configuration, selection policy, ignore rules, token budget, caching, or
    lifecycle hooks. It exists only so typed application code can depend on
    ``await reader.read(repository_root, path)``.

    This declares interface semantics only. It performs no filesystem or Git
    access, no path validation, and no encoding, size, or containment policy;
    those runtime semantics belong to a future concrete implementation.
    """

    async def read(self, repository_root: Path, path: str) -> RepositoryFile:
        """Read one repository file into the core ``RepositoryFile`` model.

        ``repository_root`` is a local-machine ``pathlib.Path``. ``path`` is a
        logical, repository-relative file identifier (for example
        ``"packages/core/src/llmforeman_core/models.py"``); it is expected to be
        repository-relative, and it is intentionally kept as a ``str`` rather
        than a filesystem ``Path`` because it ultimately becomes
        ``RepositoryFile.path``.

        Asynchronous from day one because real implementations will involve
        local filesystem I/O, Git/subprocess operations, and orchestration
        concurrency. This declaration defines the contract only and implements
        no reading behavior; it does not access the filesystem, resolve or
        validate ``path``, or verify that the file exists or is tracked.
        """
        ...


class RepositoryTextSearcher(Protocol):
    """Typed, async contract for plain-text repository search.

    A structural interface that a concrete local workspace searcher satisfies.
    It is the third, independent repository-exploration capability alongside
    ``RepositoryContextLoader`` (initial lightweight snapshot) and
    ``RepositoryFileReader`` (read one explicitly known file): it answers
    "where does this text occur?" when the caller does not yet know the file.
    It is stateless and holds no configuration, ignore rules, token budget,
    caching, result limit, or lifecycle hooks. It exists only so typed
    application code can depend on ``await searcher.search(repository_root,
    query)``.

    This declares interface semantics only. It performs no filesystem, Git,
    ripgrep, or subprocess access, no directory traversal, and no query
    validation; how a query string is mapped onto a search engine, whether only
    Git-tracked files are searched, result ordering, and any output bound all
    belong to a future concrete implementation.
    """

    async def search(
        self,
        repository_root: Path,
        query: str,
    ) -> RepositorySearchResult:
        """Search a local repository for a plain-text ``query``.

        ``repository_root`` is a local-machine ``pathlib.Path``. ``query`` is a
        plain (literal) text search string, kept as a ``str`` and carrying no
        regex, case-sensitivity, glob, file-filter, whole-word, or
        context-line semantics at this contract level.

        Precondition: ``query`` must contain meaningful, non-whitespace content
        (``""``, ``"   "``, and ``"\t\n"`` are conceptually invalid). This is a
        documented precondition only; a ``Protocol`` cannot enforce it at
        runtime, so a future concrete implementation must reject a blank query
        before starting a search rather than relying on this declaration.

        Asynchronous from day one because real implementations will involve
        local filesystem I/O, Git/subprocess operations, and orchestration
        concurrency. This declaration defines the contract only and implements
        no search behavior; it does not access the filesystem, validate
        ``query``, or bound the number of results. Finding nothing is a valid,
        non-error outcome represented by an empty ``RepositorySearchResult``.
        """
        ...
