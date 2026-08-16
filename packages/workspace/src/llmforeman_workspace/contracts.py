"""Local coding workspace repository-access contracts.

Defines the minimal typed interfaces that future concrete implementations
satisfy so that typed application code can depend on::

    context = await loader.load(repository_root)
    file = await reader.read(repository_root, path)

without knowing how the local repository is inspected or how an individual file
is retrieved. This module declares interface semantics only: it contains no
implementation, no filesystem access, no Git or subprocess behavior, and no
repository scanning.

The ``repository_root`` is a local-machine ``pathlib.Path``. It belongs at this
infrastructure boundary precisely because ``core`` (and its
``RepositoryContext``/``RepositoryFile``) is deliberately unaware of local
checkout locations, absolute paths, or the current working directory. Results
remain the core models: this boundary translates a local repository into the
normalized, provider- and runtime-agnostic core model.
"""

from pathlib import Path
from typing import Protocol

from llmforeman_core import RepositoryContext, RepositoryFile

__all__ = [
    "RepositoryContextLoader",
    "RepositoryFileReader",
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
