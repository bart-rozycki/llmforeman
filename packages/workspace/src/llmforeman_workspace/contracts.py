"""Local coding workspace repository-loading contract.

Defines the minimal typed interface that a future concrete loader satisfies so
that typed application code can depend on::

    context = await loader.load(repository_root)

without knowing how the local repository is inspected. This module declares
interface semantics only: it contains no loader implementation, no filesystem
access, no Git or subprocess behavior, and no repository scanning.

The ``repository_root`` is a local-machine ``pathlib.Path``. It belongs at this
infrastructure boundary precisely because ``core`` (and its
``RepositoryContext``) is deliberately unaware of local checkout locations,
absolute paths, or the current working directory. The result remains the core
``RepositoryContext``: this boundary translates a local repository into the
normalized, provider- and runtime-agnostic core model.
"""

from pathlib import Path
from typing import Protocol

from llmforeman_core import RepositoryContext

__all__ = [
    "RepositoryContextLoader",
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
