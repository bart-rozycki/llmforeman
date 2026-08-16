"""Workspace-specific failure contract for local repository inspection.

The concrete :class:`~llmforeman_workspace.git_loader.GitRepositoryContextLoader`
translates the low-level filesystem and Git subprocess failures it encounters
into this small, infrastructure-agnostic hierarchy at the loader boundary.
Ordinary callers can therefore distinguish *invalid caller input* from *a
genuine inspection failure* without catching raw ``subprocess`` /
``OSError`` types or knowing that Git is used under the hood.

The hierarchy is deliberately minimal: it captures only the two distinctions a
real local workspace loader currently needs. It intentionally avoids errno- or
Git-exit-code-specific classes. Original causality is preserved via
``raise ... from original`` where a meaningful cause exists; error messages
never include repository file contents, environment variables, or secrets.
"""

__all__ = [
    "InvalidRepositoryError",
    "RepositoryInspectionError",
    "WorkspaceError",
]


class WorkspaceError(Exception):
    """Base class for all workspace repository-loading failures.

    Raised (via a subclass) when a loader cannot produce a trustworthy
    ``RepositoryContext``. Callers may catch this base type to handle any
    workspace-level failure uniformly.
    """


class InvalidRepositoryError(WorkspaceError):
    """The caller-supplied repository input is not a usable Git working tree.

    Represents caller/input faults such as a path that does not exist, a path
    that is not a directory, a directory Git does not consider part of a working
    tree, and bare/no-working-tree repositories.
    """


class RepositoryInspectionError(WorkspaceError):
    """A valid repository was identified but could not be inspected safely.

    Represents failures that occur *after* a working tree has been resolved,
    such as the Git executable not being launchable, ``git ls-files``
    unexpectedly failing, or tracked path output that cannot be represented
    safely and deterministically in the core string-path contract.
    """
