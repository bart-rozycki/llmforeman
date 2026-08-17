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
    "RepositoryFileAccessError",
    "RepositoryFileWriteError",
    "RepositoryInspectionError",
    "RepositorySearchError",
    "WorkspaceCommandExecutionError",
    "WorkspaceCommandTimeoutError",
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


class RepositoryFileAccessError(WorkspaceError):
    """An explicitly requested repository file could not be safely returned.

    Raised by the on-demand file reader for the ordinary, expected ways a single
    explicit read can fail once the repository itself is valid: an invalid
    repository-relative requested path, a path that is not Git-tracked, a tracked
    file missing from the working tree, a target that is not a regular readable
    file, a file exceeding the configured size limit, invalid UTF-8, binary-like
    NUL-containing content, or a symlink that would escape the repository root.

    Unlike best-effort seed gathering (which silently skips unsuitable files),
    an explicit read that cannot return the requested file surfaces this error.
    Messages may name the requested repository-relative path but never include
    file contents, resolved absolute paths, environment variables, or secrets.
    """


class RepositoryFileWriteError(WorkspaceError):
    """A requested repository file write could not be performed safely.

    Raised by the concrete file writer for the ordinary, expected ways a single
    whole-file write can fail once the repository itself is valid: an invalid
    repository-relative requested path, content that cannot be encoded as strict
    UTF-8, requested content exceeding the configured size limit, a symlink
    encountered as any existing parent component or as the final target, a parent
    path that is a regular file or other non-directory object, a final target
    that is a directory or other non-regular object, an existing target whose
    current contents are not valid UTF-8 text (invalid encoding or NUL bytes),
    and permission/filesystem failures encountered during the mutation.

    It is deliberately distinct from :class:`InvalidRepositoryError` (invalid
    caller repository input) and :class:`RepositoryInspectionError` (Git
    inspection failure): those are never rewrapped into this error, so callers
    retain the distinction between an invalid workspace, a Git inspection
    failure, and a write failure. Original causality is preserved via
    ``raise ... from original`` where a meaningful OS cause exists. Messages may
    name the requested repository-relative path but never include file contents,
    resolved absolute paths, environment variables, or secrets.
    """


class RepositorySearchError(WorkspaceError):
    """A repository text search could not be executed or trusted to completion.

    Raised by the concrete ripgrep-backed searcher for search-layer failures
    that occur *after* a valid working tree has been resolved: a blank or
    NUL-containing query that cannot be searched meaningfully, an inability to
    launch the ``rg`` executable, a non-success ripgrep execution status,
    malformed or structurally unusable ripgrep JSON output, a reported match
    path outside the exact approved tracked-candidate set, or a single tracked
    path too large to pass safely within the subprocess argument budget.

    It is deliberately distinct from :class:`InvalidRepositoryError` (invalid
    caller repository input) and :class:`RepositoryInspectionError` (Git
    inspection failure): those are never rewrapped into this error, so callers
    retain the distinction between an invalid workspace, a Git inspection
    failure, and a text-search failure. A search that runs successfully and
    finds nothing is *not* an error; it is an empty ``RepositorySearchResult``.
    Messages never include repository file contents, resolved absolute paths,
    environment variables, or secrets.
    """


class WorkspaceCommandExecutionError(WorkspaceError):
    """A workspace command could not be executed to a trustworthy completion.

    Raised by the concrete subprocess command runner for failures that mean a
    command did not run to a normal completion whose result can be trusted:
    invalid command argv (empty command, empty/whitespace-only executable,
    empty or NUL-containing argv entries), an executable that cannot be started
    (for example a missing binary), a local subprocess-creation or
    infrastructure failure, captured output exceeding the configured per-stream
    limit, and a cleanup failure that prevents a trustworthy result.

    It is deliberately distinct from :class:`InvalidRepositoryError` (invalid
    caller repository input) and :class:`RepositoryInspectionError` (Git
    inspection failure): those are never rewrapped into this error, so callers
    retain the distinction between an invalid workspace, a Git inspection
    failure, and a command execution failure. Crucially, it is **never** raised
    for an ordinary non-zero process exit: a command that runs to completion and
    exits non-zero is a normal :class:`~llmforeman_workspace.command.CommandResult`.
    Original causality is preserved via ``raise ... from original`` where a
    meaningful cause exists. Messages may name the executable, the configured
    timeout, or which stream exceeded its limit, but never include captured
    output, environment variables, resolved absolute paths, or secrets.
    """


class WorkspaceCommandTimeoutError(WorkspaceCommandExecutionError):
    """A workspace command exceeded its configured execution timeout.

    Raised specifically when a successfully started subprocess does not complete
    within the concrete runner's configured ``timeout_seconds``. Before this is
    raised the runner terminates the whole spawned process group and reaps the
    direct process, so no orphaned command is left running. A timeout is not a
    :class:`~llmforeman_workspace.command.CommandResult`: the capability did not
    complete normally, so no partial or synthetic exit code is returned. It is a
    subclass of :class:`WorkspaceCommandExecutionError` so callers may catch the
    broader execution failure or handle a timeout specifically.
    """
