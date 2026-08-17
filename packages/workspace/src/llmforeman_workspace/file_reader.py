"""Git-backed concrete :class:`RepositoryFileReader` for local workspaces.

``GitRepositoryFileReader`` provides the one safe primitive: *read this
explicitly named, Git-tracked repository file*. It is the narrow, on-demand
complement to ``GitRepositoryContextLoader``. It does not search, glob, list, or
interact with any model, provider, or runtime.

Security model (this is a privacy-sensitive local workspace boundary):

* Git determines the effective working-tree top-level (via the shared ``_git``
  primitives); ``.git`` is never inspected directly, so linked worktrees work.
* The caller-supplied logical path is validated as repository-relative *before*
  it is ever used to touch the filesystem: empty/whitespace, NUL, absolute
  (POSIX or Windows), and parent-traversal paths are rejected, never rewritten.
* Only paths present in Git's exact tracked/index set are eligible. Filesystem
  existence is not permission to read: a guessed but untracked ``.env`` fails
  without its contents being opened. Membership is an exact Python set check
  against the NUL-delimited tracked listing, so caller-controlled Git pathspec
  magic, globbing, and quoting never apply.
* Content comes from the *current working tree* (so locally modified tracked
  files are visible), read through a bounded operation on a worker thread. The
  resolved target must be a regular file contained within the canonical root
  (symlinks may not escape it). Oversized files, invalid UTF-8, and NUL-byte
  binary-like content fail rather than being truncated or lossily decoded.
* No absolute local path ever enters the returned ``RepositoryFile``: it carries
  the requested logical tracked path and the decoded text content only.

Unlike the loader's best-effort seed gathering (which silently skips unsuitable
files), an explicit read that cannot return the requested file raises
:class:`RepositoryFileAccessError`.
"""

import asyncio
from pathlib import Path
from typing import Final

from llmforeman_core import RepositoryFile
from llmforeman_workspace._git import (
    list_tracked_paths,
    resolve_worktree_top_level,
)
from llmforeman_workspace._paths import validate_logical_repository_path
from llmforeman_workspace.errors import RepositoryFileAccessError

__all__ = [
    "GitRepositoryFileReader",
]

_DEFAULT_MAX_FILE_BYTES: Final[int] = 1024 * 1024


class GitRepositoryFileReader:
    """Read one explicitly named, Git-tracked file into a ``RepositoryFile``.

    Satisfies the async :class:`RepositoryFileReader` protocol. The optional
    ``max_file_bytes`` bounds the read; a file larger than the limit fails
    (never truncated). This limit is concrete reader infrastructure and is
    intentionally absent from the core models and the reader protocol.

    The tracked-only policy is deliberate concrete implementation policy, not a
    property of the generic protocol: only paths Git already tracks are eligible
    for an explicit read.
    """

    def __init__(
        self,
        *,
        max_file_bytes: int = _DEFAULT_MAX_FILE_BYTES,
    ) -> None:
        # ``bool`` is an ``int`` subclass; reject it explicitly so a stray
        # ``True`` cannot masquerade as a size limit.
        if isinstance(max_file_bytes, bool) or not isinstance(max_file_bytes, int):
            raise ValueError("max_file_bytes must be a positive integer")
        if max_file_bytes <= 0:
            raise ValueError("max_file_bytes must be a positive integer")
        self._max_file_bytes = max_file_bytes

    async def read(self, repository_root: Path, path: str) -> RepositoryFile:
        """Read the tracked working-tree file at logical ``path``.

        ``repository_root`` is any local path inside the target working tree;
        Git resolves the canonical top-level used as the effective root.
        ``path`` is a logical, repository-relative identifier that is validated
        before any filesystem access and must be present in Git's tracked set.

        Raises :class:`~llmforeman_workspace.errors.InvalidRepositoryError` for
        an invalid repository entry,
        :class:`~llmforeman_workspace.errors.RepositoryInspectionError` for a
        Git inspection failure, and
        :class:`~llmforeman_workspace.errors.RepositoryFileAccessError` for any
        expected inability to return the explicitly requested file.
        """

        # Repository validation and effective-root resolution first (may raise
        # InvalidRepositoryError / RepositoryInspectionError).
        effective_root = await resolve_worktree_top_level(repository_root)

        # Validate the logical path BEFORE it is used to touch the filesystem.
        _validate_logical_path(path)

        # Exact tracked membership is the permission boundary. Untracked paths
        # (including guessed secrets) fail here without their contents opened.
        tracked = await list_tracked_paths(effective_root)
        if path not in set(tracked):
            raise RepositoryFileAccessError(
                f"requested path is not tracked in the repository: {path!r}"
            )

        content = await asyncio.to_thread(
            self._read_working_tree_file, effective_root, path
        )
        return RepositoryFile(path=path, content=content)

    def _read_working_tree_file(self, effective_root: Path, path: str) -> str:
        """Return the decoded working-tree content of a tracked ``path``.

        Runs on a worker thread so filesystem I/O never blocks the event loop.
        Every expected inability to safely return the file becomes
        :class:`RepositoryFileAccessError`; contents and absolute paths never
        appear in the raised message.
        """

        candidate = effective_root / path

        # Resolve without requiring existence so broken/dangling symlinks and
        # missing files fall through to the readability checks below.
        resolved = candidate.resolve()

        # Containment: the resolved target must stay within the canonical root.
        # Proper path semantics avoid unsafe string-prefix comparisons; a
        # symlink escaping the root is rejected before any content is read.
        if not resolved.is_relative_to(effective_root):
            raise RepositoryFileAccessError(
                f"requested path resolves outside the repository: {path!r}"
            )

        # Only a regular file (directly or via an in-repo symlink) is readable
        # content; directories, submodules, sockets, devices, FIFOs, broken
        # symlinks, and tracked-but-locally-deleted files are rejected.
        if not resolved.is_file():
            raise RepositoryFileAccessError(
                f"requested path is not a readable regular file: {path!r}"
            )

        try:
            with open(resolved, "rb") as handle:  # noqa: PTH123 - bounded read
                data = handle.read(self._max_file_bytes + 1)
        except OSError as original:
            raise RepositoryFileAccessError(
                f"requested file could not be read: {path!r}"
            ) from original

        # Reading one byte beyond the limit distinguishes ``size <= limit`` from
        # ``size > limit`` without loading an arbitrarily large file. Oversized
        # files fail; content is never truncated and returned as if complete.
        if len(data) > self._max_file_bytes:
            raise RepositoryFileAccessError(
                f"requested file exceeds the maximum allowed size: {path!r}"
            )

        # Minimal binary guard: reject a NUL byte even if the bytes would
        # otherwise decode as UTF-8. This is a tiny safety rule, not a detector.
        if b"\x00" in data:
            raise RepositoryFileAccessError(
                f"requested file contains NUL bytes and is not text: {path!r}"
            )

        try:
            return data.decode("utf-8")
        except UnicodeDecodeError as original:
            raise RepositoryFileAccessError(
                f"requested file is not valid UTF-8 text: {path!r}"
            ) from original


def _validate_logical_path(path: str) -> None:
    """Reject an unsafe requested path before any filesystem access.

    Delegates to the shared workspace path validator, raising
    :class:`RepositoryFileAccessError` at this read boundary so an unsafe path is
    refused *before* it is used to open a file (rather than only when the
    eventual model is constructed). Paths are rejected, never rewritten.
    """

    validate_logical_repository_path(path, RepositoryFileAccessError)
