"""Git-backed concrete :class:`RepositoryContextLoader` for local workspaces.

``GitRepositoryContextLoader`` turns a local repository path into a normalized,
provider- and runtime-independent core ``RepositoryContext``. It produces an
*initial* snapshot suitable for later planning: a deterministic listing of
Git-tracked paths plus the contents of a small, conservative, fixed set of
root-level seed files. It deliberately does not attempt to provide all
repository code, perform relevance selection, index, budget tokens, or interact
with any model, provider, or runtime.

Design decisions (see the workspace README / package docstring for context):

* Git is the single source of truth for working-tree identity and tracked
  files (see the shared ``_git`` primitives). The filesystem is never crawled
  and ``.git`` is never inspected directly, so linked worktrees (which lack a
  normal ``.git`` directory) work.
* The effective repository root is Git's reported top-level working-tree
  directory, so an in-worktree subdirectory canonicalizes correctly.
* Git subprocesses are invoked through an argv API with ``shell=False``; the
  repository path is always a single argument, never interpolated into a shell
  string. Tracked paths are read as NUL-delimited output because Git filenames
  may contain newlines.
* Only the fixed root-level seed files are ever read. Reads are bounded, decoded
  strictly as UTF-8, and confined to the canonical repository root (symlinks may
  not escape it). Absolute local paths never enter the returned model.
"""

import asyncio
from pathlib import Path
from typing import Final

from llmforeman_core import RepositoryContext, RepositoryFile
from llmforeman_workspace._git import (
    list_tracked_paths,
    resolve_worktree_top_level,
)

__all__ = [
    "GitRepositoryContextLoader",
]

_DEFAULT_MAX_SEED_FILE_BYTES: Final[int] = 256 * 1024

# Root-level exact-match seed files eligible for content inclusion, in the
# stable order they appear in ``RepositoryContext.files``. These are matched
# only at the repository top level; nested manifests are never discovered.
_SEED_FILE_NAMES: Final[tuple[str, ...]] = (
    "AGENTS.md",
    "CLAUDE.md",
    "README.md",
    "CONTRIBUTING.md",
    "pyproject.toml",
    "package.json",
    "Cargo.toml",
)


class GitRepositoryContextLoader:
    """Load a core ``RepositoryContext`` from a local Git working tree.

    Satisfies the async :class:`RepositoryContextLoader` protocol. The optional
    ``max_seed_file_bytes`` bounds each individual seed file read; a seed file
    larger than the limit is skipped (never truncated). This limit is concrete
    loader infrastructure and is intentionally absent from the core models and
    the loader protocol.
    """

    def __init__(
        self,
        *,
        max_seed_file_bytes: int = _DEFAULT_MAX_SEED_FILE_BYTES,
    ) -> None:
        # ``bool`` is an ``int`` subclass; reject it explicitly so a stray
        # ``True`` cannot masquerade as a size limit.
        if isinstance(max_seed_file_bytes, bool) or not isinstance(
            max_seed_file_bytes, int
        ):
            raise ValueError("max_seed_file_bytes must be a positive integer")
        if max_seed_file_bytes <= 0:
            raise ValueError("max_seed_file_bytes must be a positive integer")
        self._max_seed_file_bytes = max_seed_file_bytes

    async def load(self, repository_root: Path) -> RepositoryContext:
        """Load a ``RepositoryContext`` from ``repository_root``.

        ``repository_root`` is any local path inside the target working tree; it
        is validated to exist and be a directory, then Git resolves the actual
        working-tree top-level used as the effective root for tracked-file
        discovery, relative-path interpretation, and seed reads.
        """

        effective_root = await resolve_worktree_top_level(repository_root)
        tracked_paths = await list_tracked_paths(effective_root)

        sorted_tracked = sorted(tracked_paths)
        file_tree = "\n".join(sorted_tracked)
        files = await self._read_seed_files(effective_root, set(sorted_tracked))

        return RepositoryContext(file_tree=file_tree, files=files)

    async def _read_seed_files(
        self,
        effective_root: Path,
        tracked: set[str],
    ) -> list[RepositoryFile]:
        """Read the fixed seed allowlist, preserving declared seed order.

        A seed is eligible only if its exact repository-relative name is
        tracked. Each candidate is read from the current working tree with all
        safety checks applied; any that fails a check is skipped without failing
        the overall load, and its tracked path remains in ``file_tree``.
        """

        files: list[RepositoryFile] = []
        for name in _SEED_FILE_NAMES:
            if name not in tracked:
                continue
            content = await asyncio.to_thread(
                self._read_seed_content, effective_root, name
            )
            if content is not None:
                files.append(RepositoryFile(path=name, content=content))
        return files

    def _read_seed_content(self, effective_root: Path, name: str) -> str | None:
        """Return decoded seed content, or ``None`` if it must be skipped.

        Skips (returns ``None``) for the ordinary, expected local conditions
        that must not fail the whole load: a symlink escaping the repository
        root, a broken symlink, a non-regular file, a missing/deleted file, a
        permission failure, an oversized file, or invalid UTF-8. Runs on a
        worker thread so filesystem I/O never blocks the event loop.
        """

        candidate = effective_root / name

        # Resolve without requiring existence so broken/dangling symlinks and
        # missing files fall through to the readability checks below.
        resolved = candidate.resolve()

        # Containment: the resolved target must stay within the canonical root.
        # Proper path semantics avoid unsafe string-prefix comparisons.
        if not resolved.is_relative_to(effective_root):
            return None

        # Only a regular file (directly or via an in-repo symlink) is readable
        # content; directories, sockets, devices, FIFOs, and broken symlinks are
        # skipped. ``is_file`` follows the already-resolved path.
        if not resolved.is_file():
            return None

        data = self._read_bounded(resolved)
        if data is None:
            return None
        if len(data) > self._max_seed_file_bytes:
            # Oversized: skip rather than truncate.
            return None

        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            return None

    def _read_bounded(self, path: Path) -> bytes | None:
        """Read at most ``limit + 1`` bytes, or ``None`` on an expected error.

        Reading one byte beyond the limit is enough to distinguish
        ``size <= limit`` from ``size > limit`` without loading an arbitrarily
        large file into memory. Expected local filesystem failures return
        ``None`` (skip); unexpected errors are not swallowed here.
        """

        try:
            with open(path, "rb") as handle:  # noqa: PTH123 - narrow bounded read
                return handle.read(self._max_seed_file_bytes + 1)
        except OSError:
            # Missing/deleted, permission denied, non-readable special file,
            # or a race after the ``is_file`` check: skip this optional seed.
            return None
