"""Private, workspace-internal Git working-tree primitives.

Shared security-sensitive Git semantics used by both the concrete
``GitRepositoryContextLoader`` and ``GitRepositoryFileReader``. Keeping this in a
single place ensures the two implementations cannot drift on how the effective
repository root is resolved, how tracked paths are discovered, or how Git is
launched. It is intentionally **not** part of the workspace public API: it
exposes no new abstraction, only narrow functions.

Invariants preserved here:

* Git is the single source of truth for working-tree identity and tracked
  files. The filesystem is never crawled and ``.git`` is never inspected
  directly, so linked worktrees (which lack a normal ``.git`` directory) work.
* Git subprocesses are invoked through an argv API with ``shell=False``; the
  repository path is always a single argument, never interpolated into a shell
  string. Tracked paths are read as NUL-delimited output because Git filenames
  may contain newlines.
* Failures are normalized to the workspace error hierarchy: invalid caller
  input becomes :class:`InvalidRepositoryError`; failures after a working tree
  has been resolved become :class:`RepositoryInspectionError`.
"""

import asyncio
import os
from pathlib import Path

from llmforeman_workspace.errors import (
    InvalidRepositoryError,
    RepositoryInspectionError,
)

__all__ = [
    "list_tracked_paths",
    "resolve_worktree_top_level",
    "run_git",
    "sanitized_git_detail",
]


async def resolve_worktree_top_level(repository_root: Path) -> Path:
    """Validate ``repository_root`` and resolve the Git working-tree top-level.

    ``repository_root`` is any local path inside the target working tree; it is
    validated to exist and be a directory, then Git resolves the actual
    working-tree top-level used as the canonical effective root.

    A missing path, a non-directory path, or a directory Git does not consider
    part of a working tree (including bare repositories) is invalid caller input
    (:class:`InvalidRepositoryError`). A failure to *launch* Git is an
    inspection failure (:class:`RepositoryInspectionError`, raised by
    :func:`run_git`).
    """

    if not repository_root.exists():
        raise InvalidRepositoryError("repository path does not exist")
    if not repository_root.is_dir():
        raise InvalidRepositoryError("repository path is not a directory")

    returncode, stdout, stderr = await run_git(
        repository_root, ("rev-parse", "--show-toplevel")
    )
    if returncode != 0:
        raise InvalidRepositoryError(
            "path is not inside a Git working tree" + sanitized_git_detail(stderr)
        )

    raw = stdout.rstrip(b"\n")
    if not raw:
        # Bare repositories yield no top-level working tree.
        raise InvalidRepositoryError("path has no Git working tree")

    # Use the filesystem's own decoding for the top-level path; then
    # canonicalize so later containment checks compare resolved paths.
    return Path(os.fsdecode(raw)).resolve()


async def list_tracked_paths(effective_root: Path) -> list[str]:
    """Return tracked repository-relative paths via ``git ls-files -z``.

    Only index/tracked paths are requested (no untracked, ignored, or submodule
    recursion). Output is NUL-delimited and split on NUL. Each path is decoded
    strictly as UTF-8; a path that cannot be represented safely fails inspection
    rather than being silently mangled.
    """

    returncode, stdout, stderr = await run_git(
        effective_root, ("ls-files", "--cached", "--full-name", "-z")
    )
    if returncode != 0:
        raise RepositoryInspectionError(
            "failed to list tracked files" + sanitized_git_detail(stderr)
        )

    paths: list[str] = []
    for chunk in stdout.split(b"\x00"):
        if not chunk:
            continue
        try:
            paths.append(chunk.decode("utf-8"))
        except UnicodeDecodeError as original:
            raise RepositoryInspectionError(
                "a tracked path could not be decoded as UTF-8"
            ) from original
    return paths


async def run_git(cwd: Path, args: tuple[str, ...]) -> tuple[int, bytes, bytes]:
    """Run ``git -C <cwd> <args...>`` without a shell and collect output.

    The repository path is passed as a single argv element via ``-C`` and is
    never interpolated into a shell command. A failure to launch Git is
    normalized to :class:`RepositoryInspectionError` with preserved causality.
    """

    argv = ("git", "-C", str(cwd), *args)
    try:
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except OSError as original:
        raise RepositoryInspectionError(
            "unable to launch the git executable"
        ) from original

    stdout, stderr = await process.communicate()
    returncode = process.returncode
    # ``communicate`` on an awaited, finished process yields a concrete code.
    assert returncode is not None
    return returncode, stdout, stderr


def sanitized_git_detail(stderr: bytes) -> str:
    """Return a short, sanitized single-line Git diagnostic suffix, if any.

    Includes only Git's own concise error text (never file contents, env vars,
    or secrets) to aid diagnosis; returns an empty string when nothing usable
    is present.
    """

    text = stderr.decode("utf-8", errors="replace").strip()
    if not text:
        return ""
    first_line = text.splitlines()[0].strip()
    if not first_line:
        return ""
    return f": {first_line}"
