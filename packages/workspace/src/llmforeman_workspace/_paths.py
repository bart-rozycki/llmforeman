"""Private, workspace-internal logical repository-path validation.

Shared, security-sensitive validation of a caller-supplied logical,
repository-relative path used by both the read side
(``GitRepositoryFileReader``) and the write side
(``GitRepositoryFileWriter``). Keeping this in a single place ensures the read
and write boundaries cannot drift on which paths are considered unsafe.

The check is intentionally filesystem-free and OS-independent: it never touches
the filesystem, never resolves against the working directory, and never
rewrites an unsafe path. Empty/whitespace, NUL-containing, absolute (POSIX or
Windows), and parent-traversal paths are rejected. The concrete error type is
supplied by the caller so each boundary keeps its own error contract
(``RepositoryFileAccessError`` for reads, ``RepositoryFileWriteError`` for
writes) without this helper depending on either.

It is intentionally **not** part of the workspace public API.
"""

from collections.abc import Callable
from pathlib import PurePosixPath, PureWindowsPath

__all__ = [
    "validate_logical_repository_path",
]


def validate_logical_repository_path(
    path: str,
    make_error: Callable[[str], Exception],
) -> None:
    """Reject an unsafe requested logical path before any filesystem access.

    Semantically compatible with the core ``RepositoryFile.path`` invariant, but
    raising the caller-provided error type at the security boundary so an unsafe
    path is refused *before* it is used to touch the filesystem (rather than only
    when the eventual model is constructed). Paths are rejected, never rewritten:
    no leading ``/`` stripping, no ``..`` removal, no absolute-to-relative
    rewriting, and no resolution against the working directory.
    """

    if not path.strip():
        raise make_error("requested path must not be empty or whitespace-only")
    if "\x00" in path:
        raise make_error("requested path must not contain NUL characters")
    # Cross-platform, filesystem-free absolute-path detection. Parse the value
    # as both POSIX and Windows pure paths so an absolute path is rejected
    # regardless of which OS runs this code.
    if PurePosixPath(path).is_absolute() or PureWindowsPath(path).is_absolute():
        raise make_error(
            f"requested path must be repository-relative, not absolute: {path!r}"
        )
    # Reject parent traversal in either separator convention without resolving
    # the path against the filesystem or the working directory.
    parts = PurePosixPath(path).parts + PureWindowsPath(path).parts
    if ".." in parts:
        raise make_error(
            f"requested path must not contain parent traversal segments: {path!r}"
        )
