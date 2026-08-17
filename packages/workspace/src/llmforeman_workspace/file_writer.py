"""Git-backed concrete :class:`RepositoryFileWriter` for local workspaces.

``GitRepositoryFileWriter`` provides the one safe mutation primitive: *write
this exact textual content as the complete state of one explicitly named
repository file*. It is the write-side complement to ``GitRepositoryFileReader``
and does not patch, diff, search, stage, commit, or interact with any model,
provider, or runtime.

Security model (this is a high-risk local mutation boundary):

* Git determines the effective working-tree top-level (via the shared ``_git``
  primitives); ``.git`` is never inspected directly, so linked worktrees work.
  Git membership/tracking is intentionally *not* consulted: the write side may
  overwrite tracked or untracked files and create new untracked files. Git only
  establishes the mutation boundary; the writer never runs ``git add`` or
  otherwise mutates the index/status.
* The caller-supplied logical path is validated as repository-relative *before*
  any filesystem access: empty/whitespace, NUL, absolute (POSIX or Windows), and
  parent-traversal paths are rejected, never rewritten.
* The requested content is encoded as strict UTF-8 and size-checked against
  ``max_file_bytes`` *before* any directory is created or any target is opened,
  truncated, or created. An oversized or non-encodable request mutates nothing.
* Path traversal is performed with descriptor-relative, no-follow filesystem
  operations rooted at an opened handle on the effective Git top-level. Every
  parent component and the final target are opened with ``O_NOFOLLOW`` so a
  symlink component (internal or external) is refused at the filesystem
  operation itself, not by a check-then-open sequence that a concurrent process
  could race. Missing parent directories may be created relative to a trusted
  parent handle; a single concurrent-creation race is handled by re-opening with
  the same directory + no-follow semantics.
* An existing final target is opened no-follow, ``fstat``-verified to be a
  regular file, and its *current* contents are streamed and validated as UTF-8
  text (rejecting invalid UTF-8 or NUL bytes) *before* it is truncated, all
  through the same pinned descriptor to avoid a reopen race. The existing file
  is truncated and rewritten in place, preserving its inode and mode bits.
* No absolute local path ever enters the returned ``RepositoryFile``: it carries
  the requested logical path and the requested content only.

This writer makes no atomicity or durability promise: it performs a direct
in-place truncate+write with no temporary file, ``os.replace``, or fsync, so an
underlying filesystem failure mid-write may leave a partially written file. That
limitation is accepted for v0.1; the required property here is symlink/path
traversal safety, which is a different property from atomic replacement.
"""

import asyncio
import codecs
import contextlib
import errno
import os
import stat
from pathlib import Path, PurePosixPath
from typing import Final

from llmforeman_core import RepositoryFile
from llmforeman_workspace._git import resolve_worktree_top_level
from llmforeman_workspace._paths import validate_logical_repository_path
from llmforeman_workspace.errors import RepositoryFileWriteError

__all__ = [
    "GitRepositoryFileWriter",
]

_DEFAULT_MAX_FILE_BYTES: Final[int] = 1024 * 1024

# Bounded chunk size for streaming validation of an existing target's current
# contents; keeps an arbitrarily large existing file out of memory.
_READ_CHUNK_BYTES: Final[int] = 64 * 1024


class GitRepositoryFileWriter:
    """Write one repository file's exact requested state via secure traversal.

    Satisfies the async :class:`RepositoryFileWriter` protocol. The optional
    ``max_file_bytes`` bounds the *requested* content size (UTF-8 encoded); a
    request larger than the limit fails before any mutation. This limit is
    concrete writer infrastructure and is intentionally absent from the core
    models and the writer protocol.

    Unlike the Git-tracked reader, this writer does not require the target to be
    tracked or even to exist: it overwrites tracked/untracked files and creates
    new untracked files. Git only establishes the working-tree boundary.
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

    async def write(
        self,
        repository_root: Path,
        path: str,
        content: str,
    ) -> RepositoryFile:
        """Write ``content`` as the complete state of the file at ``path``.

        ``repository_root`` is any local path inside the target working tree;
        Git resolves the canonical top-level used as the effective root and
        mutation boundary. ``path`` is a logical, repository-relative identifier
        validated before any filesystem access.

        Raises :class:`~llmforeman_workspace.errors.InvalidRepositoryError` for
        an invalid repository entry,
        :class:`~llmforeman_workspace.errors.RepositoryInspectionError` for a
        Git inspection failure, and
        :class:`~llmforeman_workspace.errors.RepositoryFileWriteError` for any
        expected inability to perform the requested write.
        """

        # Repository validation and effective-root resolution first (may raise
        # InvalidRepositoryError / RepositoryInspectionError).
        effective_root = await resolve_worktree_top_level(repository_root)

        # Validate the logical path BEFORE it is used to touch the filesystem.
        validate_logical_repository_path(path, RepositoryFileWriteError)

        # Encode and size-check the requested content BEFORE any mutation so an
        # oversized or non-encodable request never creates or truncates anything.
        payload = _encode_content(path, content)
        if len(payload) > self._max_file_bytes:
            raise RepositoryFileWriteError(
                f"requested content exceeds the maximum allowed size: {path!r}"
            )

        await asyncio.to_thread(
            self._write_secure, effective_root, path, payload
        )
        return RepositoryFile(path=path, content=content)

    def _write_secure(
        self, effective_root: Path, path: str, payload: bytes
    ) -> None:
        """Perform the secure, no-follow descriptor-relative write.

        Runs on a worker thread so filesystem I/O never blocks the event loop.
        The effective Git top-level is opened as a trusted directory handle and
        every subsequent component is opened relative to a verified parent
        descriptor with no-follow semantics, so no symlink component (parent or
        final target) is ever traversed. Missing parent directories may be
        created relative to a trusted handle. All descriptors are closed on
        success and on every failure path.
        """

        components = PurePosixPath(path).parts
        if not components:
            # ``.`` / ``""``-like inputs that carry no addressable final name.
            raise RepositoryFileWriteError(
                f"requested path does not name a file: {path!r}"
            )
        *parent_names, final_name = components

        try:
            root_fd = os.open(effective_root, os.O_RDONLY | os.O_DIRECTORY)
        except OSError as original:
            raise RepositoryFileWriteError(
                "the repository root could not be opened for writing"
            ) from original

        try:
            parent_fd = root_fd
            opened_parents: list[int] = []
            try:
                for name in parent_names:
                    parent_fd = _descend_or_create_directory(parent_fd, name, path)
                    opened_parents.append(parent_fd)
                _write_final_target(parent_fd, final_name, payload, path)
            finally:
                for fd in opened_parents:
                    _safe_close(fd)
        finally:
            _safe_close(root_fd)


def _encode_content(path: str, content: str) -> bytes:
    """Strictly UTF-8 encode ``content`` before any filesystem mutation.

    A ``str`` carrying values that cannot be encoded under strict UTF-8 (for
    example unpaired surrogate code points) becomes a normal writer failure
    rather than a partial filesystem mutation. Replacement encoding is never
    used. The returned bytes are exactly the bytes to write.
    """

    try:
        return content.encode("utf-8")
    except UnicodeEncodeError as original:
        raise RepositoryFileWriteError(
            f"requested content is not encodable as UTF-8: {path!r}"
        ) from original


def _descend_or_create_directory(parent_fd: int, name: str, path: str) -> int:
    """Return a descriptor for directory ``name`` under ``parent_fd``.

    Opens an existing directory with no-follow, directory-required semantics, or
    creates it relative to the trusted parent when missing. A symlink component
    (internal or external), a non-directory object, or a component that cannot be
    created is rejected. A single concurrent-creation race is handled by
    re-opening with the same directory + no-follow semantics rather than trusting
    whatever appeared.
    """

    existing = _try_open_directory(parent_fd, name, path)
    if existing is not None:
        return existing

    try:
        # Ordinary creation mode subject to the process umask; no explicit mode
        # hardening beyond the standard default.
        os.mkdir(name, dir_fd=parent_fd)
    except FileExistsError:
        # Concurrent creation: fall through and re-open with the same secure
        # semantics rather than blindly trusting what now exists.
        pass
    except OSError as original:
        raise RepositoryFileWriteError(
            f"a parent directory could not be created for the requested path: {path!r}"
        ) from original

    reopened = _try_open_directory(parent_fd, name, path)
    if reopened is None:
        raise RepositoryFileWriteError(
            f"a parent directory could not be opened for the requested path: {path!r}"
        )
    return reopened


def _try_open_directory(parent_fd: int, name: str, path: str) -> int | None:
    """Open directory ``name`` under ``parent_fd`` without following symlinks.

    Returns the directory descriptor, or ``None`` when the component does not
    exist (so the caller may create it). A symlink component raises immediately;
    a non-directory object is rejected. ``O_DIRECTORY`` requires a directory and
    ``O_NOFOLLOW`` refuses a symlink at the filesystem operation itself.
    """

    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    try:
        return os.open(name, flags, dir_fd=parent_fd)
    except FileNotFoundError:
        return None
    except OSError as original:
        if original.errno in _SYMLINK_ERRNOS:
            raise RepositoryFileWriteError(
                f"a parent path component is a symlink and cannot be traversed: {path!r}"
            ) from original
        if original.errno == errno.ENOTDIR:
            raise RepositoryFileWriteError(
                f"a parent path component is not a directory: {path!r}"
            ) from original
        raise RepositoryFileWriteError(
            f"a parent path component could not be opened for the requested path: {path!r}"
        ) from original


def _write_final_target(
    parent_fd: int, name: str, payload: bytes, path: str
) -> None:
    """Create or overwrite the final component under a trusted ``parent_fd``.

    A new target is created with ``O_CREAT | O_EXCL | O_NOFOLLOW`` so a
    concurrently appearing symlink is never followed. If the target already
    exists, it is opened no-follow, verified to be a regular file, its current
    contents validated as UTF-8 text, and only then truncated and rewritten
    through the same pinned descriptor (preserving inode and mode bits).
    """

    create_flags = (
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
    )
    try:
        # Ordinary non-executable base mode (0o666) subject to the process
        # umask; ``os.open`` would otherwise default to 0o777, which umask alone
        # may leave executable. The writer never grants an executable bit.
        fd = os.open(name, create_flags, 0o666, dir_fd=parent_fd)
    except FileExistsError:
        _overwrite_existing_target(parent_fd, name, payload, path)
        return
    except OSError as original:
        if original.errno in _SYMLINK_ERRNOS:
            raise RepositoryFileWriteError(
                f"the requested target is a symlink and cannot be written: {path!r}"
            ) from original
        raise RepositoryFileWriteError(
            f"the requested target could not be created: {path!r}"
        ) from original

    try:
        _write_all(fd, payload, path)
    finally:
        _safe_close(fd)


def _overwrite_existing_target(
    parent_fd: int, name: str, payload: bytes, path: str
) -> None:
    """Overwrite an existing regular-file target through one pinned descriptor.

    Opens the existing object no-follow (rejecting a symlink), ``fstat``-verifies
    it is a regular file (rejecting directories, FIFOs, sockets, and devices),
    validates its current contents as UTF-8 text without loading the whole file
    into memory, then truncates and rewrites it in place. Using a single
    descriptor for validation and mutation avoids a reopen path-lookup race.
    """

    # ``O_NONBLOCK`` avoids blocking if a non-regular object slipped in; the
    # ``fstat`` regular-file check below rejects anything that is not a file.
    open_flags = os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
    try:
        fd = os.open(name, open_flags, dir_fd=parent_fd)
    except OSError as original:
        if original.errno in _SYMLINK_ERRNOS:
            raise RepositoryFileWriteError(
                f"the requested target is a symlink and cannot be written: {path!r}"
            ) from original
        if original.errno == errno.EISDIR:
            raise RepositoryFileWriteError(
                f"the requested target is a directory and cannot be written: {path!r}"
            ) from original
        raise RepositoryFileWriteError(
            f"the requested target could not be opened for writing: {path!r}"
        ) from original

    try:
        try:
            stat_result = os.fstat(fd)
        except OSError as original:
            raise RepositoryFileWriteError(
                f"the requested target could not be inspected: {path!r}"
            ) from original
        if not stat.S_ISREG(stat_result.st_mode):
            raise RepositoryFileWriteError(
                f"the requested target is not a regular file: {path!r}"
            )

        _validate_existing_is_text(fd, path)

        try:
            os.lseek(fd, 0, os.SEEK_SET)
            os.ftruncate(fd, 0)
        except OSError as original:
            raise RepositoryFileWriteError(
                f"the requested target could not be truncated: {path!r}"
            ) from original

        _write_all(fd, payload, path)
    finally:
        _safe_close(fd)


def _validate_existing_is_text(fd: int, path: str) -> None:
    """Stream-validate an existing target's current contents as UTF-8 text.

    Reads the file in bounded chunks so an arbitrarily large existing file is
    never loaded into memory. Each chunk is scanned for a NUL byte and fed to a
    strict incremental UTF-8 decoder (so multibyte sequences crossing chunk
    boundaries are handled correctly). Invalid UTF-8, a NUL byte, a multibyte
    sequence truncated at EOF, or an unreadable file fails *before* any
    truncation, protecting tracked or untracked binary assets from replacement.
    """

    decoder = codecs.getincrementaldecoder("utf-8")(errors="strict")
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        while True:
            chunk = os.read(fd, _READ_CHUNK_BYTES)
            if not chunk:
                break
            if b"\x00" in chunk:
                raise RepositoryFileWriteError(
                    f"the existing target contains NUL bytes and is not text: {path!r}"
                )
            try:
                decoder.decode(chunk)
            except UnicodeDecodeError as original:
                raise RepositoryFileWriteError(
                    f"the existing target is not valid UTF-8 text: {path!r}"
                ) from original
    except OSError as original:
        raise RepositoryFileWriteError(
            f"the existing target could not be read for validation: {path!r}"
        ) from original

    try:
        decoder.decode(b"", final=True)
    except UnicodeDecodeError as original:
        raise RepositoryFileWriteError(
            f"the existing target is not valid UTF-8 text: {path!r}"
        ) from original


def _write_all(fd: int, payload: bytes, path: str) -> None:
    """Write every byte of ``payload`` to ``fd``, handling partial writes.

    A single ``os.write`` is not guaranteed to consume the whole buffer, so the
    remaining bytes are written until the payload is fully flushed. Empty content
    is valid and writes nothing. No newline translation or normalization occurs;
    the resulting bytes are exactly ``payload``.
    """

    view = memoryview(payload)
    total = 0
    try:
        while total < len(view):
            total += os.write(fd, view[total:])
    except OSError as original:
        raise RepositoryFileWriteError(
            f"the requested content could not be written: {path!r}"
        ) from original


def _safe_close(fd: int) -> None:
    """Close a descriptor, ignoring a benign close-time error.

    Descriptor cleanup must never mask the original success or failure being
    propagated; a close error on an already-finished operation is not
    actionable to the caller.
    """

    with contextlib.suppress(OSError):
        os.close(fd)


# Symlink refusal surfaces as ``ELOOP`` on most platforms; some report
# ``EMLINK`` for ``O_NOFOLLOW`` on a symlink, so both are treated as "symlink".
_SYMLINK_ERRNOS: Final[frozenset[int]] = frozenset(
    code for code in (getattr(errno, "ELOOP", None), getattr(errno, "EMLINK", None))
    if code is not None
)
