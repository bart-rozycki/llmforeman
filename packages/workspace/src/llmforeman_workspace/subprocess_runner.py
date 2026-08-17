"""Exec-style local command execution for trusted callers.

``SubprocessWorkspaceCommandRunner`` is the concrete
:class:`~llmforeman_workspace.contracts.WorkspaceCommandRunner`: it runs one
explicit argv command to completion at the effective Git working-tree top-level
and returns a :class:`~llmforeman_workspace.command.CommandResult`.

Trust boundary (read before using):

* This runs the executable supplied by its **trusted** caller. It is NOT a
  sandbox, allowlist, authorization layer, filesystem/network sandbox,
  container, or command-policy engine. It can technically run dangerous
  programs (for example ``["rm", ...]`` or an arbitrary interpreter) if a
  trusted caller requests them.
* The child inherits LLMForeman's current process environment (``PATH``,
  ``HOME``, toolchain/virtualenv variables, and anything else in the process
  environment). Because of that inheritance and the absence of any command
  authorization, this runner MUST NOT be exposed directly to an untrusted model
  before a command-authorization/sandbox policy exists.

Execution invariants:

* No shell is ever invoked. The command is executed exactly as argv via
  :func:`asyncio.create_subprocess_exec`; strings such as ``&&``, ``|``, ``>``,
  ``*``, ``$HOME``, or ``;`` in argv remain ordinary literal arguments.
* ``cwd`` is the real Git working-tree top-level resolved from
  ``repository_root`` via the shared private ``_git`` helper (a subdirectory or
  linked-worktree entry point still runs at the top-level). ``.git`` is never
  inspected directly and the global process cwd is never mutated.
* ``stdin`` is :data:`~asyncio.subprocess.DEVNULL` (accidentally interactive
  commands receive EOF instead of blocking on LLMForeman's terminal); ``stdout``
  and ``stderr`` are captured as separate PIPEs and never merged.
* Each command runs in its own POSIX session/process group
  (``start_new_session=True``) so timeout, cancellation, and output overflow can
  terminate the whole spawned group, not just the direct child.

This module is macOS/POSIX-oriented for v0.1; the process-group cleanup relies
on POSIX session semantics.
"""

import asyncio
import contextlib
import math
import os
import signal
from pathlib import Path
from typing import Final

from llmforeman_workspace._git import resolve_worktree_top_level
from llmforeman_workspace.command import CommandResult
from llmforeman_workspace.errors import (
    WorkspaceCommandExecutionError,
    WorkspaceCommandTimeoutError,
)

__all__ = [
    "SubprocessWorkspaceCommandRunner",
]

_DEFAULT_TIMEOUT_SECONDS: Final[float] = 300.0
_DEFAULT_MAX_OUTPUT_BYTES: Final[int] = 4 * 1024 * 1024

# Bounded grace period between SIGTERM and SIGKILL of the process group. This is
# a private v0.1 cleanup detail, deliberately not a public runner knob; tests do
# not depend on its exact value.
_TERMINATE_GRACE_SECONDS: Final[float] = 3.0

# Incremental read chunk size for bounded draining. Not a product limit.
_READ_CHUNK_BYTES: Final[int] = 64 * 1024


class _OutputLimitExceeded(Exception):
    """Internal signal that a stream exceeded its per-stream byte budget."""

    def __init__(self, stream_name: str) -> None:
        super().__init__(stream_name)
        self.stream_name = stream_name


def _signal_process_group(
    process: asyncio.subprocess.Process,
    sig: signal.Signals,
) -> None:
    """Send ``sig`` to the child's whole POSIX process group, tolerating races.

    Because the child was spawned with ``start_new_session=True`` it is the
    leader of its own process group (``pgid == pid``), so signalling the group
    reaches descendants too. Process termination is inherently racy: the group
    may already be gone between the check and the signal, which is a benign
    "already exited" condition, not a cleanup failure. No retry/backoff is used.
    """

    pid = process.pid
    try:
        pgid = os.getpgid(pid)
    except ProcessLookupError:
        return
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        return


def _validate_command(command: list[str]) -> None:
    """Validate argv before any process is spawned.

    Rejects an empty command, a whitespace-only executable, empty argv entries,
    and any NUL-containing entry. Argument values are otherwise preserved
    exactly (including surrounding/inner whitespace); nothing is stripped or
    rewritten. Invalid input raises :class:`WorkspaceCommandExecutionError`
    rather than a raw ``ValueError`` or a low-level subprocess argument error.
    """

    if not command:
        raise WorkspaceCommandExecutionError(
            "command must contain at least one argv element"
        )
    for entry in command:
        if not isinstance(entry, str):
            raise WorkspaceCommandExecutionError(
                "command argv entries must all be strings"
            )
        if entry == "":
            raise WorkspaceCommandExecutionError(
                "command argv entries must be non-empty strings"
            )
        if "\x00" in entry:
            raise WorkspaceCommandExecutionError(
                "command argv entries must not contain a NUL character"
            )
    # The executable itself must carry meaningful, non-whitespace text; a
    # whitespace-only executable can never name a real program. Ordinary
    # arguments may remain whitespace-only.
    if command[0].strip() == "":
        raise WorkspaceCommandExecutionError(
            "command executable must not be whitespace-only"
        )


async def _read_stream_bounded(
    stream: asyncio.StreamReader | None,
    max_output_bytes: int,
    stream_name: str,
) -> bytes:
    """Drain ``stream`` incrementally, enforcing a hard per-stream byte cap.

    Reads in bounded chunks and accumulates at most ``max_output_bytes`` bytes.
    The first byte beyond the limit raises :class:`_OutputLimitExceeded` (rather
    than silently discarding output or returning a truncated buffer). A missing
    stream (no PIPE) yields empty bytes.
    """

    if stream is None:
        return b""
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await stream.read(_READ_CHUNK_BYTES)
        if not chunk:
            break
        total += len(chunk)
        if total > max_output_bytes:
            raise _OutputLimitExceeded(stream_name)
        chunks.append(chunk)
    return b"".join(chunks)


class SubprocessWorkspaceCommandRunner:
    """Exec-style :class:`WorkspaceCommandRunner` backed by a local subprocess.

    Runs one explicit argv command to completion at the effective Git
    working-tree top-level and returns a
    :class:`~llmforeman_workspace.command.CommandResult`. An ordinary non-zero
    (or negative signal) process exit is a normal result and never an
    exception. See the module docstring for the full trust boundary: this is a
    trusted-caller runner, not a sandbox or authorization layer.

    ``timeout_seconds`` bounds the running subprocess lifecycle after successful
    creation; ``max_output_bytes`` is the hard limit applied to *each* captured
    stream independently (stdout and stderr do not share a budget). Both are
    concrete-runner configuration and are intentionally absent from the generic
    ``WorkspaceCommandRunner`` protocol and its ``run`` signature.
    """

    def __init__(
        self,
        *,
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
        max_output_bytes: int = _DEFAULT_MAX_OUTPUT_BYTES,
    ) -> None:
        # ``bool`` is an ``int`` subclass; reject it explicitly so a stray
        # ``True``/``False`` cannot masquerade as a duration or a byte count.
        if isinstance(timeout_seconds, bool) or not isinstance(
            timeout_seconds, (int, float)
        ):
            raise ValueError("timeout_seconds must be a finite positive number")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be a finite positive number")

        if isinstance(max_output_bytes, bool) or not isinstance(max_output_bytes, int):
            raise ValueError("max_output_bytes must be a positive integer")
        if max_output_bytes <= 0:
            raise ValueError("max_output_bytes must be a positive integer")

        self._timeout_seconds = float(timeout_seconds)
        self._max_output_bytes = max_output_bytes

    async def run(
        self,
        repository_root: Path,
        command: list[str],
    ) -> CommandResult:
        """Run one argv ``command`` to completion at the Git top-level.

        Validates ``command`` before any process is spawned, resolves the
        effective Git working-tree top-level from ``repository_root`` (reusing
        the shared ``_git`` semantics, so an invalid repository remains
        :class:`InvalidRepositoryError` and a Git inspection failure remains
        :class:`RepositoryInspectionError`), then spawns the executable directly
        in its own POSIX session with inherited environment, ``DEVNULL`` stdin,
        and separate stdout/stderr pipes.
        """

        # Snapshot argv before any await so later external mutation of the
        # caller's list cannot change what is executed or reported.
        command_snapshot = list(command)
        _validate_command(command_snapshot)

        # May raise InvalidRepositoryError / RepositoryInspectionError; those are
        # intentionally *not* wrapped into a command-execution error.
        effective_root = await resolve_worktree_top_level(repository_root)

        process = await self._spawn(command_snapshot, effective_root)
        return await self._supervise(process, command_snapshot)

    async def _spawn(
        self,
        command: list[str],
        cwd: Path,
    ) -> asyncio.subprocess.Process:
        """Spawn the executable directly (no shell) in a new POSIX session.

        A failure to *start* the process (for example a missing executable) is a
        :class:`WorkspaceCommandExecutionError`; there is no legitimate
        ``CommandResult`` because no process ran.
        """

        try:
            return await asyncio.create_subprocess_exec(
                *command,
                cwd=str(cwd),
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                # New POSIX session/process group so the whole spawned tree can
                # be signalled on timeout/cancellation/overflow. Preferred over
                # ``preexec_fn=os.setsid``.
                start_new_session=True,
            )
        except OSError as original:
            raise WorkspaceCommandExecutionError(
                f"failed to start command executable {command[0]!r}"
            ) from original

    async def _supervise(
        self,
        process: asyncio.subprocess.Process,
        command: list[str],
    ) -> CommandResult:
        """Drain both pipes concurrently under the timeout and enforce cleanup.

        On normal completion both streams are drained within their per-stream
        limit and the direct process is reaped. On timeout, output overflow, or
        cancellation, the whole process group is terminated and reaped via the
        single shared cleanup path before the corresponding error (or the
        original :class:`asyncio.CancelledError`) propagates.
        """

        stdout_task = asyncio.ensure_future(
            _read_stream_bounded(process.stdout, self._max_output_bytes, "stdout")
        )
        stderr_task = asyncio.ensure_future(
            _read_stream_bounded(process.stderr, self._max_output_bytes, "stderr")
        )
        reader_tasks = (stdout_task, stderr_task)

        try:
            async with asyncio.timeout(self._timeout_seconds):
                stdout_bytes, stderr_bytes = await asyncio.gather(
                    stdout_task, stderr_task
                )
                # Both pipes reached EOF within budget; reap the direct process.
                await process.wait()
        except TimeoutError:
            await self._cleanup_group(process, reader_tasks)
            raise WorkspaceCommandTimeoutError(
                "command timed out after "
                f"{self._timeout_seconds:g} seconds"
            ) from None
        except _OutputLimitExceeded as overflow:
            await self._cleanup_group(process, reader_tasks)
            raise WorkspaceCommandExecutionError(
                f"command {overflow.stream_name} exceeded the configured "
                f"{self._max_output_bytes} byte limit"
            ) from None
        except asyncio.CancelledError:
            await self._cleanup_group(process, reader_tasks)
            raise
        except BaseException:
            # Any other unexpected failure (for example an internal reader
            # error) must not leave the spawned process or reader tasks alive.
            await self._cleanup_group(process, reader_tasks)
            raise

        returncode = process.returncode
        # ``wait`` on a completed process yields a concrete return code.
        assert returncode is not None
        return CommandResult(
            command=command,
            exit_code=returncode,
            stdout=stdout_bytes.decode("utf-8", errors="replace"),
            stderr=stderr_bytes.decode("utf-8", errors="replace"),
        )

    async def _cleanup_group(
        self,
        process: asyncio.subprocess.Process,
        reader_tasks: tuple[asyncio.Future[bytes], ...],
    ) -> None:
        """Terminate/reap the whole group and readers, robust to cancellation.

        Runs the actual teardown in a shielded task and keeps awaiting it even
        if this coroutine is itself repeatedly cancelled, so the caller only
        observes an outcome after a bounded, serious attempt to eliminate the
        spawned process group and reap the direct process has completed.
        """

        cleanup = asyncio.ensure_future(self._teardown(process, reader_tasks))
        while not cleanup.done():
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                if cleanup.done():
                    break
                # External cancellation arrived mid-cleanup; the shielded task
                # keeps running. Keep waiting so teardown always completes.
                continue
        # Surface an unexpected teardown failure (the shielded task is never
        # cancelled, so ``exception()`` cannot itself raise ``CancelledError``).
        error = cleanup.exception()
        if error is not None:
            raise WorkspaceCommandExecutionError(
                "failed to clean up the command process group"
            ) from error

    async def _teardown(
        self,
        process: asyncio.subprocess.Process,
        reader_tasks: tuple[asyncio.Future[bytes], ...],
    ) -> None:
        """SIGTERM the group, grace-wait, SIGKILL if needed, reap, stop readers."""

        if process.returncode is None:
            _signal_process_group(process, signal.SIGTERM)
            try:
                async with asyncio.timeout(_TERMINATE_GRACE_SECONDS):
                    await process.wait()
            except TimeoutError:
                _signal_process_group(process, signal.SIGKILL)
                await process.wait()
        else:
            # Already exited; ensure it is reaped.
            await process.wait()

        for task in reader_tasks:
            task.cancel()
        for task in reader_tasks:
            with contextlib.suppress(
                asyncio.CancelledError, _OutputLimitExceeded, OSError
            ):
                await task
