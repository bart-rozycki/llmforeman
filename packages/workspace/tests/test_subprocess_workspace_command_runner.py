"""Behavioral tests for ``SubprocessWorkspaceCommandRunner``.

These run real, local subprocesses (using ``sys.executable`` for deterministic
programs) inside real, temporary Git repositories (``git init``; no commits, no
remotes, no network). Async ``run`` calls are driven with ``asyncio.run`` to
match the repository convention of not adding a pytest asyncio plugin. No model
or network calls are made anywhere in this module.
"""

import asyncio
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from llmforeman_workspace import (
    CommandResult,
    InvalidRepositoryError,
    SubprocessWorkspaceCommandRunner,
    WorkspaceCommandExecutionError,
    WorkspaceCommandRunner,
    WorkspaceCommandTimeoutError,
)

_GIT = shutil.which("git")
pytestmark = pytest.mark.skipif(_GIT is None, reason="git executable not available")


def _init_repo(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "-C", str(root), "init", "-q"],
        check=True,
        capture_output=True,
    )
    return root


def _run(
    root: Path,
    command: list[str],
    *,
    timeout_seconds: float = 30.0,
    max_output_bytes: int = 4 * 1024 * 1024,
) -> CommandResult:
    runner = SubprocessWorkspaceCommandRunner(
        timeout_seconds=timeout_seconds,
        max_output_bytes=max_output_bytes,
    )
    return asyncio.run(runner.run(root, command))


def _py(source: str, *args: str) -> list[str]:
    """Build an argv running an inline Python program (no shell involved)."""

    return [sys.executable, "-c", source, *args]


# --- Protocol / type compatibility -------------------------------------------


def test_structurally_satisfies_protocol() -> None:
    # Assigning to the Protocol-typed name is the static structural check; mypy
    # rejects an implementation whose ``run`` signature does not match. No
    # ``Any``, cast, or ``type: ignore`` is used here.
    runner: WorkspaceCommandRunner = SubprocessWorkspaceCommandRunner()
    assert runner is not None


# --- Constructor validation --------------------------------------------------


@pytest.mark.parametrize("bad", [0, -1, -1.5, float("nan"), float("inf"), True, False])
def test_rejects_invalid_timeout(bad: object) -> None:
    with pytest.raises(ValueError):
        SubprocessWorkspaceCommandRunner(timeout_seconds=bad)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", [0, -1, -1024, 3.5, True, False])
def test_rejects_invalid_max_output_bytes(bad: object) -> None:
    with pytest.raises(ValueError):
        SubprocessWorkspaceCommandRunner(max_output_bytes=bad)  # type: ignore[arg-type]


# --- Successful execution ----------------------------------------------------


def test_successful_process(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    result = _run(repo, _py("print('hello')"))
    assert result.exit_code == 0
    assert result.stdout == "hello\n"
    assert result.stderr == ""
    assert result.command == _py("print('hello')")


def test_stderr_separation(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    result = _run(
        repo,
        _py(
            "import sys;"
            "sys.stdout.write('OUT');"
            "sys.stderr.write('ERR')"
        ),
    )
    assert result.exit_code == 0
    assert result.stdout == "OUT"
    assert result.stderr == "ERR"


def test_non_zero_exit_is_normal(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    result = _run(repo, _py("import sys; sys.exit(7)"))
    assert result.exit_code == 7


def test_non_zero_with_output(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    result = _run(
        repo,
        _py(
            "import sys;"
            "sys.stdout.write('progress');"
            "sys.stderr.write('boom');"
            "sys.exit(3)"
        ),
    )
    assert result.exit_code == 3
    assert result.stdout == "progress"
    assert result.stderr == "boom"


@pytest.mark.skipif(os.name != "posix", reason="POSIX signal exit semantics")
def test_negative_signal_exit(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    # The child terminates itself with SIGTERM; Python reports this as a
    # negative return code (-SIGTERM), never a shell-style 128+signal.
    result = _run(
        repo,
        _py("import os, signal; os.kill(os.getpid(), signal.SIGTERM)"),
    )
    assert result.exit_code == -signal.SIGTERM


def test_empty_streams_are_valid(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    result = _run(repo, _py("pass"))
    assert result.exit_code == 0
    assert result.stdout == ""
    assert result.stderr == ""


# --- No shell / exact argv ---------------------------------------------------


def test_exact_argv_no_shell_interpretation(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    metachars = [
        "argument with spaces",
        "*",
        "&&",
        "$HOME",
        ";",
        ">",
        "  preserve me  ",
    ]
    # The child serializes argv[1:] as NUL-joined bytes so the parent can verify
    # each argument arrived literally and individually.
    program = _py(
        "import sys;"
        "sys.stdout.write('\\x00'.join(sys.argv[1:]))",
        *metachars,
    )
    result = _run(repo, program)
    assert result.exit_code == 0
    assert result.stdout.split("\x00") == metachars


# --- Command snapshot --------------------------------------------------------


def test_command_snapshot_is_stable(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    command = _py("print('ok')")
    runner = SubprocessWorkspaceCommandRunner(timeout_seconds=30.0)

    async def go() -> CommandResult:
        task = asyncio.ensure_future(runner.run(repo, command))
        # Let ``run`` begin executing (and snapshot argv) before mutating the
        # caller's list; the snapshot taken before spawning must be unaffected.
        await asyncio.sleep(0.2)
        command.append("INJECTED")
        return await task

    result = asyncio.run(go())
    assert "INJECTED" not in result.command
    assert result.command == _py("print('ok')")


# --- Command validation (before spawn) ---------------------------------------


@pytest.mark.parametrize(
    "command",
    [
        [],
        [""],
        ["   "],
        ["uv", ""],
        ["tool", "\x00arg"],
        ["\x00exe"],
    ],
)
def test_invalid_command_rejected(tmp_path: Path, command: list[str]) -> None:
    repo = _init_repo(tmp_path / "repo")
    with pytest.raises(WorkspaceCommandExecutionError):
        _run(repo, command)


def test_whitespace_argument_is_preserved(tmp_path: Path) -> None:
    # A whitespace-only *argument* (not the executable) is valid and must not be
    # stripped or rewritten.
    repo = _init_repo(tmp_path / "repo")
    program = _py("import sys; sys.stdout.write(sys.argv[1])", "  ")
    result = _run(repo, program)
    assert result.exit_code == 0
    assert result.stdout == "  "


# --- Working directory = Git top-level ---------------------------------------


def test_cwd_is_git_top_level(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    result = _run(repo, _py("import os; print(os.getcwd())"))
    assert Path(result.stdout.strip()) == repo.resolve()


def test_subdirectory_entry_point_resolves_to_top_level(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    subdir = repo / "packages" / "core"
    subdir.mkdir(parents=True)
    result = _run(subdir, _py("import os; print(os.getcwd())"))
    assert Path(result.stdout.strip()) == repo.resolve()


@pytest.mark.skipif(os.name != "posix", reason="linked worktree via local git")
def test_linked_worktree_resolves_to_worktree_top_level(tmp_path: Path) -> None:
    # A committed main checkout is required before a linked worktree can be
    # added. Everything stays local (no network).
    main = _init_repo(tmp_path / "main")
    subprocess.run(
        ["git", "-C", str(main), "config", "user.email", "t@example.com"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(main), "config", "user.name", "Test"],
        check=True,
        capture_output=True,
    )
    (main / "README.md").write_text("x\n", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(main), "add", "README.md"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(main), "commit", "-q", "-m", "init"],
        check=True,
        capture_output=True,
    )
    linked = tmp_path / "linked"
    subprocess.run(
        ["git", "-C", str(main), "worktree", "add", "-q", str(linked)],
        check=True,
        capture_output=True,
    )
    result = _run(linked, _py("import os; print(os.getcwd())"))
    assert Path(result.stdout.strip()) == linked.resolve()


# --- Invalid repository ------------------------------------------------------


def test_missing_repository_root(tmp_path: Path) -> None:
    with pytest.raises(InvalidRepositoryError):
        _run(tmp_path / "does-not-exist", _py("pass"))


def test_file_as_repository_root(tmp_path: Path) -> None:
    file_path = tmp_path / "afile"
    file_path.write_text("x", encoding="utf-8")
    with pytest.raises(InvalidRepositoryError):
        _run(file_path, _py("pass"))


def test_non_git_directory(tmp_path: Path) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    with pytest.raises(InvalidRepositoryError):
        _run(plain, _py("pass"))


# --- Missing executable ------------------------------------------------------


def test_missing_executable(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    with pytest.raises(WorkspaceCommandExecutionError) as excinfo:
        _run(repo, ["llmforeman-definitely-missing-binary-xyz"])
    # Underlying causality is preserved (a spawn OSError).
    assert isinstance(excinfo.value.__cause__, OSError)


# --- Environment inheritance -------------------------------------------------


def test_environment_is_inherited(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    key = "LLMFOREMAN_TEST_INHERIT_MARKER"
    previous = os.environ.get(key)
    os.environ[key] = "sentinel-value"
    try:
        program = _py(
            "import os; sys=__import__('sys');"
            f"sys.stdout.write(os.environ.get({key!r}, '<missing>'))"
        )
        result = _run(repo, program)
    finally:
        if previous is None:
            del os.environ[key]
        else:
            os.environ[key] = previous
    assert result.stdout == "sentinel-value"


# --- Non-interactive stdin ---------------------------------------------------


def test_stdin_is_eof(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    # A child reading stdin must observe EOF (empty) rather than blocking on the
    # parent's terminal.
    result = _run(
        repo,
        _py("import sys; data = sys.stdin.read(); sys.stdout.write(str(len(data)))"),
    )
    assert result.exit_code == 0
    assert result.stdout == "0"


# --- Invalid UTF-8 output ----------------------------------------------------


def test_invalid_utf8_output_uses_replacement(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    result = _run(
        repo,
        _py(
            "import sys;"
            "sys.stdout.buffer.write(b'\\xff\\xfe');"
            "sys.stdout.buffer.flush();"
            "sys.stderr.buffer.write(b'\\xff');"
            "sys.stderr.buffer.flush()"
        ),
    )
    assert result.exit_code == 0
    assert result.stdout == "\ufffd\ufffd"
    assert result.stderr == "\ufffd"


# --- Timeout -----------------------------------------------------------------


def test_timeout_raises_and_kills_process(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    marker = tmp_path / "survived.txt"
    # The child would create a sentinel after a long sleep; the short timeout
    # must fire first and terminate it so the sentinel never appears.
    program = _py(
        "import time, pathlib;"
        "time.sleep(30);"
        f"pathlib.Path({str(marker)!r}).write_text('x')"
    )
    runner = SubprocessWorkspaceCommandRunner(timeout_seconds=0.5)
    with pytest.raises(WorkspaceCommandTimeoutError):
        asyncio.run(runner.run(repo, program))
    time.sleep(1.0)
    assert not marker.exists()


def test_timeout_kills_descendants(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    marker = tmp_path / "descendant.txt"
    # Parent spawns a detached descendant that would create a sentinel after a
    # delay; process-group termination must reach the descendant too.
    child_src = (
        "import time, pathlib;"
        "time.sleep(5);"
        f"pathlib.Path({str(marker)!r}).write_text('x')"
    )
    parent_src = (
        "import subprocess, sys, time;"
        f"subprocess.Popen([sys.executable, '-c', {child_src!r}]);"
        "time.sleep(30)"
    )
    runner = SubprocessWorkspaceCommandRunner(timeout_seconds=0.5)
    with pytest.raises(WorkspaceCommandTimeoutError):
        asyncio.run(runner.run(repo, _py(parent_src)))
    time.sleep(2.0)
    assert not marker.exists()


# --- Cancellation ------------------------------------------------------------


def test_cancellation_propagates(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")

    async def go() -> None:
        runner = SubprocessWorkspaceCommandRunner(timeout_seconds=30.0)
        task = asyncio.ensure_future(
            runner.run(repo, _py("import time; time.sleep(30)"))
        )
        await asyncio.sleep(0.3)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(go())


def test_cancellation_kills_descendants(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    marker = tmp_path / "cancel-descendant.txt"
    child_src = (
        "import time, pathlib;"
        "time.sleep(5);"
        f"pathlib.Path({str(marker)!r}).write_text('x')"
    )
    parent_src = (
        "import subprocess, sys, time;"
        f"subprocess.Popen([sys.executable, '-c', {child_src!r}]);"
        "time.sleep(30)"
    )

    async def go() -> None:
        runner = SubprocessWorkspaceCommandRunner(timeout_seconds=30.0)
        task = asyncio.ensure_future(runner.run(repo, _py(parent_src)))
        await asyncio.sleep(0.5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(go())
    time.sleep(2.0)
    assert not marker.exists()


# --- Output limits -----------------------------------------------------------


def test_stdout_limit_exceeded(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    program = _py(
        "import sys; sys.stdout.buffer.write(b'x' * 10_000)"
    )
    with pytest.raises(WorkspaceCommandExecutionError) as excinfo:
        _run(repo, program, max_output_bytes=1024)
    assert not isinstance(excinfo.value, WorkspaceCommandTimeoutError)


def test_stderr_limit_exceeded(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    program = _py(
        "import sys; sys.stderr.buffer.write(b'x' * 10_000)"
    )
    with pytest.raises(WorkspaceCommandExecutionError) as excinfo:
        _run(repo, program, max_output_bytes=1024)
    assert not isinstance(excinfo.value, WorkspaceCommandTimeoutError)


def test_exact_limit_succeeds_and_plus_one_fails(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    limit = 2048
    at_limit = _py(f"import sys; sys.stdout.buffer.write(b'x' * {limit})")
    result = _run(repo, at_limit, max_output_bytes=limit)
    assert result.exit_code == 0
    assert len(result.stdout) == limit

    over = _py(f"import sys; sys.stdout.buffer.write(b'x' * {limit + 1})")
    with pytest.raises(WorkspaceCommandExecutionError):
        _run(repo, over, max_output_bytes=limit)


def test_per_stream_budget_is_independent(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    limit = 4096
    # Each stream stays under the per-stream limit, but their combined size
    # exceeds it. This must succeed, proving the budget is per stream.
    program = _py(
        "import sys;"
        f"sys.stdout.buffer.write(b'o' * {limit - 100});"
        f"sys.stderr.buffer.write(b'e' * {limit - 100})"
    )
    result = _run(repo, program, max_output_bytes=limit)
    assert result.exit_code == 0
    assert len(result.stdout) == limit - 100
    assert len(result.stderr) == limit - 100


def test_overflow_terminates_long_running_process(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    marker = tmp_path / "overflow-survived.txt"
    # Emit output beyond the limit, then would otherwise sleep and create a
    # sentinel; overflow must terminate the process promptly.
    program = _py(
        "import sys, time, pathlib;"
        "sys.stdout.buffer.write(b'x' * 10_000);"
        "sys.stdout.buffer.flush();"
        "time.sleep(30);"
        f"pathlib.Path({str(marker)!r}).write_text('x')"
    )
    runner = SubprocessWorkspaceCommandRunner(
        timeout_seconds=30.0, max_output_bytes=1024
    )
    with pytest.raises(WorkspaceCommandExecutionError):
        asyncio.run(runner.run(repo, program))
    time.sleep(1.0)
    assert not marker.exists()


# --- Concurrent draining -----------------------------------------------------


def test_concurrent_stdout_stderr_draining(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    size = 512 * 1024
    # Interleave large writes to both streams. A sequential reader (drain all of
    # one pipe, then the other) would deadlock when a pipe buffer fills.
    program = _py(
        "import sys;"
        f"out = b'o' * {size};"
        f"err = b'e' * {size};"
        "so = sys.stdout.buffer;"
        "se = sys.stderr.buffer;"
        "step = 8192;"
        "pos = 0;"
        f"total = {size};"
        "\n"
        "while pos < total:\n"
        "    so.write(out[pos:pos+step]); so.flush();\n"
        "    se.write(err[pos:pos+step]); se.flush();\n"
        "    pos += step\n",
    )
    result = _run(repo, program, max_output_bytes=size * 2)
    assert result.exit_code == 0
    assert len(result.stdout) == size
    assert len(result.stderr) == size
    assert set(result.stdout) == {"o"}
    assert set(result.stderr) == {"e"}


# --- Repeated execution ------------------------------------------------------


def test_repeated_execution_no_state_leak(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    runner = SubprocessWorkspaceCommandRunner(timeout_seconds=30.0)
    for i in range(3):
        result = asyncio.run(
            runner.run(repo, _py(f"print({i})"))
        )
        assert result.exit_code == 0
        assert result.stdout == f"{i}\n"
