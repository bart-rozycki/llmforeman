"""Behavioral tests for :class:`WorkspaceActionExecutor`.

All tests use pure, typed structural fakes of the four workspace capability
Protocols. They perform no real filesystem, Git, ripgrep, subprocess, network,
or model access; every capability outcome is configured in-memory.
"""

import asyncio
from pathlib import Path
from typing import cast

import pytest

from llmforeman_core import (
    ActionErrorObservation,
    FinishAction,
    ReadFileAction,
    ReadObservation,
    RepositoryFile,
    RunCommandAction,
    RunObservation,
    SearchAction,
    SearchObservation,
    WriteFileAction,
    WriteObservation,
)
from llmforeman_orchestration import WorkspaceActionExecutor
from llmforeman_workspace import (
    CommandResult,
    InvalidRepositoryError,
    RepositoryFileAccessError,
    RepositoryFileWriteError,
    RepositoryInspectionError,
    RepositorySearchError,
    RepositorySearchMatch,
    RepositorySearchResult,
    WorkspaceCommandExecutionError,
    WorkspaceCommandTimeoutError,
)

REPO_ROOT = Path("/example/repo/subdirectory")


class _FakeSearcher:
    def __init__(
        self,
        result: RepositorySearchResult | None = None,
        error: BaseException | None = None,
    ) -> None:
        self._result = result
        self._error = error
        self.calls: list[tuple[Path, str]] = []

    async def search(
        self, repository_root: Path, query: str
    ) -> RepositorySearchResult:
        self.calls.append((repository_root, query))
        if self._error is not None:
            raise self._error
        assert self._result is not None
        return self._result


class _FakeReader:
    def __init__(
        self,
        result: RepositoryFile | None = None,
        error: BaseException | None = None,
    ) -> None:
        self._result = result
        self._error = error
        self.calls: list[tuple[Path, str]] = []

    async def read(self, repository_root: Path, path: str) -> RepositoryFile:
        self.calls.append((repository_root, path))
        if self._error is not None:
            raise self._error
        assert self._result is not None
        return self._result


class _FakeWriter:
    def __init__(
        self,
        result: RepositoryFile | None = None,
        error: BaseException | None = None,
    ) -> None:
        self._result = result
        self._error = error
        self.calls: list[tuple[Path, str, str]] = []

    async def write(
        self, repository_root: Path, path: str, content: str
    ) -> RepositoryFile:
        self.calls.append((repository_root, path, content))
        if self._error is not None:
            raise self._error
        assert self._result is not None
        return self._result


class _FakeRunner:
    def __init__(
        self,
        result: CommandResult | None = None,
        error: BaseException | None = None,
    ) -> None:
        self._result = result
        self._error = error
        self.calls: list[tuple[Path, list[str]]] = []

    async def run(
        self, repository_root: Path, command: list[str]
    ) -> CommandResult:
        self.calls.append((repository_root, command))
        if self._error is not None:
            raise self._error
        assert self._result is not None
        return self._result


def _executor(
    searcher: _FakeSearcher | None = None,
    reader: _FakeReader | None = None,
    writer: _FakeWriter | None = None,
    runner: _FakeRunner | None = None,
) -> tuple[
    WorkspaceActionExecutor, _FakeSearcher, _FakeReader, _FakeWriter, _FakeRunner
]:
    searcher = searcher or _FakeSearcher(RepositorySearchResult(matches=[]))
    reader = reader or _FakeReader(RepositoryFile(path="x", content=""))
    writer = writer or _FakeWriter(RepositoryFile(path="x", content=""))
    runner = runner or _FakeRunner(
        CommandResult(command=["x"], exit_code=0, stdout="", stderr="")
    )
    executor = WorkspaceActionExecutor(
        searcher=searcher,
        reader=reader,
        writer=writer,
        runner=runner,
    )
    return executor, searcher, reader, writer, runner


# --- Structural construction / typing -----------------------------------------


def test_construction_accepts_structurally_typed_capabilities() -> None:
    from llmforeman_workspace import (
        RepositoryFileReader,
        RepositoryFileWriter,
        RepositoryTextSearcher,
        WorkspaceCommandRunner,
    )

    # Assigning fakes to the Protocol-typed names is the static structural check;
    # mypy rejects a fake whose signatures do not match the capability Protocols.
    searcher: RepositoryTextSearcher = _FakeSearcher(RepositorySearchResult(matches=[]))
    reader: RepositoryFileReader = _FakeReader(RepositoryFile(path="x", content=""))
    writer: RepositoryFileWriter = _FakeWriter(RepositoryFile(path="x", content=""))
    runner: WorkspaceCommandRunner = _FakeRunner(
        CommandResult(command=["x"], exit_code=0, stdout="", stderr="")
    )
    executor = WorkspaceActionExecutor(
        searcher=searcher,
        reader=reader,
        writer=writer,
        runner=runner,
    )
    assert isinstance(executor, WorkspaceActionExecutor)


# --- Search -------------------------------------------------------------------


def test_search_invokes_only_searcher_with_exact_arguments() -> None:
    executor, searcher, reader, writer, runner = _executor(
        searcher=_FakeSearcher(RepositorySearchResult(matches=[]))
    )
    action = SearchAction(action="search", query="RetryPolicy")

    observation = asyncio.run(executor.execute(REPO_ROOT, action))

    assert len(searcher.calls) == 1
    root, query = searcher.calls[0]
    assert root is REPO_ROOT
    assert query == "RetryPolicy"
    assert reader.calls == []
    assert writer.calls == []
    assert runner.calls == []
    assert isinstance(observation.root, SearchObservation)


def test_search_maps_matches_preserving_order_and_fields() -> None:
    result = RepositorySearchResult(
        matches=[
            RepositorySearchMatch(path="src/b.py", line_number=20, line="    RetryPolicy"),
            RepositorySearchMatch(path="src/a.py", line_number=3, line="RetryPolicy"),
        ]
    )
    executor, *_ = _executor(searcher=_FakeSearcher(result))
    action = SearchAction(action="search", query="RetryPolicy")

    observation = asyncio.run(executor.execute(REPO_ROOT, action))

    root = observation.root
    assert isinstance(root, SearchObservation)
    assert root.query == "RetryPolicy"
    assert [(m.path, m.line_number, m.line) for m in root.matches] == [
        ("src/b.py", 20, "    RetryPolicy"),
        ("src/a.py", 3, "RetryPolicy"),
    ]


def test_empty_search_is_normal_observation() -> None:
    executor, *_ = _executor(
        searcher=_FakeSearcher(RepositorySearchResult(matches=[]))
    )
    action = SearchAction(action="search", query="Nothing")

    observation = asyncio.run(executor.execute(REPO_ROOT, action))

    root = observation.root
    assert isinstance(root, SearchObservation)
    assert root.matches == []


# --- Read ---------------------------------------------------------------------


def test_read_invokes_only_reader_with_exact_arguments() -> None:
    executor, searcher, reader, writer, runner = _executor(
        reader=_FakeReader(RepositoryFile(path="src/example.py", content="body"))
    )
    action = ReadFileAction(action="read", path="src/example.py")

    asyncio.run(executor.execute(REPO_ROOT, action))

    assert len(reader.calls) == 1
    root, path = reader.calls[0]
    assert root is REPO_ROOT
    assert path == "src/example.py"
    assert searcher.calls == []
    assert writer.calls == []
    assert runner.calls == []


def test_read_maps_result_preserving_content_exactly() -> None:
    executor, *_ = _executor(
        reader=_FakeReader(
            RepositoryFile(path="src/example.py", content="    value = 1\n\n")
        )
    )
    action = ReadFileAction(action="read", path="src/example.py")

    observation = asyncio.run(executor.execute(REPO_ROOT, action))

    root = observation.root
    assert isinstance(root, ReadObservation)
    assert root.path == "src/example.py"
    assert root.content == "    value = 1\n\n"


def test_empty_read_content_is_valid() -> None:
    executor, *_ = _executor(
        reader=_FakeReader(RepositoryFile(path="src/empty.py", content=""))
    )
    action = ReadFileAction(action="read", path="src/empty.py")

    observation = asyncio.run(executor.execute(REPO_ROOT, action))

    root = observation.root
    assert isinstance(root, ReadObservation)
    assert root.content == ""


# --- Write --------------------------------------------------------------------


SENTINEL_CONTENT = "SECRET_SOURCE_SENTINEL_9f3a\nline two\n    indented\n"


def test_write_invokes_only_writer_with_exact_arguments() -> None:
    executor, searcher, reader, writer, runner = _executor(
        writer=_FakeWriter(
            RepositoryFile(path="src/example.py", content=SENTINEL_CONTENT)
        )
    )
    action = WriteFileAction(
        action="write", path="src/example.py", content=SENTINEL_CONTENT
    )

    asyncio.run(executor.execute(REPO_ROOT, action))

    assert len(writer.calls) == 1
    root, path, content = writer.calls[0]
    assert root is REPO_ROOT
    assert path == "src/example.py"
    assert content == SENTINEL_CONTENT
    assert searcher.calls == []
    assert reader.calls == []
    assert runner.calls == []


def test_write_success_does_not_echo_content() -> None:
    executor, *_ = _executor(
        writer=_FakeWriter(
            RepositoryFile(path="src/example.py", content=SENTINEL_CONTENT)
        )
    )
    action = WriteFileAction(
        action="write", path="src/example.py", content=SENTINEL_CONTENT
    )

    observation = asyncio.run(executor.execute(REPO_ROOT, action))

    root = observation.root
    assert isinstance(root, WriteObservation)
    assert root.path == "src/example.py"
    serialized = observation.model_dump_json()
    assert "SECRET_SOURCE_SENTINEL_9f3a" not in serialized
    assert set(root.model_dump().keys()) == {"observation", "path"}


# --- Run ----------------------------------------------------------------------


def test_run_invokes_only_runner_with_exact_argv() -> None:
    executor, searcher, reader, writer, runner = _executor(
        runner=_FakeRunner(
            CommandResult(
                command=["uv", "run", "pytest"], exit_code=0, stdout="", stderr=""
            )
        )
    )
    action = RunCommandAction(action="run", command=["uv", "run", "pytest"])

    asyncio.run(executor.execute(REPO_ROOT, action))

    assert len(runner.calls) == 1
    root, command = runner.calls[0]
    assert root is REPO_ROOT
    assert command == ["uv", "run", "pytest"]
    assert searcher.calls == []
    assert reader.calls == []
    assert writer.calls == []


def test_run_success_maps_exactly() -> None:
    executor, *_ = _executor(
        runner=_FakeRunner(
            CommandResult(
                command=["uv", "run", "pytest"],
                exit_code=0,
                stdout="10 passed\n",
                stderr="",
            )
        )
    )
    action = RunCommandAction(action="run", command=["uv", "run", "pytest"])

    observation = asyncio.run(executor.execute(REPO_ROOT, action))

    root = observation.root
    assert isinstance(root, RunObservation)
    assert root.command == ["uv", "run", "pytest"]
    assert root.exit_code == 0
    assert root.stdout == "10 passed\n"
    assert root.stderr == ""


def test_run_non_zero_exit_is_normal_observation() -> None:
    executor, *_ = _executor(
        runner=_FakeRunner(
            CommandResult(
                command=["uv", "run", "pytest"],
                exit_code=1,
                stdout="3 failed\n",
                stderr="",
            )
        )
    )
    action = RunCommandAction(action="run", command=["uv", "run", "pytest"])

    observation = asyncio.run(executor.execute(REPO_ROOT, action))

    root = observation.root
    assert isinstance(root, RunObservation)
    assert root.exit_code == 1
    assert root.stdout == "3 failed\n"


def test_run_negative_exit_code_is_preserved() -> None:
    executor, *_ = _executor(
        runner=_FakeRunner(
            CommandResult(
                command=["make"], exit_code=-15, stdout="", stderr=""
            )
        )
    )
    action = RunCommandAction(action="run", command=["make"])

    observation = asyncio.run(executor.execute(REPO_ROOT, action))

    root = observation.root
    assert isinstance(root, RunObservation)
    assert root.exit_code == -15


def test_run_output_is_preserved_exactly() -> None:
    stdout = "  leading\ttab\nline2 trailing   \n\n\x1b[31mred\x1b[0m\n"
    stderr = "warn: something   \n"
    executor, *_ = _executor(
        runner=_FakeRunner(
            CommandResult(
                command=["tool"], exit_code=2, stdout=stdout, stderr=stderr
            )
        )
    )
    action = RunCommandAction(action="run", command=["tool"])

    observation = asyncio.run(executor.execute(REPO_ROOT, action))

    root = observation.root
    assert isinstance(root, RunObservation)
    assert root.stdout == stdout
    assert root.stderr == stderr


# --- Expected action-level error sanitization ---------------------------------

SENTINEL_PATH = "/Users/private-user/secret-repo/.env"


def test_search_error_is_sanitized() -> None:
    executor, *_ = _executor(
        searcher=_FakeSearcher(error=RepositorySearchError(SENTINEL_PATH))
    )
    action = SearchAction(action="search", query="RetryPolicy")

    observation = asyncio.run(executor.execute(REPO_ROOT, action))

    root = observation.root
    assert isinstance(root, ActionErrorObservation)
    assert root.action == "search"
    assert SENTINEL_PATH not in root.message
    assert SENTINEL_PATH not in observation.model_dump_json()


def test_read_error_is_sanitized_but_keeps_logical_path() -> None:
    executor, *_ = _executor(
        reader=_FakeReader(error=RepositoryFileAccessError(SENTINEL_PATH))
    )
    action = ReadFileAction(action="read", path="src/example.py")

    observation = asyncio.run(executor.execute(REPO_ROOT, action))

    root = observation.root
    assert isinstance(root, ActionErrorObservation)
    assert root.action == "read"
    assert "src/example.py" in root.message
    assert SENTINEL_PATH not in observation.model_dump_json()


def test_write_error_is_sanitized_but_keeps_logical_path() -> None:
    executor, *_ = _executor(
        writer=_FakeWriter(error=RepositoryFileWriteError(SENTINEL_PATH))
    )
    action = WriteFileAction(
        action="write", path="src/example.py", content="anything"
    )

    observation = asyncio.run(executor.execute(REPO_ROOT, action))

    root = observation.root
    assert isinstance(root, ActionErrorObservation)
    assert root.action == "write"
    assert "src/example.py" in root.message
    assert SENTINEL_PATH not in observation.model_dump_json()


def test_run_timeout_is_sanitized_with_specific_message() -> None:
    executor, *_ = _executor(
        runner=_FakeRunner(error=WorkspaceCommandTimeoutError(SENTINEL_PATH))
    )
    action = RunCommandAction(action="run", command=["sleep", "999"])

    observation = asyncio.run(executor.execute(REPO_ROOT, action))

    root = observation.root
    assert isinstance(root, ActionErrorObservation)
    assert root.action == "run"
    assert root.message == "Command execution timed out."
    assert SENTINEL_PATH not in observation.model_dump_json()


def test_run_execution_error_is_sanitized_generic() -> None:
    executor, *_ = _executor(
        runner=_FakeRunner(error=WorkspaceCommandExecutionError(SENTINEL_PATH))
    )
    action = RunCommandAction(action="run", command=["missing-binary"])

    observation = asyncio.run(executor.execute(REPO_ROOT, action))

    root = observation.root
    assert isinstance(root, ActionErrorObservation)
    assert root.action == "run"
    assert root.message == "Command could not be executed."
    assert SENTINEL_PATH not in observation.model_dump_json()


def test_timeout_catch_precedes_generic_execution_error() -> None:
    # WorkspaceCommandTimeoutError subclasses WorkspaceCommandExecutionError;
    # verify the timeout-specific message is not shadowed by the generic branch.
    assert issubclass(WorkspaceCommandTimeoutError, WorkspaceCommandExecutionError)
    executor, *_ = _executor(
        runner=_FakeRunner(error=WorkspaceCommandTimeoutError("boom"))
    )
    action = RunCommandAction(action="run", command=["sleep", "999"])

    observation = asyncio.run(executor.execute(REPO_ROOT, action))

    root = observation.root
    assert isinstance(root, ActionErrorObservation)
    assert root.message == "Command execution timed out."


# --- Fatal / cancellation / unexpected propagation ----------------------------


def test_invalid_repository_error_propagates_from_each_branch() -> None:
    searcher = _FakeSearcher(error=InvalidRepositoryError("bad repo"))
    reader = _FakeReader(error=InvalidRepositoryError("bad repo"))
    writer = _FakeWriter(error=InvalidRepositoryError("bad repo"))
    runner = _FakeRunner(error=InvalidRepositoryError("bad repo"))
    executor = WorkspaceActionExecutor(
        searcher=searcher, reader=reader, writer=writer, runner=runner
    )

    with pytest.raises(InvalidRepositoryError):
        asyncio.run(
            executor.execute(REPO_ROOT, SearchAction(action="search", query="q"))
        )
    with pytest.raises(InvalidRepositoryError):
        asyncio.run(
            executor.execute(REPO_ROOT, ReadFileAction(action="read", path="a.py"))
        )
    with pytest.raises(InvalidRepositoryError):
        asyncio.run(
            executor.execute(
                REPO_ROOT,
                WriteFileAction(action="write", path="a.py", content="x"),
            )
        )
    with pytest.raises(InvalidRepositoryError):
        asyncio.run(
            executor.execute(
                REPO_ROOT, RunCommandAction(action="run", command=["x"])
            )
        )


def test_repository_inspection_error_propagates() -> None:
    executor, *_ = _executor(
        searcher=_FakeSearcher(error=RepositoryInspectionError("inspection"))
    )
    with pytest.raises(RepositoryInspectionError):
        asyncio.run(
            executor.execute(REPO_ROOT, SearchAction(action="search", query="q"))
        )


def test_cancelled_error_propagates() -> None:
    executor, *_ = _executor(
        runner=_FakeRunner(error=asyncio.CancelledError())
    )
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            executor.execute(
                REPO_ROOT, RunCommandAction(action="run", command=["x"])
            )
        )


def test_unexpected_error_propagates() -> None:
    executor, *_ = _executor(
        reader=_FakeReader(error=RuntimeError("programming bug"))
    )
    with pytest.raises(RuntimeError):
        asyncio.run(
            executor.execute(REPO_ROOT, ReadFileAction(action="read", path="a.py"))
        )


# --- Misuse / FinishAction exclusion ------------------------------------------


def test_finish_action_is_not_executable_fails_fast() -> None:
    executor, searcher, reader, writer, runner = _executor()
    # FinishAction is statically excluded from execute(); bypass typing only
    # here to exercise the runtime fail-fast guard.
    finish = cast(SearchAction, FinishAction(action="finish", summary="done"))

    with pytest.raises(TypeError) as exc_info:
        asyncio.run(executor.execute(REPO_ROOT, finish))

    # Type-level diagnostic only; no action repr/content leakage.
    assert "FinishAction" in str(exc_info.value)
    assert "summary" not in str(exc_info.value)
    assert searcher.calls == []
    assert reader.calls == []
    assert writer.calls == []
    assert runner.calls == []


# --- Serialization stays core-controlled --------------------------------------


def test_serialization_shape_is_flat_core_protocol() -> None:
    executor, *_ = _executor(
        runner=_FakeRunner(
            CommandResult(
                command=["uv"], exit_code=0, stdout="ok\n", stderr=""
            )
        )
    )
    observation = asyncio.run(
        executor.execute(REPO_ROOT, RunCommandAction(action="run", command=["uv"]))
    )
    dumped = observation.model_dump()
    assert dumped == {
        "observation": "run",
        "command": ["uv"],
        "exit_code": 0,
        "stdout": "ok\n",
        "stderr": "",
    }
