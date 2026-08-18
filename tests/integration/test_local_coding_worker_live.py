"""Opt-in live smoke test for the local coding-worker loop.

This test exercises the *real* production local coding-agent vertical
end-to-end against a real local model served by Ollama::

    real OllamaRuntime
        -> real LocalCodingWorker
        -> real structured WorkerAction generation
        -> test-local authorization (allow-all, harness only)
        -> real WorkspaceActionExecutor
        -> real repository context/search/read/write
        -> REAL temporary Git repository
        -> FinishAction

It answers exactly one binary integration question: *can a real local model
drive LLMForeman's structured observe-act loop to inspect and modify a small
real Git repository correctly?* It is not a benchmark, model matrix, or quality
comparison; one invocation exercises exactly one configured model.

Safety of this harness (a property of THIS test, not a production policy):

* Model-requested ``RunCommandAction``s are routed to a test-local
  ``SmokeTestCommandRunner`` that NEVER executes the requested command, so no
  model-directed host command runs. Git and ripgrep used by the real workspace
  infrastructure legitimately spawn their own local subprocesses; this test
  makes no "no subprocesses at all" claim.
* Model writes still go through the real, secure ``GitRepositoryFileWriter``
  rooted in the disposable temporary repository.
* Search and read are non-mutating and bounded by the real workspace
  implementations.
* The allow-all ``SmokeTestAuthorizer`` is acceptable ONLY here because the
  repository is temporary/disposable, writes are bounded by the real writer,
  run actions cannot execute, and finish only terminates the loop.
* Generated source is parsed with ``ast`` only; it is never imported or
  executed.

This test is DISABLED BY DEFAULT. It runs only when the user explicitly opts in
with ``LLMFOREMAN_RUN_LIVE_OLLAMA_TESTS=1``. Ordinary ``uv run pytest`` skips it
and never contacts Ollama, inspects installed models, or checks executables.

Once opted in, missing prerequisites (Git, ripgrep, a reachable Ollama server,
the selected/default model) FAIL rather than skip: the user asked to run an
environment-sensitive test and deserves an actionable failure.

Manual invocation::

    LLMFOREMAN_RUN_LIVE_OLLAMA_TESTS=1 \\
    uv run pytest tests/integration/test_local_coding_worker_live.py -s

Optional model override (uses the OllamaRuntime default when unset)::

    LLMFOREMAN_OLLAMA_MODEL=<installed-model-name> \\
    LLMFOREMAN_RUN_LIVE_OLLAMA_TESTS=1 \\
    uv run pytest tests/integration/test_local_coding_worker_live.py -s
"""

from __future__ import annotations

import ast
import asyncio
import os
import shutil
import subprocess
import tempfile
from collections.abc import Awaitable
from dataclasses import dataclass
from pathlib import Path

import pytest

from llmforeman_core import WorkerAction
from llmforeman_orchestration import (
    LocalCodingWorker,
    LocalWorkerResult,
    WorkspaceActionExecutor,
)
from llmforeman_runtimes import OllamaRuntime
from llmforeman_workspace import (
    CommandResult,
    GitRepositoryContextLoader,
    GitRepositoryFileReader,
    GitRepositoryFileWriter,
    RipgrepRepositoryTextSearcher,
)

# Explicit opt-in gate. Only the exact value ``"1"`` enables live execution,
# matching the established ``LLMFOREMAN_RUN_LIVE_TESTS`` convention. Merely
# having Ollama installed/running with a model present must NOT enable it.
_LIVE_FLAG_ENV = "LLMFOREMAN_RUN_LIVE_OLLAMA_TESTS"

# Optional model override read only by this test harness; the runtime itself
# has no such environment variable. Unset -> OllamaRuntime default model.
_MODEL_ENV = "LLMFOREMAN_OLLAMA_MODEL"

# The deliberately small, deterministic engineering task. It states the desired
# outcome and contract but prescribes no action sequence, no search/read/write
# counts, and no file contents, so the agent chooses its own valid path.
_INSTRUCTION = (
    "Add a `triple(value: int) -> int` function to `src/math_utils.py`.\n"
    "It must return `value * 3`.\n"
    "Preserve the existing `double` function.\n"
    "Inspect the repository as needed and finish when complete."
)

# Initial tiny source; intentionally lacks ``triple`` so a passing run must have
# been produced by the real worker write path.
_INITIAL_MATH_UTILS = "def double(value: int) -> int:\n    return value + value\n"

_README = "# Local worker smoke repository\n"


pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        os.getenv(_LIVE_FLAG_ENV) != "1",
        reason=(
            f"Live Ollama worker smoke test disabled. Set {_LIVE_FLAG_ENV}=1 to "
            "run it; it drives a real local model through Ollama and requires "
            "Git, ripgrep, a running Ollama server, and the selected model."
        ),
    ),
]


class SmokeTestCommandRunner:
    """Test-local ``WorkspaceCommandRunner`` that never executes a command.

    Structurally satisfies the real ``WorkspaceCommandRunner`` Protocol. It is
    the harness's command-execution security boundary: a model-requested
    ``RunCommandAction`` reaches this fake instead of any real runner such as
    ``SubprocessWorkspaceCommandRunner``, so no model-directed command runs on
    the host. It records the exact requested argv for diagnostics/assertions and
    returns a deterministic successful ``CommandResult`` preserving that argv. It
    never spawns a process, invokes a shell, inspects the executable name, or
    differentiates commands.
    """

    def __init__(self) -> None:
        self.requested_commands: list[list[str]] = []

    async def run(
        self,
        repository_root: Path,
        command: list[str],
    ) -> CommandResult:
        self.requested_commands.append(list(command))
        return CommandResult(
            command=list(command),
            exit_code=0,
            stdout="Smoke-test harness: command accepted but not executed.",
            stderr="",
        )


class SmokeTestAuthorizer:
    """Test-local allow-all ``WorkerActionAuthorizer`` for THIS harness only.

    Structurally satisfies the real ``WorkerActionAuthorizer`` Protocol by
    authorizing every action (returning normally, never raising). It records
    each authorized ``WorkerAction`` so the test can confirm that actions --
    including the terminal finish -- crossed the authorization boundary.

    Allow-all is acceptable here, and ONLY here, because: search/read remain
    bounded by the real workspace implementations; writes are bounded by the
    real secure ``GitRepositoryFileWriter`` rooted in the temp repo;
    model-requested run actions reach the non-executing fake runner; finish only
    terminates the loop; and the entire repository is temporary/disposable. This
    is a property of the smoke-test harness and is deliberately NOT a production
    authorization policy.
    """

    def __init__(self) -> None:
        self.actions: list[WorkerAction] = []

    async def authorize(self, action: WorkerAction) -> None:
        self.actions.append(action)


def _run[T](coro: Awaitable[T]) -> T:
    """Drive an async coroutine, matching the repository's asyncio.run style."""

    return asyncio.run(coro)  # type: ignore[arg-type]


def _require_executable(name: str) -> None:
    """Fail fast (never skip) when an opted-in prerequisite executable is absent.

    The user explicitly opted in, so a missing local tool is an actionable
    configuration failure rather than a silent skip.
    """

    if shutil.which(name) is None:
        pytest.fail(
            f"Live Ollama worker smoke test enabled ({_LIVE_FLAG_ENV}=1) but the "
            f"'{name}' executable was not found on PATH. The live worker smoke "
            "uses real Git and the RipgrepRepositoryTextSearcher (ripgrep, "
            f"'rg'); install '{name}' to run it.",
            pytrace=False,
        )


def _git(repo: Path, *args: str) -> None:
    """Run a local, shell-free Git command for temporary-repo setup only."""

    subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
    )


def _init_temp_repository(root: Path) -> None:
    """Create the tiny real temporary Git repository the worker operates on.

    Initializes a local Git working tree and tracks ``README.md`` and
    ``src/math_utils.py`` (``git init`` + ``git add``; no commit and no Git
    identity are required, and there are no remotes/network). ``math_utils.py``
    is deliberately NOT a root seed file, so it is discoverable only through the
    tree and the normal search/read capabilities rather than being injected into
    the initial context.
    """

    root.mkdir(parents=True, exist_ok=True)
    _git(root, "init", "-q")

    (root / "README.md").write_text(_README, encoding="utf-8")
    src_dir = root / "src"
    src_dir.mkdir(parents=True, exist_ok=True)
    (src_dir / "math_utils.py").write_text(_INITIAL_MATH_UTILS, encoding="utf-8")

    _git(root, "add", "--", "README.md", "src/math_utils.py")


def _build_runtime() -> OllamaRuntime:
    """Construct the real OllamaRuntime, honoring the optional model override.

    Unset override -> the current ``OllamaRuntime`` default model (no default
    literal is duplicated in the test). A nonblank override -> that exact model.
    A blank/whitespace-only override -> a clear failure rather than a surprising
    silent fallback. The selected model is never auto-pulled.
    """

    override = os.getenv(_MODEL_ENV)
    if override is None:
        return OllamaRuntime()
    if not override.strip():
        pytest.fail(
            f"{_MODEL_ENV} is set but blank/whitespace-only. Unset it to use the "
            "OllamaRuntime default model, or set it to an installed model name.",
            pytrace=False,
        )
    return OllamaRuntime(model=override)


def _top_level_function(module: ast.Module, name: str) -> ast.FunctionDef | None:
    """Return the top-level ``def``/``async def`` named ``name``, or ``None``."""

    for node in module.body:
        if (
            isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
            and node.name == name
        ):
            return node if isinstance(node, ast.FunctionDef) else None
    return None


def _assert_triple_contract(func: ast.FunctionDef) -> None:
    """Assert ``triple`` matches ``triple(value: int) -> int`` returning value * 3.

    Uses focused, location-independent AST structure checks: exactly one normal
    positional parameter named ``value`` annotated ``int``, an ``int`` return
    annotation, and a body containing ``return value * 3`` (the requested direct
    multiplication semantics). Formatting/style are intentionally not asserted.
    """

    args = func.args
    assert not args.posonlyargs, "triple must not use positional-only parameters"
    assert not args.kwonlyargs, "triple must not use keyword-only parameters"
    assert args.vararg is None, "triple must not use *args"
    assert args.kwarg is None, "triple must not use **kwargs"
    assert len(args.args) == 1, "triple must take exactly one parameter"

    (param,) = args.args
    assert param.arg == "value", f"triple parameter must be named 'value', got {param.arg!r}"
    assert isinstance(param.annotation, ast.Name) and param.annotation.id == "int", (
        "triple parameter 'value' must be annotated 'int'"
    )
    assert isinstance(func.returns, ast.Name) and func.returns.id == "int", (
        "triple must have an 'int' return annotation"
    )

    def _is_value_times_three(expr: ast.expr | None) -> bool:
        return (
            isinstance(expr, ast.BinOp)
            and isinstance(expr.op, ast.Mult)
            and isinstance(expr.left, ast.Name)
            and expr.left.id == "value"
            and isinstance(expr.right, ast.Constant)
            and expr.right.value == 3
        )

    returns_value_times_three = any(
        isinstance(node, ast.Return) and _is_value_times_three(node.value)
        for node in ast.walk(func)
    )
    assert returns_value_times_three, (
        "triple must return the requested 'value * 3' multiplication"
    )


@dataclass(frozen=True, slots=True)
class _RunOutcome:
    """The worker result plus the harness observers captured during one run."""

    result: LocalWorkerResult
    command_runner: SmokeTestCommandRunner
    authorizer: SmokeTestAuthorizer
    max_steps: int


async def _run_worker(runtime: OllamaRuntime, repo: Path) -> _RunOutcome:
    """Compose the real worker with the fakes and run one bounded loop.

    Real: OllamaRuntime, GitRepositoryContextLoader, RipgrepRepositoryTextSearcher,
    GitRepositoryFileReader, GitRepositoryFileWriter, WorkspaceActionExecutor,
    LocalCodingWorker. Fake: the command runner and the authorizer. The runtime
    is used as an async context manager so its owned client is closed even if
    generation fails, the worker hits the step limit, or the run otherwise
    raises.
    """

    command_runner = SmokeTestCommandRunner()
    authorizer = SmokeTestAuthorizer()

    executor = WorkspaceActionExecutor(
        searcher=RipgrepRepositoryTextSearcher(),
        reader=GitRepositoryFileReader(),
        writer=GitRepositoryFileWriter(),
        runner=command_runner,
    )
    # The executor's command boundary must be exactly the non-executing fake.
    assert executor._runner is command_runner  # noqa: SLF001

    worker = LocalCodingWorker(
        runtime=runtime,
        repository_context_loader=GitRepositoryContextLoader(),
        authorizer=authorizer,
        executor=executor,
    )

    async with runtime:
        result = await worker.run(repo, _INSTRUCTION)

    return _RunOutcome(
        result=result,
        command_runner=command_runner,
        authorizer=authorizer,
        max_steps=worker._max_steps,  # noqa: SLF001
    )


def test_local_coding_worker_live() -> None:
    # Opted in: prerequisites are failures, not skips.
    _require_executable("git")
    _require_executable("rg")

    tmp_root = Path(tempfile.mkdtemp(prefix="llmforeman-live-worker-"))
    repo = tmp_root / "repo"
    try:
        _init_temp_repository(repo)

        math_utils_path = repo / "src" / "math_utils.py"
        initial_source = math_utils_path.read_text(encoding="utf-8")
        assert "def double" in initial_source
        assert "triple" not in initial_source

        # Record the initial ``double`` structure for a strong preservation check.
        initial_module = ast.parse(initial_source)
        initial_double = _top_level_function(initial_module, "double")
        assert initial_double is not None, "fixture must define top-level 'double'"
        initial_double_dump = ast.dump(initial_double, include_attributes=False)

        runtime = _build_runtime()
        configured_model = runtime.model
        outcome = _run(_run_worker(runtime, repo))
        result = outcome.result

        # --- Worker finished normally through an authorized FinishAction. ------
        assert isinstance(result, LocalWorkerResult)

        # --- Bounded, positive step count. -------------------------------------
        assert 1 <= result.steps <= outcome.max_steps

        # --- Meaningful live model usage survived the whole composition. -------
        assert result.usage.input_tokens > 0
        assert result.usage.output_tokens > 0

        # --- The final source was actually mutated by the real worker. ---------
        final_source = math_utils_path.read_text(encoding="utf-8")
        assert final_source != initial_source, "worker did not modify math_utils.py"

        # --- Static-only verification: parse, never import/execute. ------------
        final_module = ast.parse(final_source)

        final_double = _top_level_function(final_module, "double")
        assert final_double is not None, "'double' must still exist"
        assert (
            ast.dump(final_double, include_attributes=False) == initial_double_dump
        ), "existing 'double' function must be preserved unchanged"

        final_triple = _top_level_function(final_module, "triple")
        assert final_triple is not None, "'triple' function must be added"
        _assert_triple_contract(final_triple)

        # --- The terminal finish crossed the test authorization boundary. ------
        authorizer = outcome.authorizer
        assert authorizer.actions, "authorizer observed no actions"
        finish_authorized = any(
            action.root.action == "finish" for action in authorizer.actions
        )
        assert finish_authorized, "FinishAction did not cross the authorizer"
        assert authorizer.actions[-1].root.action == "finish", (
            "the final authorized action must be the terminating finish"
        )

        # --- At least one write crossed authorization (real mutation path). ----
        write_authorized = any(
            action.root.action == "write" for action in authorizer.actions
        )
        assert write_authorized, "no WriteFileAction crossed the authorizer"

        # --- The command boundary was the non-executing fake runner. -----------
        # Zero requested commands is valid; if the model requested any, they were
        # recorded here and provably never executed on the host.
        command_runner = outcome.command_runner

        # --- Diagnostics (visible only under ``-s``). --------------------------
        # Deliberately no absolute temp-repo path, environment, system prompt,
        # observation history, or Ollama host/credentials are printed.
        print(f"\nModel: {configured_model}")
        print(f"Steps: {result.steps}")
        print(f"Input tokens: {result.usage.input_tokens}")
        print(f"Output tokens: {result.usage.output_tokens}")
        print(f"Cache read input tokens: {result.usage.cache_read_input_tokens}")
        print(
            f"Cache creation input tokens: {result.usage.cache_creation_input_tokens}"
        )
        print(f"Summary: {result.summary}")
        if command_runner.requested_commands:
            print("Model-requested commands (not executed):")
            for argv in command_runner.requested_commands:
                print(f"- {argv}")
        else:
            print("Model-requested commands (not executed): <none>")
        print("Final src/math_utils.py:")
        print(final_source)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
