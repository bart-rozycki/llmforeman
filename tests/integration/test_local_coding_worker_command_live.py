"""Opt-in live smoke test for the local coding-worker *command-execution* loop.

Unlike ``test_local_coding_worker_live.py`` (which routes model-requested
commands to a non-executing fake runner), this test exercises the *real*
command-execution vertical end-to-end against a real local model served by
Ollama::

    real OllamaRuntime
        -> real LocalCodingWorker
        -> real structured WorkerAction generation
        -> real CommandApprovalAuthorizer (with a test-local approval callback)
        -> real WorkspaceActionExecutor
        -> real SubprocessWorkspaceCommandRunner
        -> a REAL trusted verifier process
        -> real CommandResult -> RunObservation
        -> next model decision
        -> FinishAction

Central security property of THIS harness (a property of the test, not a
production policy): the worker may receive approval for exactly ONE trusted
verifier ``argv`` fixed before the worker starts. Every other
``RunCommandAction`` is denied by the real ``CommandApprovalAuthorizer`` and
aborts the smoke. In particular ``pytest``, ``python -c``, any shell, any other
executable, prefixes, subsets, extra/reordered argv, and even an extra flag are
all denied -- authorization is exact ``list`` equality.

The trusted verifier script and its markers live OUTSIDE the disposable Git
repository supplied to the worker, so ``GitRepositoryFileWriter`` (rooted in the
repo) can never modify them. The verifier treats the repository source purely as
text: it ``ast.parse``s the file and never imports, executes, ``exec``/``eval``s,
runs pytest, or spawns another process. Model-generated code is therefore never
executed anywhere in this test.

This test is DISABLED BY DEFAULT. It runs only when the user explicitly opts in
with ``LLMFOREMAN_RUN_LIVE_OLLAMA_COMMAND_TESTS=1``. This is a DIFFERENT,
stronger gate than ``LLMFOREMAN_RUN_LIVE_OLLAMA_TESTS=1`` (which merely permits
live local model interaction): running a real model-triggered subprocess through
the production command-runner path requires this explicit command gate.
``LLMFOREMAN_RUN_LIVE_OLLAMA_TESTS=1`` alone does NOT enable it, and ordinary
``uv run pytest`` skips it without contacting Ollama, inspecting models, or
launching the verifier.

Once opted in, missing prerequisites (Git, ripgrep, a reachable Ollama server,
the selected/default model) FAIL rather than skip.

Manual invocation::

    LLMFOREMAN_RUN_LIVE_OLLAMA_COMMAND_TESTS=1 \\
    uv run pytest tests/integration/test_local_coding_worker_command_live.py -s

Optional model override (uses the OllamaRuntime default when unset)::

    LLMFOREMAN_OLLAMA_MODEL=<installed-model-name> \\
    LLMFOREMAN_RUN_LIVE_OLLAMA_COMMAND_TESTS=1 \\
    uv run pytest tests/integration/test_local_coding_worker_command_live.py -s
"""

from __future__ import annotations

import ast
import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Awaitable
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from llmforeman_core import RunCommandAction
from llmforeman_orchestration import (
    CommandApprovalAuthorizer,
    LocalCodingWorker,
    LocalWorkerResult,
    WorkspaceActionExecutor,
)
from llmforeman_runtimes import OllamaRuntime
from llmforeman_workspace import (
    GitRepositoryContextLoader,
    GitRepositoryFileReader,
    GitRepositoryFileWriter,
    RipgrepRepositoryTextSearcher,
    SubprocessWorkspaceCommandRunner,
)

# Explicit, command-specific opt-in gate. This is INTENTIONALLY distinct from
# the weaker ``LLMFOREMAN_RUN_LIVE_OLLAMA_TESTS`` gate: permitting live local
# model interaction is not the same as permitting this harness to run a real
# model-triggered subprocess through the production command-runner path. Only the
# exact value ``"1"`` enables execution.
_LIVE_COMMAND_FLAG_ENV = "LLMFOREMAN_RUN_LIVE_OLLAMA_COMMAND_TESTS"

# Optional model override read only by this test harness (same convention as the
# Task #34 worker smoke). Unset -> OllamaRuntime default model.
_MODEL_ENV = "LLMFOREMAN_OLLAMA_MODEL"

# The fixed relative target the model must edit and the verifier must inspect.
_TARGET_REL = "src/math_utils.py"

# Initial tiny source; intentionally lacks ``triple`` so a passing run must have
# been produced by the real worker write path.
_INITIAL_MATH_UTILS = "def double(value: int) -> int:\n    return value + value\n"

_README = "# Local worker command smoke repository\n"

# A conservative timeout for the trusted AST verifier, which completes almost
# immediately. Kept test-local; production runner defaults are not modified.
_COMMAND_TIMEOUT_SECONDS = 30.0


pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        os.getenv(_LIVE_COMMAND_FLAG_ENV) != "1",
        reason=(
            f"Live Ollama command-execution smoke test disabled. Set "
            f"{_LIVE_COMMAND_FLAG_ENV}=1 to run it; it drives a real local model "
            "through Ollama AND executes a real trusted verifier subprocess "
            "through the production command-runner path, and requires Git, "
            "ripgrep, a running Ollama server, and the selected model. The "
            "weaker LLMFOREMAN_RUN_LIVE_OLLAMA_TESTS gate does NOT enable it."
        ),
    ),
]


# --- Trusted verifier source -------------------------------------------------
#
# The verifier is STATIC test-harness source (never model-generated). Harness
# creation time injects only literal marker paths, the expected relative target,
# and the harness-computed expected ``double`` AST dump. The verifier treats the
# repository source purely as text: it ``ast.parse``s it and never imports,
# executes, ``exec``/``eval``s, runs pytest, or spawns a subprocess. It uses only
# the stdlib (``ast``, ``pathlib``, ``sys``).

_VERIFIER_BODY = """\
import ast
import sys
from pathlib import Path


def _fail(message: str) -> None:
    # Concise, deterministic diagnostic; no absolute harness/interpreter paths.
    print("Verification failed: " + message)
    sys.exit(1)


def _record_ran() -> None:
    # Append one line per invocation so runs can be counted. Marker content is
    # deliberately path-free.
    with open(RAN_MARKER, "a", encoding="utf-8") as handle:
        handle.write("ran\\n")


def _top_level_function(module, name):
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def _is_value_times_three(expr) -> bool:
    return (
        isinstance(expr, ast.BinOp)
        and isinstance(expr.op, ast.Mult)
        and isinstance(expr.left, ast.Name)
        and expr.left.id == "value"
        and isinstance(expr.right, ast.Constant)
        and expr.right.value == 3
    )


def main() -> None:
    _record_ran()

    # Defense in depth: the authorizer already fixes argv, but re-validate.
    if len(sys.argv) != 2 or sys.argv[1] != EXPECTED_TARGET:
        _fail("verifier invoked with unexpected arguments.")

    # Resolve the target relative to the command cwd (the Git top-level chosen
    # by the production runner); reject arbitrary absolute source paths.
    target = Path(sys.argv[1])
    if target.is_absolute():
        _fail("target must be a relative path.")

    try:
        source = target.read_text(encoding="utf-8")
    except OSError:
        _fail("target source file could not be read.")
        return

    try:
        module = ast.parse(source)
    except SyntaxError:
        _fail("target source is not valid Python.")
        return

    double_node = _top_level_function(module, "double")
    if double_node is None:
        _fail("double function is missing.")
        return
    if ast.dump(double_node, include_attributes=False) != EXPECTED_DOUBLE_DUMP:
        _fail("double function changed.")

    triple_node = _top_level_function(module, "triple")
    if triple_node is None:
        _fail("triple function is missing.")
        return

    args = triple_node.args
    if args.posonlyargs or args.kwonlyargs or args.vararg or args.kwarg:
        _fail("triple must take exactly one normal positional parameter.")
    if len(args.args) != 1:
        _fail("triple must take exactly one parameter.")
    param = args.args[0]
    if param.arg != "value":
        _fail("triple parameter must be named value.")
    if not (isinstance(param.annotation, ast.Name) and param.annotation.id == "int"):
        _fail("triple parameter value must be annotated int.")
    if not (isinstance(triple_node.returns, ast.Name) and triple_node.returns.id == "int"):
        _fail("triple must have an int return annotation.")

    returns_value_times_three = any(
        isinstance(node, ast.Return) and _is_value_times_three(node.value)
        for node in ast.walk(triple_node)
    )
    if not returns_value_times_three:
        _fail("triple must return value * 3.")

    # Only now, after every check passed, record success.
    with open(SUCCESS_MARKER, "w", encoding="utf-8") as handle:
        handle.write("ok\\n")

    print("Verification succeeded.")
    sys.exit(0)


main()
"""


def _build_verifier_source(
    *,
    ran_marker: Path,
    success_marker: Path,
    expected_double_dump: str,
) -> str:
    """Compose the static verifier source with harness-injected literals.

    Only literal marker paths, the expected relative target, and the harness
    expected ``double`` AST dump are embedded (via ``repr``). No model input is
    involved and no code is generated dynamically beyond these constants.
    """

    header = (
        "RAN_MARKER = " + repr(str(ran_marker)) + "\n"
        "SUCCESS_MARKER = " + repr(str(success_marker)) + "\n"
        "EXPECTED_TARGET = " + repr(_TARGET_REL) + "\n"
        "EXPECTED_DOUBLE_DUMP = " + repr(expected_double_dump) + "\n\n"
    )
    return header + _VERIFIER_BODY


# --- Test-local async helpers ------------------------------------------------


def _run[T](coro: Awaitable[T]) -> T:
    """Drive an async coroutine, matching the repository's asyncio.run style."""

    return asyncio.run(coro)  # type: ignore[arg-type]


def _require_executable(name: str) -> None:
    """Fail fast (never skip) when an opted-in prerequisite executable is absent."""

    if shutil.which(name) is None:
        pytest.fail(
            f"Live Ollama command smoke test enabled "
            f"({_LIVE_COMMAND_FLAG_ENV}=1) but the '{name}' executable was not "
            "found on PATH. The smoke uses real Git and the "
            f"RipgrepRepositoryTextSearcher (ripgrep, 'rg'); install '{name}'.",
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
    """Create the tiny real temporary Git repository the worker operates on."""

    root.mkdir(parents=True, exist_ok=True)
    _git(root, "init", "-q")

    (root / "README.md").write_text(_README, encoding="utf-8")
    src_dir = root / "src"
    src_dir.mkdir(parents=True, exist_ok=True)
    (src_dir / "math_utils.py").write_text(_INITIAL_MATH_UTILS, encoding="utf-8")

    _git(root, "add", "--", "README.md", "src/math_utils.py")


def _build_runtime() -> OllamaRuntime:
    """Construct the real OllamaRuntime, honoring the optional model override."""

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
    """Return the top-level ``def`` named ``name``, or ``None``."""

    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def _assert_triple_contract(func: ast.FunctionDef) -> None:
    """Assert ``triple`` matches ``triple(value: int) -> int`` returning value * 3."""

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
    assert returns_value_times_three, "triple must return the requested 'value * 3' multiplication"


# --- Exact-command approval callback -----------------------------------------


@dataclass(slots=True)
class _ApprovalRecorder:
    """Test-local exact-argv approval policy injected into the REAL authorizer.

    This is the ONLY test-local dependency in the composition. It approves a
    ``RunCommandAction`` if and only if its argv is exactly equal to the fixed
    ``expected_command`` known before the worker starts; every other command is
    denied (the real ``CommandApprovalAuthorizer`` then raises
    ``WorkerActionDeniedError``). No decision is cached: each action is decided
    independently, so repeated identical verifier runs are each re-approved.
    """

    expected_command: list[str]
    approval_requests: list[list[str]] = field(default_factory=list)
    approved_commands: list[list[str]] = field(default_factory=list)

    async def approve_command(self, action: RunCommandAction) -> bool:
        self.approval_requests.append(list(action.command))
        if action.command == self.expected_command:
            self.approved_commands.append(list(action.command))
            return True
        return False


@dataclass(frozen=True, slots=True)
class _RunOutcome:
    """The worker result plus the harness observers captured during one run."""

    result: LocalWorkerResult
    recorder: _ApprovalRecorder
    max_steps: int


async def _run_worker(
    runtime: OllamaRuntime,
    repo: Path,
    recorder: _ApprovalRecorder,
    instruction: str,
) -> _RunOutcome:
    """Compose the real worker with the REAL runner/authorizer and run one loop.

    Real: OllamaRuntime, GitRepositoryContextLoader, RipgrepRepositoryTextSearcher,
    GitRepositoryFileReader, GitRepositoryFileWriter,
    SubprocessWorkspaceCommandRunner, WorkspaceActionExecutor,
    CommandApprovalAuthorizer, LocalCodingWorker. The only test-local dependency
    is ``recorder.approve_command`` injected into the production authorizer. The
    runtime is used as an async context manager so its owned client is closed on
    success, authorization denial, runtime failure, step limit, verifier
    failure, or any raised assertion.
    """

    authorizer = CommandApprovalAuthorizer(recorder.approve_command)

    executor = WorkspaceActionExecutor(
        searcher=RipgrepRepositoryTextSearcher(),
        reader=GitRepositoryFileReader(),
        writer=GitRepositoryFileWriter(),
        runner=SubprocessWorkspaceCommandRunner(
            timeout_seconds=_COMMAND_TIMEOUT_SECONDS,
        ),
    )

    worker = LocalCodingWorker(
        runtime=runtime,
        repository_context_loader=GitRepositoryContextLoader(),
        authorizer=authorizer,
        executor=executor,
    )

    async with runtime:
        result = await worker.run(repo, instruction)

    return _RunOutcome(
        result=result,
        recorder=recorder,
        max_steps=worker._max_steps,  # noqa: SLF001
    )


def _count_ran(ran_marker: Path) -> int:
    """Count trusted-verifier executions recorded in the ran marker."""

    if not ran_marker.exists():
        return 0
    return sum(1 for line in ran_marker.read_text(encoding="utf-8").splitlines() if line.strip())


def test_local_coding_worker_command_live() -> None:
    # Opted in: prerequisites are failures, not skips.
    _require_executable("git")
    _require_executable("rg")

    tmp_root = Path(tempfile.mkdtemp(prefix="llmforeman-live-command-"))
    repo = tmp_root / "repo"
    verifier_path = tmp_root / "verify_math_utils.py"
    ran_marker = tmp_root / "verifier-ran"
    success_marker = tmp_root / "verifier-success"
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

        # The trusted verifier and its markers live OUTSIDE the Git worktree, so
        # the model cannot modify them through GitRepositoryFileWriter. Assert
        # this with path semantics rather than string-prefix containment.
        verifier_path.write_text(
            _build_verifier_source(
                ran_marker=ran_marker,
                success_marker=success_marker,
                expected_double_dump=initial_double_dump,
            ),
            encoding="utf-8",
        )
        repo_resolved = repo.resolve()
        for outside in (verifier_path, ran_marker, success_marker):
            assert repo_resolved not in outside.resolve().parents, (
                f"{outside.name} must live outside the model-writable repository"
            )

        # The exact, only-approvable command, fixed BEFORE the worker starts.
        expected_command = [
            sys.executable,
            str(verifier_path),
            _TARGET_REL,
        ]
        command_json = json.dumps(expected_command)

        instruction = (
            f"Add a `triple(value: int) -> int` function to `{_TARGET_REL}`.\n"
            "It must return `value * 3`.\n"
            "Preserve the existing `double` function.\n\n"
            "Before finishing, you MUST verify your implementation by running "
            "exactly this command (argv as a JSON array):\n"
            f"{command_json}\n\n"
            "Run that command with the `run` action, using each JSON array "
            "element as one argv entry. Finish only after that verification "
            "command has exited successfully."
        )

        recorder = _ApprovalRecorder(expected_command=expected_command)

        runtime = _build_runtime()
        configured_model = runtime.model
        outcome = _run(_run_worker(runtime, repo, recorder, instruction))
        result = outcome.result

        # --- Worker finished normally through an authorized FinishAction. ------
        assert isinstance(result, LocalWorkerResult)

        # --- Bounded, positive step count. -------------------------------------
        assert 1 <= result.steps <= outcome.max_steps

        # --- Meaningful live model usage survived the whole composition. -------
        assert result.usage.input_tokens > 0
        assert result.usage.output_tokens > 0

        # --- The model actually requested and ran the trusted verifier. --------
        assert recorder.approval_requests, "approval callback was never invoked"
        assert recorder.approved_commands, "no command was approved"
        for approved in recorder.approved_commands:
            assert approved == expected_command, (
                "only the exact expected verifier argv may be approved"
            )

        execution_count = _count_ran(ran_marker)
        assert execution_count >= 1, "trusted verifier never executed"
        # Every real verifier execution must have been individually authorized.
        assert len(recorder.approved_commands) >= execution_count >= 1, (
            "each verifier execution must correspond to an approved run action"
        )

        # --- A successful trusted verification occurred before finish. ---------
        assert success_marker.exists(), (
            "verifier-success marker missing: no successful verification occurred "
            "before the worker finished"
        )
        assert success_marker.read_text(encoding="utf-8").strip() == "ok", (
            "verifier-success marker contains unexpected content"
        )

        # --- The final source was actually mutated by the real worker. ---------
        final_source = math_utils_path.read_text(encoding="utf-8")
        assert final_source != initial_source, "worker did not modify math_utils.py"

        # --- Static-only verification: parse, never import/execute. ------------
        final_module = ast.parse(final_source)

        final_double = _top_level_function(final_module, "double")
        assert final_double is not None, "'double' must still exist"
        assert ast.dump(final_double, include_attributes=False) == initial_double_dump, (
            "existing 'double' function must be preserved unchanged"
        )

        final_triple = _top_level_function(final_module, "triple")
        assert final_triple is not None, "'triple' function must be added"
        _assert_triple_contract(final_triple)

        # --- Diagnostics (visible only under ``-s``). --------------------------
        # Absolute temp paths (interpreter, verifier, harness root, temp repo)
        # are deliberately redacted; the approved command is shown as a semantic
        # label only, never as raw argv containing temporary absolute paths.
        print(f"\nModel: {configured_model}")
        print(f"Steps: {result.steps}")
        print(f"Input tokens: {result.usage.input_tokens}")
        print(f"Output tokens: {result.usage.output_tokens}")
        print(
            f"Approved commands: {len(recorder.approved_commands)} x "
            "trusted verifier [temporary paths redacted]"
        )
        print(f"Verifier executions: {execution_count}")
        print("Verifier successful: yes")
        print(f"Summary: {result.summary}")
        print("Final src/math_utils.py:")
        print(final_source)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
