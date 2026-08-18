"""Executable composition root for the local coding worker.

This module is the concrete composition root for ``llmforeman run``: it chooses
the production adapters (Ollama runtime, Git workspace capabilities, ripgrep
search, subprocess command execution, and a terminal command-approval prompt),
wires them into a :class:`~llmforeman_orchestration.LocalCodingWorker`, runs one
user instruction against a repository, and presents interaction and results to
the local user.

Composition lives here on purpose. There is deliberately no worker factory, no
dependency-injection container, and no runtime registry: the executable layer is
the only place that is allowed to know this v0.1 CLI runs
``Ollama + Git + ripgrep + terminal approval``. Orchestration must remain
unaware of that concrete choice. A single narrow private seam
(:func:`_run_local_worker`) keeps composition testable without exposing any
public fakeable interface.
"""

from __future__ import annotations

import argparse
import asyncio
import shlex
from collections.abc import Callable
from pathlib import Path

from llmforeman_core import ModelUsage, RunCommandAction
from llmforeman_orchestration import (
    CommandApprovalAuthorizer,
    LocalCodingWorker,
    LocalWorkerResult,
    WorkerActionDeniedError,
    WorkerStepLimitError,
    WorkspaceActionExecutor,
)
from llmforeman_runtimes import ModelRuntimeError, OllamaRuntime
from llmforeman_workspace import (
    GitRepositoryContextLoader,
    GitRepositoryFileReader,
    GitRepositoryFileWriter,
    RipgrepRepositoryTextSearcher,
    SubprocessWorkspaceCommandRunner,
    WorkspaceError,
)

__all__ = ["main"]

# Exit codes: conventional success/failure plus 130 for Ctrl-C interruption.
_EXIT_SUCCESS = 0
_EXIT_FAILURE = 1
_EXIT_INTERRUPTED = 130

# Approval accepts only these case-insensitive, whitespace-trimmed responses.
_APPROVAL_RESPONSES = frozenset({"y", "yes"})


async def _approve_command(action: RunCommandAction) -> bool:
    """Prompt the local user to approve one exact ``RunCommandAction``.

    The command is displayed with :func:`shlex.join` for a readable, safely
    escaped rendering; this is presentation only and never re-parsed or executed
    as a shell string. Only a case-insensitive ``y``/``yes`` (after trimming
    surrounding whitespace) approves. Anything else -- including an empty line
    or ``EOFError`` -- denies (fail closed). ``KeyboardInterrupt`` is
    intentionally not caught here: Ctrl-C cancels the operation, it is not a
    "No".
    """
    rendered = shlex.join(action.command)
    print()
    print("Agent wants to run:")
    print()
    print(f"  {rendered}")
    print()
    try:
        response = input("Allow this command? [y/N]: ")
    except EOFError:
        return False
    return response.strip().lower() in _APPROVAL_RESPONSES


async def _run_local_worker(
    *,
    repository_root: Path,
    instruction: str,
    model: str | None,
) -> LocalWorkerResult:
    """Compose the production local coding worker and run one instruction.

    This is the single narrow private seam that owns concrete composition and
    the real :class:`OllamaRuntime` lifecycle. The runtime is used as an async
    context manager so its client is closed on every exit path (success,
    denial, worker failure, invalid repository, step limit, cancellation).
    """
    runtime = OllamaRuntime(model=model) if model is not None else OllamaRuntime()
    async with runtime:
        print("LLMForeman local worker")
        print(f"Repository: {repository_root}")
        print(f"Model: {runtime.model}")
        print()
        print("Working...")

        executor = WorkspaceActionExecutor(
            searcher=RipgrepRepositoryTextSearcher(),
            reader=GitRepositoryFileReader(),
            writer=GitRepositoryFileWriter(),
            runner=SubprocessWorkspaceCommandRunner(),
        )
        authorizer = CommandApprovalAuthorizer(approve_command=_approve_command)
        worker = LocalCodingWorker(
            runtime=runtime,
            repository_context_loader=GitRepositoryContextLoader(),
            authorizer=authorizer,
            executor=executor,
        )
        return await worker.run(repository_root, instruction)


def _print_result(result: LocalWorkerResult) -> None:
    """Print a successful run's step count, all four usage counters, and summary.

    No monetary cost is derived or displayed: ``LocalWorkerResult`` tracks token
    usage only, and inventing pricing would be false economics data.
    """
    usage: ModelUsage = result.usage
    print()
    print("Completed.")
    print()
    print(f"Steps: {result.steps}")
    print(f"Input tokens: {usage.input_tokens:,}")
    print(f"Output tokens: {usage.output_tokens:,}")
    print(f"Cache read input tokens: {usage.cache_read_input_tokens:,}")
    print(f"Cache creation input tokens: {usage.cache_creation_input_tokens:,}")
    print()
    print("Summary:")
    print(result.summary)


def _run_command(args: argparse.Namespace) -> int:
    """Handle ``llmforeman run``: resolve inputs, run the worker, present output.

    Only established expected operational/application failures are translated
    into concise, traceback-free messages; unexpected programming errors (and
    ``KeyboardInterrupt``) are intentionally not caught here so they remain
    visible or reach top-level cancellation handling.
    """
    repository_root: Path = args.repo if args.repo is not None else Path.cwd()

    try:
        result = asyncio.run(
            _run_local_worker(
                repository_root=repository_root,
                instruction=args.instruction,
                model=args.model,
            )
        )
    except WorkerActionDeniedError:
        print("Command denied. Worker stopped.")
        return _EXIT_FAILURE
    except WorkerStepLimitError:
        print("Error: worker reached its step limit.")
        return _EXIT_FAILURE
    except WorkspaceError:
        print("Error: repository could not be inspected.")
        return _EXIT_FAILURE
    except ModelRuntimeError:
        print("Error: local model execution failed.")
        return _EXIT_FAILURE

    _print_result(result)
    return _EXIT_SUCCESS


def _repository_path(value: str) -> Path:
    """Normalize a user-supplied repository path with ordinary ``expanduser``.

    Repository semantics (Git top-level resolution, tracking, existence) belong
    to the workspace implementations; the CLI performs only conventional
    user-path handling.
    """
    return Path(value).expanduser()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llmforeman",
        description="LLMForeman command-line interface.",
    )
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser(
        "run",
        help="Run the local coding worker against a repository.",
        description="Run the local coding worker against a repository.",
    )
    run_parser.add_argument(
        "instruction",
        help="The engineering task for the local coding worker to perform.",
    )
    run_parser.add_argument(
        "--repo",
        type=_repository_path,
        default=None,
        help="Repository path (defaults to the current working directory).",
    )
    run_parser.add_argument(
        "--model",
        default=None,
        help="Ollama model to use (defaults to the runtime's default model).",
    )
    run_parser.set_defaults(handler=_run_command)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``llmforeman`` console script.

    Parses arguments and dispatches to the selected subcommand. With no
    subcommand it prints help and exits successfully. ``KeyboardInterrupt`` is
    translated into a concise ``Cancelled.`` message with conventional
    interruption exit status; unexpected exceptions are not swallowed.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    handler: Callable[[argparse.Namespace], int] | None = getattr(
        args, "handler", None
    )
    if handler is None:
        parser.print_help()
        return _EXIT_SUCCESS

    try:
        return handler(args)
    except KeyboardInterrupt:
        print("Cancelled.")
        return _EXIT_INTERRUPTED
