"""First LLMForeman application/composition service.

``WorkspaceActionExecutor`` is the first code that deliberately composes core
worker semantics with workspace capabilities:

    core executable WorkerAction
        + workspace search/read/write/run capabilities
        -> core WorkerObservation

It takes a single, already-executable worker action (``search``/``read``/
``write``/``run``), invokes exactly one injected workspace capability, and maps
the capability result -- or a selected, expected action-level failure -- into
the closed core :class:`~llmforeman_core.WorkerObservation` vocabulary.

Boundaries this service deliberately does *not* cross:

* No model/runtime/provider invocation and no worker loop: it executes one
  supplied action and returns one observation. Deciding sequences, iteration,
  and finishing belongs to a future higher-level worker loop.
* No ``FinishAction`` handling: ``FinishAction`` is worker-loop control flow, not
  a workspace capability request, so it is excluded from the accepted union and
  never reaches this service in correctly typed code.
* It is *not* an authorization or command-policy boundary. Authorization
  assumption: an action reaching :meth:`WorkspaceActionExecutor.execute` is
  assumed to have already been authorized by its caller. The typed worker-action
  protocol validates action *shape*, never *intent*: a well-formed
  ``RunCommandAction`` may still be destructive. This service is therefore not a
  sandbox and applies no allowlist/denylist/approval; a future layer owns
  authorization before real model-controlled execution.

Error handling is a deliberate agent-facing privacy boundary. Each action
branch catches only the specific, expected action-level workspace failure(s) it
can meaningfully translate into a safe :class:`ActionErrorObservation`, using a
fixed message (plus the already-validated logical path where useful). Raw
exception text is never copied into an observation. Everything else -- fatal
environment failures (``InvalidRepositoryError``/``RepositoryInspectionError``),
``asyncio.CancelledError``, unexpected bugs, and core observation validation
failures -- propagates unchanged so the application (not the worker) notices it.
"""

from pathlib import Path

from llmforeman_core import (
    ActionErrorObservation,
    ReadFileAction,
    ReadObservation,
    RunCommandAction,
    RunObservation,
    SearchAction,
    SearchObservation,
    WorkerObservation,
    WorkerSearchMatch,
    WriteFileAction,
    WriteObservation,
)
from llmforeman_workspace import (
    RepositoryFileAccessError,
    RepositoryFileReader,
    RepositoryFileWriteError,
    RepositoryFileWriter,
    RepositorySearchError,
    RepositoryTextSearcher,
    WorkspaceCommandExecutionError,
    WorkspaceCommandRunner,
    WorkspaceCommandTimeoutError,
)

__all__ = [
    "WorkspaceActionExecutor",
]


class WorkspaceActionExecutor:
    """Execute one authorized, workspace-backed worker action.

    Injects the four narrow workspace capabilities and, per :meth:`execute`
    call, dispatches a single executable action to exactly one of them, mapping
    the outcome into a core :class:`~llmforeman_core.WorkerObservation`.

    The service is stateless beyond its injected dependencies: it retains no
    action history, observations, current task, repository root, or counters.
    All execution input is explicit per call. It depends on capability
    *Protocols*, never concrete implementations, so a future composition root
    chooses implementations and tests can inject pure fakes.

    Authorization assumption: an action reaching :meth:`execute` is assumed to
    have already been authorized by its caller. This service validates neither
    intent nor authorization and is not a sandbox or command-policy layer.
    """

    def __init__(
        self,
        searcher: RepositoryTextSearcher,
        reader: RepositoryFileReader,
        writer: RepositoryFileWriter,
        runner: WorkspaceCommandRunner,
    ) -> None:
        self._searcher = searcher
        self._reader = reader
        self._writer = writer
        self._runner = runner

    async def execute(
        self,
        repository_root: Path,
        action: (
            SearchAction
            | ReadFileAction
            | WriteFileAction
            | RunCommandAction
        ),
    ) -> WorkerObservation:
        """Execute ``action`` against ``repository_root`` and observe the result.

        ``repository_root`` is passed through unchanged to the selected
        capability; this service performs no ``resolve()``, Git lookup, cwd
        substitution, or path rewriting -- repository-root semantics belong to
        the concrete workspace capabilities.

        Returns a :class:`~llmforeman_core.WorkerObservation` wrapping the
        concrete success variant, or an :class:`ActionErrorObservation` for a
        selected expected action-level workspace failure. Fatal environment
        errors, cancellation, unexpected errors, and core validation failures
        propagate.

        Only ``search``/``read``/``write``/``run`` are executable; ``finish`` is
        excluded by the static type. A caller that bypasses typing and supplies
        an unsupported action triggers a fail-fast ``TypeError`` (API misuse),
        never an :class:`ActionErrorObservation`.
        """
        match action:
            case SearchAction():
                return await self._execute_search(repository_root, action)
            case ReadFileAction():
                return await self._execute_read(repository_root, action)
            case WriteFileAction():
                return await self._execute_write(repository_root, action)
            case RunCommandAction():
                return await self._execute_run(repository_root, action)
            case _:
                # Static typing already excludes FinishAction and the WorkerAction
                # RootModel; this guards runtime misuse only. Use a type-level
                # diagnostic exclusively: repr(action) could leak a
                # WriteFileAction's full generated content into the error surface.
                raise TypeError(
                    f"Unsupported workspace action type: {type(action).__name__}"
                )

    async def _execute_search(
        self,
        repository_root: Path,
        action: SearchAction,
    ) -> WorkerObservation:
        try:
            result = await self._searcher.search(repository_root, action.query)
        except RepositorySearchError:
            return WorkerObservation(
                ActionErrorObservation(
                    observation="error",
                    action="search",
                    message="Repository search failed for the requested query.",
                )
            )
        matches = [
            WorkerSearchMatch(
                path=match.path,
                line_number=match.line_number,
                line=match.line,
            )
            for match in result.matches
        ]
        return WorkerObservation(
            SearchObservation(
                observation="search",
                query=action.query,
                matches=matches,
            )
        )

    async def _execute_read(
        self,
        repository_root: Path,
        action: ReadFileAction,
    ) -> WorkerObservation:
        try:
            result = await self._reader.read(repository_root, action.path)
        except RepositoryFileAccessError:
            return WorkerObservation(
                ActionErrorObservation(
                    observation="error",
                    action="read",
                    message=f"Unable to read repository file '{action.path}'.",
                )
            )
        return WorkerObservation(
            ReadObservation(
                observation="read",
                path=result.path,
                content=result.content,
            )
        )

    async def _execute_write(
        self,
        repository_root: Path,
        action: WriteFileAction,
    ) -> WorkerObservation:
        try:
            result = await self._writer.write(
                repository_root,
                action.path,
                action.content,
            )
        except RepositoryFileWriteError:
            return WorkerObservation(
                ActionErrorObservation(
                    observation="error",
                    action="write",
                    message=f"Unable to write repository file '{action.path}'.",
                )
            )
        # Deliberately no content echo: the worker just produced the content in
        # the preceding action, so repeating it would only duplicate context.
        return WorkerObservation(
            WriteObservation(
                observation="write",
                path=result.path,
            )
        )

    async def _execute_run(
        self,
        repository_root: Path,
        action: RunCommandAction,
    ) -> WorkerObservation:
        try:
            result = await self._runner.run(repository_root, action.command)
        except WorkspaceCommandTimeoutError:
            # Caught before its parent WorkspaceCommandExecutionError so the
            # timeout-specific message is never shadowed by the generic one.
            return WorkerObservation(
                ActionErrorObservation(
                    observation="error",
                    action="run",
                    message="Command execution timed out.",
                )
            )
        except WorkspaceCommandExecutionError:
            return WorkerObservation(
                ActionErrorObservation(
                    observation="error",
                    action="run",
                    message="Command could not be executed.",
                )
            )
        # A non-zero (or negative) exit code is a normal, successfully executed
        # command whose diagnostic result the worker needs; never an error.
        return WorkerObservation(
            RunObservation(
                observation="run",
                command=result.command,
                exit_code=result.exit_code,
                stdout=result.stdout,
                stderr=result.stderr,
            )
        )
