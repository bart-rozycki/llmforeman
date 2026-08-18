# llmforeman-orchestration

Application/composition layer for LLMForeman.

This package is the code that intentionally composes previously independent
boundaries:

- `llmforeman-core` product/worker semantics (`WorkerAction`, `WorkerObservation`);
- `llmforeman-workspace` local repository/process capabilities
  (`RepositoryTextSearcher`, `RepositoryFileReader`, `RepositoryFileWriter`,
  `WorkspaceCommandRunner`);
- `llmforeman-runtimes` local structured model generation
  (`StructuredModelRuntime`), used by the local coding-agent loop.

It depends only on `llmforeman-core`, `llmforeman-workspace`, and
`llmforeman-runtimes`. It must **not** depend on cloud providers
(`llmforeman-providers`), the CLI, or the desktop app, and nothing in `core` or
`workspace` may depend back on it. The `runtimes` dependency is driven by the
implemented worker loop composing a real structured runtime, not speculative
layering.

It exposes:

- `WorkspaceActionExecutor`, which executes a single already-authorized,
  workspace-backed worker action (`search`/`read`/`write`/`run`) against injected
  workspace capabilities and maps the result into a core `WorkerObservation`. It
  is a mechanical execution boundary only: no model invocation, no worker loop,
  no authorization, and no command policy.
- `WorkerActionAuthorizer` / `WorkerActionDeniedError`, the authorization seam
  every model-generated `WorkerAction` crosses before it is unwrapped or executed.
- `LocalCodingWorker` (with `LocalWorkerResult` and `WorkerStepLimitError`), the
  first bounded local coding-agent loop. It loads repository context once,
  generates one structured `WorkerAction` per step, authorizes the full action
  before unwrapping it, returns on an authorized `FinishAction`, otherwise
  executes via `WorkspaceActionExecutor` and appends the resulting observation,
  bounded by a constructor `max_steps` budget. It injects an authorizer but
  introduces no concrete approval policy or sandbox, so it is not yet safe for
  unrestricted autonomous execution.
