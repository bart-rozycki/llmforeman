# llmforeman-orchestration

Application/composition layer for LLMForeman.

This package is the first code that intentionally composes two previously
independent boundaries:

- `llmforeman-core` product/worker semantics (`WorkerAction`, `WorkerObservation`);
- `llmforeman-workspace` local repository/process capabilities
  (`RepositoryTextSearcher`, `RepositoryFileReader`, `RepositoryFileWriter`,
  `WorkspaceCommandRunner`).

It depends only on `llmforeman-core` and `llmforeman-workspace`. It must **not**
depend on cloud providers (`llmforeman-providers`), local inference runtimes
(`llmforeman-runtimes`), the CLI, or the desktop app, and nothing in `core` or
`workspace` may depend back on it.

It currently exposes one concrete application service, `WorkspaceActionExecutor`,
which executes a single already-authorized, workspace-backed worker action
(`search`/`read`/`write`/`run`) against injected workspace capabilities and maps
the result into a core `WorkerObservation`. It is a mechanical execution
boundary only: it performs no model/runtime invocation, no worker loop, no
authorization, and no command policy.
