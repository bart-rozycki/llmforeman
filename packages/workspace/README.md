# llmforeman-workspace

Local coding workspace infrastructure boundary for LLMForeman.

This package owns infrastructure concerned with the local coding workspace
(repository/filesystem access, and in future Git operations, diffs, and
workspace isolation). It depends only on `llmforeman-core`; it must not depend
on cloud providers (`llmforeman-providers`) or local inference runtimes
(`llmforeman-runtimes`).

It defines the typed async contract `RepositoryContextLoader` for loading a core
`RepositoryContext` from a local repository root, together with its concrete
Git-backed implementation `GitRepositoryContextLoader`.

`GitRepositoryContextLoader` produces an initial repository snapshot:

- Git (not `.git` filesystem inspection) is the source of truth for working-tree
  identity, so linked worktrees and in-worktree subdirectories both resolve to
  the repository top-level via `git rev-parse --show-toplevel`.
- Tracked paths come from `git ls-files --cached --full-name -z` (NUL-delimited,
  no filesystem crawl, no untracked/ignored files); they form a deterministic,
  sorted, repository-relative `file_tree`.
- Only a small fixed set of root-level seed files (`AGENTS.md`, `CLAUDE.md`,
  `README.md`, `CONTRIBUTING.md`, `pyproject.toml`, `package.json`, `Cargo.toml`)
  are read, from the current working tree, with a configurable per-file byte
  limit (`max_seed_file_bytes`, default 256 KiB), strict UTF-8 decoding, and
  symlink-containment checks that never escape the repository root or leak
  absolute paths.

Git subprocesses are invoked without a shell. Invalid caller input raises
`InvalidRepositoryError`; failures inspecting an otherwise valid repository raise
`RepositoryInspectionError` (both subclasses of `WorkspaceError`).
