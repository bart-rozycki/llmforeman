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

It also defines the typed async contract `RepositoryFileReader` for reading a
single, explicitly named file into a core `RepositoryFile`, together with its
concrete Git-backed implementation `GitRepositoryFileReader`.

`GitRepositoryFileReader` provides one safe, on-demand primitive — *read this
explicitly named, Git-tracked file* — and does not search, glob, or list:

- The effective repository top-level is resolved through Git (as with the
  loader), so subdirectory entry points and linked worktrees work.
- The caller-supplied logical path is validated as repository-relative (no
  empty/whitespace, NUL, absolute POSIX/Windows, or `..` traversal) *before* any
  filesystem access; unsafe paths are rejected, never rewritten.
- Only paths present in Git's exact tracked/index set are eligible: filesystem
  existence is not permission to read, so a guessed but untracked `.env` fails
  without its contents being opened. Membership is an exact set check against the
  NUL-delimited tracked listing, so caller-controlled Git pathspec magic never
  applies. Force-added ignored files are eligible because Git tracks them.
- Content comes from the current working tree (locally modified tracked files
  are visible), read through a bounded operation on a worker thread with a
  configurable byte limit (`max_file_bytes`, default 1 MiB), strict UTF-8
  decoding, a minimal NUL-byte binary guard, and symlink-containment checks that
  never escape the repository root or leak absolute paths. Oversized files fail
  rather than being truncated.

It also defines the typed async contract `RepositoryFileWriter` for writing the
complete requested textual state of a single, explicitly named repository file
into a core `RepositoryFile`. This is the first repository *mutation*
capability and, unlike the Git-tracked reader/searcher, it is deliberately
Git-independent: it imposes no precondition that the target path already exist
or be Git-tracked, so a future worker can create new files. The target `path`
is a logical, repository-relative `str` (kept distinct from the local-machine
`repository_root` `Path`), empty `content` is a valid request, and no
normalization is implied. Consistent with the other workspace contracts, this
declaration defines interface semantics only: there is no concrete
implementation yet, and no filesystem write, directory creation, atomicity,
durability, encoding, overwrite, symlink, or path-validation behavior is
promised by the port. Task #24 will own the concrete, safe write mechanics.

Git subprocesses are invoked without a shell. Invalid caller input raises
`InvalidRepositoryError`; failures inspecting an otherwise valid repository raise
`RepositoryInspectionError`; and an explicit file read that cannot be satisfied
(invalid path, untracked path, missing/oversized/non-text/escaping file) raises
`RepositoryFileAccessError` (all subclasses of `WorkspaceError`).
