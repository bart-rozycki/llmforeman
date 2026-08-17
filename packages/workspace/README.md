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
into a core `RepositoryFile`, together with its concrete Git-bounded
implementation `GitRepositoryFileWriter`.

`GitRepositoryFileWriter` is the first repository *mutation* primitive. Unlike
the Git-tracked reader/searcher, it is deliberately **not** tracked-only: Git is
used only to establish the working-tree boundary, so the writer can overwrite
tracked *and* untracked files and create new untracked files. It never runs
`git add` or otherwise mutates the Git index/status:

- The effective repository top-level is resolved through Git (as with the reader),
  so subdirectory entry points and linked worktrees write inside the correct
  effective working tree; the logical `path` is always relative to that top-level.
- The logical `path` is validated as repository-relative (no empty/whitespace,
  NUL, absolute POSIX/Windows, or `..` traversal), and `content` is strictly
  UTF-8 encoded and size-checked against a configurable byte limit
  (`max_file_bytes`, default 1 MiB), all *before* any filesystem mutation, so an
  unsafe path, non-encodable content, or oversized request never creates,
  truncates, or opens anything.
- Path traversal uses descriptor-relative, no-follow (`O_NOFOLLOW`/`O_DIRECTORY`)
  operations rooted at an opened handle on the effective top-level, so no symlink
  component — parent or final target, internal or external — is ever followed for
  writing, and symlink safety is enforced at the filesystem operation rather than
  by a race-prone check-then-open. Missing parent directories may be created
  relative to a trusted handle.
- An existing target is opened no-follow, verified to be a regular file, and its
  current contents streamed and validated as UTF-8 text (rejecting invalid UTF-8
  or NUL bytes) *before* it is truncated and rewritten in place through the same
  pinned descriptor, preserving inode and mode bits (an existing executable stays
  executable). New files receive ordinary non-executable permissions subject to
  the process umask. Content is written exactly, with no whitespace, newline, or
  encoding normalization.

The write is a direct in-place truncate+write: it is intentionally **not**
atomic and makes no durability, rollback, or backup promise in v0.1 (no temp
file, `os.replace`, or fsync). The required guarantee here is symlink/path
traversal safety, which is distinct from atomic content replacement.

It also defines the typed async contract `WorkspaceCommandRunner` for running a
single command in a workspace, together with its workspace-owned `CommandResult`
result model. This is the "verify / execute development commands" capability
that complements repository understanding, search, reading, and writing; the
capabilities stay independent and are not aggregated into a generic workspace
object.

The concrete implementation is `SubprocessWorkspaceCommandRunner` (see below).

- The command is an explicit argv `list[str]` (`command[0]` is the executable,
  `command[1:]` its arguments), never a shell command string. There is no
  `str | list[str]` overload and no shell/pipe/redirection/expansion semantics at
  the contract level; tokens such as `|`, `>`, `&&`, or `$HOME` are ordinary argv
  values. This establishes safe exec-style semantics for a future concrete runner.
- A non-empty command (at least one executable argv element) is a documented
  precondition. The async `Protocol` performs no runtime validation; the
  structural invariant is enforced instead by the `CommandResult` model.
- `CommandResult` holds exactly `command`, `exit_code`, `stdout`, and `stderr`.
  Argv order/contents are preserved verbatim (no joining, quoting, trimming, or
  reordering); the model rejects an empty command and empty-string argv entries
  while leaving whitespace-only arguments valid. A non-zero `exit_code` is a
  normal result (never an exception), negative exit codes remain valid, and
  `stdout`/`stderr` may each be empty and are kept separate (never merged).

`SubprocessWorkspaceCommandRunner` is the concrete exec-style runner. It runs
one argv command to completion and returns a `CommandResult`; an ordinary
non-zero (or negative signal) exit is a normal result, never an exception.

- **Not a sandbox.** It executes the executable supplied by its *trusted*
  caller: it is not an allowlist, authorization layer, filesystem/network
  sandbox, container, or command-policy engine, and it can technically run
  dangerous programs (for example `["rm", ...]` or an arbitrary interpreter) if
  a trusted caller requests them. The child inherits LLMForeman's process
  environment and OS permissions. Because of that, this runner must not be
  exposed directly to an untrusted model before a command-authorization/sandbox
  policy exists.
- **No shell, ever.** The command is executed exactly as argv via
  `asyncio.create_subprocess_exec`; argv is never joined, quoted, or interpreted,
  so `&&`, `|`, `>`, `*`, `$HOME`, and `;` remain literal arguments. Argv is
  validated *before* spawning (rejecting an empty command, a whitespace-only
  executable, empty argv entries, and NUL-containing entries) and snapshotted so
  later caller mutation cannot change what ran.
- **Effective Git top-level cwd.** The working directory is the Git working-tree
  top-level resolved via the shared `_git` helper (subdirectory and linked
  worktree entry points resolve correctly); the global process cwd is never
  changed. `stdin` is `DEVNULL` (accidentally interactive commands get EOF), and
  `stdout`/`stderr` are captured as separate, never-merged pipes and decoded as
  UTF-8 with replacement (invalid diagnostic bytes do not fail the command).
- **Bounds and cleanup.** `timeout_seconds` (default 300) bounds the running
  process; `max_output_bytes` (default 4 MiB) is a hard *per-stream* cap enforced
  by explicit concurrent bounded draining of both pipes. Each command runs in its
  own POSIX session (`start_new_session=True`), and timeout, output overflow, and
  cancellation share one cleanup path that terminates the **whole process group**
  (SIGTERM, bounded grace, then SIGKILL) and reaps the direct process. Timeout
  raises `WorkspaceCommandTimeoutError`, output overflow raises
  `WorkspaceCommandExecutionError`, and cancellation re-raises `CancelledError`
  only after cleanup completes. No partial `CommandResult` is returned on failure.

Git subprocesses are invoked without a shell. Invalid caller input raises
`InvalidRepositoryError`; failures inspecting an otherwise valid repository raise
`RepositoryInspectionError`; an explicit file read that cannot be satisfied
(invalid path, untracked path, missing/oversized/non-text/escaping file) raises
`RepositoryFileAccessError`; an explicit file write that cannot be performed
safely (invalid path, non-encodable/oversized content, symlink component,
parent conflict, directory/special target, or existing binary/non-text target)
raises `RepositoryFileWriteError`; and a command that cannot be executed to a
trustworthy completion (invalid argv, unstartable executable, output-limit
overflow, or subprocess/cleanup failure) raises
`WorkspaceCommandExecutionError`, with `WorkspaceCommandTimeoutError` for the
configured execution timeout (all subclasses of `WorkspaceError`).
