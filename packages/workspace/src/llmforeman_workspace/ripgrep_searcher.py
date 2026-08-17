"""Ripgrep-backed concrete :class:`RepositoryTextSearcher` for local workspaces.

``RipgrepRepositoryTextSearcher`` answers "where does this literal text occur?"
across a local repository. It is the third repository-exploration capability
alongside ``GitRepositoryContextLoader`` (initial snapshot) and
``GitRepositoryFileReader`` (read one explicit file). It uses ripgrep as the
search engine but Git tracked membership as the search boundary; it does not
read files itself, glob, rank, or interact with any model, provider, or runtime.

Security model (this is a privacy-sensitive local workspace boundary):

* **Git owns the search set, not ripgrep's ignore rules.** Git determines the
  effective working-tree top-level (via the shared ``_git`` primitives) and the
  exact tracked/index paths; ``.git`` is never inspected directly, so linked
  worktrees work. ripgrep receives *only* explicit approved tracked file
  operands, never a directory or ``.`` — so an untracked or ignored-but-present
  ``secret.txt`` / ``.env`` is completely outside the search set. Because
  operands are explicit, ripgrep's own ``.gitignore``/``.ignore``/``.rgignore``
  and hidden-file filtering could otherwise silently drop an approved tracked
  file, so ``--no-ignore`` and ``--hidden`` neutralize that filtering without
  widening scope. Force-added ignored files and hidden tracked files remain
  searchable precisely because Git membership is authoritative.
* **Only safe current working-tree files reach ripgrep.** The tracked set may
  contain paths whose current working-tree object is not an ordinary file
  (deleted, submodule/gitlink directories, FIFOs, sockets, devices, broken or
  escaping symlinks). Each candidate is classified with filesystem metadata
  only (no content is read to build candidates): the resolved target must stay
  within the canonical root (a symlink may not escape it) and must be a regular
  file. Unsafe/unavailable candidates are skipped, never failing the whole
  search; passing a directory operand — which could make ripgrep recurse into
  never-tracked nested content — is thereby impossible.
* **Deterministic regardless of environment.** ``--no-config`` neutralizes any
  developer ``RIPGREP_CONFIG_PATH``; the query is passed literally via
  ``--fixed-strings`` and ``--regexp=<query>`` so regex metacharacters have no
  meaning and a query beginning with ``-`` is never mistaken for a flag; ``--``
  separates operands so a filename beginning with ``-`` stays a path. There is
  no shell: the argv is executed directly.
* **Output is validated at the boundary.** ripgrep ``--json`` output is parsed
  as JSON Lines; only ``match`` events become results, one result per matching
  line. Every reported path must be one of the exact approved repo-relative
  operands for that batch (otherwise the search fails); non-text (byte-only)
  match lines are omitted rather than lossily decoded; and no absolute local
  path can appear in a result because operands are repo-relative and ripgrep is
  run with ``cwd`` at the effective top-level.

Search reads the *current working tree* (unstaged modifications are visible),
never HEAD/index blobs. Finding nothing — including ripgrep exit code ``1`` — is
a successful empty result. Any failure that prevents a trustworthy *complete*
search (missing/failed ``rg``, malformed output, an unexpected path, a failed
batch) raises :class:`RepositorySearchError`; repository/Git faults keep their
existing :class:`InvalidRepositoryError` / :class:`RepositoryInspectionError`
identity and are never rewrapped. There are no retries: a local, deterministic
subprocess failure stays visible.
"""

import asyncio
import json
import os
from pathlib import Path
from typing import Final

from llmforeman_workspace._git import (
    list_tracked_paths,
    resolve_worktree_top_level,
    sanitized_git_detail,
)
from llmforeman_workspace.errors import RepositorySearchError
from llmforeman_workspace.search import RepositorySearchMatch, RepositorySearchResult

__all__ = [
    "RipgrepRepositoryTextSearcher",
]

# The local ripgrep executable this adapter depends on. A module-level constant
# (rather than a public constructor option) keeps the single-execution-model
# contract while providing a narrow seam for tests that simulate a missing rg.
_RIPGREP_EXECUTABLE: Final[str] = "rg"

# Conservative, portable command-line byte budget for a single ``rg`` invocation
# (executable + fixed flags + query + explicit path operands). It bounds argv
# size for transport safety only; it is deliberately far below typical platform
# limits (POSIX ARG_MAX is commonly >= 256 KiB, usually >= 1 MiB) to leave room
# for the environment block. It is NOT a product search limit: every eligible
# tracked candidate is still searched, across as many sequential batches as
# needed. A module-level constant provides a narrow seam for batching tests.
_MAX_ARG_BYTES: Final[int] = 128 * 1024

# Fixed ripgrep flags shared by every invocation. See the module docstring for
# why each is required. Order is stable for deterministic argv accounting.
_RIPGREP_FLAGS: Final[tuple[str, ...]] = (
    "--no-config",
    "--json",
    "--fixed-strings",
    "--no-ignore",
    "--hidden",
)

# ripgrep search exit codes: 0 = matches found, 1 = completed with no matches
# (NOT an error), any other status = execution/search failure.
_RG_EXIT_MATCHES: Final[int] = 0
_RG_EXIT_NO_MATCHES: Final[int] = 1


class RipgrepRepositoryTextSearcher:
    """Search Git-tracked working-tree files for a literal ``query`` via ripgrep.

    Satisfies the async :class:`RepositoryTextSearcher` protocol. It is
    stateless and holds no configuration, ignore rules, result limit, caching,
    or lifecycle hooks. The tracked-only, literal, single-execution-model policy
    is deliberate concrete implementation policy, not a property of the generic
    protocol.
    """

    async def search(
        self,
        repository_root: Path,
        query: str,
    ) -> RepositorySearchResult:
        """Search the tracked working tree for the literal text ``query``.

        ``repository_root`` is any local path inside the target working tree;
        Git resolves the canonical top-level used as the effective root. All
        returned paths are repository-relative to that top-level.

        Raises :class:`~llmforeman_workspace.errors.InvalidRepositoryError` for
        an invalid repository entry,
        :class:`~llmforeman_workspace.errors.RepositoryInspectionError` for a
        Git inspection failure, and
        :class:`~llmforeman_workspace.errors.RepositorySearchError` for any
        search-layer failure (blank/NUL query, ripgrep launch/execution failure,
        malformed output, or a path outside the approved candidate set). An
        empty result is a valid, non-error outcome.
        """

        # Query precondition first: reject before touching Git or launching any
        # subprocess. Trimming only decides blankness; the value is never
        # rewritten and is passed to ripgrep exactly as given.
        self._validate_query(query)

        # Repository validation / effective-root resolution (may raise
        # InvalidRepositoryError / RepositoryInspectionError). These are never
        # rewrapped into RepositorySearchError.
        effective_root = await resolve_worktree_top_level(repository_root)

        # Git tracked/index set is the authoritative candidate universe.
        tracked = await list_tracked_paths(effective_root)

        # Classify to safe, current working-tree file operands using filesystem
        # metadata only (no file contents are read). Runs on a worker thread so
        # the stat/resolve calls never block the event loop.
        candidates = await asyncio.to_thread(
            _select_searchable_candidates, effective_root, tracked
        )

        # An empty candidate set is a successful empty search: ripgrep is never
        # invoked without explicit operands (which could search stdin/cwd).
        if not candidates:
            return RepositorySearchResult(matches=[])

        # Deterministic ordering of candidates -> deterministic batches.
        candidates.sort()
        batches = _batch_operands(candidates, query)

        matches: list[RepositorySearchMatch] = []
        for batch in batches:
            returncode, stdout, stderr = await self._run_ripgrep(
                effective_root, query, batch
            )
            if returncode == _RG_EXIT_NO_MATCHES:
                # A no-match batch contributes nothing; this is not a failure.
                continue
            if returncode != _RG_EXIT_MATCHES:
                # Any later-batch failure fails the whole search: a successful
                # result must represent the complete search over all candidates.
                raise RepositorySearchError(
                    "ripgrep search failed" + sanitized_git_detail(stderr)
                )
            matches.extend(_parse_matches(stdout, frozenset(batch)))

        # Result determinism is owned here, not by any ripgrep execution mode.
        matches.sort(key=lambda match: (match.path, match.line_number))
        return RepositorySearchResult(matches=matches)

    @staticmethod
    def _validate_query(query: str) -> None:
        """Reject a query that cannot be searched meaningfully or safely.

        A blank/whitespace-only query has no meaningful literal content; a NUL
        cannot be represented in process argv. Both are refused before any Git
        or subprocess work. The query is otherwise preserved exactly.
        """

        if not query.strip():
            raise RepositorySearchError(
                "query must contain non-whitespace text"
            )
        if "\x00" in query:
            raise RepositorySearchError(
                "query must not contain NUL characters"
            )

    async def _run_ripgrep(
        self,
        effective_root: Path,
        query: str,
        operands: list[str],
    ) -> tuple[int, bytes, bytes]:
        """Run one ripgrep invocation over explicit repo-relative ``operands``.

        Executed without a shell via an argv API, with ``cwd`` at the effective
        top-level so ripgrep emits repo-relative paths. Returns the exit code
        and captured stdout/stderr; a failure to *launch* ``rg`` becomes
        :class:`RepositorySearchError` with preserved causality. This is a
        narrow private execution seam, not a generic subprocess framework.
        """

        argv = _build_argv(_RIPGREP_EXECUTABLE, query, operands)
        try:
            process = await asyncio.create_subprocess_exec(
                *argv,
                cwd=str(effective_root),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as original:
            raise RepositorySearchError(
                "unable to launch the ripgrep executable"
            ) from original

        stdout, stderr = await process.communicate()
        returncode = process.returncode
        # ``communicate`` on an awaited, finished process yields a concrete code.
        assert returncode is not None
        return returncode, stdout, stderr


def _build_argv(executable: str, query: str, operands: list[str]) -> list[str]:
    """Build the shell-free ripgrep argv for ``operands``.

    ``--regexp=<query>`` keeps a query beginning with ``-`` from being parsed as
    a flag; ``--`` separates operands so a filename beginning with ``-`` stays a
    path. The query is literal (``--fixed-strings``); no shell quoting or
    interpolation occurs.
    """

    return [
        executable,
        *_RIPGREP_FLAGS,
        f"--regexp={query}",
        "--",
        *operands,
    ]


def _select_searchable_candidates(
    effective_root: Path, tracked: list[str]
) -> list[str]:
    """Return the repo-relative tracked paths that are safe file operands.

    Uses filesystem metadata/resolution only (never reads file contents). A
    path is eligible only when its resolved working-tree target stays within the
    canonical ``effective_root`` (containment: escaping symlinks are excluded)
    and is a regular file. This excludes tracked-but-deleted paths, directories,
    submodule/gitlink directories, FIFOs, sockets, devices, and broken or
    escaping symlinks — so no directory operand can ever reach ripgrep and cause
    recursion into never-tracked content. Unavailable candidates are skipped,
    not fatal: the search covers the currently readable/safe tracked files.
    """

    candidates: list[str] = []
    for path in tracked:
        candidate = effective_root / path
        try:
            # ``resolve`` without requiring existence lets broken/dangling
            # symlinks and missing files fall through to the checks below.
            resolved = candidate.resolve()
            if not resolved.is_relative_to(effective_root):
                # A symlink (or path) escaping the repository root is excluded;
                # its external target must never be searched.
                continue
            if not resolved.is_file():
                # Directories, submodules, deleted paths, special files, and
                # broken symlinks are not searchable file operands.
                continue
        except OSError:
            # An individual metadata failure skips only that candidate; it must
            # not abort the whole repository search.
            continue
        candidates.append(path)
    return candidates


def _batch_operands(paths: list[str], query: str) -> list[list[str]]:
    """Divide ``paths`` into batches bounded by ``_MAX_ARG_BYTES``.

    Batching is subprocess transport safety only, never a product result limit:
    every path ends up in exactly one batch and all batches are searched. The
    fixed argv prefix (executable + flags + query + ``--``) is charged once per
    batch; each operand is charged its encoded byte length plus one for the argv
    separator. A single path that cannot fit even alone fails the search rather
    than being silently omitted.
    """

    base_cost = _argv_byte_cost(_build_argv(_RIPGREP_EXECUTABLE, query, []))

    batches: list[list[str]] = []
    current: list[str] = []
    current_cost = base_cost
    for path in paths:
        path_cost = _operand_byte_cost(path)
        if base_cost + path_cost > _MAX_ARG_BYTES:
            # Even alone this path overflows the budget; refuse rather than
            # return an incomplete result while claiming success.
            raise RepositorySearchError(
                "a tracked path is too large to search within the command-line "
                "argument budget"
            )
        if current and current_cost + path_cost > _MAX_ARG_BYTES:
            batches.append(current)
            current = []
            current_cost = base_cost
        current.append(path)
        current_cost += path_cost
    if current:
        batches.append(current)
    return batches


def _argv_byte_cost(argv: list[str]) -> int:
    """Return a conservative encoded-byte cost for a full argv vector."""

    return sum(_operand_byte_cost(arg) for arg in argv)


def _operand_byte_cost(arg: str) -> int:
    """Return the encoded byte cost of one argv element (+1 for its separator)."""

    return len(os.fsencode(arg)) + 1


def _parse_matches(
    stdout: bytes, approved: frozenset[str]
) -> list[RepositorySearchMatch]:
    """Parse ripgrep JSON Lines ``stdout`` into validated line matches.

    Every non-empty output line must be valid ripgrep JSON. Only ``match``
    events become results (``begin``/``end``/``summary`` and other event types
    are ignored after being parsed as valid JSON). One matching line yields one
    :class:`RepositorySearchMatch`; ``submatches`` are not expanded into
    multiple results. Each reported path must be one of the exact ``approved``
    repo-relative operands; a byte-only (non-text) line payload is omitted
    rather than lossily decoded; only the final line terminator is stripped.
    Any structurally unusable ``match`` event fails the search rather than
    yielding partial results.
    """

    matches: list[RepositorySearchMatch] = []
    for raw_line in stdout.splitlines():
        if not raw_line.strip():
            continue
        try:
            event = json.loads(raw_line)
        except json.JSONDecodeError as original:
            raise RepositorySearchError(
                "ripgrep produced malformed JSON output"
            ) from original
        if not isinstance(event, dict):
            raise RepositorySearchError(
                "ripgrep produced an unexpected JSON event"
            )
        if event.get("type") != "match":
            # Known non-match events (begin/end/summary/context) carry no
            # workspace match; ordering is not relied upon for the final result.
            continue

        data = event.get("data")
        if not isinstance(data, dict):
            raise RepositorySearchError(
                "ripgrep match event had an unexpected structure"
            )

        path_text = _match_path_text(data)
        if path_text not in approved:
            # Defense in depth: never trust an engine-reported path outside the
            # exact approved batch, even though operands were explicit.
            raise RepositorySearchError(
                "ripgrep reported a path outside the approved candidate set"
            )

        line_text = _match_line_text(data)
        if line_text is None:
            # A byte-only line payload is non-text search output for this v0.1
            # text contract; omit it rather than guessing an encoding.
            continue

        line_number = data.get("line_number")
        if (
            not isinstance(line_number, int)
            or isinstance(line_number, bool)
            or line_number < 1
        ):
            raise RepositorySearchError(
                "ripgrep match event lacked a valid line number"
            )

        matches.append(
            RepositorySearchMatch(
                path=path_text,
                line_number=line_number,
                line=_strip_line_terminator(line_text),
            )
        )
    return matches


def _match_path_text(data: dict[str, object]) -> str:
    """Return the textual match path, or fail if it cannot be mapped safely.

    A path represented only as bytes cannot be mapped back to one of the exact
    approved tracked paths, so it fails rather than being guessed.
    """

    path_obj = data.get("path")
    if not isinstance(path_obj, dict):
        raise RepositorySearchError(
            "ripgrep match event had an unexpected path structure"
        )
    text = path_obj.get("text")
    if not isinstance(text, str):
        raise RepositorySearchError(
            "ripgrep reported a match path that could not be mapped to a "
            "tracked path"
        )
    return text


def _match_line_text(data: dict[str, object]) -> str | None:
    """Return the textual matching line, or ``None`` for a byte-only payload.

    A missing/invalid ``lines`` structure is a malformed event and fails; a
    byte-only payload (``lines.bytes`` without ``lines.text``) is a recognized
    non-text line and returns ``None`` to signal a safe omission.
    """

    lines_obj = data.get("lines")
    if not isinstance(lines_obj, dict):
        raise RepositorySearchError(
            "ripgrep match event had an unexpected line structure"
        )
    text = lines_obj.get("text")
    if text is None:
        # Recognized non-text payload (e.g. base64 ``bytes``): safe omission.
        return None
    if not isinstance(text, str):
        raise RepositorySearchError(
            "ripgrep match event had an unexpected line payload"
        )
    return text


def _strip_line_terminator(line: str) -> str:
    """Remove at most the final line terminator, preserving all other content.

    Handles ``\\r\\n``, ``\\n``, and ``\\r``; leading/trailing spaces and tabs
    and all interior formatting are preserved (never ``strip``/``rstrip``). A
    line without a terminator is returned unchanged.
    """

    if line.endswith("\r\n"):
        return line[:-2]
    if line.endswith("\n") or line.endswith("\r"):
        return line[:-1]
    return line
