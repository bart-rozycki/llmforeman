"""Behavioral tests for the concrete ripgrep-backed repository text searcher.

These build real, local, temporary Git repositories (``git init`` + ``git add``;
no commits unless a linked worktree requires one, no remotes, no network) and
drive the concrete ``RipgrepRepositoryTextSearcher`` end to end against the real
local ``rg`` binary, plus a few narrow unit tests for the private batching and
JSON-parsing seams. Async calls are driven with ``asyncio.run`` to match the
repository convention of not adding a pytest asyncio plugin. No network, model,
or paid calls occur.
"""

import asyncio
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from llmforeman_workspace import (
    InvalidRepositoryError,
    RepositorySearchError,
    RepositorySearchResult,
    RepositoryTextSearcher,
    RipgrepRepositoryTextSearcher,
)
from llmforeman_workspace import ripgrep_searcher as rg_module

# Local ``git`` and ``rg`` executables are the only external requirements; skip
# the real-binary tests cleanly if either is unavailable rather than failing.
_GIT = shutil.which("git")
_RG = shutil.which("rg")
_requires_git = pytest.mark.skipif(_GIT is None, reason="git executable not available")
_requires_rg = pytest.mark.skipif(_RG is None, reason="rg executable not available")
_requires_symlink = pytest.mark.skipif(
    not hasattr(os, "symlink"), reason="symlinks unavailable on this platform"
)


def _git(repo: Path, *args: str) -> None:
    """Run a local, shell-free Git command for fixture setup only."""

    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def _init_repo(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    _git(root, "init", "-q")
    return root


def _add(repo: Path, *relpaths: str) -> None:
    _git(repo, "add", "--", *relpaths)


def _write(repo: Path, relpath: str, content: str) -> Path:
    path = repo / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _search(root: Path, query: str) -> RepositorySearchResult:
    searcher = RipgrepRepositoryTextSearcher()
    return asyncio.run(searcher.search(root, query))


def _pairs(result: RepositorySearchResult) -> list[tuple[str, int, str]]:
    return [(m.path, m.line_number, m.line) for m in result.matches]


# --- Static structural check -------------------------------------------------


def test_concrete_searcher_structurally_satisfies_protocol() -> None:
    # Static structural check under mypy: the concrete searcher satisfies the
    # generic Protocol without ``Any``, casts, or ignores.
    searcher: RepositoryTextSearcher = RipgrepRepositoryTextSearcher()
    assert searcher is not None


# --- Happy path / multiple results / ordering --------------------------------


@_requires_git
@_requires_rg
def test_happy_path_finds_tracked_matches(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    _write(repo, "src/a.py", "class RetryPolicy:\n    pass\n")
    _write(repo, "src/b.py", "policy = RetryPolicy()\n")
    _add(repo, "src/a.py", "src/b.py")

    result = _search(repo, "RetryPolicy")

    assert _pairs(result) == [
        ("src/a.py", 1, "class RetryPolicy:"),
        ("src/b.py", 1, "policy = RetryPolicy()"),
    ]


@_requires_git
@_requires_rg
def test_results_sorted_by_path_then_line_number(tmp_path: Path) -> None:
    # Creation order deliberately differs from the desired (path, line) order.
    repo = _init_repo(tmp_path / "repo")
    _write(repo, "src/zeta.py", "noise\nRetryPolicy here\n")  # match on line 2
    _write(repo, "src/alpha.py", "RetryPolicy top\nRetryPolicy again\n")
    _add(repo, "src/zeta.py", "src/alpha.py")

    result = _search(repo, "RetryPolicy")

    assert [(m.path, m.line_number) for m in result.matches] == [
        ("src/alpha.py", 1),
        ("src/alpha.py", 2),
        ("src/zeta.py", 2),
    ]


@_requires_git
@_requires_rg
def test_multiple_occurrences_on_one_line_yield_one_match(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    _write(repo, "a.py", "RetryPolicy RetryPolicy RetryPolicy\n")
    _add(repo, "a.py")

    result = _search(repo, "RetryPolicy")

    assert _pairs(result) == [("a.py", 1, "RetryPolicy RetryPolicy RetryPolicy")]


@_requires_git
@_requires_rg
def test_no_match_returns_empty_result(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    _write(repo, "a.py", "nothing relevant here\n")
    _add(repo, "a.py")

    result = _search(repo, "RetryPolicy")

    assert result.matches == []


# --- Literal / fixed-string semantics ----------------------------------------


@_requires_git
@_requires_rg
def test_regex_metacharacters_are_literal(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    _write(repo, "literal.txt", "here is (foo|bar).* literally\n")
    # Would match only if the query were treated as a regular expression.
    _write(repo, "regexy.txt", "fooZZZ and barZZZ\n")
    _add(repo, "literal.txt", "regexy.txt")

    result = _search(repo, "(foo|bar).*")

    assert _pairs(result) == [("literal.txt", 1, "here is (foo|bar).* literally")]


@_requires_git
@_requires_rg
def test_query_beginning_with_dash_is_data_not_flag(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    _write(repo, "a.py", 'value = "--example"\n')
    _add(repo, "a.py")

    result = _search(repo, "--example")

    assert _pairs(result) == [("a.py", 1, 'value = "--example"')]


# --- Query validation (no subprocess) ----------------------------------------


@pytest.mark.parametrize("blank", ["", "   ", "\t\n"])
@_requires_git
def test_blank_query_rejected_without_launching_ripgrep(
    tmp_path: Path, blank: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _init_repo(tmp_path / "repo")
    _write(repo, "a.py", "RetryPolicy\n")
    _add(repo, "a.py")

    searcher = RipgrepRepositoryTextSearcher()

    async def _boom(*args: object, **kwargs: object) -> tuple[int, bytes, bytes]:
        raise AssertionError("ripgrep must not be launched for a blank query")

    monkeypatch.setattr(searcher, "_run_ripgrep", _boom)

    with pytest.raises(RepositorySearchError):
        asyncio.run(searcher.search(repo, blank))


@_requires_git
def test_nul_query_rejected_before_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _init_repo(tmp_path / "repo")
    _write(repo, "a.py", "RetryPolicy\n")
    _add(repo, "a.py")

    searcher = RipgrepRepositoryTextSearcher()

    async def _boom(*args: object, **kwargs: object) -> tuple[int, bytes, bytes]:
        raise AssertionError("ripgrep must not be launched for a NUL query")

    monkeypatch.setattr(searcher, "_run_ripgrep", _boom)

    with pytest.raises(RepositorySearchError):
        asyncio.run(searcher.search(repo, "Retry\x00Policy"))


# --- Working tree vs index ---------------------------------------------------


@_requires_git
@_requires_rg
def test_unstaged_working_tree_modification_is_searched(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    _write(repo, "a.py", "original = 1\n")
    _add(repo, "a.py")
    # Introduce the query only in the unstaged working-tree version.
    _write(repo, "a.py", "original = 1\nDISTINCT_UNSTAGED_TOKEN = 2\n")

    result = _search(repo, "DISTINCT_UNSTAGED_TOKEN")

    assert _pairs(result) == [("a.py", 2, "DISTINCT_UNSTAGED_TOKEN = 2")]


# --- Tracked-only privacy boundary -------------------------------------------


@_requires_git
@_requires_rg
def test_untracked_file_is_never_searched(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    _write(repo, "tracked.txt", "ordinary content\n")
    _add(repo, "tracked.txt")
    secret = "UNTRACKED_SECRET_TOKEN"
    _write(repo, "secret.txt", secret + "\n")  # never added

    result = _search(repo, secret)

    assert result.matches == []


@_requires_git
@_requires_rg
def test_ignored_untracked_file_is_never_searched(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    _write(repo, ".gitignore", ".env\n")
    _add(repo, ".gitignore")
    _write(repo, ".env", "IGNORED_SECRET_TOKEN=1\n")  # ignored + untracked

    result = _search(repo, "IGNORED_SECRET_TOKEN")

    assert result.matches == []


@_requires_git
@_requires_rg
def test_force_added_ignored_file_is_searchable(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    _write(repo, ".gitignore", "config.txt\n")
    _write(repo, "config.txt", "FORCED_TRACKED_TOKEN\n")
    _git(repo, "add", "--force", "--", "config.txt", ".gitignore")

    result = _search(repo, "FORCED_TRACKED_TOKEN")

    assert _pairs(result) == [("config.txt", 1, "FORCED_TRACKED_TOKEN")]


@_requires_git
@_requires_rg
def test_hidden_tracked_file_is_searchable(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    _write(repo, ".hidden-source", "HIDDEN_TRACKED_TOKEN\n")
    _add(repo, ".hidden-source")

    result = _search(repo, "HIDDEN_TRACKED_TOKEN")

    assert _pairs(result) == [(".hidden-source", 1, "HIDDEN_TRACKED_TOKEN")]


@_requires_git
@_requires_rg
def test_rgignore_does_not_hide_tracked_candidate(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    _write(repo, "source.txt", "RGIGNORE_TOKEN present\n")
    # A local .rgignore would make ripgrep's own traversal skip source.txt; the
    # explicit-operand + --no-ignore invocation must neutralize that filtering.
    _write(repo, ".rgignore", "source.txt\n")
    _add(repo, "source.txt", ".rgignore")

    result = _search(repo, "RGIGNORE_TOKEN")

    assert ("source.txt", 1, "RGIGNORE_TOKEN present") in _pairs(result)


@_requires_git
@_requires_rg
def test_ripgrep_user_config_does_not_change_semantics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _init_repo(tmp_path / "repo")
    _write(repo, "a.py", "RetryPolicy exact case\n")
    _add(repo, "a.py")

    # A user config enabling case-insensitivity would let a wrong-case query
    # match; ``--no-config`` must make it irrelevant.
    config = tmp_path / "rg.conf"
    config.write_text("--ignore-case\n", encoding="utf-8")
    monkeypatch.setenv("RIPGREP_CONFIG_PATH", str(config))

    # Wrong case: with config honored this would match; it must not.
    assert _search(repo, "retrypolicy").matches == []
    # Exact case still matches, proving search still works.
    assert _pairs(_search(repo, "RetryPolicy")) == [
        ("a.py", 1, "RetryPolicy exact case")
    ]


# --- Repository entry-point semantics ----------------------------------------


@_requires_git
@_requires_rg
def test_subdirectory_entry_point_uses_top_level(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    _write(repo, "README.md", "RetryPolicy at root\n")
    _write(repo, "src/a.py", "RetryPolicy nested\n")
    _add(repo, "README.md", "src/a.py")

    # Entry point is a subdirectory; results remain top-level repo-relative and
    # include both the root-level and nested tracked files.
    result = _search(repo / "src", "RetryPolicy")

    assert _pairs(result) == [
        ("README.md", 1, "RetryPolicy at root"),
        ("src/a.py", 1, "RetryPolicy nested"),
    ]


# --- Missing / non-regular tracked working-tree objects ----------------------


@_requires_git
@_requires_rg
def test_tracked_but_deleted_file_contributes_no_match(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    _write(repo, "gone.txt", "RetryPolicy\n")
    _write(repo, "kept.txt", "RetryPolicy\n")
    _add(repo, "gone.txt", "kept.txt")
    # Delete from the working tree only; it remains tracked in the index.
    (repo / "gone.txt").unlink()

    result = _search(repo, "RetryPolicy")

    assert _pairs(result) == [("kept.txt", 1, "RetryPolicy")]


# --- Symlink safety ----------------------------------------------------------


@_requires_git
@_requires_rg
@_requires_symlink
def test_external_symlink_target_is_not_searched(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    external = "EXTERNAL_SYMLINK_SECRET"
    (outside / "secret").write_text(external + "\n", encoding="utf-8")

    repo = _init_repo(tmp_path / "repo")
    link = repo / "link.txt"
    try:
        link.symlink_to(outside / "secret")
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation not permitted")
    _add(repo, "link.txt")

    result = _search(repo, external)

    assert result.matches == []
    # The external target's content/path must never appear anywhere.
    assert all(external not in m.line for m in result.matches)


@_requires_git
@_requires_rg
@_requires_symlink
def test_safe_internal_symlink_is_searched_under_logical_path(
    tmp_path: Path,
) -> None:
    repo = _init_repo(tmp_path / "repo")
    _write(repo, "docs/real.txt", "INTERNAL_SYMLINK_TOKEN\n")
    link = repo / "link.txt"
    try:
        link.symlink_to(repo / "docs" / "real.txt")
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation not permitted")
    _add(repo, "docs/real.txt", "link.txt")

    result = _search(repo, "INTERNAL_SYMLINK_TOKEN")

    # Both the real tracked file and the safe internal symlink are tracked; the
    # symlink match is reported under its logical repo-relative path, never the
    # resolved target's absolute path.
    pairs = _pairs(result)
    assert ("docs/real.txt", 1, "INTERNAL_SYMLINK_TOKEN") in pairs
    assert ("link.txt", 1, "INTERNAL_SYMLINK_TOKEN") in pairs
    assert all(str(repo) not in m.path for m in result.matches)


# --- Unusual names -----------------------------------------------------------


@_requires_git
@_requires_rg
def test_path_with_spaces(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    _write(repo, "docs/file with spaces.txt", "RetryPolicy spaced\n")
    _add(repo, "docs/file with spaces.txt")

    result = _search(repo, "RetryPolicy")

    assert _pairs(result) == [
        ("docs/file with spaces.txt", 1, "RetryPolicy spaced")
    ]


@_requires_git
@_requires_rg
def test_pathspec_like_and_dash_leading_names(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    bracket = "weird/[a-z]file.txt"
    _write(repo, bracket, "RetryPolicy bracket\n")
    _write(repo, "-dash.txt", "RetryPolicy dash\n")
    _add(repo, bracket, "-dash.txt")

    result = _search(repo, "RetryPolicy")

    assert _pairs(result) == [
        ("-dash.txt", 1, "RetryPolicy dash"),
        ("weird/[a-z]file.txt", 1, "RetryPolicy bracket"),
    ]


# --- Line normalization ------------------------------------------------------


@_requires_git
@_requires_rg
def test_line_whitespace_is_preserved(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    indented = "        RetryPolicy  \t"
    _write(repo, "a.py", indented + "\n")
    _add(repo, "a.py")

    result = _search(repo, "RetryPolicy")

    assert _pairs(result) == [("a.py", 1, indented)]


@_requires_git
@_requires_rg
def test_crlf_terminator_removed_without_other_changes(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    # Write bytes directly so no editor/Git normalization alters the CRLF.
    (repo / "a.py").write_bytes(b"before RetryPolicy after\r\n")
    _add(repo, "a.py")

    result = _search(repo, "RetryPolicy")

    assert _pairs(result) == [("a.py", 1, "before RetryPolicy after")]


# --- Determinism -------------------------------------------------------------


@_requires_git
@_requires_rg
def test_repeated_search_is_deterministic(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    _write(repo, "src/a.py", "RetryPolicy one\n")
    _write(repo, "src/b.py", "RetryPolicy two\nRetryPolicy three\n")
    _add(repo, "src/a.py", "src/b.py")

    first = _search(repo, "RetryPolicy")
    second = _search(repo, "RetryPolicy")

    assert first.model_dump() == second.model_dump()


# --- Empty searchable repository --------------------------------------------


@_requires_git
def test_empty_candidate_set_returns_empty_without_launching_rg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _init_repo(tmp_path / "repo")  # no tracked files at all

    searcher = RipgrepRepositoryTextSearcher()

    async def _boom(*args: object, **kwargs: object) -> tuple[int, bytes, bytes]:
        raise AssertionError("ripgrep must not run with no candidates")

    monkeypatch.setattr(searcher, "_run_ripgrep", _boom)

    result = asyncio.run(searcher.search(repo, "RetryPolicy"))
    assert result.matches == []


# --- Repository error semantics (kept distinct from search errors) -----------


@_requires_git
def test_missing_root_is_invalid_repository(tmp_path: Path) -> None:
    with pytest.raises(InvalidRepositoryError):
        _search(tmp_path / "does-not-exist", "RetryPolicy")


@_requires_git
def test_non_git_directory_is_invalid_repository(tmp_path: Path) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    with pytest.raises(InvalidRepositoryError):
        _search(plain, "RetryPolicy")


# --- Batching (cross-batch merge, no functional file limit) ------------------


@_requires_git
@_requires_rg
def test_batching_searches_every_file_and_merges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Grouped names so that, after the deterministic candidate sort, whole
    # batches are either all-matching (``m*``) or all-non-matching (``z*``);
    # this guarantees both a matching batch (exit 0) and a no-match batch
    # (exit 1) contribute to a single successful merged result.
    repo = _init_repo(tmp_path / "repo")
    matching = [f"m{i:02d}.txt" for i in range(6)]
    non_matching = [f"z{i:02d}.txt" for i in range(6)]
    for name in matching:
        _write(repo, name, "RetryPolicy\n")
    for name in non_matching:
        _write(repo, name, "nothing here\n")
    _add(repo, *matching, *non_matching)

    # Force a tiny argv budget so the small files require several batches.
    monkeypatch.setattr(rg_module, "_MAX_ARG_BYTES", 110)

    result = _search(repo, "RetryPolicy")

    assert [m.path for m in result.matches] == sorted(matching)
    assert all(m.line == "RetryPolicy" for m in result.matches)


def test_batch_operands_partitions_all_paths_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = [f"dir/file{i:03d}.py" for i in range(50)]
    # A small budget relative to the encoded paths forces multiple batches.
    monkeypatch.setattr(rg_module, "_MAX_ARG_BYTES", 300)

    batches = rg_module._batch_operands(list(paths), "query")

    assert len(batches) > 1
    flattened = [p for batch in batches for p in batch]
    assert flattened == paths  # every path once, order preserved


def test_batch_operands_single_path_too_large_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A budget smaller than even the fixed prefix means no path can ever fit.
    monkeypatch.setattr(rg_module, "_MAX_ARG_BYTES", 8)

    with pytest.raises(RepositorySearchError):
        rg_module._batch_operands(["some/file.py"], "query")


# --- Candidate classification seam (no directory operand reaches rg) ---------


def test_directory_candidate_is_excluded(tmp_path: Path) -> None:
    root = tmp_path
    (root / "a_dir").mkdir()
    (root / "a_file.txt").write_text("x\n", encoding="utf-8")

    candidates = rg_module._select_searchable_candidates(
        root, ["a_dir", "a_file.txt", "missing.txt"]
    )

    # A directory (e.g. a submodule/gitlink working-tree object) and a
    # tracked-but-deleted path are excluded; only the regular file remains.
    assert candidates == ["a_file.txt"]


# --- ripgrep process / JSON failure seams ------------------------------------


@_requires_git
def test_missing_ripgrep_becomes_search_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _init_repo(tmp_path / "repo")
    _write(repo, "a.py", "RetryPolicy\n")
    _add(repo, "a.py")

    # Point the adapter at a non-existent executable so launching fails.
    monkeypatch.setattr(rg_module, "_RIPGREP_EXECUTABLE", "rg-does-not-exist-xyz")

    with pytest.raises(RepositorySearchError):
        _search(repo, "RetryPolicy")


@_requires_git
def test_ripgrep_execution_failure_becomes_search_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _init_repo(tmp_path / "repo")
    _write(repo, "a.py", "RetryPolicy\n")
    _add(repo, "a.py")

    searcher = RipgrepRepositoryTextSearcher()

    async def _fail(*args: object, **kwargs: object) -> tuple[int, bytes, bytes]:
        return 2, b"", b"ripgrep: some execution error"

    monkeypatch.setattr(searcher, "_run_ripgrep", _fail)

    with pytest.raises(RepositorySearchError):
        asyncio.run(searcher.search(repo, "RetryPolicy"))


def test_parser_rejects_malformed_json() -> None:
    with pytest.raises(RepositorySearchError):
        rg_module._parse_matches(b"this is not json\n", frozenset({"a.py"}))


def test_parser_rejects_path_outside_approved_batch() -> None:
    event = (
        b'{"type":"match","data":{"path":{"text":"other.py"},'
        b'"lines":{"text":"RetryPolicy\\n"},"line_number":1}}\n'
    )
    with pytest.raises(RepositorySearchError):
        rg_module._parse_matches(event, frozenset({"approved.py"}))


def test_parser_ignores_non_match_events() -> None:
    stdout = (
        b'{"type":"begin","data":{"path":{"text":"a.py"}}}\n'
        b'{"type":"match","data":{"path":{"text":"a.py"},'
        b'"lines":{"text":"RetryPolicy\\n"},"line_number":3}}\n'
        b'{"type":"end","data":{"path":{"text":"a.py"}}}\n'
        b'{"data":{"stats":{}},"type":"summary"}\n'
    )
    matches = rg_module._parse_matches(stdout, frozenset({"a.py"}))
    assert [(m.path, m.line_number, m.line) for m in matches] == [
        ("a.py", 3, "RetryPolicy")
    ]


def test_parser_omits_byte_only_line_payload() -> None:
    # A non-text (base64 ``bytes``) line payload is omitted, not decoded.
    event = (
        b'{"type":"match","data":{"path":{"text":"a.py"},'
        b'"lines":{"bytes":"UmV0cnk="},"line_number":1}}\n'
    )
    matches = rg_module._parse_matches(event, frozenset({"a.py"}))
    assert matches == []


def test_parser_rejects_invalid_line_number() -> None:
    event = (
        b'{"type":"match","data":{"path":{"text":"a.py"},'
        b'"lines":{"text":"RetryPolicy\\n"},"line_number":0}}\n'
    )
    with pytest.raises(RepositorySearchError):
        rg_module._parse_matches(event, frozenset({"a.py"}))
