"""Behavioral tests for ``GitRepositoryContextLoader``.

These tests build real, local, temporary Git repositories (``git init`` +
``git add``; no commits, no remotes, no network) and exercise the concrete
loader end to end. Async ``load`` calls are driven with ``asyncio.run`` to match
the repository convention of not adding a pytest asyncio plugin.
"""

import asyncio
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

import pytest

from llmforeman_core import RepositoryContext
from llmforeman_workspace import (
    GitRepositoryContextLoader,
    InvalidRepositoryError,
    RepositoryInspectionError,
    WorkspaceError,
)

# A local ``git`` executable is the only external requirement. Skip the whole
# module cleanly if it is unavailable rather than failing the suite.
_GIT = shutil.which("git")
pytestmark = pytest.mark.skipif(_GIT is None, reason="git executable not available")


def _git(repo: Path, *args: str) -> None:
    """Run a local, shell-free Git command for fixture setup only."""

    subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
    )


def _init_repo(root: Path) -> Path:
    """Initialize an empty Git working tree at ``root`` and return it."""

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


def _load(root: Path, **kwargs: int) -> RepositoryContext:
    loader = GitRepositoryContextLoader(**kwargs)
    return asyncio.run(loader.load(root))


def _paths(context: RepositoryContext) -> Sequence[str]:
    return [f.path for f in context.files]


# --- Constructor validation --------------------------------------------------


@pytest.mark.parametrize("bad", [0, -1, -1024, True, False])
def test_rejects_non_positive_or_bool_limit(bad: object) -> None:
    with pytest.raises(ValueError):
        GitRepositoryContextLoader(max_seed_file_bytes=bad)  # type: ignore[arg-type]


# --- Valid repository / tracked tree vs. seed contents -----------------------


def test_valid_repository_tree_and_seed_selection(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    _write(repo, "README.md", "# Title\n")
    _write(repo, "pyproject.toml", "[project]\nname='x'\n")
    _write(repo, "src/example.py", "print('hi')\n")
    _add(repo, "README.md", "pyproject.toml", "src/example.py")

    context = _load(repo)

    assert context.file_tree == "README.md\npyproject.toml\nsrc/example.py"
    assert _paths(context) == ["README.md", "pyproject.toml"]
    # Arbitrary source code is not seed content.
    assert "src/example.py" not in _paths(context)


def test_full_tracked_tree_includes_nested_non_seed(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    _write(repo, "deeply/nested/module.py", "x = 1\n")
    _add(repo, "deeply/nested/module.py")

    context = _load(repo)

    assert context.file_tree == "deeply/nested/module.py"
    assert context.files == []


def test_seed_order_is_stable_not_add_order(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    # Create/add in an order intentionally different from declared seed order.
    _write(repo, "package.json", "{}\n")
    _write(repo, "README.md", "readme\n")
    _write(repo, "AGENTS.md", "agents\n")
    _add(repo, "package.json", "README.md", "AGENTS.md")

    context = _load(repo)

    # Declared order: AGENTS.md, CLAUDE.md, README.md, ..., package.json, ...
    assert _paths(context) == ["AGENTS.md", "README.md", "package.json"]


# --- Untracked / ignored privacy --------------------------------------------


def test_untracked_file_excluded(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    _write(repo, "tracked.txt", "ok\n")
    _add(repo, "tracked.txt")
    _write(repo, "secret.txt", "TOP SECRET\n")  # never added

    context = _load(repo)

    assert context.file_tree == "tracked.txt"
    assert "secret.txt" not in context.file_tree
    assert _paths(context) == []


def test_gitignored_paths_excluded(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    _write(repo, ".gitignore", ".env\nnode_modules/\n")
    _add(repo, ".gitignore")
    _write(repo, ".env", "SECRET=1\n")
    _write(repo, "node_modules/foo", "junk\n")

    context = _load(repo)

    assert context.file_tree == ".gitignore"
    assert ".env" not in context.file_tree
    assert "node_modules" not in context.file_tree


def test_tracked_file_matching_ignore_is_not_hidden(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    _write(repo, ".gitignore", "config.txt\n")
    _write(repo, "config.txt", "tracked despite ignore\n")
    # Force-add so it is tracked even though a rule matches it.
    _git(repo, "add", "--force", "--", "config.txt", ".gitignore")

    context = _load(repo)

    assert "config.txt" in context.file_tree.splitlines()


def test_untracked_seed_name_not_read(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    _write(repo, "keep.txt", "keep\n")
    _add(repo, "keep.txt")
    # A seed-allowlisted name that exists but is NOT tracked.
    _write(repo, "README.md", "should not be read\n")

    context = _load(repo)

    assert "README.md" not in context.file_tree
    assert _paths(context) == []


# --- Invalid roots -----------------------------------------------------------


def test_missing_root_rejected(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist"
    with pytest.raises(InvalidRepositoryError):
        _load(missing)


def test_file_as_root_rejected(tmp_path: Path) -> None:
    file_path = tmp_path / "a-file"
    file_path.write_text("x\n", encoding="utf-8")
    with pytest.raises(InvalidRepositoryError):
        _load(file_path)


def test_non_git_directory_rejected(tmp_path: Path) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    with pytest.raises(InvalidRepositoryError):
        _load(plain)


def test_bare_repository_rejected(tmp_path: Path) -> None:
    bare = tmp_path / "bare.git"
    bare.mkdir()
    _git(bare, "init", "-q", "--bare")
    with pytest.raises(InvalidRepositoryError):
        _load(bare)


def test_invalid_repository_is_workspace_error(tmp_path: Path) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    with pytest.raises(WorkspaceError):
        _load(plain)


# --- Empty repository --------------------------------------------------------


def test_empty_repository_succeeds(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    context = _load(repo)
    assert context.file_tree == ""
    assert context.files == []


# --- Effective-root resolution through Git ------------------------------------


def test_subdirectory_resolves_to_top_level(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    _write(repo, "README.md", "root readme\n")
    _write(repo, "src/example.py", "x = 1\n")
    _add(repo, "README.md", "src/example.py")

    context = _load(repo / "src")

    # Result is rooted at the repository top-level: root-relative paths, no
    # traversal, no absolute leakage.
    assert context.file_tree == "README.md\nsrc/example.py"
    assert _paths(context) == ["README.md"]
    assert ".." not in context.file_tree
    assert str(tmp_path) not in context.file_tree


def test_no_absolute_paths_leak_into_context(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    _write(repo, "README.md", "content\n")
    _add(repo, "README.md")

    context = _load(repo)

    assert str(repo) not in context.file_tree
    for f in context.files:
        assert not Path(f.path).is_absolute()
        assert str(repo) not in f.path


# --- Seed byte limit ---------------------------------------------------------


def test_oversized_seed_skipped_but_tracked(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    _write(repo, "README.md", "x" * 100)
    _add(repo, "README.md")

    context = _load(repo, max_seed_file_bytes=10)

    assert "README.md" in context.file_tree.splitlines()
    assert _paths(context) == []  # skipped, not truncated


def test_exact_size_boundary(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    _write(repo, "README.md", "a" * 8)  # exactly the limit
    _write(repo, "AGENTS.md", "b" * 9)  # one over the limit
    _add(repo, "README.md", "AGENTS.md")

    context = _load(repo, max_seed_file_bytes=8)

    # size == limit is allowed; size == limit + 1 is skipped.
    assert _paths(context) == ["README.md"]
    assert context.files[0].content == "a" * 8


# --- Encoding / content edge cases -------------------------------------------


def test_invalid_utf8_seed_skipped(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    (repo / "README.md").write_bytes(b"\xff\xfe\x00\x01 not utf-8")
    _add(repo, "README.md")

    context = _load(repo)

    assert "README.md" in context.file_tree.splitlines()
    assert _paths(context) == []


def test_empty_seed_file_included(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    _write(repo, "README.md", "")
    _add(repo, "README.md")

    context = _load(repo)

    assert _paths(context) == ["README.md"]
    assert context.files[0].content == ""


def test_tracked_but_deleted_seed_skipped(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    _write(repo, "README.md", "hello\n")
    _add(repo, "README.md")
    # Remove from the working tree without updating the index.
    (repo / "README.md").unlink()

    context = _load(repo)

    assert "README.md" in context.file_tree.splitlines()
    assert _paths(context) == []


# --- Symlink security --------------------------------------------------------


@pytest.mark.skipif(
    not hasattr(__import__("os"), "symlink"),
    reason="symlinks unavailable on this platform",
)
def test_symlink_escape_is_not_followed(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = outside / "secret"
    secret.write_text("EXTERNAL SECRET CREDENTIALS\n", encoding="utf-8")

    repo = _init_repo(tmp_path / "repo")
    link = repo / "README.md"
    try:
        link.symlink_to(secret)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation not permitted")
    _add(repo, "README.md")

    context = _load(repo)

    assert "README.md" in context.file_tree.splitlines()
    # The external secret must never appear in returned content.
    for f in context.files:
        assert "EXTERNAL SECRET" not in f.content
    assert _paths(context) == []
    assert str(secret) not in context.file_tree


@pytest.mark.skipif(
    not hasattr(__import__("os"), "symlink"),
    reason="symlinks unavailable on this platform",
)
def test_safe_internal_symlink_is_read_under_logical_path(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    _write(repo, "docs/readme.txt", "internal readme body\n")
    link = repo / "README.md"
    try:
        link.symlink_to(repo / "docs" / "readme.txt")
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation not permitted")
    _add(repo, "docs/readme.txt", "README.md")

    context = _load(repo)

    assert _paths(context) == ["README.md"]
    assert context.files[0].content == "internal readme body\n"
    # The absolute target path is never exposed.
    assert str(repo) not in context.files[0].path


# --- Determinism -------------------------------------------------------------


def test_repeated_loads_are_semantically_equal(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    _write(repo, "README.md", "readme\n")
    _write(repo, "pyproject.toml", "[project]\n")
    _write(repo, "src/a.py", "a = 1\n")
    _write(repo, "src/b.py", "b = 2\n")
    _add(repo, "README.md", "pyproject.toml", "src/a.py", "src/b.py")

    first = _load(repo)
    second = _load(repo)

    assert first.model_dump() == second.model_dump()


# --- Paths with spaces (no shell quoting) ------------------------------------


def test_repository_root_with_spaces(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "dir with spaces")
    _write(repo, "README.md", "spaced\n")
    _write(repo, "a file.txt", "data\n")
    _add(repo, "README.md", "a file.txt")

    context = _load(repo)

    assert context.file_tree == "README.md\na file.txt"
    assert _paths(context) == ["README.md"]


# --- Error normalization -----------------------------------------------------


def test_git_launch_failure_becomes_inspection_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _init_repo(tmp_path / "repo")
    _write(repo, "README.md", "x\n")
    _add(repo, "README.md")

    loader = GitRepositoryContextLoader()

    async def _boom(*args: object, **kwargs: object) -> object:
        raise FileNotFoundError("git not found")

    # Narrow seam: only the subprocess launch is replaced, exercising the
    # documented normalization to RepositoryInspectionError with causality.
    monkeypatch.setattr(asyncio, "create_subprocess_exec", _boom)

    with pytest.raises(RepositoryInspectionError) as excinfo:
        asyncio.run(loader.load(repo))
    assert isinstance(excinfo.value.__cause__, FileNotFoundError)
