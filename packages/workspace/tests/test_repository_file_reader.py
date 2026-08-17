"""Behavioral tests for the workspace repository file-reading contract.

The first group exercises the generic ``RepositoryFileReader`` Protocol with a
test-local fake (structural typing only). The remaining groups build real,
local, temporary Git repositories (``git init`` + ``git add``; no commits unless
a linked worktree requires one, no remotes, no network) and exercise the
concrete ``GitRepositoryFileReader`` end to end. Async calls are driven with
``asyncio.run`` to match the repository convention of not adding a pytest
asyncio plugin.
"""

import asyncio
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from llmforeman_core import RepositoryFile
from llmforeman_workspace import (
    GitRepositoryFileReader,
    InvalidRepositoryError,
    RepositoryFileAccessError,
    RepositoryFileReader,
    RepositoryInspectionError,
    WorkspaceError,
)
from llmforeman_workspace import file_reader as file_reader_module

# --- Generic Protocol structural checks (no filesystem / Git) ----------------


class _FakeRepositoryFileReader:
    """Test-local structural implementation of ``RepositoryFileReader``.

    Returns a manually constructed ``RepositoryFile`` and performs no filesystem
    or Git access; it exercises only the typed contract.
    """

    async def read(self, repository_root: Path, path: str) -> RepositoryFile:
        return RepositoryFile(path=path, content="example")


def test_fake_structurally_satisfies_protocol() -> None:
    # Assigning to the Protocol-typed name is the static structural check;
    # mypy rejects a fake whose signature is not
    # ``(Path, str) -> RepositoryFile``.
    reader: RepositoryFileReader = _FakeRepositoryFileReader()
    assert reader is not None


def test_async_read_returns_repository_file() -> None:
    reader: RepositoryFileReader = _FakeRepositoryFileReader()

    async def run() -> RepositoryFile:
        # The repository path need not exist; it is merely input to the fake.
        return await reader.read(
            Path("/example/repository"),
            "packages/core/src/example.py",
        )

    result = asyncio.run(run())
    assert isinstance(result, RepositoryFile)
    assert result.path == "packages/core/src/example.py"
    assert result.content == "example"


def test_logical_repository_relative_path_is_preserved() -> None:
    reader: RepositoryFileReader = _FakeRepositoryFileReader()

    logical_path = "packages/core/src/llmforeman_core/models.py"
    result = asyncio.run(reader.read(Path("/example/repository"), logical_path))

    # The fake echoes the logical path through to ``RepositoryFile.path``; this
    # does not imply the Protocol runtime-validates the path.
    assert isinstance(result, RepositoryFile)
    assert result.path == logical_path


def test_concrete_reader_structurally_satisfies_protocol() -> None:
    # Static structural check under mypy: the concrete reader satisfies the
    # generic Protocol without ``Any``, casts, or ignores.
    reader: RepositoryFileReader = GitRepositoryFileReader()
    assert reader is not None


# --- Concrete Git-backed reader ---------------------------------------------

# A local ``git`` executable is the only external requirement. Skip the concrete
# tests cleanly if it is unavailable rather than failing the suite.
_GIT = shutil.which("git")
_requires_git = pytest.mark.skipif(_GIT is None, reason="git executable not available")
_requires_symlink = pytest.mark.skipif(
    not hasattr(os, "symlink"), reason="symlinks unavailable on this platform"
)


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


def _read(root: Path, path: str, **kwargs: int) -> RepositoryFile:
    reader = GitRepositoryFileReader(**kwargs)
    return asyncio.run(reader.read(root, path))


# --- Constructor validation --------------------------------------------------


@pytest.mark.parametrize("bad", [0, -1, -1024, True, False])
def test_rejects_non_positive_or_bool_limit(bad: object) -> None:
    with pytest.raises(ValueError):
        GitRepositoryFileReader(max_file_bytes=bad)  # type: ignore[arg-type]


# --- Happy path --------------------------------------------------------------


@_requires_git
def test_happy_path_reads_tracked_file(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    _write(repo, "src/example.py", "print('hi')\n")
    _add(repo, "src/example.py")

    result = _read(repo, "src/example.py")

    assert result == RepositoryFile(path="src/example.py", content="print('hi')\n")


@_requires_git
def test_empty_tracked_file_is_valid(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    _write(repo, "empty.txt", "")
    _add(repo, "empty.txt")

    result = _read(repo, "empty.txt")

    assert result == RepositoryFile(path="empty.txt", content="")


@_requires_git
def test_subdirectory_entry_point_resolves_to_top_level(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    _write(repo, "src/example.py", "x = 1\n")
    _add(repo, "src/example.py")

    # Entry point is a subdirectory; ``path`` is relative to the Git top-level.
    result = _read(repo / "src", "src/example.py")

    assert result.path == "src/example.py"
    assert result.content == "x = 1\n"


@_requires_git
def test_linked_worktree_reads_consistently(tmp_path: Path) -> None:
    # A linked worktree lacks a normal ``.git`` directory; Git still resolves
    # its top-level. A commit is required to attach a worktree.
    repo = _init_repo(tmp_path / "repo")
    _write(repo, "README.md", "main tree\n")
    _add(repo, "README.md")
    _git(repo, "-c", "user.email=t@e", "-c", "user.name=t", "commit", "-q", "-m", "init")

    linked = tmp_path / "linked"
    _git(repo, "worktree", "add", "-q", str(linked))
    _write(linked, "extra.txt", "linked body\n")
    _add(linked, "extra.txt")

    result = _read(linked, "extra.txt")
    assert result == RepositoryFile(path="extra.txt", content="linked body\n")


# --- Untracked / ignored privacy --------------------------------------------


@_requires_git
def test_untracked_file_is_rejected_without_reading(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    _write(repo, "tracked.txt", "ok\n")
    _add(repo, "tracked.txt")
    secret_body = "TOP-SECRET-UNTRACKED-CREDENTIAL"
    _write(repo, "secret.txt", secret_body + "\n")  # never added

    with pytest.raises(RepositoryFileAccessError) as excinfo:
        _read(repo, "secret.txt")

    # The untracked file's contents must never surface in the error.
    assert secret_body not in str(excinfo.value)


@_requires_git
def test_dotenv_untracked_is_rejected(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    _write(repo, "README.md", "readme\n")
    _add(repo, "README.md")
    _write(repo, ".env", "API_KEY=super-secret\n")  # exists but untracked

    with pytest.raises(RepositoryFileAccessError) as excinfo:
        _read(repo, ".env")
    assert "super-secret" not in str(excinfo.value)


@_requires_git
def test_ignored_untracked_file_is_rejected(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    _write(repo, ".gitignore", ".env\n")
    _add(repo, ".gitignore")
    _write(repo, ".env", "SECRET=1\n")  # matches ignore, untracked

    with pytest.raises(RepositoryFileAccessError):
        _read(repo, ".env")


@_requires_git
def test_force_added_ignored_file_is_eligible(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    _write(repo, ".gitignore", "config.txt\n")
    _write(repo, "config.txt", "tracked despite ignore\n")
    # Force-add so it is tracked even though an ignore rule matches it.
    _git(repo, "add", "--force", "--", "config.txt", ".gitignore")

    result = _read(repo, "config.txt")
    assert result == RepositoryFile(
        path="config.txt", content="tracked despite ignore\n"
    )


# --- Path validation (before any filesystem access) --------------------------


@pytest.mark.parametrize("bad", ["", "   ", "\t\n"])
@_requires_git
def test_empty_or_whitespace_path_rejected(tmp_path: Path, bad: str) -> None:
    repo = _init_repo(tmp_path / "repo")
    _write(repo, "keep.txt", "keep\n")
    _add(repo, "keep.txt")

    with pytest.raises(RepositoryFileAccessError):
        _read(repo, bad)


@pytest.mark.parametrize(
    "bad",
    ["/etc/passwd", "C:\\Users\\example\\secret.txt", "\\\\server\\share\\x"],
)
@_requires_git
def test_absolute_paths_rejected(tmp_path: Path, bad: str) -> None:
    repo = _init_repo(tmp_path / "repo")
    _write(repo, "keep.txt", "keep\n")
    _add(repo, "keep.txt")

    with pytest.raises(RepositoryFileAccessError):
        _read(repo, bad)


@pytest.mark.parametrize(
    "bad",
    ["../secret.txt", "../../.ssh/id_rsa", "src/../../secret.txt"],
)
@_requires_git
def test_parent_traversal_rejected(tmp_path: Path, bad: str) -> None:
    repo = _init_repo(tmp_path / "repo")
    _write(repo, "keep.txt", "keep\n")
    _add(repo, "keep.txt")

    with pytest.raises(RepositoryFileAccessError):
        _read(repo, bad)


@_requires_git
def test_nul_path_rejected(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    _write(repo, "keep.txt", "keep\n")
    _add(repo, "keep.txt")

    with pytest.raises(RepositoryFileAccessError):
        _read(repo, "keep\x00.txt")


@_requires_git
def test_invalid_path_rejected_before_filesystem_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _init_repo(tmp_path / "repo")
    _write(repo, "keep.txt", "keep\n")
    _add(repo, "keep.txt")

    reader = GitRepositoryFileReader()

    def _boom(*args: object, **kwargs: object) -> object:
        raise AssertionError("no filesystem read must occur for an invalid path")

    # If validation happened only during read, this would be reached.
    monkeypatch.setattr(reader, "_read_working_tree_file", _boom)

    with pytest.raises(RepositoryFileAccessError):
        asyncio.run(reader.read(repo, "../secret.txt"))


# --- Missing / non-regular targets ------------------------------------------


@_requires_git
def test_tracked_but_deleted_from_working_tree_fails(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    _write(repo, "src/example.py", "x = 1\n")
    _add(repo, "src/example.py")
    # Remove from the working tree without updating the index.
    (repo / "src" / "example.py").unlink()

    with pytest.raises(RepositoryFileAccessError):
        _read(repo, "src/example.py")


@_requires_git
def test_directory_path_fails(tmp_path: Path) -> None:
    # ``src`` is not itself a tracked path (only ``src/example.py`` is), so this
    # fails membership; a request that names a directory cannot yield a file.
    repo = _init_repo(tmp_path / "repo")
    _write(repo, "src/example.py", "x = 1\n")
    _add(repo, "src/example.py")

    with pytest.raises(RepositoryFileAccessError):
        _read(repo, "src")


# --- Encoding / content policy ----------------------------------------------


@_requires_git
def test_invalid_utf8_fails(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    (repo / "bad.bin").write_bytes(b"\xff\xfe not utf-8")
    _add(repo, "bad.bin")

    with pytest.raises(RepositoryFileAccessError):
        _read(repo, "bad.bin")


@_requires_git
def test_nul_byte_content_fails(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    # Otherwise-valid UTF-8 bytes that contain a NUL byte.
    (repo / "data.txt").write_bytes(b"hello\x00world")
    _add(repo, "data.txt")

    with pytest.raises(RepositoryFileAccessError):
        _read(repo, "data.txt")


# --- Size limits -------------------------------------------------------------


@_requires_git
def test_oversized_file_fails_without_truncation(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    _write(repo, "big.txt", "x" * 100)
    _add(repo, "big.txt")

    with pytest.raises(RepositoryFileAccessError):
        _read(repo, "big.txt", max_file_bytes=10)


@_requires_git
def test_exact_size_boundary(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    _write(repo, "exact.txt", "a" * 8)  # exactly the limit
    _write(repo, "over.txt", "b" * 9)  # one over the limit
    _add(repo, "exact.txt", "over.txt")

    result = _read(repo, "exact.txt", max_file_bytes=8)
    assert result.content == "a" * 8

    with pytest.raises(RepositoryFileAccessError):
        _read(repo, "over.txt", max_file_bytes=8)


# --- Symlink security --------------------------------------------------------


@_requires_git
@_requires_symlink
def test_external_symlink_escape_fails(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    external_body = "EXTERNAL-SECRET-CREDENTIALS"
    (outside / "secret").write_text(external_body + "\n", encoding="utf-8")

    repo = _init_repo(tmp_path / "repo")
    link = repo / "config.txt"
    try:
        link.symlink_to(outside / "secret")
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation not permitted")
    _add(repo, "config.txt")

    with pytest.raises(RepositoryFileAccessError) as excinfo:
        _read(repo, "config.txt")

    # The external target contents/path must never surface.
    assert external_body not in str(excinfo.value)
    assert str(outside) not in str(excinfo.value)


@_requires_git
@_requires_symlink
def test_safe_internal_symlink_read_under_logical_path(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    _write(repo, "docs/real.txt", "internal body\n")
    link = repo / "readme-link.txt"
    try:
        link.symlink_to(repo / "docs" / "real.txt")
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation not permitted")
    _add(repo, "docs/real.txt", "readme-link.txt")

    result = _read(repo, "readme-link.txt")

    # Content from the safe internal target; logical path stays as requested.
    assert result.path == "readme-link.txt"
    assert result.content == "internal body\n"
    assert str(repo) not in result.path


@_requires_git
@_requires_symlink
def test_broken_symlink_fails(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    link = repo / "dangling.txt"
    try:
        link.symlink_to(repo / "does-not-exist.txt")
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation not permitted")
    _add(repo, "dangling.txt")

    with pytest.raises(RepositoryFileAccessError):
        _read(repo, "dangling.txt")


# --- Unusual names -----------------------------------------------------------


@_requires_git
def test_path_with_spaces(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    _write(repo, "docs/file with spaces.txt", "spaced body\n")
    _add(repo, "docs/file with spaces.txt")

    result = _read(repo, "docs/file with spaces.txt")
    assert result == RepositoryFile(
        path="docs/file with spaces.txt", content="spaced body\n"
    )


@_requires_git
def test_pathspec_like_name(tmp_path: Path) -> None:
    # Brackets and a leading punctuation char have Git pathspec meaning; exact
    # membership must match them literally without pathspec interpretation.
    name = "weird/[a-z]file.txt"
    repo = _init_repo(tmp_path / "repo")
    _write(repo, name, "bracket body\n")
    _add(repo, name)

    result = _read(repo, name)
    assert result == RepositoryFile(path=name, content="bracket body\n")


# --- Working-tree content ----------------------------------------------------


@_requires_git
def test_returns_current_working_tree_content(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    _write(repo, "src/example.py", "original = 1\n")
    _add(repo, "src/example.py")
    # Modify locally without staging; the reader must return this content.
    _write(repo, "src/example.py", "modified = 2\n")

    result = _read(repo, "src/example.py")
    assert result.content == "modified = 2\n"


# --- Determinism -------------------------------------------------------------


@_requires_git
def test_repeated_reads_are_semantically_equal(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    _write(repo, "src/example.py", "x = 1\n")
    _add(repo, "src/example.py")

    first = _read(repo, "src/example.py")
    second = _read(repo, "src/example.py")

    assert first.model_dump() == second.model_dump()


# --- Repository error semantics ---------------------------------------------


@_requires_git
def test_missing_root_is_invalid_repository(tmp_path: Path) -> None:
    with pytest.raises(InvalidRepositoryError):
        _read(tmp_path / "does-not-exist", "x.txt")


@_requires_git
def test_file_as_root_is_invalid_repository(tmp_path: Path) -> None:
    file_path = tmp_path / "a-file"
    file_path.write_text("x\n", encoding="utf-8")
    with pytest.raises(InvalidRepositoryError):
        _read(file_path, "x.txt")


@_requires_git
def test_non_git_directory_is_invalid_repository(tmp_path: Path) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    with pytest.raises(InvalidRepositoryError):
        _read(plain, "x.txt")


@_requires_git
def test_invalid_repository_is_workspace_error(tmp_path: Path) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    with pytest.raises(WorkspaceError):
        _read(plain, "x.txt")


@_requires_git
def test_git_listing_failure_becomes_inspection_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _init_repo(tmp_path / "repo")
    _write(repo, "src/example.py", "x = 1\n")
    _add(repo, "src/example.py")

    reader = GitRepositoryFileReader()

    async def _fail_listing(effective_root: Path) -> list[str]:
        raise RepositoryInspectionError("tracked listing failed")

    # Narrow seam: the shared tracked-listing primitive fails after the
    # repository has been validated; this must surface as inspection failure.
    monkeypatch.setattr(file_reader_module, "list_tracked_paths", _fail_listing)

    with pytest.raises(RepositoryInspectionError):
        asyncio.run(reader.read(repo, "src/example.py"))
