"""Behavioral tests for the concrete ``GitRepositoryFileWriter``.

These build real, local, temporary Git repositories (``git init`` + ``git
add``; no commits unless a linked worktree requires one, no remotes, no
network) and exercise the concrete writer end to end against a real filesystem.
Security-sensitive cases (symlink targets/parents, binary protection, oversize
pre-mutation behavior) use real filesystem objects rather than mocking the
mutation boundary. Async calls are driven with ``asyncio.run`` to match the
repository convention of not adding a pytest asyncio plugin.
"""

import asyncio
import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

from llmforeman_core import RepositoryFile
from llmforeman_workspace import (
    GitRepositoryFileWriter,
    InvalidRepositoryError,
    RepositoryFileWriteError,
    RepositoryFileWriter,
)

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


def _commit(repo: Path, message: str) -> None:
    _git(
        repo,
        "-c",
        "user.email=t@e",
        "-c",
        "user.name=t",
        "commit",
        "-q",
        "-m",
        message,
    )


def _status(repo: Path) -> str:
    # ``--untracked-files=all`` lists individual untracked files rather than
    # collapsing a new directory to a single ``?? dir/`` entry.
    result = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _write_disk(repo: Path, relpath: str, content: str) -> Path:
    path = repo / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _write(root: Path, path: str, content: str, **kwargs: int) -> RepositoryFile:
    writer = GitRepositoryFileWriter(**kwargs)
    return asyncio.run(writer.write(root, path, content))


# --- Protocol / constructor --------------------------------------------------


def test_concrete_writer_structurally_satisfies_protocol() -> None:
    # Static structural check under mypy: the concrete writer satisfies the
    # generic Protocol without ``Any``, casts, or ignores.
    writer: RepositoryFileWriter = GitRepositoryFileWriter()
    assert writer is not None


@pytest.mark.parametrize("bad", [0, -1, -1024, True, False])
def test_rejects_non_positive_or_bool_limit(bad: object) -> None:
    with pytest.raises(ValueError):
        GitRepositoryFileWriter(max_file_bytes=bad)  # type: ignore[arg-type]


# --- Tracked / untracked / new-file semantics --------------------------------


@_requires_git
def test_overwrite_tracked_file(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    _write_disk(repo, "src/foo.py", "old = 1\n")
    _add(repo, "src/foo.py")
    _commit(repo, "init")

    result = _write(repo, "src/foo.py", "new = 2\n")

    assert result == RepositoryFile(path="src/foo.py", content="new = 2\n")
    assert (repo / "src/foo.py").read_bytes() == b"new = 2\n"
    # The path is still tracked, and the modification is unstaged (" M"), never
    # staged ("M ") — the writer must not have run ``git add``.
    tracked = subprocess.run(
        ["git", "-C", str(repo), "ls-files", "--", "src/foo.py"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert tracked.strip() == "src/foo.py"
    status = _status(repo)
    assert " M src/foo.py" in status
    assert "M  src/foo.py" not in status


@_requires_git
def test_create_new_untracked_file(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    _write_disk(repo, "README.md", "readme\n")
    _add(repo, "README.md")

    result = _write(repo, "pkg/new.py", "print('hi')\n")

    assert result == RepositoryFile(path="pkg/new.py", content="print('hi')\n")
    assert (repo / "pkg/new.py").read_bytes() == b"print('hi')\n"
    # Git reports the new path as untracked; the writer did not stage it.
    status = _status(repo)
    assert "?? pkg/new.py" in status


@_requires_git
def test_overwrite_existing_untracked_file(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    _write_disk(repo, "notes.txt", "original\n")  # never added

    result = _write(repo, "notes.txt", "replaced\n")

    assert result.content == "replaced\n"
    assert (repo / "notes.txt").read_bytes() == b"replaced\n"
    assert "?? notes.txt" in _status(repo)


@_requires_git
def test_writer_does_not_stage(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    _write_disk(repo, "keep.txt", "keep\n")
    _add(repo, "keep.txt")

    _write(repo, "generated.py", "x = 1\n")

    # No new staged (index) entry appeared for the created file.
    diff_cached = subprocess.run(
        ["git", "-C", str(repo), "diff", "--cached", "--name-only"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "generated.py" not in diff_cached


# --- Content fidelity --------------------------------------------------------


@_requires_git
def test_empty_content_existing_becomes_zero_bytes(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    _write_disk(repo, "data.txt", "not empty\n")

    result = _write(repo, "data.txt", "")

    assert result == RepositoryFile(path="data.txt", content="")
    assert (repo / "data.txt").read_bytes() == b""


@_requires_git
def test_empty_content_new_file_is_zero_bytes(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")

    result = _write(repo, "empty.txt", "")

    assert result.content == ""
    assert (repo / "empty.txt").read_bytes() == b""


@_requires_git
def test_exact_whitespace_and_newlines_preserved(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    content = "    leading\n\n\n\ttab\ntrailing spaces   \nno-final-newline"

    _write(repo, "src/exact.py", content)

    assert (repo / "src/exact.py").read_bytes() == content.encode("utf-8")


@_requires_git
def test_crlf_is_not_normalized(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    content = "a\r\nb\r\n"

    _write(repo, "win.txt", content)

    assert (repo / "win.txt").read_bytes() == b"a\r\nb\r\n"


@_requires_git
def test_nested_new_directories_are_created(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")

    result = _write(repo, "src/new_feature/subpackage/service.py", "svc = 1\n")

    assert result.path == "src/new_feature/subpackage/service.py"
    assert (repo / "src/new_feature/subpackage/service.py").read_bytes() == b"svc = 1\n"
    assert (repo / "src/new_feature/subpackage").is_dir()


@_requires_git
def test_repeated_write_replaces_including_shorter(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")

    _write(repo, "f.txt", "a very long first version\n")
    _write(repo, "f.txt", "short\n")

    assert (repo / "f.txt").read_bytes() == b"short\n"


@_requires_git
def test_path_with_spaces(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")

    result = _write(repo, "docs/new file.py", "spaced = 1\n")

    assert result.path == "docs/new file.py"
    assert (repo / "docs/new file.py").read_bytes() == b"spaced = 1\n"


@_requires_git
def test_return_value_has_no_absolute_path(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")

    result = _write(repo, "src/x.py", "x = 1\n")

    assert result == RepositoryFile(path="src/x.py", content="x = 1\n")
    assert str(repo) not in result.path


# --- Invalid logical paths ---------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "   ",
        "\t\n",
        "/etc/passwd",
        "C:\\Users\\example\\secret.txt",
        "../outside.txt",
        "src/../../outside.txt",
    ],
)
@_requires_git
def test_invalid_logical_paths_rejected(tmp_path: Path, bad: str) -> None:
    repo = _init_repo(tmp_path / "repo")

    with pytest.raises(RepositoryFileWriteError):
        _write(repo, bad, "data\n")


@_requires_git
def test_nul_path_rejected(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")

    with pytest.raises(RepositoryFileWriteError):
        _write(repo, "keep\x00.txt", "data\n")


@_requires_git
def test_invalid_path_creates_nothing(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")

    with pytest.raises(RepositoryFileWriteError):
        _write(repo, "src/../../outside.txt", "data\n")

    # No target and no parent directory were created for the rejected path.
    assert not (repo / "src").exists()
    assert not (repo.parent / "outside.txt").exists()


# --- Size limits -------------------------------------------------------------


@_requires_git
def test_oversized_new_content_creates_nothing(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")

    with pytest.raises(RepositoryFileWriteError):
        _write(repo, "big/new.txt", "x" * 11, max_file_bytes=10)

    assert not (repo / "big").exists()
    assert not (repo / "big/new.txt").exists()


@_requires_git
def test_oversized_content_leaves_existing_unchanged(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    _write_disk(repo, "keep.txt", "original\n")

    with pytest.raises(RepositoryFileWriteError):
        _write(repo, "keep.txt", "x" * 100, max_file_bytes=10)

    assert (repo / "keep.txt").read_bytes() == b"original\n"


@_requires_git
def test_exact_byte_limit_boundary_multibyte(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    # "€" encodes to 3 UTF-8 bytes; two of them are exactly 6 bytes.
    at_limit = "€€"  # 6 bytes
    over_limit = "€€x"  # 7 bytes

    result = _write(repo, "exact.txt", at_limit, max_file_bytes=6)
    assert (repo / "exact.txt").read_bytes() == at_limit.encode("utf-8")
    assert result.content == at_limit

    with pytest.raises(RepositoryFileWriteError):
        _write(repo, "over.txt", over_limit, max_file_bytes=6)
    assert not (repo / "over.txt").exists()


@_requires_git
def test_existing_large_valid_text_may_be_replaced_by_small(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    _write_disk(repo, "big.txt", "a" * 5000)  # valid text, larger than new limit

    result = _write(repo, "big.txt", "tiny\n", max_file_bytes=100)

    assert result.content == "tiny\n"
    assert (repo / "big.txt").read_bytes() == b"tiny\n"


# --- Existing binary / invalid text protection -------------------------------


@_requires_git
def test_existing_binary_nul_is_rejected_unchanged(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    original = b"\x00\x01\x02binary"
    (repo / "asset.bin").write_bytes(original)

    with pytest.raises(RepositoryFileWriteError):
        _write(repo, "asset.bin", "text\n")

    assert (repo / "asset.bin").read_bytes() == original


@_requires_git
def test_existing_invalid_utf8_is_rejected_unchanged(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    original = b"\xff\xfe not utf-8"
    (repo / "bad.dat").write_bytes(original)

    with pytest.raises(RepositoryFileWriteError):
        _write(repo, "bad.dat", "text\n")

    assert (repo / "bad.dat").read_bytes() == original


# --- Directory / conflict targets -------------------------------------------


@_requires_git
def test_directory_target_is_rejected(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    (repo / "adir").mkdir()

    with pytest.raises(RepositoryFileWriteError):
        _write(repo, "adir", "data\n")

    assert (repo / "adir").is_dir()


@_requires_git
def test_parent_path_is_file_is_rejected(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    _write_disk(repo, "src/not_a_directory", "iam a file\n")

    with pytest.raises(RepositoryFileWriteError):
        _write(repo, "src/not_a_directory/foo.py", "data\n")

    assert (repo / "src/not_a_directory").read_bytes() == b"iam a file\n"


# --- Symlink security --------------------------------------------------------


@_requires_git
@_requires_symlink
def test_final_target_symlink_inside_repo_is_rejected(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    real_body = "REAL-BODY\n"
    _write_disk(repo, "real.txt", real_body)
    link = repo / "link.txt"
    try:
        link.symlink_to(repo / "real.txt")
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation not permitted")

    with pytest.raises(RepositoryFileWriteError):
        _write(repo, "link.txt", "attempted\n")

    # Neither the symlink nor its (internal) target was written through.
    assert (repo / "real.txt").read_bytes() == real_body.encode("utf-8")
    assert link.is_symlink()


@_requires_git
@_requires_symlink
def test_final_target_symlink_outside_repo_is_rejected(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    external_body = "EXTERNAL\n"
    (outside / "secret").write_text(external_body, encoding="utf-8")

    repo = _init_repo(tmp_path / "repo")
    link = repo / "config.txt"
    try:
        link.symlink_to(outside / "secret")
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation not permitted")

    with pytest.raises(RepositoryFileWriteError):
        _write(repo, "config.txt", "attempted\n")

    # The external target must not be redirected/written.
    assert (outside / "secret").read_bytes() == external_body.encode("utf-8")


@_requires_git
@_requires_symlink
def test_parent_symlink_outside_repo_is_rejected(tmp_path: Path) -> None:
    outside = tmp_path / "outside_dir"
    outside.mkdir()

    repo = _init_repo(tmp_path / "repo")
    try:
        (repo / "link").symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation not permitted")

    with pytest.raises(RepositoryFileWriteError):
        _write(repo, "link/new_file.py", "data\n")

    # Nothing was created through the symlinked parent.
    assert not (outside / "new_file.py").exists()


@_requires_git
@_requires_symlink
def test_parent_symlink_inside_repo_is_rejected(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    (repo / "packages" / "generated").mkdir(parents=True)
    try:
        (repo / "generated").symlink_to(
            repo / "packages" / "generated", target_is_directory=True
        )
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation not permitted")

    with pytest.raises(RepositoryFileWriteError):
        _write(repo, "generated/new.py", "data\n")

    # The stricter writer policy refuses even an internal parent symlink.
    assert not (repo / "packages" / "generated" / "new.py").exists()


# --- Descriptor-relative helper seam (TOCTOU-resistant design) ---------------


@_requires_git
@_requires_symlink
def test_helper_rejects_symlink_parent_at_open_time(tmp_path: Path) -> None:
    # Exercises the private no-follow directory-open seam directly to show the
    # symlink refusal is enforced at the filesystem operation, not by a prior
    # ``is_symlink()`` check.
    from llmforeman_workspace import file_writer as fw

    repo = _init_repo(tmp_path / "repo")
    (repo / "realdir").mkdir()
    try:
        (repo / "linkdir").symlink_to(repo / "realdir", target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation not permitted")

    root_fd = os.open(repo, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with pytest.raises(RepositoryFileWriteError):
            fw._try_open_directory(root_fd, "linkdir", "linkdir/x.py")
        # A real directory opens fine and must be closed by the caller.
        real_fd = fw._try_open_directory(root_fd, "realdir", "realdir/x.py")
        assert real_fd is not None
        os.close(real_fd)
    finally:
        os.close(root_fd)


# --- Permissions / metadata --------------------------------------------------


@_requires_git
@pytest.mark.skipif(os.name != "posix", reason="POSIX mode bits required")
def test_existing_executable_bit_is_preserved(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    script = _write_disk(repo, "run.sh", "#!/bin/sh\necho old\n")
    script.chmod(0o755)
    before = stat.S_IMODE(script.stat().st_mode)

    _write(repo, "run.sh", "#!/bin/sh\necho new\n")

    after = stat.S_IMODE(script.stat().st_mode)
    assert after == before
    assert after & stat.S_IXUSR
    assert script.read_bytes() == b"#!/bin/sh\necho new\n"


@_requires_git
@pytest.mark.skipif(os.name != "posix", reason="POSIX mode bits required")
def test_new_file_is_not_executable(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")

    _write(repo, "plain.py", "x = 1\n")

    mode = stat.S_IMODE((repo / "plain.py").stat().st_mode)
    # The writer never sets an executable bit explicitly.
    assert not (mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH))


# --- Boundary / entry-point semantics ----------------------------------------


@_requires_git
def test_subdirectory_entry_point_writes_relative_to_top_level(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    (repo / "src").mkdir()

    writer = GitRepositoryFileWriter()
    result = asyncio.run(writer.write(repo / "src", "new/file.py", "x = 1\n"))

    assert result.path == "new/file.py"
    assert (repo / "new/file.py").read_bytes() == b"x = 1\n"
    # The logical path is relative to the Git top-level, not the entry point.
    assert not (repo / "src/new/file.py").exists()


@_requires_git
def test_linked_worktree_writes_inside_linked_top_level(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    _write_disk(repo, "README.md", "main tree\n")
    _add(repo, "README.md")
    _commit(repo, "init")

    linked = tmp_path / "linked"
    _git(repo, "worktree", "add", "-q", str(linked))

    writer = GitRepositoryFileWriter()
    result = asyncio.run(writer.write(linked, "pkg/created.py", "y = 2\n"))

    assert result.path == "pkg/created.py"
    assert (linked / "pkg/created.py").read_bytes() == b"y = 2\n"
    # It must not have leaked into the main worktree.
    assert not (repo / "pkg/created.py").exists()


@_requires_git
def test_untracked_path_that_looks_ignored_is_written(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    _write_disk(repo, ".gitignore", "generated/\n")
    _add(repo, ".gitignore")

    result = _write(repo, "generated/new.py", "g = 1\n")

    assert result.content == "g = 1\n"
    assert (repo / "generated/new.py").read_bytes() == b"g = 1\n"
    # It remains ignored/untracked, but the write still succeeded.
    status = _status(repo)
    assert "generated/new.py" not in status  # ignored → not listed as untracked


# --- Invalid repository semantics --------------------------------------------


@_requires_git
def test_missing_root_is_invalid_repository(tmp_path: Path) -> None:
    with pytest.raises(InvalidRepositoryError):
        _write(tmp_path / "does-not-exist", "x.txt", "data\n")


@_requires_git
def test_file_as_root_is_invalid_repository(tmp_path: Path) -> None:
    file_path = tmp_path / "a-file"
    file_path.write_text("x\n", encoding="utf-8")
    with pytest.raises(InvalidRepositoryError):
        _write(file_path, "x.txt", "data\n")


@_requires_git
def test_non_git_directory_is_invalid_repository(tmp_path: Path) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    with pytest.raises(InvalidRepositoryError):
        _write(plain, "x.txt", "data\n")


# --- Filesystem permission failure normalization -----------------------------


@_requires_git
@pytest.mark.skipif(os.name != "posix", reason="POSIX permission bits required")
def test_permission_failure_becomes_write_error(tmp_path: Path) -> None:
    if os.geteuid() == 0:
        pytest.skip("root bypasses directory permission bits")
    repo = _init_repo(tmp_path / "repo")
    locked = repo / "locked"
    locked.mkdir()
    locked.chmod(0o500)  # read+execute, no write
    try:
        with pytest.raises(RepositoryFileWriteError):
            _write(repo, "locked/new.py", "data\n")
    finally:
        # Restore permissions so tmp_path cleanup can remove the tree.
        locked.chmod(0o700)
