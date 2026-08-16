"""Behavioral tests for the repository-context domain models.

These models represent already-prepared, provider-independent repository
context. They perform no filesystem or Git access, so no test here creates real
repository files; path validation is pure and OS-independent.
"""

import pytest
from pydantic import ValidationError

from llmforeman_core import RepositoryContext, RepositoryFile


def test_repository_relative_path_is_accepted() -> None:
    file = RepositoryFile(
        path="packages/core/src/llmforeman_core/models.py",
        content="...",
    )
    assert file.path == "packages/core/src/llmforeman_core/models.py"
    assert file.content == "..."


def test_empty_content_is_valid() -> None:
    file = RepositoryFile(path="src/empty.py", content="")
    assert file.content == ""


def test_nested_repository_relative_path_is_preserved() -> None:
    file = RepositoryFile(path="a/b/c/d/e.py", content="x")
    assert file.path == "a/b/c/d/e.py"


def test_repository_file_serialization_preserves_fields() -> None:
    file = RepositoryFile(path="packages/core/README.md", content="# Core")
    assert file.model_dump() == {
        "path": "packages/core/README.md",
        "content": "# Core",
    }


@pytest.mark.parametrize(
    "bad_path",
    [
        "",
        "   ",
        "/Users/example/project/file.py",
        "C:\\Users\\example\\project\\file.py",
        "../secret.txt",
        "packages/core/../../secret.txt",
    ],
)
def test_invalid_paths_are_rejected(bad_path: str) -> None:
    with pytest.raises(ValidationError):
        RepositoryFile(path=bad_path, content="x")


def test_context_can_contain_tree_and_selected_files() -> None:
    context = RepositoryContext(
        file_tree="packages/\n  core/",
        files=[
            RepositoryFile(path="packages/core/README.md", content="# Core"),
        ],
    )
    assert context.file_tree == "packages/\n  core/"
    assert len(context.files) == 1
    assert context.files[0].path == "packages/core/README.md"


def test_selected_file_order_is_preserved() -> None:
    context = RepositoryContext(
        file_tree="tree",
        files=[
            RepositoryFile(path="a.py", content="a"),
            RepositoryFile(path="b.py", content="b"),
            RepositoryFile(path="c.py", content="c"),
        ],
    )
    assert [file.path for file in context.files] == ["a.py", "b.py", "c.py"]


def test_empty_files_collection_is_valid() -> None:
    context = RepositoryContext(file_tree="tree", files=[])
    assert context.files == []


def test_files_defaults_to_empty_list() -> None:
    context = RepositoryContext(file_tree="tree")
    assert context.files == []


def test_default_files_not_shared_between_contexts() -> None:
    first = RepositoryContext(file_tree="one")
    second = RepositoryContext(file_tree="two")
    first.files.append(RepositoryFile(path="a.py", content="a"))
    assert first.files != second.files
    assert second.files == []


def test_empty_file_tree_is_valid() -> None:
    context = RepositoryContext(file_tree="")
    assert context.file_tree == ""


def test_nested_serialization_is_predictable() -> None:
    context = RepositoryContext(
        file_tree="packages/\n  core/",
        files=[
            RepositoryFile(path="packages/core/README.md", content="# Core"),
        ],
    )
    assert context.model_dump() == {
        "file_tree": "packages/\n  core/",
        "files": [
            {
                "path": "packages/core/README.md",
                "content": "# Core",
            }
        ],
    }
