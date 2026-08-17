"""Contract and model tests for the workspace repository text-search capability.

These exercise the workspace-owned result models
(``RepositorySearchMatch``/``RepositorySearchResult``) and the structural
``RepositoryTextSearcher`` Protocol with a test-local fake. They perform no
filesystem, Git, subprocess, ripgrep, network, or model access; async calls are
driven with ``asyncio.run`` to match the repository convention of not adding a
pytest asyncio plugin.
"""

import asyncio
from pathlib import Path

import pytest
from pydantic import ValidationError

from llmforeman_workspace import (
    RepositorySearchMatch,
    RepositorySearchResult,
    RepositoryTextSearcher,
)

# --- RepositorySearchMatch construction --------------------------------------


def test_search_match_preserves_fields() -> None:
    match = RepositorySearchMatch(
        path="packages/core/src/llmforeman_core/models.py",
        line_number=42,
        line="class RetryPolicy:",
    )

    assert match.path == "packages/core/src/llmforeman_core/models.py"
    assert match.line_number == 42
    assert match.line == "class RetryPolicy:"


@pytest.mark.parametrize(
    "path",
    [
        "README.md",
        "src/example.py",
        "packages/core/src/llmforeman_core/models.py",
        "docs/file with spaces.txt",
        "weird/[a-z]file.txt",
    ],
)
def test_safe_repository_relative_paths_accepted(path: str) -> None:
    match = RepositorySearchMatch(path=path, line_number=1, line="x")
    assert match.path == path


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "   ",
        "\t\n",
        "/etc/passwd",
        "C:\\Users\\example\\secret.txt",
        "../secret.txt",
        "packages/core/../../secret.txt",
        "keep\x00.txt",
    ],
)
def test_invalid_match_path_rejected(bad: str) -> None:
    with pytest.raises(ValidationError):
        RepositorySearchMatch(path=bad, line_number=1, line="x")


# --- line_number validation --------------------------------------------------


@pytest.mark.parametrize("line_number", [1, 42])
def test_valid_line_numbers_accepted(line_number: int) -> None:
    match = RepositorySearchMatch(path="a.py", line_number=line_number, line="x")
    assert match.line_number == line_number


@pytest.mark.parametrize("line_number", [0, -1])
def test_non_positive_line_numbers_rejected(line_number: int) -> None:
    with pytest.raises(ValidationError):
        RepositorySearchMatch(path="a.py", line_number=line_number, line="x")


# --- line content ------------------------------------------------------------


@pytest.mark.parametrize(
    "line",
    [
        "class RetryPolicy:",
        "    return await searcher.search(root, query)  # trailing comment",
        "",
    ],
)
def test_line_content_preserved_exactly(line: str) -> None:
    match = RepositorySearchMatch(path="a.py", line_number=1, line=line)
    assert match.line == line


# --- RepositorySearchResult --------------------------------------------------


def test_result_preserves_matches_and_supplied_order() -> None:
    # Deliberately non-sorted paths and line numbers to detect accidental
    # sorting or ranking during construction.
    first = RepositorySearchMatch(
        path="src/zeta.py",
        line_number=99,
        line="RetryPolicy A",
    )
    second = RepositorySearchMatch(
        path="src/alpha.py",
        line_number=3,
        line="RetryPolicy B",
    )

    result = RepositorySearchResult(matches=[first, second])

    assert result.matches == [first, second]
    assert result.matches[0] is first
    assert result.matches[1] is second


def test_empty_result_is_valid_explicit() -> None:
    result = RepositorySearchResult(matches=[])
    assert result.matches == []


def test_empty_result_is_valid_default() -> None:
    result = RepositorySearchResult()
    assert result.matches == []


def test_mutable_default_is_not_shared() -> None:
    first = RepositorySearchResult()
    second = RepositorySearchResult()

    first.matches.append(
        RepositorySearchMatch(path="a.py", line_number=1, line="x")
    )

    assert first.matches != second.matches
    assert second.matches == []


def test_result_serializes_predictably() -> None:
    result = RepositorySearchResult(
        matches=[
            RepositorySearchMatch(path="src/example.py", line_number=3, line="RetryPolicy"),
        ]
    )

    assert result.model_dump() == {
        "matches": [
            {
                "path": "src/example.py",
                "line_number": 3,
                "line": "RetryPolicy",
            }
        ]
    }


# --- Structural Protocol (no filesystem / Git / subprocess) ------------------


class _FakeRepositoryTextSearcher:
    """Test-local structural implementation of ``RepositoryTextSearcher``.

    Returns a manually constructed ``RepositorySearchResult`` and performs no
    filesystem, Git, or subprocess access; it exercises only the typed contract.
    """

    async def search(
        self,
        repository_root: Path,
        query: str,
    ) -> RepositorySearchResult:
        return RepositorySearchResult(
            matches=[
                RepositorySearchMatch(
                    path="src/example.py",
                    line_number=3,
                    line="RetryPolicy",
                )
            ]
        )


def test_fake_structurally_satisfies_protocol() -> None:
    # Assigning to the Protocol-typed name is the static structural check; mypy
    # rejects a fake whose signature is not
    # ``(Path, str) -> RepositorySearchResult``.
    searcher: RepositoryTextSearcher = _FakeRepositoryTextSearcher()
    assert searcher is not None


def test_async_search_returns_repository_search_result() -> None:
    searcher: RepositoryTextSearcher = _FakeRepositoryTextSearcher()

    async def run() -> RepositorySearchResult:
        # The path need not exist; it is merely input to the fake searcher.
        return await searcher.search(
            Path("/example/repository"),
            "RetryPolicy",
        )

    result = asyncio.run(run())
    assert isinstance(result, RepositorySearchResult)
    assert result.matches[0].path == "src/example.py"
    assert result.matches[0].line_number == 3
    assert result.matches[0].line == "RetryPolicy"
