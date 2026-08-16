"""Behavioral tests for the workspace repository file-reading contract."""

import asyncio
from pathlib import Path

from llmforeman_core import RepositoryFile
from llmforeman_workspace import RepositoryFileReader


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
