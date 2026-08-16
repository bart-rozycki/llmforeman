"""Behavioral tests for the workspace repository-loading contract."""

import asyncio
from pathlib import Path

from llmforeman_core import RepositoryContext, RepositoryFile
from llmforeman_workspace import RepositoryContextLoader


class _FakeRepositoryContextLoader:
    """Test-local structural implementation of ``RepositoryContextLoader``.

    Returns a manually constructed ``RepositoryContext`` and performs no
    filesystem access; it exercises only the typed contract.
    """

    async def load(self, repository_root: Path) -> RepositoryContext:
        return RepositoryContext(
            file_tree="",
            files=[],
        )


def test_fake_structurally_satisfies_protocol() -> None:
    # Assigning to the Protocol-typed name is the static structural check;
    # mypy rejects a fake whose signature is not ``Path -> RepositoryContext``.
    loader: RepositoryContextLoader = _FakeRepositoryContextLoader()
    assert loader is not None


def test_async_load_returns_repository_context() -> None:
    loader: RepositoryContextLoader = _FakeRepositoryContextLoader()

    async def run() -> RepositoryContext:
        # The path need not exist; it is merely input to the fake loader.
        return await loader.load(Path("/example/repository"))

    context = asyncio.run(run())
    assert isinstance(context, RepositoryContext)
    assert context.file_tree == ""
    assert context.files == []


def test_returned_context_can_carry_repository_data() -> None:
    class _PopulatingLoader:
        async def load(self, repository_root: Path) -> RepositoryContext:
            return RepositoryContext(
                file_tree="src/\n  main.py\n",
                files=[RepositoryFile(path="src/main.py", content="print('hi')\n")],
            )

    loader: RepositoryContextLoader = _PopulatingLoader()
    context = asyncio.run(loader.load(Path("/example/repository")))
    assert isinstance(context, RepositoryContext)
    assert context.files[0].path == "src/main.py"
