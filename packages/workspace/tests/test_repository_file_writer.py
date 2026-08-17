"""Contract/type tests for the workspace repository file-writing contract.

These exercise the generic ``RepositoryFileWriter`` Protocol with a test-local
fake using structural typing only. There is no concrete writer yet, so these
tests perform no filesystem writes, no directory creation, and no Git
operations; a plain synthetic ``Path`` handed to the fake is sufficient. Async
calls are driven with ``asyncio.run`` to match the repository convention of not
adding a pytest asyncio plugin.
"""

import asyncio
from pathlib import Path

from llmforeman_core import RepositoryFile
from llmforeman_workspace import RepositoryFileWriter


class FakeRepositoryFileWriter:
    """Test-local structural implementation of ``RepositoryFileWriter``.

    Echoes the requested logical ``path`` and ``content`` back as a
    ``RepositoryFile`` and performs no filesystem access; it exercises only the
    typed contract.
    """

    async def write(
        self,
        repository_root: Path,
        path: str,
        content: str,
    ) -> RepositoryFile:
        return RepositoryFile(
            path=path,
            content=content,
        )


def test_fake_structurally_satisfies_protocol() -> None:
    # Assigning to the Protocol-typed name is the static structural check;
    # mypy rejects a fake whose signature is not
    # ``(Path, str, str) -> RepositoryFile``.
    writer: RepositoryFileWriter = FakeRepositoryFileWriter()
    assert writer is not None


def test_async_write_returns_repository_file() -> None:
    writer: RepositoryFileWriter = FakeRepositoryFileWriter()

    async def run() -> RepositoryFile:
        # The repository path need not exist; it is merely input to the fake.
        return await writer.write(
            Path("/example/repository"),
            "src/example.py",
            "print('hello')\n",
        )

    written = asyncio.run(run())
    assert isinstance(written, RepositoryFile)


def test_logical_path_is_preserved() -> None:
    writer: RepositoryFileWriter = FakeRepositoryFileWriter()
    written = asyncio.run(
        writer.write(
            Path("/example/repository"),
            "packages/core/tests/test_models.py",
            "def test_models() -> None:\n    ...\n",
        )
    )
    # Preservation by the fake/result only; the Protocol does not runtime-
    # validate the path.
    assert written.path == "packages/core/tests/test_models.py"


def test_content_is_preserved_exactly() -> None:
    writer: RepositoryFileWriter = FakeRepositoryFileWriter()
    content = "def f():\n    x = 1  \n\treturn x\n"
    written = asyncio.run(
        writer.write(
            Path("/example/repository"),
            "src/example.py",
            content,
        )
    )
    # Indentation, mixed tabs/spaces, trailing whitespace, and line endings are
    # preserved verbatim; the contract implies no normalization.
    assert written.content == content


def test_empty_content_is_valid() -> None:
    writer: RepositoryFileWriter = FakeRepositoryFileWriter()
    written = asyncio.run(
        writer.write(
            Path("/example/repository"),
            "src/empty.py",
            "",
        )
    )
    assert written == RepositoryFile(path="src/empty.py", content="")


def test_new_untracked_style_path_needs_no_precondition() -> None:
    writer: RepositoryFileWriter = FakeRepositoryFileWriter()
    # A logical path for a file that need not already exist or be Git-tracked;
    # this protects the semantic difference from the Git-tracked reader. No file
    # is created on disk.
    written = asyncio.run(
        writer.write(
            Path("/example/repository"),
            "packages/foo/tests/test_new_feature.py",
            "def test_new_feature() -> None:\n    ...\n",
        )
    )
    assert written.path == "packages/foo/tests/test_new_feature.py"


def test_result_is_core_repository_file() -> None:
    writer: RepositoryFileWriter = FakeRepositoryFileWriter()
    written = asyncio.run(
        writer.write(
            Path("/example/repository"),
            "src/example.py",
            "print('hi')\n",
        )
    )
    assert type(written) is RepositoryFile
