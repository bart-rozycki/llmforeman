"""Contract/type tests for the ``WorkspaceCommandRunner`` Protocol.

These exercise the async ``WorkspaceCommandRunner`` Protocol with a test-local
fake using structural typing only. There is no concrete runner yet, so these
tests spawn no process, invoke no shell, and perform no filesystem/Git access;
a synthetic ``Path`` handed to the fake is sufficient. Async calls are driven
with ``asyncio.run`` to match the repository convention of not adding a pytest
asyncio plugin.
"""

import asyncio
from pathlib import Path

from llmforeman_workspace import CommandResult, WorkspaceCommandRunner


class FakeWorkspaceCommandRunner:
    """Test-local structural implementation of ``WorkspaceCommandRunner``.

    Echoes the requested argv back in a ``CommandResult`` and performs no
    subprocess execution; it exercises only the typed contract.
    """

    async def run(
        self,
        repository_root: Path,
        command: list[str],
    ) -> CommandResult:
        return CommandResult(
            command=list(command),
            exit_code=0,
            stdout="ok",
            stderr="",
        )


def test_fake_structurally_satisfies_protocol() -> None:
    # Assigning to the Protocol-typed name is the static structural check;
    # mypy rejects a fake whose signature is not
    # ``(Path, list[str]) -> CommandResult``.
    runner: WorkspaceCommandRunner = FakeWorkspaceCommandRunner()
    assert runner is not None


def test_async_run_returns_command_result() -> None:
    runner: WorkspaceCommandRunner = FakeWorkspaceCommandRunner()

    async def run() -> CommandResult:
        # The path need not exist; it is merely input to the fake runner.
        return await runner.run(
            Path("/example/repository"),
            ["uv", "run", "pytest"],
        )

    result = asyncio.run(run())
    assert isinstance(result, CommandResult)
    assert result.exit_code == 0
    assert result.stdout == "ok"


def test_async_run_preserves_argv() -> None:
    runner: WorkspaceCommandRunner = FakeWorkspaceCommandRunner()
    result = asyncio.run(
        runner.run(
            Path("/example/repository"),
            ["uv", "run", "pytest", "-q", "packages/core"],
        )
    )
    assert result.command == ["uv", "run", "pytest", "-q", "packages/core"]
