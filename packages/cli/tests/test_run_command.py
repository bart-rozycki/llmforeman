"""Deterministic tests for the ``llmforeman run`` command.

These tests never touch a real Ollama server, Git repository, ripgrep,
subprocess, network, or the filesystem repository state. Composition is
exercised only through the single narrow private seam
``llmforeman_cli._cli._run_local_worker``; approval is exercised through the
private ``_approve_command`` helper with controlled terminal input.
"""

from __future__ import annotations

import builtins
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import llmforeman_cli
from llmforeman_cli import _cli
from llmforeman_core import ModelUsage, RunCommandAction
from llmforeman_orchestration import (
    LocalWorkerResult,
    WorkerActionDeniedError,
    WorkerStepLimitError,
)
from llmforeman_runtimes import ModelRuntimeError
from llmforeman_workspace import InvalidRepositoryError


def _install_capturing_seam(
    monkeypatch: pytest.MonkeyPatch,
    *,
    result: LocalWorkerResult | None = None,
    error: BaseException | None = None,
) -> dict[str, Any]:
    """Replace the composition seam with a recorder that returns/raises.

    Returns a mutable dict that captures the keyword arguments the seam
    received, so tests can assert exactly what reached composition without
    constructing any real component.
    """
    captured: dict[str, Any] = {}

    async def fake_seam(
        *,
        repository_root: Path,
        instruction: str,
        model: str | None,
    ) -> LocalWorkerResult:
        captured["repository_root"] = repository_root
        captured["instruction"] = instruction
        captured["model"] = model
        if error is not None:
            raise error
        assert result is not None
        return result

    monkeypatch.setattr(_cli, "_run_local_worker", fake_seam)
    return captured


def _sample_result() -> LocalWorkerResult:
    return LocalWorkerResult(
        summary="Implemented validation.",
        steps=8,
        usage=ModelUsage(
            input_tokens=4312,
            output_tokens=687,
            cache_read_input_tokens=100,
            cache_creation_input_tokens=50,
        ),
    )


# --- Parsing / options -----------------------------------------------------


def test_run_parses_instruction_and_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _install_capturing_seam(monkeypatch, result=_sample_result())

    exit_code = llmforeman_cli.main(["run", "Implement feature"])

    assert exit_code == 0
    assert captured["instruction"] == "Implement feature"
    assert captured["model"] is None


def test_run_defaults_repo_to_cwd(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured = _install_capturing_seam(monkeypatch, result=_sample_result())
    monkeypatch.chdir(tmp_path)

    llmforeman_cli.main(["run", "Implement feature"])

    assert captured["repository_root"] == Path.cwd()
    assert captured["repository_root"] == tmp_path


def test_run_explicit_repo(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _install_capturing_seam(monkeypatch, result=_sample_result())

    llmforeman_cli.main(["run", "--repo", "/example/repo", "Implement feature"])

    assert captured["repository_root"] == Path("/example/repo")


def test_run_explicit_model(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _install_capturing_seam(monkeypatch, result=_sample_result())

    llmforeman_cli.main(["run", "--model", "qwen3.8:27b", "Implement feature"])

    assert captured["model"] == "qwen3.8:27b"


def test_run_missing_instruction_errors() -> None:
    with pytest.raises(SystemExit) as excinfo:
        llmforeman_cli.main(["run"])
    assert excinfo.value.code != 0


# --- Terminal approval callback -------------------------------------------


def _run_approval(
    monkeypatch: pytest.MonkeyPatch,
    responses: list[str] | Callable[[str], str],
    action: RunCommandAction | None = None,
) -> bool:
    import asyncio

    if action is None:
        action = RunCommandAction(action="run", command=["ls"])

    if callable(responses):
        monkeypatch.setattr(builtins, "input", responses)
    else:
        iterator = iter(responses)

        def fake_input(_prompt: str = "") -> str:
            return next(iterator)

        monkeypatch.setattr(builtins, "input", fake_input)

    return asyncio.run(_cli._approve_command(action))


@pytest.mark.parametrize("response", ["y", "Y", "yes", "YES", " Yes"])
def test_approval_accepts(monkeypatch: pytest.MonkeyPatch, response: str) -> None:
    assert _run_approval(monkeypatch, [response]) is True


@pytest.mark.parametrize("response", ["", "n", "N", "no", "No", "whatever", "1", "true"])
def test_approval_denies(monkeypatch: pytest.MonkeyPatch, response: str) -> None:
    assert _run_approval(monkeypatch, [response]) is False


def test_approval_eof_denies(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_eof(_prompt: str = "") -> str:
        raise EOFError

    assert _run_approval(monkeypatch, raise_eof) is False


def test_approval_keyboard_interrupt_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_interrupt(_prompt: str = "") -> str:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        _run_approval(monkeypatch, raise_interrupt)


def test_approval_display_uses_readable_command(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    action = RunCommandAction(
        action="run", command=["uv", "run", "pytest", "packages/core"]
    )
    _run_approval(monkeypatch, ["n"], action=action)
    out = capsys.readouterr().out
    assert "uv run pytest packages/core" in out


def test_approval_shell_metacharacters_are_display_only(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    command = ["tool", "*", "&&", "$HOME", "argument with spaces"]
    action = RunCommandAction(action="run", command=list(command))

    result = _run_approval(monkeypatch, ["n"], action=action)

    assert result is False
    # Original argv is unchanged data; the CLI never mutates it.
    assert action.command == command
    out = capsys.readouterr().out
    # Presentation uses shlex.join-style quoting for spaces and specials.
    assert "'argument with spaces'" in out


def test_each_approval_is_independent(monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio

    action = RunCommandAction(action="run", command=["ls"])
    responses = iter(["y", "n"])
    calls: list[str] = []

    def fake_input(_prompt: str = "") -> str:
        value = next(responses)
        calls.append(value)
        return value

    monkeypatch.setattr(builtins, "input", fake_input)

    first = asyncio.run(_cli._approve_command(action))
    second = asyncio.run(_cli._approve_command(action))

    assert first is True
    assert second is False
    assert calls == ["y", "n"]


# --- Result / error output -------------------------------------------------


def test_successful_result_output(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _install_capturing_seam(monkeypatch, result=_sample_result())

    exit_code = llmforeman_cli.main(["run", "Implement feature"])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "Completed." in out
    assert "Steps: 8" in out
    assert "Input tokens: 4,312" in out
    assert "Output tokens: 687" in out
    assert "Cache read input tokens: 100" in out
    assert "Cache creation input tokens: 50" in out
    assert "Summary:" in out
    assert "Implemented validation." in out
    # No fabricated cost.
    assert "$" not in out


def test_command_denied_ux(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _install_capturing_seam(
        monkeypatch, error=WorkerActionDeniedError("Command execution was denied.")
    )

    exit_code = llmforeman_cli.main(["run", "Implement feature"])
    out = capsys.readouterr().out

    assert exit_code != 0
    assert "Command denied. Worker stopped." in out
    assert "Traceback" not in out


def test_step_limit_ux(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _install_capturing_seam(
        monkeypatch, error=WorkerStepLimitError("Worker did not finish.")
    )

    exit_code = llmforeman_cli.main(["run", "Implement feature"])
    out = capsys.readouterr().out

    assert exit_code != 0
    assert "step limit" in out
    assert "Traceback" not in out


def test_invalid_repository_ux(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _install_capturing_seam(
        monkeypatch, error=InvalidRepositoryError("not a git working tree")
    )

    exit_code = llmforeman_cli.main(["run", "Implement feature"])
    out = capsys.readouterr().out

    assert exit_code != 0
    assert out.startswith("Error:") or "Error:" in out
    assert "Traceback" not in out


def test_runtime_failure_ux(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _install_capturing_seam(
        monkeypatch, error=ModelRuntimeError("Ollama request failed unexpectedly.")
    )

    exit_code = llmforeman_cli.main(["run", "Implement feature"])
    out = capsys.readouterr().out

    assert exit_code != 0
    assert "local model execution failed" in out
    assert "Traceback" not in out


def test_unexpected_error_is_not_hidden(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_capturing_seam(monkeypatch, error=RuntimeError("programming bug"))

    with pytest.raises(RuntimeError, match="programming bug"):
        llmforeman_cli.main(["run", "Implement feature"])


def test_ctrl_c_ux(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _install_capturing_seam(monkeypatch, error=KeyboardInterrupt())

    exit_code = llmforeman_cli.main(["run", "Implement feature"])
    out = capsys.readouterr().out

    assert exit_code == 130
    assert "Cancelled." in out
    assert "Traceback" not in out
