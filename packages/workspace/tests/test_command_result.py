"""Model tests for the workspace-owned ``CommandResult``.

These exercise the ``CommandResult`` result model only. They perform no
subprocess execution, no shell invocation, and no filesystem/Git access; a
non-zero (or negative) exit code is a normal value, not an exception.
"""

import pytest
from pydantic import ValidationError

from llmforeman_workspace import CommandResult


def test_successful_result_preserves_all_fields() -> None:
    result = CommandResult(
        command=["uv", "run", "pytest", "packages/core"],
        exit_code=0,
        stdout="41 passed\n",
        stderr="",
    )
    assert result.command == ["uv", "run", "pytest", "packages/core"]
    assert result.exit_code == 0
    assert result.stdout == "41 passed\n"
    assert result.stderr == ""


def test_non_zero_exit_code_is_valid() -> None:
    # A tool completing and reporting failure through its exit status is a
    # normal result; construction must not raise.
    result = CommandResult(
        command=["uv", "run", "pytest"],
        exit_code=1,
        stdout="2 failed\n",
        stderr="",
    )
    assert result.exit_code == 1


def test_negative_exit_code_is_valid() -> None:
    # Negative values (for example signal-based termination) are not constrained
    # away by the model.
    result = CommandResult(
        command=["example"],
        exit_code=-15,
        stdout="",
        stderr="",
    )
    assert result.exit_code == -15


def test_empty_stdout_and_stderr_are_valid() -> None:
    result = CommandResult(
        command=["ruff", "check", "."],
        exit_code=0,
        stdout="",
        stderr="",
    )
    assert result.stdout == ""
    assert result.stderr == ""


def test_stdout_and_stderr_remain_separate() -> None:
    result = CommandResult(
        command=["mypy", "packages/core"],
        exit_code=0,
        stdout="Success: no issues found\n",
        stderr="note: some note\n",
    )
    # The two streams are stored independently and never merged.
    assert result.stdout == "Success: no issues found\n"
    assert result.stderr == "note: some note\n"


def test_command_order_and_contents_preserved_exactly() -> None:
    argv = ["uv", "run", "pytest", "-q", "packages/core"]
    result = CommandResult(
        command=argv,
        exit_code=0,
        stdout="",
        stderr="",
    )
    # Exact list order/contents preserved; no joining, quoting, or reordering.
    assert result.command == ["uv", "run", "pytest", "-q", "packages/core"]


def test_empty_command_is_rejected() -> None:
    with pytest.raises(ValidationError):
        CommandResult(
            command=[],
            exit_code=0,
            stdout="",
            stderr="",
        )


def test_empty_string_argv_entry_is_rejected() -> None:
    with pytest.raises(ValidationError):
        CommandResult(
            command=["uv", ""],
            exit_code=0,
            stdout="",
            stderr="",
        )


def test_argv_whitespace_is_preserved_not_normalized() -> None:
    # Arguments containing spaces (including whitespace-only entries) are
    # legitimate and must be retained verbatim; the model never strips them.
    argv = ["tool", "argument with spaces", "  preserved  "]
    result = CommandResult(
        command=argv,
        exit_code=0,
        stdout="",
        stderr="",
    )
    assert result.command == ["tool", "argument with spaces", "  preserved  "]


def test_shell_like_tokens_are_ordinary_argv_values() -> None:
    # Strings such as "|", ">", "&&", "$HOME" carry no shell semantics; they are
    # stored as plain argv values.
    argv = ["echo-like", "|", ">", "&&", "$HOME"]
    result = CommandResult(
        command=argv,
        exit_code=0,
        stdout="",
        stderr="",
    )
    assert result.command == ["echo-like", "|", ">", "&&", "$HOME"]


def test_serialization_keeps_command_as_list() -> None:
    result = CommandResult(
        command=["uv", "run", "pytest"],
        exit_code=1,
        stdout="...",
        stderr="",
    )
    dumped = result.model_dump()
    assert dumped == {
        "command": ["uv", "run", "pytest"],
        "exit_code": 1,
        "stdout": "...",
        "stderr": "",
    }
    # command must remain a list, never a joined shell string.
    assert isinstance(dumped["command"], list)
