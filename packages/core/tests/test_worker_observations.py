"""Behavioral tests for the typed v0.1 worker-observation vocabulary.

These tests exercise pure domain/orchestration information-message validation
only. They perform no filesystem, Git, subprocess, network, or model access;
path validation is filesystem-free and OS-independent.
"""

from typing import get_args

import pytest
from pydantic import BaseModel, TypeAdapter, ValidationError

from llmforeman_core import (
    ActionErrorObservation,
    ReadObservation,
    RunObservation,
    SearchObservation,
    WorkerObservation,
    WorkerSearchMatch,
    WriteObservation,
)

# Shared invalid repository-relative paths, mirroring the worker-action suite.
_INVALID_PATHS = [
    "",
    "   ",
    "/etc/passwd",
    "C:\\Users\\example\\secret.txt",
    "../secret.py",
    "src/../../secret.py",
    "path\x00name",
]


# --- WorkerSearchMatch ----------------------------------------------------


def test_worker_search_match_valid() -> None:
    match = WorkerSearchMatch(
        path="packages/core/src/example.py",
        line_number=42,
        line="class Example:",
    )
    assert match.path == "packages/core/src/example.py"
    assert match.line_number == 42
    assert match.line == "class Example:"


@pytest.mark.parametrize("bad_path", _INVALID_PATHS)
def test_worker_search_match_path_privacy(bad_path: str) -> None:
    with pytest.raises(ValidationError):
        WorkerSearchMatch(path=bad_path, line_number=1, line="x")


@pytest.mark.parametrize("line_number", [1, 42])
def test_worker_search_match_line_number_valid(line_number: int) -> None:
    match = WorkerSearchMatch(path="src/a.py", line_number=line_number, line="x")
    assert match.line_number == line_number


@pytest.mark.parametrize("line_number", [0, -1])
def test_worker_search_match_line_number_invalid(line_number: int) -> None:
    with pytest.raises(ValidationError):
        WorkerSearchMatch(path="src/a.py", line_number=line_number, line="x")


def test_worker_search_match_line_preserved_exactly() -> None:
    line = "    if value == 'x':   \t"
    match = WorkerSearchMatch(path="src/a.py", line_number=3, line=line)
    assert match.line == line


def test_worker_search_match_extra_forbidden() -> None:
    with pytest.raises(ValidationError):
        WorkerSearchMatch.model_validate(
            {"path": "src/a.py", "line_number": 1, "line": "x", "column": 2}
        )


# --- SearchObservation ----------------------------------------------------


def test_search_observation_preserves_query_and_match_order() -> None:
    # Deliberately non-sorted matches; order must be preserved verbatim.
    matches = [
        WorkerSearchMatch(path="src/z.py", line_number=99, line="z"),
        WorkerSearchMatch(path="src/a.py", line_number=1, line="a"),
        WorkerSearchMatch(path="src/m.py", line_number=50, line="m"),
    ]
    observation = SearchObservation(
        observation="search", query="Example", matches=matches
    )
    assert observation.query == "Example"
    assert observation.matches == matches
    assert [m.path for m in observation.matches] == ["src/z.py", "src/a.py", "src/m.py"]


def test_search_observation_empty_matches_valid() -> None:
    observation = SearchObservation(
        observation="search", query="MissingSymbol", matches=[]
    )
    assert observation.matches == []


@pytest.mark.parametrize("bad_query", ["", "   ", "\t\n", "Retry\x00Policy"])
def test_search_observation_query_invalid(bad_query: str) -> None:
    with pytest.raises(ValidationError):
        SearchObservation(observation="search", query=bad_query, matches=[])


def test_search_observation_query_whitespace_preserved() -> None:
    observation = SearchObservation(
        observation="search", query="  spaced  ", matches=[]
    )
    assert observation.query == "  spaced  "


# --- ReadObservation ------------------------------------------------------


def test_read_observation_valid() -> None:
    observation = ReadObservation(
        observation="read", path="src/example.py", content="print('hi')\n"
    )
    assert observation.path == "src/example.py"
    assert observation.content == "print('hi')\n"


def test_read_observation_empty_content_valid() -> None:
    observation = ReadObservation(observation="read", path="src/empty.py", content="")
    assert observation.content == ""


def test_read_observation_content_preserved_exactly() -> None:
    content = "  line1\r\nline2\t  \n\n"
    observation = ReadObservation(observation="read", path="src/a.py", content=content)
    assert observation.content == content


@pytest.mark.parametrize("bad_path", _INVALID_PATHS)
def test_read_observation_path_privacy(bad_path: str) -> None:
    with pytest.raises(ValidationError):
        ReadObservation(observation="read", path=bad_path, content="x")


# --- WriteObservation -----------------------------------------------------


def test_write_observation_valid() -> None:
    observation = WriteObservation(observation="write", path="src/example.py")
    assert observation.path == "src/example.py"


def test_write_observation_rejects_content_echo() -> None:
    # Successful write intentionally exposes only observation + path; echoing
    # content back is a forbidden extra field.
    with pytest.raises(ValidationError):
        WriteObservation.model_validate(
            {"observation": "write", "path": "src/foo.py", "content": "unexpected"}
        )


@pytest.mark.parametrize("bad_path", _INVALID_PATHS)
def test_write_observation_path_privacy(bad_path: str) -> None:
    with pytest.raises(ValidationError):
        WriteObservation(observation="write", path=bad_path)


# --- RunObservation -------------------------------------------------------


def test_run_observation_success() -> None:
    observation = RunObservation(
        observation="run",
        command=["uv", "run", "pytest"],
        exit_code=0,
        stdout="10 passed\n",
        stderr="",
    )
    assert observation.command == ["uv", "run", "pytest"]
    assert observation.exit_code == 0
    assert observation.stdout == "10 passed\n"
    assert observation.stderr == ""


@pytest.mark.parametrize("exit_code", [0, 1, 7, -15])
def test_run_observation_exit_codes_valid(exit_code: int) -> None:
    observation = RunObservation(
        observation="run",
        command=["tool"],
        exit_code=exit_code,
        stdout="",
        stderr="",
    )
    assert observation.exit_code == exit_code


def test_run_observation_non_zero_is_not_error_variant() -> None:
    observation = WorkerObservation.model_validate(
        {
            "observation": "run",
            "command": ["uv", "run", "pytest"],
            "exit_code": 1,
            "stdout": "FAILED ...",
            "stderr": "",
        }
    )
    assert isinstance(observation.root, RunObservation)
    assert observation.root.exit_code == 1


def test_run_observation_output_preserved_exactly() -> None:
    stdout = "  line1\nline2\t\n\x1b[31mred\x1b[0m\n   "
    stderr = "warning:\r\n  detail  \n"
    observation = RunObservation(
        observation="run",
        command=["tool"],
        exit_code=2,
        stdout=stdout,
        stderr=stderr,
    )
    assert observation.stdout == stdout
    assert observation.stderr == stderr


@pytest.mark.parametrize(
    "bad_command",
    [[], [""], ["   "], ["uv", ""], ["uv", "foo\x00bar"]],
)
def test_run_observation_command_invalid(bad_command: list[str]) -> None:
    with pytest.raises(ValidationError):
        RunObservation(
            observation="run",
            command=bad_command,
            exit_code=0,
            stdout="",
            stderr="",
        )


def test_run_observation_shell_metacharacters_remain_data() -> None:
    argv = ["tool", "argument with spaces", "*", "&&", "$HOME", ";", "  preserved  "]
    observation = RunObservation(
        observation="run", command=argv, exit_code=0, stdout="", stderr=""
    )
    assert observation.command == argv


# --- ActionErrorObservation -----------------------------------------------


@pytest.mark.parametrize("action", ["search", "read", "write", "run"])
def test_action_error_observation_valid(action: str) -> None:
    observation = ActionErrorObservation.model_validate(
        {"observation": "error", "action": action, "message": "Something safe happened."}
    )
    assert observation.action == action
    assert observation.message == "Something safe happened."


def test_action_error_finish_rejected() -> None:
    with pytest.raises(ValidationError):
        ActionErrorObservation.model_validate(
            {"observation": "error", "action": "finish", "message": "nope"}
        )


@pytest.mark.parametrize("bad_message", ["", "   ", "\t\n", "boom\x00now"])
def test_action_error_message_invalid(bad_message: str) -> None:
    with pytest.raises(ValidationError):
        ActionErrorObservation(
            observation="error", action="read", message=bad_message
        )


def test_action_error_message_preserved_exactly() -> None:
    message = "  Requested file does not exist.  "
    observation = ActionErrorObservation(
        observation="error", action="read", message=message
    )
    assert observation.message == message


# --- Extra-field policy ---------------------------------------------------


def test_read_observation_extra_field_forbidden() -> None:
    with pytest.raises(ValidationError):
        ReadObservation.model_validate(
            {
                "observation": "read",
                "path": "src/foo.py",
                "content": "...",
                "absolute_path": "/tmp/repo/src/foo.py",
            }
        )


def test_action_error_traceback_field_forbidden() -> None:
    with pytest.raises(ValidationError):
        ActionErrorObservation.model_validate(
            {
                "observation": "error",
                "action": "read",
                "message": "failed",
                "traceback": "Traceback (most recent call last): ...",
            }
        )


# --- WorkerObservation root parsing / discriminated union -----------------


def test_root_parsing_all_five_variants() -> None:
    cases = [
        (
            {"observation": "search", "query": "x", "matches": []},
            SearchObservation,
        ),
        (
            {"observation": "read", "path": "src/a.py", "content": "c"},
            ReadObservation,
        ),
        ({"observation": "write", "path": "src/a.py"}, WriteObservation),
        (
            {
                "observation": "run",
                "command": ["uv"],
                "exit_code": 0,
                "stdout": "",
                "stderr": "",
            },
            RunObservation,
        ),
        (
            {"observation": "error", "action": "read", "message": "boom"},
            ActionErrorObservation,
        ),
    ]
    for payload, expected_type in cases:
        observation = WorkerObservation.model_validate(payload)
        assert isinstance(observation.root, expected_type)


@pytest.mark.parametrize(
    "payload",
    [
        {"observation": "finish"},
        {"observation": "whatever", "path": "src/foo.py"},
    ],
)
def test_unknown_observation_rejected(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        WorkerObservation.model_validate(payload)


def test_missing_discriminator_rejected() -> None:
    with pytest.raises(ValidationError):
        WorkerObservation.model_validate({"path": "src/foo.py", "content": "..."})


@pytest.mark.parametrize(
    "payload",
    [
        {"observation": "search", "matches": []},
        {"observation": "read", "path": "src/foo.py"},
        {"observation": "write"},
        {"observation": "run", "command": ["pytest"], "stdout": "", "stderr": ""},
        {"observation": "error", "action": "read"},
    ],
)
def test_missing_variant_fields_rejected(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        WorkerObservation.model_validate(payload)


# --- Flat serialization / round trip --------------------------------------


_ALL_VARIANT_PAYLOADS: list[dict[str, object]] = [
    {"observation": "search", "query": "x", "matches": []},
    {"observation": "read", "path": "src/a.py", "content": "c\nd\n"},
    {"observation": "write", "path": "src/a.py"},
    {
        "observation": "run",
        "command": ["uv", "run"],
        "exit_code": 1,
        "stdout": "out",
        "stderr": "err",
    },
    {"observation": "error", "action": "run", "message": "boom"},
]


@pytest.mark.parametrize("payload", _ALL_VARIANT_PAYLOADS)
def test_flat_serialization_dump(payload: dict[str, object]) -> None:
    observation = WorkerObservation.model_validate(payload)
    dumped = observation.model_dump()
    assert "root" not in dumped
    assert dumped == payload


def test_flat_serialization_json_has_no_root_wrapper() -> None:
    observation = WorkerObservation.model_validate(
        {"observation": "write", "path": "src/foo.py"}
    )
    dumped = observation.model_dump_json()
    assert '"root"' not in dumped
    assert '"observation":"write"' in dumped


@pytest.mark.parametrize("payload", _ALL_VARIANT_PAYLOADS)
def test_json_round_trip(payload: dict[str, object]) -> None:
    observation = WorkerObservation.model_validate(payload)
    json_text = observation.model_dump_json()
    restored = WorkerObservation.model_validate_json(json_text)
    assert type(restored.root) is type(observation.root)
    assert restored.model_dump() == payload


# --- JSON schema ----------------------------------------------------------


def test_json_schema_is_closed_five_variant_discriminated_union() -> None:
    schema = WorkerObservation.model_json_schema()
    schema_text = repr(schema)
    assert "observation" in schema_text
    for literal in ("search", "read", "write", "run", "error"):
        assert literal in schema_text
    for name in (
        "SearchObservation",
        "ReadObservation",
        "WriteObservation",
        "RunObservation",
        "ActionErrorObservation",
    ):
        assert name in schema_text
    assert "FinishObservation" not in schema_text


# --- BaseModel / closed union typing --------------------------------------


def test_worker_observation_is_basemodel_subclass() -> None:
    assert issubclass(WorkerObservation, BaseModel)


def test_worker_observation_root_is_closed_concrete_union() -> None:
    root_hint = WorkerObservation.model_fields["root"].annotation
    assert set(get_args(root_hint)) == {
        SearchObservation,
        ReadObservation,
        WriteObservation,
        RunObservation,
        ActionErrorObservation,
    }


def test_type_adapter_validates_flat_union_directly() -> None:
    adapter: TypeAdapter[ReadObservation] = TypeAdapter(ReadObservation)
    parsed = adapter.validate_python(
        {"observation": "read", "path": "src/a.py", "content": "c"}
    )
    assert isinstance(parsed, ReadObservation)
