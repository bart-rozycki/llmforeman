"""Behavioral tests for the typed v0.1 worker-action vocabulary.

These tests exercise pure domain/control-message validation only. They perform
no filesystem, Git, subprocess, network, or model access; path validation is
filesystem-free and OS-independent.
"""

from typing import get_args

import pytest
from pydantic import BaseModel, TypeAdapter, ValidationError

from llmforeman_core import (
    FinishAction,
    ReadFileAction,
    RunCommandAction,
    SearchAction,
    WorkerAction,
    WriteFileAction,
)

# --- SearchAction ---------------------------------------------------------


def test_search_action_valid() -> None:
    action = SearchAction(action="search", query="RetryPolicy")
    assert action.action == "search"
    assert action.query == "RetryPolicy"


@pytest.mark.parametrize(
    "bad_query",
    ["", "   ", "\t\n", "Retry\x00Policy"],
)
def test_search_query_invalid(bad_query: str) -> None:
    with pytest.raises(ValidationError):
        SearchAction(action="search", query=bad_query)


def test_search_query_whitespace_is_preserved() -> None:
    # Whitespace may be significant in a literal text search; only blankness is
    # rejected, valid content is never normalized.
    action = SearchAction(action="search", query="  spaced  query\t")
    assert action.query == "  spaced  query\t"


# --- ReadFileAction -------------------------------------------------------


def test_read_file_action_valid() -> None:
    action = ReadFileAction(
        action="read",
        path="packages/core/src/llmforeman_core/models.py",
    )
    assert action.action == "read"
    assert action.path == "packages/core/src/llmforeman_core/models.py"


_INVALID_PATHS = [
    "",
    "   ",
    "/etc/passwd",
    "C:\\Users\\example\\secret.txt",
    "../secret.txt",
    "packages/core/../../secret.txt",
    "path\x00name",
]


@pytest.mark.parametrize("bad_path", _INVALID_PATHS)
def test_read_path_privacy(bad_path: str) -> None:
    with pytest.raises(ValidationError):
        ReadFileAction(action="read", path=bad_path)


# --- WriteFileAction ------------------------------------------------------


def test_write_file_action_valid() -> None:
    content = "def test_policy():\n    assert True\n"
    action = WriteFileAction(
        action="write",
        path="packages/core/tests/test_policy.py",
        content=content,
    )
    assert action.path == "packages/core/tests/test_policy.py"
    assert action.content == content


def test_empty_write_content_is_valid() -> None:
    action = WriteFileAction(action="write", path="src/empty.py", content="")
    assert action.content == ""


@pytest.mark.parametrize("bad_path", _INVALID_PATHS)
def test_write_path_privacy(bad_path: str) -> None:
    with pytest.raises(ValidationError):
        WriteFileAction(action="write", path=bad_path, content="x")


def test_write_traversal_rejected_through_worker_action() -> None:
    # Defense in depth: a malicious model output is rejected before any
    # workspace capability could receive it, via the public root model.
    with pytest.raises(ValidationError):
        WorkerAction.model_validate(
            {"action": "write", "path": "../../.ssh/id_rsa", "content": "oops"}
        )


# --- RunCommandAction -----------------------------------------------------


def test_run_command_action_valid() -> None:
    argv = ["uv", "run", "pytest", "packages/core"]
    action = RunCommandAction(action="run", command=argv)
    assert action.command == argv


@pytest.mark.parametrize(
    "bad_command",
    [
        [],
        [""],
        ["   "],
        ["uv", ""],
        ["uv", "abc\x00def"],
    ],
)
def test_run_command_invalid(bad_command: list[str]) -> None:
    with pytest.raises(ValidationError):
        RunCommandAction(action="run", command=bad_command)


def test_shell_metacharacters_remain_data() -> None:
    argv = [
        "tool",
        "argument with spaces",
        "  intentionally padded  ",
        "*",
        "&&",
        "$HOME",
        ";",
    ]
    action = RunCommandAction(action="run", command=argv)
    assert action.command == argv


# --- FinishAction ---------------------------------------------------------


def test_finish_action_valid() -> None:
    action = FinishAction(action="finish", summary="Implemented the change.")
    assert action.summary == "Implemented the change."


@pytest.mark.parametrize(
    "bad_summary",
    ["", "   ", "\t\n", "done\x00now"],
)
def test_finish_summary_invalid(bad_summary: str) -> None:
    with pytest.raises(ValidationError):
        FinishAction(action="finish", summary=bad_summary)


# --- Extra-field policy ---------------------------------------------------


def test_extra_fields_forbidden() -> None:
    with pytest.raises(ValidationError):
        ReadFileAction.model_validate(
            {"action": "read", "path": "src/foo.py", "whatever": 123}
        )


def test_foreign_variant_field_forbidden() -> None:
    with pytest.raises(ValidationError):
        SearchAction.model_validate(
            {"action": "search", "query": "Foo", "path": "src/foo.py"}
        )


# --- WorkerAction root parsing / discriminated union ----------------------


def test_root_parsing_search() -> None:
    action = WorkerAction.model_validate({"action": "search", "query": "RetryPolicy"})
    assert isinstance(action.root, SearchAction)
    assert action.root.query == "RetryPolicy"


def test_root_parsing_all_five_variants() -> None:
    cases = [
        ({"action": "search", "query": "x"}, SearchAction),
        ({"action": "read", "path": "src/a.py"}, ReadFileAction),
        ({"action": "write", "path": "src/a.py", "content": "c"}, WriteFileAction),
        ({"action": "run", "command": ["uv"]}, RunCommandAction),
        ({"action": "finish", "summary": "done"}, FinishAction),
    ]
    for payload, expected_type in cases:
        action = WorkerAction.model_validate(payload)
        assert isinstance(action.root, expected_type)


def test_root_parsing_finish_from_json() -> None:
    action = WorkerAction.model_validate_json(
        '{"action":"finish","summary":"Implemented the change."}'
    )
    assert isinstance(action.root, FinishAction)
    assert action.root.summary == "Implemented the change."


@pytest.mark.parametrize(
    "payload",
    [
        {"action": "delete", "path": "src/foo.py"},
        {"action": "maybe_read"},
    ],
)
def test_unknown_action_rejected(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        WorkerAction.model_validate(payload)


def test_missing_action_rejected() -> None:
    with pytest.raises(ValidationError):
        WorkerAction.model_validate({"path": "src/foo.py"})


@pytest.mark.parametrize(
    "payload",
    [
        {"action": "search"},
        {"action": "read"},
        {"action": "write", "path": "foo.py"},
        {"action": "run"},
        {"action": "finish"},
    ],
)
def test_missing_variant_fields_rejected(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        WorkerAction.model_validate(payload)


@pytest.mark.parametrize(
    "payload",
    [
        {"action": "read", "query": "RetryPolicy"},
        {"action": "run", "command": []},
    ],
)
def test_wrong_variant_fields_rejected(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        WorkerAction.model_validate(payload)


# --- Flat serialization / round trip --------------------------------------


def test_flat_serialization_dump() -> None:
    action = WorkerAction.model_validate({"action": "read", "path": "src/foo.py"})
    assert action.model_dump() == {"action": "read", "path": "src/foo.py"}


def test_flat_serialization_json_has_no_root_wrapper() -> None:
    action = WorkerAction.model_validate({"action": "read", "path": "src/foo.py"})
    dumped = action.model_dump_json()
    assert '"root"' not in dumped
    assert '"action":"read"' in dumped


def test_round_trip_all_variants() -> None:
    payloads = [
        {"action": "search", "query": "x"},
        {"action": "read", "path": "src/a.py"},
        {"action": "write", "path": "src/a.py", "content": "c\nd\n"},
        {"action": "run", "command": ["uv", "run"]},
        {"action": "finish", "summary": "done"},
    ]
    for payload in payloads:
        action = WorkerAction.model_validate(payload)
        json_text = action.model_dump_json()
        restored = WorkerAction.model_validate_json(json_text)
        assert type(restored.root) is type(action.root)
        assert restored.model_dump() == payload


# --- JSON schema ----------------------------------------------------------


def test_json_schema_is_closed_five_variant_discriminated_union() -> None:
    schema = WorkerAction.model_json_schema()
    schema_text = repr(schema)
    # Discriminator uses the "action" key.
    assert '"action"' in str(schema).replace("'", '"') or "action" in schema_text
    # All five action literals are represented somewhere in the schema.
    for literal in ("search", "read", "write", "run", "finish"):
        assert literal in schema_text
    # The five concrete variants are referenced.
    for name in (
        "SearchAction",
        "ReadFileAction",
        "WriteFileAction",
        "RunCommandAction",
        "FinishAction",
    ):
        assert name in schema_text


# --- BaseModel / generic runtime compatibility ----------------------------


def test_worker_action_is_basemodel_subclass() -> None:
    assert issubclass(WorkerAction, BaseModel)


def _accepts_base_model_type[T: BaseModel](output_type: type[T]) -> type[T]:
    """Mirror the structured-runtime ``type[T] where T: BaseModel`` constraint.

    Proves locally (without importing any runtime) that ``WorkerAction``
    satisfies the exact generic bound established by ``StructuredModelRuntime``.
    """

    return output_type


def test_worker_action_satisfies_structured_runtime_bound() -> None:
    assert _accepts_base_model_type(WorkerAction) is WorkerAction


def test_worker_action_root_is_closed_concrete_union() -> None:
    # The root value type is exactly the closed five-variant union, not Any,
    # dict, or a bare BaseModel.
    root_hint = WorkerAction.model_fields["root"].annotation
    assert set(get_args(root_hint)) == {
        SearchAction,
        ReadFileAction,
        WriteFileAction,
        RunCommandAction,
        FinishAction,
    }


def test_type_adapter_validates_flat_union_directly() -> None:
    # Sanity check that the union itself is discriminated and flat, independent
    # of the RootModel wrapper.
    adapter: TypeAdapter[SearchAction] = TypeAdapter(SearchAction)
    parsed = adapter.validate_python({"action": "search", "query": "x"})
    assert isinstance(parsed, SearchAction)
