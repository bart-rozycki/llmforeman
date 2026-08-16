"""Behavioral and typing tests for the Anthropic Foreman planning adapter.

Every test injects a small fake :class:`StructuredModelProvider`; no Anthropic
SDK is ever constructed and no live API call is performed. Async tests are
driven with ``asyncio.run`` to avoid a pytest asyncio plugin, matching the
existing provider test style.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable

import pytest
from pydantic import BaseModel

from llmforeman_core import (
    AgentRole,
    Foreman,
    ForemanPlanValidationError,
    ModelUsage,
    RepositoryContext,
    RepositoryFile,
    TaskPlan,
    TaskStatus,
)
from llmforeman_providers import (
    AnthropicForeman,
    ModelProviderPermanentError,
    ModelRequest,
    StructuredModelProvider,
    StructuredModelResponse,
)
from llmforeman_providers.foreman import (
    FOREMAN_SYSTEM_PROMPT,
    _ForemanPlanOutput,
    _ForemanTaskOutput,
)


def run[T](coro: Awaitable[T]) -> T:
    return asyncio.run(coro)  # type: ignore[arg-type]


class FakeStructuredProvider:
    """Records structured-generation calls and returns a configured result.

    Either returns a pre-built planning output or raises a configured provider
    error. It structurally satisfies ``StructuredModelProvider`` so the adapter
    can depend on the capability without any Anthropic SDK.
    """

    def __init__(
        self,
        *,
        output: BaseModel | None = None,
        error: Exception | None = None,
    ) -> None:
        self._output = output
        self._error = error
        self.calls: list[tuple[ModelRequest, type[BaseModel]]] = []

    @property
    def call_count(self) -> int:
        return len(self.calls)

    async def generate_structured[T: BaseModel](
        self,
        request: ModelRequest,
        output_type: type[T],
    ) -> StructuredModelResponse[T]:
        self.calls.append((request, output_type))
        if self._error is not None:
            raise self._error
        assert self._output is not None, "fake provider has no configured output"
        validated = output_type.model_validate(self._output.model_dump())
        return StructuredModelResponse(
            output=validated,
            usage=ModelUsage(input_tokens=0, output_tokens=0),
        )


def _task(
    task_id: str,
    *,
    title: str = "Title",
    description: str = "Description",
    role: AgentRole = AgentRole.DEVELOPER,
    dependencies: list[str] | None = None,
) -> _ForemanTaskOutput:
    return _ForemanTaskOutput(
        id=task_id,
        title=title,
        description=description,
        assigned_role=role,
        dependencies=dependencies or [],
    )


def _plan(*tasks: _ForemanTaskOutput) -> _ForemanPlanOutput:
    return _ForemanPlanOutput(tasks=list(tasks))


# --- Port compatibility ---------------------------------------------------


def test_anthropic_foreman_satisfies_foreman_protocol() -> None:
    # The annotation is the assertion: AnthropicForeman must structurally
    # satisfy Foreman for mypy to accept this under strict mode.
    provider: StructuredModelProvider = FakeStructuredProvider()
    foreman: Foreman = AnthropicForeman(provider)
    assert isinstance(foreman, AnthropicForeman)


# --- Valid objective mapping ---------------------------------------------


def test_valid_objective_maps_to_single_structured_call() -> None:
    provider = FakeStructuredProvider(output=_plan(_task("TASK-001")))
    foreman = AnthropicForeman(provider)

    run(foreman.create_plan("Add Retry-After validation"))

    assert provider.call_count == 1
    request, output_type = provider.calls[0]
    assert isinstance(request, ModelRequest)
    assert request.prompt == "Add Retry-After validation"
    assert request.system_prompt == FOREMAN_SYSTEM_PROMPT
    assert output_type is _ForemanPlanOutput


def test_no_context_prompt_has_no_repository_sections() -> None:
    provider = FakeStructuredProvider(output=_plan(_task("TASK-001")))
    foreman = AnthropicForeman(provider)

    run(foreman.create_plan("Add Retry-After validation"))

    request, _ = provider.calls[0]
    # None -> compact objective-only prompt: no artificial repository headings.
    assert request.prompt == "Add Retry-After validation"
    assert "Repository tree:" not in request.prompt
    assert "Repository files:" not in request.prompt


# --- Repository context formatting ---------------------------------------


def test_populated_context_is_formatted_into_user_prompt() -> None:
    provider = FakeStructuredProvider(output=_plan(_task("TASK-001")))
    foreman = AnthropicForeman(provider)
    context = RepositoryContext(
        file_tree="packages/\n  core/\n  providers/",
        files=[
            RepositoryFile(
                path="packages/core/src/llmforeman_core/models.py",
                content="CORE_CONTENT",
            ),
            RepositoryFile(
                path="packages/providers/src/llmforeman_providers/foreman.py",
                content="FOREMAN_CONTENT",
            ),
        ],
    )

    run(foreman.create_plan("Add Retry-After validation", repository_context=context))

    assert provider.call_count == 1
    request, _ = provider.calls[0]
    prompt = request.prompt

    # Objective is present and clearly labeled/bounded.
    assert "Engineering objective:" in prompt
    assert "Add Retry-After validation" in prompt
    # Repository tree label and exact tree content.
    assert "Repository tree:" in prompt
    assert "packages/\n  core/\n  providers/" in prompt
    # Repository files label and exact paths + contents.
    assert "Repository files:" in prompt
    assert "packages/core/src/llmforeman_core/models.py" in prompt
    assert "CORE_CONTENT" in prompt
    assert "packages/providers/src/llmforeman_providers/foreman.py" in prompt
    assert "FOREMAN_CONTENT" in prompt


def test_context_file_order_is_preserved() -> None:
    provider = FakeStructuredProvider(output=_plan(_task("TASK-001")))
    foreman = AnthropicForeman(provider)
    # Intentionally non-alphabetical order.
    context = RepositoryContext(
        file_tree="",
        files=[
            RepositoryFile(path="zeta.py", content="ZETA"),
            RepositoryFile(path="alpha.py", content="ALPHA"),
            RepositoryFile(path="mid.py", content="MID"),
        ],
    )

    run(foreman.create_plan("Objective", repository_context=context))

    prompt = provider.calls[0][0].prompt
    assert prompt.index("zeta.py") < prompt.index("alpha.py") < prompt.index("mid.py")


def test_empty_file_content_is_preserved() -> None:
    provider = FakeStructuredProvider(output=_plan(_task("TASK-001")))
    foreman = AnthropicForeman(provider)
    context = RepositoryContext(
        file_tree="src/",
        files=[RepositoryFile(path="src/empty.py", content="")],
    )

    run(foreman.create_plan("Objective", repository_context=context))

    assert provider.call_count == 1
    prompt = provider.calls[0][0].prompt
    # The empty file is not dropped: its path is represented.
    assert "src/empty.py" in prompt


def test_explicit_empty_context_makes_one_call_without_fake_files() -> None:
    provider = FakeStructuredProvider(output=_plan(_task("TASK-001")))
    foreman = AnthropicForeman(provider)
    context = RepositoryContext(file_tree="", files=[])

    run(foreman.create_plan("Objective", repository_context=context))

    assert provider.call_count == 1
    prompt = provider.calls[0][0].prompt
    # Explicitly supplied but empty: distinguishable from None (headings present),
    # yet no file entries are fabricated.
    assert "Engineering objective:" in prompt
    assert "Repository files:" in prompt
    assert "--- path:" not in prompt


def test_repository_data_stays_out_of_system_prompt() -> None:
    provider = FakeStructuredProvider(output=_plan(_task("TASK-001")))
    foreman = AnthropicForeman(provider)
    context = RepositoryContext(
        file_tree="UNIQUE_TREE_MARKER",
        files=[RepositoryFile(path="marker.py", content="UNIQUE_FILE_MARKER")],
    )

    run(foreman.create_plan("Objective", repository_context=context))

    request, _ = provider.calls[0]
    # Repository data appears only in the user prompt, never the system prompt.
    assert "UNIQUE_TREE_MARKER" in request.prompt
    assert "UNIQUE_FILE_MARKER" in request.prompt
    assert request.system_prompt == FOREMAN_SYSTEM_PROMPT
    assert "UNIQUE_TREE_MARKER" not in request.system_prompt
    assert "UNIQUE_FILE_MARKER" not in request.system_prompt
    assert "marker.py" not in request.system_prompt


def test_context_prompt_is_deterministic() -> None:
    provider_a = FakeStructuredProvider(output=_plan(_task("TASK-001")))
    provider_b = FakeStructuredProvider(output=_plan(_task("TASK-001")))
    context = RepositoryContext(
        file_tree="a/\n  b/",
        files=[RepositoryFile(path="a/b/c.py", content="C")],
    )

    run(AnthropicForeman(provider_a).create_plan("Obj", repository_context=context))
    run(AnthropicForeman(provider_b).create_plan("Obj", repository_context=context))

    assert provider_a.calls[0][0].prompt == provider_b.calls[0][0].prompt


def test_context_supplied_still_rejects_blank_objective_before_call() -> None:
    provider = FakeStructuredProvider()
    foreman = AnthropicForeman(provider)
    context = RepositoryContext(
        file_tree="x/",
        files=[RepositoryFile(path="x/y.py", content="Y")],
    )

    with pytest.raises(ValueError):
        run(foreman.create_plan("   ", repository_context=context))

    assert provider.call_count == 0


def test_context_does_not_affect_domain_conversion() -> None:
    provider = FakeStructuredProvider(
        output=_plan(
            _task("TASK-001", title="Design", role=AgentRole.DEVELOPER),
            _task("TASK-002", title="Test", role=AgentRole.TESTER, dependencies=["TASK-001"]),
        )
    )
    foreman = AnthropicForeman(provider)
    context = RepositoryContext(
        file_tree="src/",
        files=[RepositoryFile(path="src/mod.py", content="C")],
    )

    plan = run(foreman.create_plan("Objective", repository_context=context))

    assert [t.id for t in plan.tasks] == ["TASK-001", "TASK-002"]
    assert [t.title for t in plan.tasks] == ["Design", "Test"]
    assert [t.assigned_role for t in plan.tasks] == [
        AgentRole.DEVELOPER,
        AgentRole.TESTER,
    ]
    assert [t.dependencies for t in plan.tasks] == [[], ["TASK-001"]]
    assert all(t.status is TaskStatus.TODO for t in plan.tasks)


# --- Blank objective ------------------------------------------------------


@pytest.mark.parametrize("objective", ["", "   ", "\t\n"])
def test_blank_objective_is_rejected_before_any_provider_call(objective: str) -> None:
    provider = FakeStructuredProvider()
    foreman = AnthropicForeman(provider)

    with pytest.raises(ValueError):
        run(foreman.create_plan(objective))

    assert provider.call_count == 0


# --- Domain conversion ----------------------------------------------------


def test_valid_plan_is_converted_preserving_order_and_fields() -> None:
    provider = FakeStructuredProvider(
        output=_plan(
            _task(
                "TASK-001",
                title="Design contract",
                description="Define the interface",
                role=AgentRole.DEVELOPER,
            ),
            _task(
                "TASK-002",
                title="Add tests",
                description="Cover the contract",
                role=AgentRole.TESTER,
                dependencies=["TASK-001"],
            ),
            _task(
                "TASK-003",
                title="Review",
                description="Review the change",
                role=AgentRole.REVIEWER,
                dependencies=["TASK-001", "TASK-002"],
            ),
        )
    )
    foreman = AnthropicForeman(provider)

    plan = run(foreman.create_plan("Ship the feature"))

    assert isinstance(plan, TaskPlan)
    assert [t.id for t in plan.tasks] == ["TASK-001", "TASK-002", "TASK-003"]
    assert [t.title for t in plan.tasks] == ["Design contract", "Add tests", "Review"]
    assert [t.description for t in plan.tasks] == [
        "Define the interface",
        "Cover the contract",
        "Review the change",
    ]
    assert [t.assigned_role for t in plan.tasks] == [
        AgentRole.DEVELOPER,
        AgentRole.TESTER,
        AgentRole.REVIEWER,
    ]
    assert [t.dependencies for t in plan.tasks] == [
        [],
        ["TASK-001"],
        ["TASK-001", "TASK-002"],
    ]


def test_every_converted_task_starts_todo() -> None:
    provider = FakeStructuredProvider(
        output=_plan(_task("TASK-001"), _task("TASK-002"))
    )
    foreman = AnthropicForeman(provider)

    plan = run(foreman.create_plan("Do work"))

    assert all(task.status is TaskStatus.TODO for task in plan.tasks)


def test_model_output_has_no_status_field() -> None:
    # The private planning DTO must give the model no execution-state authority.
    assert "status" not in _ForemanTaskOutput.model_fields
    assert set(_ForemanTaskOutput.model_fields) == {
        "id",
        "title",
        "description",
        "assigned_role",
        "dependencies",
    }
    assert set(_ForemanPlanOutput.model_fields) == {"tasks"}


# --- Semantic validation failures -----------------------------------------


def test_empty_plan_fails_validation_without_replanning() -> None:
    provider = FakeStructuredProvider(output=_plan())
    foreman = AnthropicForeman(provider)

    with pytest.raises(ForemanPlanValidationError):
        run(foreman.create_plan("Do work"))

    assert provider.call_count == 1


def test_duplicate_ids_fail_validation() -> None:
    provider = FakeStructuredProvider(
        output=_plan(_task("TASK-001"), _task("TASK-001"))
    )
    foreman = AnthropicForeman(provider)

    with pytest.raises(ForemanPlanValidationError):
        run(foreman.create_plan("Do work"))

    assert provider.call_count == 1


@pytest.mark.parametrize("bad_id", ["TASK-1", "task-001", "TASK-"])
def test_invalid_task_id_shape_fails_validation(bad_id: str) -> None:
    provider = FakeStructuredProvider(output=_plan(_task(bad_id)))
    foreman = AnthropicForeman(provider)

    with pytest.raises(ForemanPlanValidationError):
        run(foreman.create_plan("Do work"))


def test_valid_task_id_shape_is_accepted() -> None:
    provider = FakeStructuredProvider(output=_plan(_task("TASK-001")))
    foreman = AnthropicForeman(provider)

    plan = run(foreman.create_plan("Do work"))

    assert plan.tasks[0].id == "TASK-001"


def test_dangling_dependency_fails_validation() -> None:
    provider = FakeStructuredProvider(
        output=_plan(
            _task("TASK-001"),
            _task("TASK-002", dependencies=["TASK-999"]),
        )
    )
    foreman = AnthropicForeman(provider)

    with pytest.raises(ForemanPlanValidationError):
        run(foreman.create_plan("Do work"))

    assert provider.call_count == 1


def test_self_dependency_fails_validation() -> None:
    provider = FakeStructuredProvider(
        output=_plan(_task("TASK-001", dependencies=["TASK-001"]))
    )
    foreman = AnthropicForeman(provider)

    with pytest.raises(ForemanPlanValidationError):
        run(foreman.create_plan("Do work"))


# --- Provider error propagation -------------------------------------------


def test_provider_error_propagates_unchanged() -> None:
    error = ModelProviderPermanentError("boom")
    provider = FakeStructuredProvider(error=error)
    foreman = AnthropicForeman(provider)

    with pytest.raises(ModelProviderPermanentError) as excinfo:
        run(foreman.create_plan("Do work"))

    assert excinfo.value is error
    assert provider.call_count == 1
