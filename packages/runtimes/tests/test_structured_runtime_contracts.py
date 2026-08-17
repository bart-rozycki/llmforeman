"""Behavioral and typing tests for the structured local-generation contract."""

import asyncio
from typing import assert_type

import pytest
from pydantic import BaseModel, ValidationError

from llmforeman_core import ModelUsage
from llmforeman_runtimes import (
    RuntimeRequest,
    StructuredModelRuntime,
    StructuredRuntimeResponse,
)


class ExampleOutput(BaseModel):
    """Tiny test-only schema standing in for a caller-supplied output model."""

    action: str
    count: int


class EmptyOutput(BaseModel):
    """Minimal output model with no fields, carrying no worker semantics."""


class _PlainRuntime:
    """Test-local plain-text runtime with no structured capability.

    Implements only ``generate``; it must NOT be treated as a
    ``StructuredModelRuntime`` because the two capabilities are orthogonal.
    """

    async def generate(self, request: RuntimeRequest) -> str:
        return f"echo: {request.prompt}"


class _FakeStructuredRuntime:
    """Test-local structural implementation of ``StructuredModelRuntime``.

    Implements only ``generate_structured`` (no plain ``generate``) and stays
    generic over ``T`` so it genuinely proves the Protocol and preserves the
    caller-supplied output type.
    """

    def __init__(self, values: dict[str, object]) -> None:
        self._values = values

    async def generate_structured[T: BaseModel](
        self,
        request: RuntimeRequest,
        output_type: type[T],
    ) -> StructuredRuntimeResponse[T]:
        output = output_type.model_validate(self._values)
        return StructuredRuntimeResponse[T](
            output=output,
            usage=ModelUsage(input_tokens=10, output_tokens=5),
        )


# --- StructuredRuntimeResponse --------------------------------------------


def test_response_carries_typed_output_and_usage() -> None:
    result = StructuredRuntimeResponse[ExampleOutput](
        output=ExampleOutput(action="read", count=1),
        usage=ModelUsage(input_tokens=100, output_tokens=20),
    )
    assert result.output == ExampleOutput(action="read", count=1)
    assert result.usage.input_tokens == 100
    assert result.usage.output_tokens == 20


def test_response_output_behaves_as_requested_type() -> None:
    result = StructuredRuntimeResponse[ExampleOutput](
        output=ExampleOutput(action="read", count=1),
        usage=ModelUsage(input_tokens=1, output_tokens=1),
    )
    assert isinstance(result.output, ExampleOutput)
    assert result.output.action == "read"
    assert result.output.count == 1


def test_response_preserves_usage_data() -> None:
    usage = ModelUsage(input_tokens=100, output_tokens=25)
    result = StructuredRuntimeResponse[ExampleOutput](
        output=ExampleOutput(action="read", count=1),
        usage=usage,
    )
    assert result.usage == usage


def test_empty_output_model_is_valid() -> None:
    result = StructuredRuntimeResponse[EmptyOutput](
        output=EmptyOutput(),
        usage=ModelUsage(input_tokens=1, output_tokens=0),
    )
    assert isinstance(result.output, EmptyOutput)


def test_response_serialization_includes_output_and_usage() -> None:
    result = StructuredRuntimeResponse[ExampleOutput](
        output=ExampleOutput(action="read", count=1),
        usage=ModelUsage(input_tokens=100, output_tokens=20),
    )
    assert result.model_dump() == {
        "output": {"action": "read", "count": 1},
        "usage": {
            "input_tokens": 100,
            "output_tokens": 20,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        },
    }


def test_invalid_output_data_is_rejected() -> None:
    with pytest.raises(ValidationError):
        StructuredRuntimeResponse[ExampleOutput].model_validate(
            {
                "output": {"action": "read", "count": "not-an-int"},
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }
        )


# --- StructuredModelRuntime -----------------------------------------------


def test_async_structured_runtime_contract_is_usable() -> None:
    runtime: StructuredModelRuntime = _FakeStructuredRuntime(
        {"action": "read", "count": 1}
    )

    async def run() -> StructuredRuntimeResponse[ExampleOutput]:
        return await runtime.generate_structured(
            RuntimeRequest(prompt="Choose an action"),
            ExampleOutput,
        )

    result = asyncio.run(run())
    assert result.output == ExampleOutput(action="read", count=1)
    assert result.usage.output_tokens == 5


async def use_runtime(runtime: StructuredModelRuntime) -> ExampleOutput:
    """Static type-preservation probe: requesting ``ExampleOutput`` returns it.

    mypy must infer ``response`` as ``StructuredRuntimeResponse[ExampleOutput]``
    and ``response.output`` as ``ExampleOutput`` (not ``BaseModel``/``Any``)
    without any ``cast``/``Any``/``type: ignore`` workaround.
    """

    response = await runtime.generate_structured(
        RuntimeRequest(prompt="Choose an action"),
        ExampleOutput,
    )
    assert_type(response, StructuredRuntimeResponse[ExampleOutput])
    assert_type(response.output, ExampleOutput)
    return response.output


def test_type_preservation_probe_runs() -> None:
    result = asyncio.run(
        use_runtime(_FakeStructuredRuntime({"action": "write", "count": 2}))
    )
    assert result == ExampleOutput(action="write", count=2)


def test_structured_only_fake_satisfies_protocol_without_plain_generate() -> None:
    # A structured-only implementation satisfies the Protocol: assigning it to a
    # ``StructuredModelRuntime`` binding type-checks even though it has no
    # plain-text ``generate`` method, proving the capabilities are orthogonal.
    runtime: StructuredModelRuntime = _FakeStructuredRuntime(
        {"action": "read", "count": 1}
    )
    assert not hasattr(runtime, "generate")


def test_plain_runtime_does_not_provide_structured_capability() -> None:
    # The orthogonal converse: a plain-text-only runtime exposes ``generate``
    # but not ``generate_structured``. Assigning ``_PlainRuntime`` to a
    # ``StructuredModelRuntime`` binding would be a static type error, so the
    # relationship is proven structurally rather than via ``runtime_checkable``.
    plain = _PlainRuntime()
    assert hasattr(plain, "generate")
    assert not hasattr(plain, "generate_structured")
