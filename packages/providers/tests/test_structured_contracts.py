"""Behavioral and typing tests for the structured-generation contract."""

import asyncio

import pytest
from pydantic import BaseModel, ValidationError

from llmforeman_core import ModelUsage
from llmforeman_providers import (
    ModelRequest,
    StructuredModelProvider,
    StructuredModelResponse,
)


class ExampleOutput(BaseModel):
    """Tiny test-only schema standing in for a caller-supplied output model."""

    title: str
    count: int


class _FakeStructuredProvider:
    """Test-local structural implementation of ``StructuredModelProvider``."""

    async def generate_structured[T: BaseModel](
        self,
        request: ModelRequest,
        output_type: type[T],
    ) -> StructuredModelResponse[T]:
        output = output_type.model_validate({"title": request.prompt, "count": 1})
        return StructuredModelResponse[T](
            output=output,
            usage=ModelUsage(input_tokens=10, output_tokens=5),
        )


# --- StructuredModelResponse ----------------------------------------------


def test_response_carries_typed_output_and_usage() -> None:
    result = StructuredModelResponse[ExampleOutput](
        output=ExampleOutput(title="Example", count=3),
        usage=ModelUsage(input_tokens=100, output_tokens=20),
    )
    assert result.output == ExampleOutput(title="Example", count=3)
    assert result.usage.input_tokens == 100
    assert result.usage.output_tokens == 20


def test_response_output_behaves_as_requested_type() -> None:
    result = StructuredModelResponse[ExampleOutput](
        output=ExampleOutput(title="Example", count=3),
        usage=ModelUsage(input_tokens=1, output_tokens=1),
    )
    assert isinstance(result.output, ExampleOutput)
    assert result.output.title == "Example"
    assert result.output.count == 3


def test_response_serialization_includes_output_and_usage() -> None:
    result = StructuredModelResponse[ExampleOutput](
        output=ExampleOutput(title="Example", count=3),
        usage=ModelUsage(input_tokens=100, output_tokens=20),
    )
    assert result.model_dump() == {
        "output": {"title": "Example", "count": 3},
        "usage": {
            "input_tokens": 100,
            "output_tokens": 20,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        },
    }


def test_invalid_output_data_is_rejected() -> None:
    with pytest.raises(ValidationError):
        StructuredModelResponse[ExampleOutput].model_validate(
            {
                "output": {"title": "Example", "count": "not-an-int"},
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }
        )


# --- StructuredModelProvider ----------------------------------------------


def test_async_structured_provider_contract_is_usable() -> None:
    provider: StructuredModelProvider = _FakeStructuredProvider()

    async def run() -> StructuredModelResponse[ExampleOutput]:
        return await provider.generate_structured(
            ModelRequest(prompt="Example"),
            ExampleOutput,
        )

    result = asyncio.run(run())
    assert result.output == ExampleOutput(title="Example", count=1)
    assert result.usage.output_tokens == 5


async def use_provider(provider: StructuredModelProvider) -> ExampleOutput:
    """Static type-preservation probe: requesting ``ExampleOutput`` returns it.

    mypy must infer ``response.output`` as ``ExampleOutput`` (not ``BaseModel``)
    without any ``cast``/``Any``/``type: ignore`` workaround.
    """

    response = await provider.generate_structured(
        ModelRequest(prompt="Generate example data."),
        ExampleOutput,
    )
    return response.output


def test_type_preservation_probe_runs() -> None:
    result = asyncio.run(use_provider(_FakeStructuredProvider()))
    assert result == ExampleOutput(title="Generate example data.", count=1)
