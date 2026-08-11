"""Behavioral tests for the provider-agnostic generation contract."""

import asyncio

import pytest
from pydantic import ValidationError

from llmforeman_core import ModelUsage
from llmforeman_providers import ModelProvider, ModelRequest, ModelResponse


class _FakeProvider:
    """Test-local structural implementation of the ``ModelProvider`` contract."""

    async def generate(self, request: ModelRequest) -> ModelResponse:
        return ModelResponse(
            content=f"echo: {request.prompt}",
            usage=ModelUsage(input_tokens=10, output_tokens=5),
        )


# --- ModelRequest ---------------------------------------------------------


def test_minimal_request_can_be_created() -> None:
    request = ModelRequest(prompt="Implement the requested change.")
    assert request.prompt == "Implement the requested change."


def test_system_prompt_defaults_to_none() -> None:
    request = ModelRequest(prompt="Do the work.")
    assert request.system_prompt is None


def test_request_with_prompt_and_system_prompt() -> None:
    request = ModelRequest(
        prompt="Do the work.",
        system_prompt="You are a careful engineer.",
    )
    assert request.prompt == "Do the work."
    assert request.system_prompt == "You are a careful engineer."


@pytest.mark.parametrize("bad_prompt", ["", "   ", "\t\n"])
def test_empty_or_whitespace_prompt_rejected(bad_prompt: str) -> None:
    with pytest.raises(ValidationError):
        ModelRequest(prompt=bad_prompt)


@pytest.mark.parametrize("bad_system_prompt", ["", "   ", "\t\n"])
def test_empty_or_whitespace_system_prompt_rejected(bad_system_prompt: str) -> None:
    with pytest.raises(ValidationError):
        ModelRequest(prompt="Do the work.", system_prompt=bad_system_prompt)


def test_request_serialization_preserves_fields() -> None:
    request = ModelRequest(
        prompt="Do the work.",
        system_prompt="Be careful.",
    )
    assert request.model_dump() == {
        "prompt": "Do the work.",
        "system_prompt": "Be careful.",
    }


# --- ModelResponse --------------------------------------------------------


def test_response_can_be_created() -> None:
    response = ModelResponse(
        content="Done.",
        usage=ModelUsage(input_tokens=100, output_tokens=20),
    )
    assert response.content == "Done."


def test_response_preserves_usage_data() -> None:
    usage = ModelUsage(input_tokens=100, output_tokens=20)
    response = ModelResponse(content="Done.", usage=usage)
    assert response.usage == usage
    assert response.usage.input_tokens == 100
    assert response.usage.output_tokens == 20


def test_response_serialization_includes_usage() -> None:
    response = ModelResponse(
        content="Done.",
        usage=ModelUsage(input_tokens=100, output_tokens=20),
    )
    assert response.model_dump() == {
        "content": "Done.",
        "usage": {
            "input_tokens": 100,
            "output_tokens": 20,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        },
    }


def test_empty_content_is_structurally_valid() -> None:
    response = ModelResponse(
        content="",
        usage=ModelUsage(input_tokens=1, output_tokens=0),
    )
    assert response.content == ""


# --- ModelProvider --------------------------------------------------------


def test_async_provider_contract_is_usable() -> None:
    provider: ModelProvider = _FakeProvider()

    async def run() -> ModelResponse:
        return await provider.generate(ModelRequest(prompt="hello"))

    response = asyncio.run(run())
    assert response.content == "echo: hello"
    assert response.usage.output_tokens == 5
