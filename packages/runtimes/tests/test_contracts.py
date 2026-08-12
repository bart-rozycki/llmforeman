"""Behavioral tests for the runtime-agnostic local generation contract."""

import asyncio

import pytest
from pydantic import ValidationError

from llmforeman_core import ModelUsage
from llmforeman_runtimes import ModelRuntime, RuntimeRequest, RuntimeResponse


class _FakeRuntime:
    """Test-local structural implementation of the ``ModelRuntime`` contract."""

    async def generate(self, request: RuntimeRequest) -> RuntimeResponse:
        return RuntimeResponse(
            content=f"echo: {request.prompt}",
            usage=ModelUsage(input_tokens=10, output_tokens=5),
        )


# --- RuntimeRequest -------------------------------------------------------


def test_minimal_request_can_be_created() -> None:
    request = RuntimeRequest(prompt="Implement the requested task.")
    assert request.prompt == "Implement the requested task."


def test_system_prompt_defaults_to_none() -> None:
    request = RuntimeRequest(prompt="Do the work.")
    assert request.system_prompt is None


def test_request_with_prompt_and_system_prompt() -> None:
    request = RuntimeRequest(
        prompt="Do the work.",
        system_prompt="You are a careful engineer.",
    )
    assert request.prompt == "Do the work."
    assert request.system_prompt == "You are a careful engineer."


@pytest.mark.parametrize("bad_prompt", ["", "   ", "\t\n"])
def test_empty_or_whitespace_prompt_rejected(bad_prompt: str) -> None:
    with pytest.raises(ValidationError):
        RuntimeRequest(prompt=bad_prompt)


@pytest.mark.parametrize("bad_system_prompt", ["", "   ", "\t\n"])
def test_empty_or_whitespace_system_prompt_rejected(bad_system_prompt: str) -> None:
    with pytest.raises(ValidationError):
        RuntimeRequest(prompt="Do the work.", system_prompt=bad_system_prompt)


def test_request_serialization_preserves_fields() -> None:
    request = RuntimeRequest(
        prompt="Do the work.",
        system_prompt="Be careful.",
    )
    assert request.model_dump() == {
        "prompt": "Do the work.",
        "system_prompt": "Be careful.",
    }


# --- RuntimeResponse ------------------------------------------------------


def test_response_can_be_created() -> None:
    response = RuntimeResponse(
        content="Implemented.",
        usage=ModelUsage(input_tokens=100, output_tokens=25),
    )
    assert response.content == "Implemented."


def test_response_preserves_usage_data() -> None:
    usage = ModelUsage(input_tokens=100, output_tokens=25)
    response = RuntimeResponse(content="Implemented.", usage=usage)
    assert response.usage == usage
    assert response.usage.input_tokens == 100
    assert response.usage.output_tokens == 25


def test_response_serialization_includes_usage() -> None:
    response = RuntimeResponse(
        content="Implemented.",
        usage=ModelUsage(input_tokens=100, output_tokens=25),
    )
    assert response.model_dump() == {
        "content": "Implemented.",
        "usage": {
            "input_tokens": 100,
            "output_tokens": 25,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        },
    }


def test_empty_content_is_structurally_valid() -> None:
    response = RuntimeResponse(
        content="",
        usage=ModelUsage(input_tokens=1, output_tokens=0),
    )
    assert response.content == ""


# --- ModelRuntime ---------------------------------------------------------


def test_async_runtime_contract_is_usable() -> None:
    runtime: ModelRuntime = _FakeRuntime()

    async def run() -> RuntimeResponse:
        return await runtime.generate(RuntimeRequest(prompt="hello"))

    response = asyncio.run(run())
    assert response.content == "echo: hello"
    assert response.usage.output_tokens == 5
