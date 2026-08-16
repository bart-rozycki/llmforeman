"""Behavioral and typing tests for Anthropic structured generation.

Every test uses an injected fake client's ``messages.parse`` surface; no real
Anthropic API call is ever performed and no ``ANTHROPIC_API_KEY`` is required.
Async tests are driven with ``asyncio.run`` to avoid a pytest asyncio plugin.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import Any

import pytest
from _anthropic_fakes import (
    FakeClient,
    FakeUsage,
    always_raises,
    parsed_message,
    raises_then_returns,
    status_response,
)
from anthropic import APIConnectionError, BadRequestError
from pydantic import BaseModel, ValidationError

from llmforeman_core import ModelUsage
from llmforeman_providers import (
    AnthropicProvider,
    ModelProviderPermanentError,
    ModelProviderTransientError,
    ModelRequest,
    StructuredModelProvider,
    StructuredModelResponse,
)


class ExampleOutput(BaseModel):
    """Tiny test-only schema standing in for a caller-supplied output model."""

    title: str
    count: int


def run[T](coro: Awaitable[T]) -> T:
    return asyncio.run(coro)  # type: ignore[arg-type]


def _connection_error() -> APIConnectionError:
    return APIConnectionError(request=status_response(500).request)


def _validation_error() -> ValidationError:
    """A realistic Pydantic error, exactly what ``parse`` raises on bad JSON."""

    try:
        ExampleOutput.model_validate({"title": "ok", "count": "not-an-int"})
    except ValidationError as exc:
        return exc
    raise AssertionError("expected a ValidationError")  # pragma: no cover


def _fast_provider(client: FakeClient, **kwargs: Any) -> AnthropicProvider:
    params: dict[str, Any] = {"max_tokens": 256, "max_attempts": 3}
    params.update(kwargs)
    return AnthropicProvider(client=client, **params)


# --- Capability / generic typing -----------------------------------------


async def use_structured_provider(provider: StructuredModelProvider) -> ExampleOutput:
    """Static type-preservation probe: requesting ``ExampleOutput`` returns it.

    mypy must infer ``response.output`` as ``ExampleOutput`` (not ``BaseModel``)
    without any ``cast``/``Any``/``type: ignore`` workaround.
    """

    response = await provider.generate_structured(
        ModelRequest(prompt="Generate structured data."),
        ExampleOutput,
    )
    return response.output


def _success_client(output: ExampleOutput, *, usage: FakeUsage | None = None) -> FakeClient:
    async def behavior(**_: Any) -> Any:
        return parsed_message(output, usage=usage)

    return FakeClient(parse=behavior)


def test_provider_satisfies_structured_model_provider() -> None:
    output = ExampleOutput(title="Generate structured data.", count=1)
    provider: StructuredModelProvider = _fast_provider(_success_client(output))

    result = run(use_structured_provider(provider))

    assert result == output


# --- Request mapping ------------------------------------------------------


def test_request_maps_to_single_user_message_and_output_format() -> None:
    client = _success_client(ExampleOutput(title="t", count=1))
    provider = AnthropicProvider(client=client, max_tokens=321)

    run(
        provider.generate_structured(
            ModelRequest(prompt="Do the work."),
            ExampleOutput,
        )
    )

    kwargs = client.parse_calls[0]
    assert kwargs["model"] == "claude-opus-5"
    assert kwargs["max_tokens"] == 321
    assert kwargs["messages"] == [{"role": "user", "content": "Do the work."}]
    assert kwargs["output_format"] is ExampleOutput


def test_system_prompt_omitted_when_absent() -> None:
    client = _success_client(ExampleOutput(title="t", count=1))
    provider = _fast_provider(client)

    run(provider.generate_structured(ModelRequest(prompt="Hello"), ExampleOutput))

    assert "system" not in client.parse_calls[0]


def test_system_prompt_mapped_when_present() -> None:
    client = _success_client(ExampleOutput(title="t", count=1))
    provider = _fast_provider(client)

    run(
        provider.generate_structured(
            ModelRequest(prompt="Hello", system_prompt="You are a careful engineer."),
            ExampleOutput,
        )
    )

    assert client.parse_calls[0]["system"] == "You are a careful engineer."


def test_adapter_does_not_send_thinking_effort_sampling_or_output_config() -> None:
    client = _success_client(ExampleOutput(title="t", count=1))
    provider = _fast_provider(client)

    run(provider.generate_structured(ModelRequest(prompt="Hello"), ExampleOutput))

    kwargs = client.parse_calls[0]
    for forbidden in (
        "thinking",
        "output_config",
        "effort",
        "temperature",
        "top_p",
        "top_k",
        "stream",
        "tools",
        "tool_choice",
        "cache_control",
        "metadata",
    ):
        assert forbidden not in kwargs


# --- Success --------------------------------------------------------------


def test_successful_parse_returns_typed_output() -> None:
    expected = ExampleOutput(title="Plan", count=3)
    client = _success_client(expected)
    provider = _fast_provider(client)

    result = run(provider.generate_structured(ModelRequest(prompt="Hi"), ExampleOutput))

    assert isinstance(result, StructuredModelResponse)
    assert result.output == expected
    assert isinstance(result.output, ExampleOutput)
    assert client.parse_call_count == 1


# --- Usage normalization --------------------------------------------------


def test_usage_normalized_including_cache_counters() -> None:
    usage = FakeUsage(
        input_tokens=11,
        output_tokens=7,
        cache_read_input_tokens=3,
        cache_creation_input_tokens=5,
    )
    client = _success_client(ExampleOutput(title="t", count=1), usage=usage)
    provider = _fast_provider(client)

    result = run(provider.generate_structured(ModelRequest(prompt="Hi"), ExampleOutput))

    assert result.usage == ModelUsage(
        input_tokens=11,
        output_tokens=7,
        cache_read_input_tokens=3,
        cache_creation_input_tokens=5,
    )


def test_missing_cache_counters_normalize_to_zero() -> None:
    usage = FakeUsage(
        input_tokens=4,
        output_tokens=2,
        cache_read_input_tokens=None,
        cache_creation_input_tokens=None,
    )
    client = _success_client(ExampleOutput(title="t", count=1), usage=usage)
    provider = _fast_provider(client)

    result = run(provider.generate_structured(ModelRequest(prompt="Hi"), ExampleOutput))

    assert result.usage.cache_read_input_tokens == 0
    assert result.usage.cache_creation_input_tokens == 0


# --- Refusal --------------------------------------------------------------


def test_refusal_is_non_retryable_failure() -> None:
    async def behavior(**_: Any) -> Any:
        return parsed_message(None, stop_reason="refusal")

    client = FakeClient(parse=behavior)
    provider = _fast_provider(client)

    with pytest.raises(ModelProviderPermanentError):
        run(provider.generate_structured(ModelRequest(prompt="Hi"), ExampleOutput))

    assert client.parse_call_count == 1


# --- Max tokens -----------------------------------------------------------


def test_max_tokens_truncation_is_non_retryable_failure() -> None:
    async def behavior(**_: Any) -> Any:
        return parsed_message(None, stop_reason="max_tokens")

    client = FakeClient(parse=behavior)
    provider = _fast_provider(client)

    with pytest.raises(ModelProviderPermanentError):
        run(provider.generate_structured(ModelRequest(prompt="Hi"), ExampleOutput))

    assert client.parse_call_count == 1


# --- Missing parsed output ------------------------------------------------


def test_missing_parsed_output_is_non_retryable_failure() -> None:
    async def behavior(**_: Any) -> Any:
        return parsed_message(None, stop_reason="end_turn")

    client = FakeClient(parse=behavior)
    provider = _fast_provider(client)

    with pytest.raises(ModelProviderPermanentError):
        run(provider.generate_structured(ModelRequest(prompt="Hi"), ExampleOutput))

    assert client.parse_call_count == 1


# --- Parse / validation failure -------------------------------------------


def test_validation_failure_is_normalized_and_not_retried() -> None:
    error = _validation_error()
    client = FakeClient(parse=always_raises(error))
    provider = _fast_provider(client, max_attempts=3)

    with pytest.raises(ModelProviderPermanentError) as excinfo:
        run(provider.generate_structured(ModelRequest(prompt="Hi"), ExampleOutput))

    assert client.parse_call_count == 1
    assert isinstance(excinfo.value.__cause__, ValidationError)


# --- Transient retry through RelPrim --------------------------------------


def test_transient_failure_is_retried_then_succeeds() -> None:
    expected = ExampleOutput(title="done", count=9)
    client = FakeClient(
        parse=raises_then_returns(_connection_error(), parsed_message(expected), failures=1)
    )
    provider = _fast_provider(client, max_attempts=3)

    result = run(provider.generate_structured(ModelRequest(prompt="Hi"), ExampleOutput))

    assert result.output == expected
    assert client.parse_call_count == 2


def test_transient_exhaustion_raises_normalized_transient_error() -> None:
    client = FakeClient(parse=always_raises(_connection_error()))
    provider = _fast_provider(client, max_attempts=2)

    with pytest.raises(ModelProviderTransientError) as excinfo:
        run(provider.generate_structured(ModelRequest(prompt="Hi"), ExampleOutput))

    assert client.parse_call_count == 2
    assert isinstance(excinfo.value.__cause__, APIConnectionError)


# --- Permanent transport error --------------------------------------------


def test_permanent_transport_error_is_not_retried() -> None:
    error = BadRequestError("bad", response=status_response(400), body=None)
    client = FakeClient(parse=always_raises(error))
    provider = _fast_provider(client, max_attempts=3)

    with pytest.raises(ModelProviderPermanentError):
        run(provider.generate_structured(ModelRequest(prompt="Hi"), ExampleOutput))

    assert client.parse_call_count == 1
