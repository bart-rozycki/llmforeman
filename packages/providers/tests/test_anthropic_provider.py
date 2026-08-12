"""Behavioral tests for the Anthropic provider adapter.

Every test uses an injected fake client; no real Anthropic API call is ever
performed and no ``ANTHROPIC_API_KEY`` is required. Async tests are driven with
``asyncio.run`` to avoid adding a pytest asyncio plugin dependency.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import Any

import pytest
from _anthropic_fakes import (
    FakeClient,
    FakeMessage,
    FakeTextBlock,
    FakeThinkingBlock,
    FakeUsage,
    always_raises,
    hangs_until_cancelled,
    raises_then_returns,
    returns,
    status_response,
    text_message,
)
from anthropic import (
    APIConnectionError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    PermissionDeniedError,
    RateLimitError,
    RequestTooLargeError,
    UnprocessableEntityError,
)

from llmforeman_core import ModelUsage
from llmforeman_providers import (
    AnthropicProvider,
    ModelProvider,
    ModelProviderError,
    ModelProviderPermanentError,
    ModelProviderRateLimitError,
    ModelProviderTimeoutError,
    ModelProviderTransientError,
    ModelRequest,
    ModelResponse,
)
from llmforeman_providers.anthropic import _parse_retry_after_header


def run[T](coro: Awaitable[T]) -> T:
    return asyncio.run(coro)  # type: ignore[arg-type]


def _rate_limit_error(retry_after: str | None = None) -> RateLimitError:
    headers = {"retry-after": retry_after} if retry_after is not None else {}
    return RateLimitError("rate limited", response=status_response(429, headers), body=None)


def _connection_error() -> APIConnectionError:
    return APIConnectionError(request=status_response(500).request)


# --- Contract conformance -------------------------------------------------


def test_satisfies_model_provider_protocol() -> None:
    provider: ModelProvider = AnthropicProvider(
        client=FakeClient(returns(text_message("ok"))),
        max_tokens=8,
    )
    assert isinstance(provider, AnthropicProvider)


# --- Request mapping ------------------------------------------------------


def _fast_provider(client: FakeClient, **kwargs: Any) -> AnthropicProvider:
    params: dict[str, Any] = {"max_tokens": 256, "max_attempts": 3}
    params.update(kwargs)
    return AnthropicProvider(client=client, **params)


def test_request_maps_to_single_user_message() -> None:
    client = FakeClient(returns(text_message("ok")))
    provider = AnthropicProvider(client=client, max_tokens=321)

    run(provider.generate(ModelRequest(prompt="Do the work.")))

    kwargs = client.calls[0]
    assert kwargs["model"] == "claude-opus-5"
    assert kwargs["max_tokens"] == 321
    assert kwargs["messages"] == [{"role": "user", "content": "Do the work."}]


def test_system_prompt_omitted_when_absent() -> None:
    client = FakeClient(returns(text_message("ok")))
    provider = _fast_provider(client)

    run(provider.generate(ModelRequest(prompt="Hello")))

    assert "system" not in client.calls[0]


def test_system_prompt_mapped_when_present() -> None:
    client = FakeClient(returns(text_message("ok")))
    provider = _fast_provider(client)

    run(
        provider.generate(
            ModelRequest(prompt="Hello", system_prompt="You are a careful engineer.")
        )
    )

    assert client.calls[0]["system"] == "You are a careful engineer."


def test_adapter_does_not_send_thinking_effort_or_sampling() -> None:
    client = FakeClient(returns(text_message("ok")))
    provider = _fast_provider(client)

    run(provider.generate(ModelRequest(prompt="Hello")))

    kwargs = client.calls[0]
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


# --- Text extraction ------------------------------------------------------


def test_text_blocks_extracted_in_order_and_non_text_ignored() -> None:
    message = FakeMessage(
        content=[
            FakeThinkingBlock(),
            FakeTextBlock("A"),
            FakeThinkingBlock(),
            FakeTextBlock("B"),
        ]
    )
    client = FakeClient(returns(message))
    provider = _fast_provider(client)

    response = run(provider.generate(ModelRequest(prompt="Hello")))

    assert response.content == "AB"


def test_no_text_blocks_yields_empty_content() -> None:
    message = FakeMessage(content=[FakeThinkingBlock(), FakeThinkingBlock()])
    client = FakeClient(returns(message))
    provider = _fast_provider(client)

    response = run(provider.generate(ModelRequest(prompt="Hello")))

    assert response.content == ""


# --- Usage normalization --------------------------------------------------


def test_usage_normalized_including_cache_counters() -> None:
    usage = FakeUsage(
        input_tokens=11,
        output_tokens=7,
        cache_read_input_tokens=3,
        cache_creation_input_tokens=5,
    )
    client = FakeClient(returns(text_message("ok", usage=usage)))
    provider = _fast_provider(client)

    response = run(provider.generate(ModelRequest(prompt="Hello")))

    assert response.usage == ModelUsage(
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
    client = FakeClient(returns(text_message("ok", usage=usage)))
    provider = _fast_provider(client)

    response = run(provider.generate(ModelRequest(prompt="Hello")))

    assert response.usage.cache_read_input_tokens == 0
    assert response.usage.cache_creation_input_tokens == 0


# --- SDK retry disabling --------------------------------------------------


def test_owned_client_disables_sdk_retries_and_timeout() -> None:
    client = AnthropicProvider._build_client("test-key")

    assert client.max_retries == 0
    assert client.timeout is None


def test_provider_construction_uses_zero_sdk_retries() -> None:
    provider = AnthropicProvider(max_tokens=16, api_key="test-key")

    assert provider._owned_client is not None
    assert provider._owned_client.max_retries == 0


# --- Transient retry ------------------------------------------------------


def test_transient_failure_is_retried_then_succeeds() -> None:
    client = FakeClient(
        raises_then_returns(_connection_error(), text_message("done"), failures=1)
    )
    provider = _fast_provider(client, max_attempts=3)

    response = run(provider.generate(ModelRequest(prompt="Hello")))

    assert isinstance(response, ModelResponse)
    assert response.content == "done"
    assert client.call_count == 2


def test_transient_failure_exhaustion_raises_normalized_transient_error() -> None:
    client = FakeClient(always_raises(_connection_error()))
    provider = _fast_provider(client, max_attempts=2)

    with pytest.raises(ModelProviderTransientError) as excinfo:
        run(provider.generate(ModelRequest(prompt="Hello")))

    assert client.call_count == 2
    assert not isinstance(excinfo.value, RateLimitError)
    assert isinstance(excinfo.value.__cause__, APIConnectionError)


# --- Permanent errors -----------------------------------------------------


@pytest.mark.parametrize(
    "error",
    [
        BadRequestError("bad", response=status_response(400), body=None),
        AuthenticationError("auth", response=status_response(401), body=None),
        PermissionDeniedError("perm", response=status_response(403), body=None),
        RequestTooLargeError("too large", response=status_response(413), body=None),
        UnprocessableEntityError("unprocessable", response=status_response(422), body=None),
    ],
)
def test_permanent_errors_are_not_retried(error: Exception) -> None:
    client = FakeClient(always_raises(error))
    provider = _fast_provider(client, max_attempts=3)

    with pytest.raises(ModelProviderPermanentError):
        run(provider.generate(ModelRequest(prompt="Hello")))

    assert client.call_count == 1


def test_server_error_is_transient_and_retried() -> None:
    server_error = InternalServerError("boom", response=status_response(503), body=None)
    client = FakeClient(raises_then_returns(server_error, text_message("ok"), failures=1))
    provider = _fast_provider(client, max_attempts=3)

    response = run(provider.generate(ModelRequest(prompt="Hello")))

    assert response.content == "ok"
    assert client.call_count == 2


# --- Rate limiting --------------------------------------------------------


def test_rate_limit_is_retried_then_succeeds() -> None:
    client = FakeClient(
        raises_then_returns(_rate_limit_error("0"), text_message("ok"), failures=1)
    )
    provider = _fast_provider(client, max_attempts=3)

    response = run(provider.generate(ModelRequest(prompt="Hello")))

    assert response.content == "ok"
    assert client.call_count == 2


def test_rate_limit_participates_in_bounded_budget() -> None:
    client = FakeClient(always_raises(_rate_limit_error("0")))
    provider = _fast_provider(client, max_attempts=2)

    with pytest.raises(ModelProviderRateLimitError):
        run(provider.generate(ModelRequest(prompt="Hello")))

    assert client.call_count == 2


def test_rate_limit_wait_exceeding_max_does_not_sleep_or_retry() -> None:
    client = FakeClient(always_raises(_rate_limit_error("1000")))
    provider = _fast_provider(client, max_attempts=3, max_rate_limit_wait_seconds=0.5)

    with pytest.raises(ModelProviderRateLimitError):
        run(provider.generate(ModelRequest(prompt="Hello")))

    # Provider-recommended wait (1000s) exceeds the configured bound, so the
    # operation fails immediately rather than sleeping or retrying.
    assert client.call_count == 1


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("0", 0.0),
        ("2", 2.0),
        ("2.5", 2.5),
        (None, None),
        ("later", None),
        ("-1", None),
        ("nan", None),
        ("inf", None),
    ],
)
def test_retry_after_header_parsing(raw: str | None, expected: float | None) -> None:
    assert _parse_retry_after_header(raw) == expected


# --- Timeout and cancellation --------------------------------------------


def test_timeout_is_enforced_by_relprim_without_orphan_task() -> None:
    counter: dict[str, int] = {}
    client = FakeClient(hangs_until_cancelled(counter))
    provider = AnthropicProvider(
        client=client,
        max_tokens=16,
        timeout_seconds=0.05,
        max_attempts=1,
    )

    with pytest.raises(ModelProviderTimeoutError):
        run(provider.generate(ModelRequest(prompt="Hello")))

    assert counter.get("started") == 1
    assert counter.get("cancelled") == 1


def test_caller_cancellation_propagates() -> None:
    async def scenario() -> None:
        counter: dict[str, int] = {}
        client = FakeClient(hangs_until_cancelled(counter))
        provider = AnthropicProvider(
            client=client,
            max_tokens=16,
            timeout_seconds=5.0,
            max_attempts=3,
        )

        task = asyncio.create_task(provider.generate(ModelRequest(prompt="Hello")))
        for _ in range(5):
            await asyncio.sleep(0)
        assert counter.get("started") == 1

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert counter.get("cancelled") == 1

    run(scenario())


def test_unexpected_error_is_normalized_to_base_provider_error() -> None:
    client = FakeClient(always_raises(RuntimeError("boom")))
    provider = _fast_provider(client, max_attempts=3)

    with pytest.raises(ModelProviderError) as excinfo:
        run(provider.generate(ModelRequest(prompt="Hello")))

    # Unknown errors default safely: not retried and not treated as transient.
    assert client.call_count == 1
    assert not isinstance(excinfo.value, ModelProviderTransientError)
    assert isinstance(excinfo.value.__cause__, RuntimeError)
