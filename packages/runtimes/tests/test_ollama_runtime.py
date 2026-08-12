"""Behavioral tests for the Ollama runtime adapter.

Every test uses an injected fake client or a locally constructed client; no
real Ollama server, no downloaded model, and no network access are ever
required. Async tests are driven with ``asyncio.run`` to avoid adding a pytest
asyncio plugin dependency. Retry-heavy tests use a zero-delay backoff so they
remain deterministic and fast.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import Any

import httpx
import pytest
from _ollama_fakes import (
    FakeClient,
    always_raises,
    generate_response,
    hangs_until_cancelled,
    raises_then_returns,
    returns,
)
from ollama import ResponseError
from relprim import ExponentialBackoff

from llmforeman_core import ModelUsage
from llmforeman_runtimes import (
    ModelRuntime,
    ModelRuntimeError,
    ModelRuntimePermanentError,
    ModelRuntimeTimeoutError,
    ModelRuntimeTransientError,
    OllamaRuntime,
    RuntimeRequest,
    RuntimeResponse,
)

_NO_DELAY = ExponentialBackoff(base_delay_seconds=0.0, max_delay_seconds=0.0, jitter=False)


def run[T](coro: Awaitable[T]) -> T:
    return asyncio.run(coro)  # type: ignore[arg-type]


def _fast_runtime(client: FakeClient, **kwargs: Any) -> OllamaRuntime:
    params: dict[str, Any] = {"backoff": _NO_DELAY}
    params.update(kwargs)
    return OllamaRuntime(client=client, **params)


def _connection_error() -> ConnectionError:
    # The Ollama SDK normalizes a failed connection to the built-in
    # ConnectionError, so tests use exactly that type.
    return ConnectionError("could not connect to ollama")


# --- Contract conformance -------------------------------------------------


def test_satisfies_model_runtime_protocol() -> None:
    runtime: ModelRuntime = OllamaRuntime(client=FakeClient(returns(generate_response())))
    assert isinstance(runtime, OllamaRuntime)


# --- Configuration --------------------------------------------------------


def test_default_model_is_qwen() -> None:
    runtime = OllamaRuntime(client=FakeClient(returns(generate_response())))
    assert runtime.model == "qwen3.6:35b-a3b"


def test_explicit_model_override() -> None:
    runtime = OllamaRuntime(
        client=FakeClient(returns(generate_response())),
        model="llama3.1:8b",
    )
    assert runtime.model == "llama3.1:8b"


@pytest.mark.parametrize("bad_model", ["", "   ", "\t\n"])
def test_empty_or_whitespace_model_rejected(bad_model: str) -> None:
    with pytest.raises(ValueError):
        OllamaRuntime(client=FakeClient(returns(generate_response())), model=bad_model)


@pytest.mark.parametrize("bad_host", ["", "   ", "\t\n"])
def test_empty_or_whitespace_host_rejected(bad_host: str) -> None:
    with pytest.raises(ValueError):
        OllamaRuntime(host=bad_host)


@pytest.mark.parametrize("bad_timeout", [0.0, -1.0])
def test_non_positive_timeout_rejected(bad_timeout: float) -> None:
    with pytest.raises(ValueError):
        OllamaRuntime(
            client=FakeClient(returns(generate_response())),
            timeout_seconds=bad_timeout,
        )


def test_zero_max_attempts_rejected() -> None:
    with pytest.raises(ValueError):
        OllamaRuntime(
            client=FakeClient(returns(generate_response())),
            max_attempts=0,
        )


def test_explicit_host_is_passed_to_owned_client() -> None:
    runtime = OllamaRuntime(host="http://ollama.local:1234")
    assert runtime._owned_client is not None
    assert "ollama.local:1234" in str(runtime._owned_client._client.base_url)


def test_owned_client_has_no_competing_transport_timeout() -> None:
    client = OllamaRuntime._build_client(None)
    assert client._client.timeout == httpx.Timeout(None)


# --- Request mapping ------------------------------------------------------


def test_request_maps_model_prompt_and_non_streaming() -> None:
    client = FakeClient(returns(generate_response()))
    runtime = _fast_runtime(client, model="codellama:13b")

    run(runtime.generate(RuntimeRequest(prompt="Implement this function.")))

    kwargs = client.calls[0]
    assert kwargs["model"] == "codellama:13b"
    assert kwargs["prompt"] == "Implement this function."
    assert kwargs["stream"] is False


def test_system_prompt_omitted_when_absent() -> None:
    client = FakeClient(returns(generate_response()))
    runtime = _fast_runtime(client)

    run(runtime.generate(RuntimeRequest(prompt="Implement this function.")))

    assert "system" not in client.calls[0]


def test_system_prompt_mapped_when_present() -> None:
    client = FakeClient(returns(generate_response()))
    runtime = _fast_runtime(client)

    run(
        runtime.generate(
            RuntimeRequest(
                prompt="Implement this function.",
                system_prompt="You are a coding worker.",
            )
        )
    )

    assert client.calls[0]["system"] == "You are a coding worker."


def test_adapter_sends_no_generation_options() -> None:
    client = FakeClient(returns(generate_response()))
    runtime = _fast_runtime(client)

    run(runtime.generate(RuntimeRequest(prompt="Implement this function.")))

    kwargs = client.calls[0]
    for forbidden in (
        "options",
        "format",
        "temperature",
        "top_p",
        "top_k",
        "seed",
        "stop",
        "think",
        "tools",
        "images",
        "context",
        "keep_alive",
        "raw",
        "suffix",
        "template",
        "logprobs",
    ):
        assert forbidden not in kwargs


# --- Content extraction and thinking exclusion ----------------------------


def test_final_response_used_and_thinking_excluded() -> None:
    response = generate_response(response="Final answer", thinking="Internal reasoning")
    client = FakeClient(returns(response))
    runtime = _fast_runtime(client)

    result = run(runtime.generate(RuntimeRequest(prompt="Solve it.")))

    assert isinstance(result, RuntimeResponse)
    assert result.content == "Final answer"
    assert "Internal reasoning" not in result.content


def test_missing_final_response_normalizes_to_empty_string() -> None:
    response = generate_response(response=None, thinking="hidden")
    client = FakeClient(returns(response))
    runtime = _fast_runtime(client)

    result = run(runtime.generate(RuntimeRequest(prompt="Solve it.")))

    assert result.content == ""


# --- Usage normalization --------------------------------------------------


def test_usage_normalized_from_ollama_counters() -> None:
    response = generate_response(prompt_eval_count=123, eval_count=45)
    client = FakeClient(returns(response))
    runtime = _fast_runtime(client)

    result = run(runtime.generate(RuntimeRequest(prompt="Solve it.")))

    assert result.usage == ModelUsage(
        input_tokens=123,
        output_tokens=45,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=0,
    )


def test_explicit_zero_usage_is_preserved() -> None:
    response = generate_response(prompt_eval_count=0, eval_count=0)
    client = FakeClient(returns(response))
    runtime = _fast_runtime(client)

    result = run(runtime.generate(RuntimeRequest(prompt="Solve it.")))

    assert result.usage.input_tokens == 0
    assert result.usage.output_tokens == 0


def test_timings_are_not_inserted_into_usage() -> None:
    response = generate_response(prompt_eval_count=5, eval_count=7)
    client = FakeClient(returns(response))
    runtime = _fast_runtime(client)

    result = run(runtime.generate(RuntimeRequest(prompt="Solve it.")))

    assert set(result.usage.model_dump()) == {
        "input_tokens",
        "output_tokens",
        "cache_read_input_tokens",
        "cache_creation_input_tokens",
    }


@pytest.mark.parametrize(
    ("prompt_eval_count", "eval_count"),
    [(None, 5), (5, None), (None, None)],
)
def test_missing_usage_counters_raise_permanent_error_without_retry(
    prompt_eval_count: int | None,
    eval_count: int | None,
) -> None:
    response = generate_response(
        prompt_eval_count=prompt_eval_count,
        eval_count=eval_count,
    )
    client = FakeClient(returns(response))
    runtime = _fast_runtime(client, max_attempts=3)

    with pytest.raises(ModelRuntimePermanentError):
        run(runtime.generate(RuntimeRequest(prompt="Solve it.")))

    # Normalization happens once after a single successful generation attempt.
    assert client.call_count == 1


# --- Transient retry ------------------------------------------------------


def test_transient_failure_is_retried_then_succeeds() -> None:
    client = FakeClient(
        raises_then_returns(_connection_error(), generate_response(response="done"), failures=1)
    )
    runtime = _fast_runtime(client, max_attempts=3)

    result = run(runtime.generate(RuntimeRequest(prompt="Solve it.")))

    assert isinstance(result, RuntimeResponse)
    assert result.content == "done"
    assert client.call_count == 2


def test_transient_exhaustion_raises_normalized_transient_error() -> None:
    client = FakeClient(always_raises(_connection_error()))
    runtime = _fast_runtime(client, max_attempts=3)

    with pytest.raises(ModelRuntimeTransientError) as excinfo:
        run(runtime.generate(RuntimeRequest(prompt="Solve it.")))

    assert client.call_count == 3
    # The public failure is the normalized runtime error, not the raw SDK type,
    # but the original cause remains chained for diagnostics.
    assert not isinstance(excinfo.value, ConnectionError)
    assert isinstance(excinfo.value.__cause__, ConnectionError)


def test_server_error_is_transient_and_retried() -> None:
    server_error = ResponseError("internal error", 503)
    client = FakeClient(raises_then_returns(server_error, generate_response(), failures=1))
    runtime = _fast_runtime(client, max_attempts=3)

    result = run(runtime.generate(RuntimeRequest(prompt="Solve it.")))

    assert result.content == "ok"
    assert client.call_count == 2


def test_transport_interruption_is_transient_and_retried() -> None:
    read_error = httpx.ReadError("connection reset")
    client = FakeClient(raises_then_returns(read_error, generate_response(), failures=1))
    runtime = _fast_runtime(client, max_attempts=3)

    result = run(runtime.generate(RuntimeRequest(prompt="Solve it.")))

    assert result.content == "ok"
    assert client.call_count == 2


# --- Permanent errors -----------------------------------------------------


@pytest.mark.parametrize("status_code", [400, 404, 422])
def test_permanent_errors_are_not_retried(status_code: int) -> None:
    error = ResponseError("bad request", status_code)
    client = FakeClient(always_raises(error))
    runtime = _fast_runtime(client, max_attempts=3)

    with pytest.raises(ModelRuntimePermanentError) as excinfo:
        run(runtime.generate(RuntimeRequest(prompt="Solve it.")))

    assert client.call_count == 1
    assert isinstance(excinfo.value.__cause__, ResponseError)


def test_model_not_found_is_permanent_and_never_pulls() -> None:
    error = ResponseError("model 'qwen3.6:35b-a3b' not found", 404)
    client = FakeClient(always_raises(error))
    runtime = _fast_runtime(client, max_attempts=3)

    with pytest.raises(ModelRuntimePermanentError):
        run(runtime.generate(RuntimeRequest(prompt="Solve it.")))

    # Exactly one generation attempt, no retry, and no auto-pull behavior: the
    # adapter only ever calls ``generate`` and never exposes a ``pull`` seam.
    assert client.call_count == 1
    assert not hasattr(runtime, "pull")
    assert not hasattr(runtime, "_pull")


def test_unknown_error_is_normalized_to_base_runtime_error() -> None:
    client = FakeClient(always_raises(RuntimeError("boom")))
    runtime = _fast_runtime(client, max_attempts=3)

    with pytest.raises(ModelRuntimeError) as excinfo:
        run(runtime.generate(RuntimeRequest(prompt="Solve it.")))

    # Unknown errors default safely: not retried and not treated as transient.
    assert client.call_count == 1
    assert not isinstance(excinfo.value, ModelRuntimeTransientError)
    assert isinstance(excinfo.value.__cause__, RuntimeError)


# --- Timeout and cancellation --------------------------------------------


def test_timeout_is_enforced_by_relprim_without_orphan_task() -> None:
    counter: dict[str, int] = {}
    client = FakeClient(hangs_until_cancelled(counter))
    runtime = OllamaRuntime(
        client=client,
        timeout_seconds=0.05,
        max_attempts=1,
        backoff=_NO_DELAY,
    )

    with pytest.raises(ModelRuntimeTimeoutError):
        run(runtime.generate(RuntimeRequest(prompt="Solve it.")))

    assert counter.get("started") == 1
    assert counter.get("cancelled") == 1


def test_caller_cancellation_propagates() -> None:
    async def scenario() -> None:
        counter: dict[str, int] = {}
        client = FakeClient(hangs_until_cancelled(counter))
        runtime = OllamaRuntime(
            client=client,
            timeout_seconds=5.0,
            max_attempts=3,
            backoff=_NO_DELAY,
        )

        task = asyncio.create_task(runtime.generate(RuntimeRequest(prompt="Solve it.")))
        for _ in range(5):
            await asyncio.sleep(0)
        assert counter.get("started") == 1

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert counter.get("cancelled") == 1

    run(scenario())


# --- Client ownership -----------------------------------------------------


def test_owned_client_is_closed_on_aclose() -> None:
    async def scenario() -> None:
        runtime = OllamaRuntime()
        owned = runtime._owned_client
        assert owned is not None
        assert owned._client.is_closed is False

        await runtime.aclose()
        assert owned._client.is_closed is True

    run(scenario())


def test_injected_client_is_not_closed() -> None:
    class ClosableFakeClient(FakeClient):
        def __init__(self) -> None:
            super().__init__(returns(generate_response()))
            self.closed = False

        async def close(self) -> None:
            self.closed = True

    async def scenario() -> None:
        client = ClosableFakeClient()
        runtime = OllamaRuntime(client=client)
        assert runtime._owned_client is None

        await runtime.aclose()
        assert client.closed is False

    run(scenario())
