"""Behavioral and typing tests for Ollama structured generation.

Every test uses an injected fake client; no real Ollama server, no downloaded
model, and no network access are ever required. Async tests are driven with
``asyncio.run`` and retry-heavy tests use a zero-delay backoff, matching the
plain-generation test suite. The focus is on the structured-output additions:
the exact ``format`` schema, request-mapping parity with plain generation,
typed/validated output, generic type preservation, usage reuse, the
non-retryable structured-output failure modes, and preserved transport
reliability.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import Any, assert_type

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
from pydantic import BaseModel
from relprim import ExponentialBackoff

from llmforeman_core import ModelUsage
from llmforeman_runtimes import (
    ModelRuntime,
    ModelRuntimePermanentError,
    ModelRuntimeStructuredOutputError,
    ModelRuntimeTimeoutError,
    ModelRuntimeTransientError,
    OllamaRuntime,
    RuntimeRequest,
    StructuredModelRuntime,
    StructuredRuntimeResponse,
)

_NO_DELAY = ExponentialBackoff(base_delay_seconds=0.0, max_delay_seconds=0.0, jitter=False)

_VALID_JSON = '{"action":"read","count":1}'


class ExampleOutput(BaseModel):
    """Tiny test-only schema standing in for a caller-supplied output model."""

    action: str
    count: int


def run[T](coro: Awaitable[T]) -> T:
    return asyncio.run(coro)  # type: ignore[arg-type]


def _fast_runtime(client: FakeClient, **kwargs: Any) -> OllamaRuntime:
    params: dict[str, Any] = {"backoff": _NO_DELAY}
    params.update(kwargs)
    return OllamaRuntime(client=client, **params)


def _connection_error() -> ConnectionError:
    return ConnectionError("could not connect to ollama")


# --- Structural capability ------------------------------------------------


def test_satisfies_both_runtime_protocols_structurally() -> None:
    # Static structural typing: the concrete runtime is assignable to both
    # orthogonal capability bindings (mypy proves the structural match; these
    # bindings are not runtime ``isinstance`` probes).
    runtime = OllamaRuntime(client=FakeClient(returns(generate_response())))
    plain: ModelRuntime = runtime
    structured: StructuredModelRuntime = runtime
    assert plain is runtime
    assert structured is runtime


# --- Format schema mapping ------------------------------------------------


def test_structured_passes_exact_pydantic_schema_as_format() -> None:
    client = FakeClient(returns(generate_response(response=_VALID_JSON)))
    runtime = _fast_runtime(client)

    run(
        runtime.generate_structured(
            RuntimeRequest(prompt="Choose an action."),
            ExampleOutput,
        )
    )

    assert client.calls[0]["format"] == ExampleOutput.model_json_schema()


# --- Request mapping parity with plain generation -------------------------


def test_structured_request_maps_model_prompt_and_non_streaming() -> None:
    client = FakeClient(returns(generate_response(response=_VALID_JSON)))
    runtime = _fast_runtime(client, model="codellama:13b")

    run(
        runtime.generate_structured(
            RuntimeRequest(prompt="Choose an action."),
            ExampleOutput,
        )
    )

    kwargs = client.calls[0]
    assert kwargs["model"] == "codellama:13b"
    assert kwargs["prompt"] == "Choose an action."
    assert kwargs["stream"] is False


def test_structured_system_prompt_omitted_when_absent() -> None:
    client = FakeClient(returns(generate_response(response=_VALID_JSON)))
    runtime = _fast_runtime(client)

    run(
        runtime.generate_structured(
            RuntimeRequest(prompt="Choose an action."),
            ExampleOutput,
        )
    )

    assert "system" not in client.calls[0]


def test_structured_system_prompt_mapped_when_present() -> None:
    client = FakeClient(returns(generate_response(response=_VALID_JSON)))
    runtime = _fast_runtime(client)

    run(
        runtime.generate_structured(
            RuntimeRequest(
                prompt="Choose an action.",
                system_prompt="You are a coding worker.",
            ),
            ExampleOutput,
        )
    )

    assert client.calls[0]["system"] == "You are a coding worker."


def test_structured_request_mapping_matches_plain_except_format() -> None:
    plain_client = FakeClient(returns(generate_response(response=_VALID_JSON)))
    structured_client = FakeClient(returns(generate_response(response=_VALID_JSON)))
    plain_runtime = _fast_runtime(plain_client)
    structured_runtime = _fast_runtime(structured_client)
    request = RuntimeRequest(prompt="Choose an action.", system_prompt="Be precise.")

    run(plain_runtime.generate(request))
    run(structured_runtime.generate_structured(request, ExampleOutput))

    plain_kwargs = plain_client.calls[0]
    structured_kwargs = structured_client.calls[0]
    # Structured mapping differs from plain solely by the added ``format`` key.
    assert set(structured_kwargs) - set(plain_kwargs) == {"format"}
    assert {k: structured_kwargs[k] for k in plain_kwargs} == plain_kwargs


def test_structured_sends_no_thinking_or_sampling_keywords() -> None:
    client = FakeClient(returns(generate_response(response=_VALID_JSON)))
    runtime = _fast_runtime(client)

    run(
        runtime.generate_structured(
            RuntimeRequest(prompt="Choose an action."),
            ExampleOutput,
        )
    )

    kwargs = client.calls[0]
    # Task #28 does not change thinking/sampling configuration for structured
    # generation; only ``format`` is added relative to plain generation.
    for forbidden in (
        "think",
        "options",
        "temperature",
        "top_p",
        "top_k",
        "seed",
        "stop",
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


# --- Valid structured output ----------------------------------------------


def test_valid_structured_response_is_parsed_and_typed() -> None:
    client = FakeClient(returns(generate_response(response=_VALID_JSON)))
    runtime = _fast_runtime(client)

    result = run(
        runtime.generate_structured(
            RuntimeRequest(prompt="Choose an action."),
            ExampleOutput,
        )
    )

    assert isinstance(result, StructuredRuntimeResponse)
    assert isinstance(result.output, ExampleOutput)
    assert result.output.action == "read"
    assert result.output.count == 1


async def _preserves_output_type(runtime: OllamaRuntime) -> ExampleOutput:
    """Static type-preservation probe for the concrete runtime.

    mypy must infer ``result`` as ``StructuredRuntimeResponse[ExampleOutput]``
    and ``result.output`` as ``ExampleOutput`` (never ``BaseModel``/``Any``)
    with no ``cast``/``Any``/``type: ignore`` workaround.
    """

    result = await runtime.generate_structured(
        RuntimeRequest(prompt="Choose an action."),
        ExampleOutput,
    )
    assert_type(result, StructuredRuntimeResponse[ExampleOutput])
    assert_type(result.output, ExampleOutput)
    return result.output


def test_generic_type_preservation_probe_runs() -> None:
    client = FakeClient(returns(generate_response(response=_VALID_JSON)))
    runtime = _fast_runtime(client)

    output = run(_preserves_output_type(runtime))
    assert output == ExampleOutput(action="read", count=1)


# --- Usage normalization --------------------------------------------------


def test_structured_usage_normalized_from_ollama_counters() -> None:
    response = generate_response(response=_VALID_JSON, prompt_eval_count=123, eval_count=45)
    client = FakeClient(returns(response))
    runtime = _fast_runtime(client)

    result = run(
        runtime.generate_structured(
            RuntimeRequest(prompt="Choose an action."),
            ExampleOutput,
        )
    )

    assert result.usage == ModelUsage(
        input_tokens=123,
        output_tokens=45,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=0,
    )


@pytest.mark.parametrize(
    ("prompt_eval_count", "eval_count"),
    [(None, 5), (5, None), (None, None)],
)
def test_structured_missing_usage_counters_fail_without_retry(
    prompt_eval_count: int | None,
    eval_count: int | None,
) -> None:
    response = generate_response(
        response=_VALID_JSON,
        prompt_eval_count=prompt_eval_count,
        eval_count=eval_count,
    )
    client = FakeClient(returns(response))
    runtime = _fast_runtime(client, max_attempts=3)

    with pytest.raises(ModelRuntimePermanentError):
        run(
            runtime.generate_structured(
                RuntimeRequest(prompt="Choose an action."),
                ExampleOutput,
            )
        )

    # Usage normalization mirrors plain generation: one successful attempt only.
    assert client.call_count == 1


# --- Structured-output failure modes --------------------------------------


def test_invalid_json_is_structured_output_error() -> None:
    client = FakeClient(returns(generate_response(response="not-json")))
    runtime = _fast_runtime(client)

    with pytest.raises(ModelRuntimeStructuredOutputError) as excinfo:
        run(
            runtime.generate_structured(
                RuntimeRequest(prompt="Choose an action."),
                ExampleOutput,
            )
        )

    # The originating Pydantic validation error is preserved as the cause, but
    # the raw model output is not embedded in the message.
    assert excinfo.value.__cause__ is not None
    assert "not-json" not in str(excinfo.value)


def test_schema_validation_failure_is_structured_output_error() -> None:
    # Syntactically valid JSON that violates the model: ``count`` is required and
    # a non-numeric string is not coercible to ``int``.
    client = FakeClient(returns(generate_response(response='{"action":"read","count":"nope"}')))
    runtime = _fast_runtime(client)

    with pytest.raises(ModelRuntimeStructuredOutputError):
        run(
            runtime.generate_structured(
                RuntimeRequest(prompt="Choose an action."),
                ExampleOutput,
            )
        )


def test_missing_required_field_is_structured_output_error() -> None:
    client = FakeClient(returns(generate_response(response='{"action":"read"}')))
    runtime = _fast_runtime(client)

    with pytest.raises(ModelRuntimeStructuredOutputError):
        run(
            runtime.generate_structured(
                RuntimeRequest(prompt="Choose an action."),
                ExampleOutput,
            )
        )


@pytest.mark.parametrize("body", ["", "   ", None])
def test_empty_or_missing_response_is_structured_output_error(body: str | None) -> None:
    client = FakeClient(returns(generate_response(response=body)))
    runtime = _fast_runtime(client)

    with pytest.raises(ModelRuntimeStructuredOutputError):
        run(
            runtime.generate_structured(
                RuntimeRequest(prompt="Choose an action."),
                ExampleOutput,
            )
        )


def test_structured_output_error_is_a_runtime_error_and_permanent() -> None:
    # The structured-output error is a normal permanent runtime error, so the
    # reliability boundary never treats it as retryable.
    assert issubclass(ModelRuntimeStructuredOutputError, ModelRuntimePermanentError)
    assert not issubclass(ModelRuntimeStructuredOutputError, ModelRuntimeTransientError)


# --- Thinking is ignored --------------------------------------------------


def test_thinking_is_ignored_and_only_response_is_parsed() -> None:
    # ``thinking`` is deliberately NOT valid JSON while ``response`` is valid;
    # parsing only the final ``response`` must still succeed.
    response = generate_response(
        response=_VALID_JSON,
        thinking="I think the answer is not-json {",
    )
    client = FakeClient(returns(response))
    runtime = _fast_runtime(client)

    result = run(
        runtime.generate_structured(
            RuntimeRequest(prompt="Choose an action."),
            ExampleOutput,
        )
    )

    assert result.output == ExampleOutput(action="read", count=1)


# --- No hidden structured retry -------------------------------------------


def test_invalid_json_is_not_retried() -> None:
    client = FakeClient(returns(generate_response(response="not-json")))
    runtime = _fast_runtime(client, max_attempts=3)

    with pytest.raises(ModelRuntimeStructuredOutputError):
        run(
            runtime.generate_structured(
                RuntimeRequest(prompt="Choose an action."),
                ExampleOutput,
            )
        )

    # Exactly one generation: invalid structured output must never trigger the
    # transport retry mechanism.
    assert client.call_count == 1


def test_schema_validation_failure_is_not_retried() -> None:
    client = FakeClient(returns(generate_response(response='{"action":"read","count":"nope"}')))
    runtime = _fast_runtime(client, max_attempts=3)

    with pytest.raises(ModelRuntimeStructuredOutputError):
        run(
            runtime.generate_structured(
                RuntimeRequest(prompt="Choose an action."),
                ExampleOutput,
            )
        )

    assert client.call_count == 1


def test_empty_response_is_not_retried() -> None:
    client = FakeClient(returns(generate_response(response="")))
    runtime = _fast_runtime(client, max_attempts=3)

    with pytest.raises(ModelRuntimeStructuredOutputError):
        run(
            runtime.generate_structured(
                RuntimeRequest(prompt="Choose an action."),
                ExampleOutput,
            )
        )

    assert client.call_count == 1


# --- Preserved transport reliability --------------------------------------


def test_transient_transport_failure_still_retries_then_succeeds() -> None:
    client = FakeClient(
        raises_then_returns(
            _connection_error(),
            generate_response(response=_VALID_JSON),
            failures=1,
        )
    )
    runtime = _fast_runtime(client, max_attempts=3)

    result = run(
        runtime.generate_structured(
            RuntimeRequest(prompt="Choose an action."),
            ExampleOutput,
        )
    )

    assert result.output == ExampleOutput(action="read", count=1)
    # The existing bounded RelPrim retry is reused: one failure then success.
    assert client.call_count == 2


def test_permanent_transport_failure_does_not_retry_and_is_not_structured_error() -> None:
    error = ResponseError("model 'qwen3.6:35b-a3b' not found", 404)
    client = FakeClient(always_raises(error))
    runtime = _fast_runtime(client, max_attempts=3)

    with pytest.raises(ModelRuntimePermanentError) as excinfo:
        run(
            runtime.generate_structured(
                RuntimeRequest(prompt="Choose an action."),
                ExampleOutput,
            )
        )

    assert client.call_count == 1
    # A transport failure stays a transport error, never a structured-output one.
    assert not isinstance(excinfo.value, ModelRuntimeStructuredOutputError)
    assert isinstance(excinfo.value.__cause__, ResponseError)


def test_transient_exhaustion_raises_transient_error() -> None:
    client = FakeClient(always_raises(_connection_error()))
    runtime = _fast_runtime(client, max_attempts=3)

    with pytest.raises(ModelRuntimeTransientError):
        run(
            runtime.generate_structured(
                RuntimeRequest(prompt="Choose an action."),
                ExampleOutput,
            )
        )

    assert client.call_count == 3


def test_structured_timeout_uses_same_relprim_boundary() -> None:
    counter: dict[str, int] = {}
    client = FakeClient(hangs_until_cancelled(counter))
    runtime = OllamaRuntime(
        client=client,
        timeout_seconds=0.05,
        max_attempts=1,
        backoff=_NO_DELAY,
    )

    with pytest.raises(ModelRuntimeTimeoutError):
        run(
            runtime.generate_structured(
                RuntimeRequest(prompt="Choose an action."),
                ExampleOutput,
            )
        )

    assert counter.get("started") == 1
    assert counter.get("cancelled") == 1
