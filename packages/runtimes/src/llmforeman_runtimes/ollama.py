"""Ollama local inference runtime adapter.

Implements the runtime-independent :class:`ModelRuntime` contract against a
running Ollama server via the official asynchronous ``ollama.AsyncClient``. All
Ollama- and reliability-specific concepts (SDK client, exception
classification, retry/timeout policy) live inside this module so that the rest
of the codebase depends only on the runtime-agnostic contract and error
hierarchy.

Reliability is delegated to RelPrim, which is the single owner of retry and
timeout semantics for this local-server boundary. The Ollama client is
constructed without its own request timeout so that RelPrim owns the
request-attempt timeout boundary and no competing timeout exists.

A local *runtime* is deliberately distinct from a cloud *provider*: this module
shares no types, error classes, or reliability helpers with
``llmforeman_providers``.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Protocol, cast

import httpx
from ollama import AsyncClient, GenerateResponse, ResponseError
from pydantic import BaseModel, ValidationError
from relprim import (
    AsyncOperation,
    ExponentialBackoff,
    OperationExecutionError,
    OperationTimeoutError,
    RetryPolicy,
    TimeoutPolicy,
    async_operation,
)

from llmforeman_core import ModelUsage
from llmforeman_runtimes.contracts import (
    RuntimeRequest,
    RuntimeResponse,
    StructuredRuntimeResponse,
)
from llmforeman_runtimes.errors import (
    ModelRuntimeError,
    ModelRuntimePermanentError,
    ModelRuntimeStructuredOutputError,
    ModelRuntimeTimeoutError,
    ModelRuntimeTransientError,
)

__all__ = ["OllamaGenerateClient", "OllamaRuntime"]

_DEFAULT_MODEL = "qwen3.6:35b-a3b"
_DEFAULT_TIMEOUT_SECONDS = 300.0
_DEFAULT_MAX_ATTEMPTS = 3

_GenerateCallable = Callable[..., Awaitable[GenerateResponse]]

# httpx transport failures the Ollama SDK does not normalize itself but which
# are genuinely worth retrying: transport timeouts, network faults (read/write/
# close), and a server that disconnects mid-response. Local protocol misuse and
# unsupported-protocol errors are deliberately excluded as they are permanent.
_TRANSIENT_TRANSPORT_ERRORS: tuple[type[Exception], ...] = (
    httpx.TimeoutException,
    httpx.NetworkError,
    httpx.RemoteProtocolError,
)


class OllamaGenerateClient(Protocol):
    """Minimal async Ollama client seam used for testable injection.

    Only ``client.generate(...)`` is exercised by the adapter; this Protocol
    exists solely so a narrow fake can be supplied in tests without a running
    Ollama server. It is intentionally not a general SDK abstraction.
    """

    def generate(self, **kwargs: Any) -> Awaitable[GenerateResponse]: ...


def _translate_ollama_error(exception: Exception) -> ModelRuntimeError:
    """Translate an expected Ollama/transport failure into the runtime contract.

    Classification is status/type based, never message-string based. Only
    genuinely transient conditions become transient errors; ordinary client
    faults (including ``404`` model-not-found) are permanent so they are not
    retried.
    """

    if isinstance(exception, ResponseError):
        status_code = exception.status_code
        if status_code == 408:
            return ModelRuntimeTransientError("Ollama request timed out.")
        if status_code >= 500:
            return ModelRuntimeTransientError(
                f"Ollama server error (status {status_code})."
            )
        if status_code >= 400:
            return ModelRuntimePermanentError(
                f"Ollama rejected the request (status {status_code})."
            )
        # A status the SDK could not associate with a response (e.g. -1) is not
        # safely retryable; treat it as permanent rather than guessing.
        return ModelRuntimePermanentError("Ollama request failed.")

    if isinstance(exception, ConnectionError):
        # The Ollama SDK normalizes a failed connection to the built-in
        # ConnectionError; retrying may succeed once the server is reachable.
        return ModelRuntimeTransientError("Ollama connection error.")

    if isinstance(exception, _TRANSIENT_TRANSPORT_ERRORS):
        return ModelRuntimeTransientError("Ollama transport error.")

    return ModelRuntimePermanentError("Ollama request failed.")


def _normalize_usage(response: GenerateResponse) -> ModelUsage:
    """Map Ollama token counters into the runtime-independent usage model.

    A successful non-streaming response is expected to report both counters.
    When either is absent the response is unusable for the current normalized
    contract; that is surfaced as a permanent runtime failure rather than being
    silently reinterpreted as a genuine measured zero. Ollama timing fields are
    intentionally not mapped: timing belongs to future execution telemetry.
    """

    input_tokens = response.prompt_eval_count
    output_tokens = response.eval_count
    if input_tokens is None or output_tokens is None:
        raise ModelRuntimePermanentError(
            "Ollama response is missing required token usage counters."
        )

    return ModelUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=0,
    )


def _normalize_response(response: GenerateResponse) -> RuntimeResponse:
    """Normalize an Ollama ``GenerateResponse`` into a :class:`RuntimeResponse`.

    Only the final ``response`` text is used; any separate ``thinking`` output
    is ignored and never inspected. A missing final response normalizes to an
    empty string, which the runtime contract explicitly permits.
    """

    return RuntimeResponse(
        content=response.response or "",
        usage=_normalize_usage(response),
    )


def _validate_structured_output[T: BaseModel](
    response: GenerateResponse,
    output_type: type[T],
) -> T:
    """Validate the final Ollama ``response`` text as exactly ``output_type``.

    Only the final ``response`` field is parsed; any separate ``thinking``
    output is ignored and never concatenated. Pydantic is the single source of
    truth: the text is validated directly with ``model_validate_json`` and no
    JSON extraction, fence stripping, or repair is attempted. A missing, empty,
    whitespace-only, syntactically invalid, or schema-violating response is an
    unusable structured result and becomes a permanent structured-output error
    that never embeds the raw model output.
    """

    text = response.response
    if text is None:
        raise ModelRuntimeStructuredOutputError(
            f"Ollama returned no structured output for {output_type.__name__}."
        )
    try:
        return output_type.model_validate_json(text)
    except ValidationError as exc:
        raise ModelRuntimeStructuredOutputError(
            f"Ollama returned invalid structured output for {output_type.__name__}."
        ) from exc


def _normalize_structured_response[T: BaseModel](
    response: GenerateResponse,
    output_type: type[T],
) -> StructuredRuntimeResponse[T]:
    """Build a :class:`StructuredRuntimeResponse` from an Ollama response.

    Usage is normalized with the exact same helper as plain generation (so a
    missing required token counter fails identically), and the final response
    text is validated as ``output_type``. Both failure modes are permanent and
    occur after the transport reliability boundary, so neither triggers a
    generation retry, and no partial result is ever returned.
    """

    usage = _normalize_usage(response)
    output = _validate_structured_output(response, output_type)
    return StructuredRuntimeResponse[T](output=output, usage=usage)


class OllamaRuntime:
    """Ollama implementation of the local runtime generation contracts.

    Structurally satisfies both :class:`ModelRuntime` (plain ``generate``) and
    :class:`StructuredModelRuntime` (typed ``generate_structured``) without
    either Protocol inheriting from the other. Performs a single asynchronous
    ``generate`` request per attempt, wrapped by a RelPrim reliability boundary
    that owns retry and timeout behavior; structured generation reuses that same
    single boundary and adds only Ollama's schema ``format`` argument. The
    adapter holds no per-request mutable state, so both methods are safe for
    concurrent use with a client that supports concurrent requests.
    """

    _OPERATION_NAME: str = "ollama.generate"

    def __init__(
        self,
        *,
        model: str = _DEFAULT_MODEL,
        host: str | None = None,
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
        max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
        backoff: ExponentialBackoff | None = None,
        client: OllamaGenerateClient | None = None,
    ) -> None:
        if not model.strip():
            raise ValueError("model must not be empty or whitespace-only.")
        if host is not None and not host.strip():
            raise ValueError("host must not be empty or whitespace-only.")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than 0.")
        if max_attempts < 1:
            raise ValueError("max_attempts must be greater than or equal to 1.")

        self._model = model

        self._owned_client: AsyncClient | None
        if client is None:
            owned = self._build_client(host)
            self._owned_client = owned
            # stream=False (enforced below) guarantees a GenerateResponse; the
            # SDK's broad Union return type is narrowed here at the single seam.
            self._generate: _GenerateCallable = cast("_GenerateCallable", owned.generate)
        else:
            self._owned_client = None
            self._generate = client.generate

        self._retry_policy = RetryPolicy(
            max_attempts=max_attempts,
            backoff=backoff if backoff is not None else ExponentialBackoff(),
            retry_on=(ModelRuntimeTransientError, OperationTimeoutError),
        )
        self._timeout_policy = TimeoutPolicy(seconds=timeout_seconds)

    @property
    def model(self) -> str:
        """The model this runtime instance generates with."""

        return self._model

    @staticmethod
    def _build_client(host: str | None) -> AsyncClient:
        """Construct the runtime-owned Ollama client.

        The transport request timeout is disabled (``timeout=None``) so RelPrim
        owns the request-attempt timeout boundary and no competing timeout
        exists. When ``host`` is ``None`` the SDK's normal default/environment
        host resolution is preserved rather than duplicated.
        """

        return AsyncClient(host=host, timeout=None)

    def _build_request_kwargs(
        self,
        request: RuntimeRequest,
        *,
        format_schema: dict[str, Any] | None = None,
        think: bool | None = None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "prompt": request.prompt,
            "stream": False,
        }
        if request.system_prompt is not None:
            kwargs["system"] = request.system_prompt
        # ``format`` is added only for structured generation; plain generation
        # never sends it, so its request mapping is unchanged.
        if format_schema is not None:
            kwargs["format"] = format_schema
        # ``think`` is sent only when explicitly requested (structured
        # generation passes ``think=False``); plain generation leaves it
        # omitted so Ollama/model default thinking semantics are preserved.
        # Sampling keywords are deliberately still never set for either path.
        if think is not None:
            kwargs["think"] = think
        return kwargs

    def _resilient_operation(
        self,
        request: RuntimeRequest,
        *,
        format_schema: dict[str, Any] | None = None,
        think: bool | None = None,
    ) -> AsyncOperation[[], GenerateResponse]:
        request_kwargs = self._build_request_kwargs(
            request, format_schema=format_schema, think=think
        )

        async def attempt() -> GenerateResponse:
            try:
                return await self._generate(**request_kwargs)
            except (ResponseError, ConnectionError, httpx.TransportError) as exc:
                raise _translate_ollama_error(exc) from exc

        return (
            async_operation(self._OPERATION_NAME, attempt)
            .with_timeout(self._timeout_policy)
            .with_retry(self._retry_policy)
        )

    async def _generate_once(
        self,
        request: RuntimeRequest,
        *,
        format_schema: dict[str, Any] | None = None,
        think: bool | None = None,
    ) -> GenerateResponse:
        """Run one Ollama generation through the shared reliability boundary.

        Both plain and structured generation route through this single helper so
        RelPrim remains the one owner of retry and timeout semantics and there
        is exactly one transport reliability path. Structured generation differs
        only by supplying ``format_schema``; the returned ``GenerateResponse``
        is normalized by the caller *after* this boundary, so any structured
        parsing/validation happens outside the retry loop.

        On failure, exposes a normalized runtime error while preserving
        exception chaining for diagnostics. Already-normalized runtime errors
        are re-raised with their original cause intact; a RelPrim timeout
        becomes a runtime timeout error; anything else defaults safely to the
        base runtime error rather than being treated as retryable. Caller
        cancellation is never converted into a runtime error.
        """

        try:
            result = await self._resilient_operation(
                request, format_schema=format_schema, think=think
            ).run()
        except OperationExecutionError as exc:
            cause = exc.cause
            if isinstance(cause, ModelRuntimeError):
                raise cause from cause.__cause__
            if isinstance(cause, OperationTimeoutError):
                raise ModelRuntimeTimeoutError("Ollama request timed out.") from cause
            raise ModelRuntimeError("Ollama request failed unexpectedly.") from cause

        # RelPrim ships without a ``py.typed`` marker, so its result value is
        # seen as ``Any``; the attempt callable fixes it to GenerateResponse.
        return cast("GenerateResponse", result.value)

    async def generate(self, request: RuntimeRequest) -> RuntimeResponse:
        """Generate a response for ``request`` via the Ollama generate API.

        Behaviorally unchanged: uses the configured model, maps prompt/system as
        before, forces ``stream=False``, returns the final ``response`` text as
        plain content (ignoring ``thinking``), normalizes usage as before, and
        uses the existing RelPrim reliability/error semantics.
        """

        response = await self._generate_once(request)
        return _normalize_response(response)

    async def generate_structured[T: BaseModel](
        self,
        request: RuntimeRequest,
        output_type: type[T],
    ) -> StructuredRuntimeResponse[T]:
        """Generate a validated ``output_type`` instance via structured output.

        Reuses the exact plain-generation model/prompt/system mapping and
        ``stream=False`` transport, adding only Ollama's ``format`` argument set
        to ``output_type.model_json_schema()`` so the model is constrained to
        the requested schema, and ``think=False`` so the final ``response``
        field carries the structured protocol payload rather than reasoning
        text (which is never a substitute for, or fallback to, the structured
        output). The prompt is never mutated and no schema text is injected into
        the prompt or system prompt. The single RelPrim transport boundary is
        shared with plain generation; the final ``response`` text is validated
        as ``output_type`` *after* that boundary, so an invalid or unusable
        structured output fails as a permanent
        :class:`ModelRuntimeStructuredOutputError` and is never retried.
        """

        schema = output_type.model_json_schema()
        response = await self._generate_once(request, format_schema=schema, think=False)
        return _normalize_structured_response(response, output_type)

    async def aclose(self) -> None:
        """Close the runtime-owned Ollama client, if any.

        Injected clients are caller-owned and are never closed here.
        """

        if self._owned_client is not None:
            # ollama's AsyncClient.close() carries no return annotation, so a
            # strict-mode call is flagged; the SDK method is the intended,
            # smallest correct lifecycle API for the owned client.
            await self._owned_client.close()  # type: ignore[no-untyped-call]

    async def __aenter__(self) -> OllamaRuntime:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()
