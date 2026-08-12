"""Anthropic Claude text-generation adapter.

Implements the provider-independent :class:`ModelProvider` contract against
Anthropic's asynchronous Messages API. All Anthropic- and reliability-specific
concepts (SDK client, exception classification, retry/timeout/rate-limit
policy, ``Retry-After`` parsing) live inside this module so that the rest of
the codebase depends only on the provider-agnostic contract and error
hierarchy.

Reliability is delegated to RelPrim, which is the single owner of retry,
timeout, and rate-limit semantics for this network boundary. The Anthropic
SDK's own retry logic is disabled so that RelPrim is the only retry owner.
"""

from __future__ import annotations

import math
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from anthropic import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    AsyncAnthropic,
    RateLimitError,
)
from anthropic.types import Message
from relprim import (
    AsyncOperation,
    ExponentialBackoff,
    OperationExecutionError,
    OperationTimeoutError,
    RateLimitPolicy,
    RetryPolicy,
    TimeoutPolicy,
    async_operation,
)

from llmforeman_core import ModelUsage
from llmforeman_providers.contracts import ModelRequest, ModelResponse
from llmforeman_providers.errors import (
    ModelProviderError,
    ModelProviderPermanentError,
    ModelProviderRateLimitError,
    ModelProviderTimeoutError,
    ModelProviderTransientError,
)

__all__ = ["AnthropicMessagesClient", "AnthropicProvider"]

_MessageCreate = Callable[..., Awaitable[Message]]


class _MessagesResource(Protocol):
    """The single Anthropic Messages entry point this adapter depends on."""

    def create(self, **kwargs: Any) -> Awaitable[Message]: ...


class AnthropicMessagesClient(Protocol):
    """Minimal async Anthropic client seam used for testable injection.

    Only ``client.messages.create`` is exercised by the adapter; this Protocol
    exists solely so a narrow fake can be supplied in tests without a network
    call. It is intentionally not a general SDK abstraction.
    """

    @property
    def messages(self) -> _MessagesResource: ...


def _parse_retry_after_header(raw: str | None) -> float | None:
    """Parse an Anthropic ``Retry-After`` header value into seconds.

    Returns a finite, non-negative number of seconds when the header is present
    and usable; returns ``None`` when it is missing, malformed, negative, or
    non-finite. Never raises for a malformed header.
    """

    if raw is None:
        return None

    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(value) or value < 0:
        return None

    return value


def _rate_limit_retry_after(exception: Exception) -> float | None:
    """RelPrim retry-after extractor over the normalized rate-limit error."""

    if isinstance(exception, ModelProviderRateLimitError):
        return exception.retry_after_seconds
    return None


def _extract_retry_after(exception: APIStatusError) -> float | None:
    """Read and parse the ``Retry-After`` header from an Anthropic response."""

    return _parse_retry_after_header(exception.response.headers.get("retry-after"))


def _translate_anthropic_error(exception: APIError) -> ModelProviderError:
    """Translate an Anthropic SDK error into the provider-independent contract.

    Classification is status/type based, never message-string based. Only
    genuinely transient conditions become transient errors; every other
    request/client fault is permanent so it is not retried.
    """

    if isinstance(exception, RateLimitError):
        return ModelProviderRateLimitError(
            "Anthropic rate limit exceeded.",
            retry_after_seconds=_extract_retry_after(exception),
        )

    if isinstance(exception, APITimeoutError):
        return ModelProviderTimeoutError("Anthropic request timed out.")

    if isinstance(exception, APIConnectionError):
        return ModelProviderTransientError("Anthropic connection error.")

    if isinstance(exception, APIStatusError):
        status_code = exception.status_code
        if status_code == 408:
            return ModelProviderTimeoutError("Anthropic request timed out.")
        if status_code == 409:
            return ModelProviderTransientError("Anthropic request conflict.")
        if status_code >= 500:
            return ModelProviderTransientError(
                f"Anthropic server error (status {status_code})."
            )
        return ModelProviderPermanentError(
            f"Anthropic rejected the request (status {status_code})."
        )

    return ModelProviderPermanentError("Anthropic request failed.")


def _extract_text(message: Message) -> str:
    """Concatenate visible text blocks in provider order.

    Non-text blocks (including thinking and redacted thinking) are ignored.
    Their contents are never inspected. An absence of text blocks yields an
    empty string rather than a failure.
    """

    return "".join(block.text for block in message.content if block.type == "text")


def _normalize_usage(message: Message) -> ModelUsage:
    """Map Anthropic usage counters into the provider-independent model.

    Missing/``None`` cache counters normalize to zero; no totals are derived.
    """

    usage = message.usage
    return ModelUsage(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_read_input_tokens=usage.cache_read_input_tokens or 0,
        cache_creation_input_tokens=usage.cache_creation_input_tokens or 0,
    )


def _normalize_response(message: Message) -> ModelResponse:
    """Normalize an Anthropic ``Message`` into a :class:`ModelResponse`."""

    return ModelResponse(
        content=_extract_text(message),
        usage=_normalize_usage(message),
    )


class AnthropicProvider:
    """Anthropic Claude implementation of the :class:`ModelProvider` contract.

    Performs a single asynchronous Messages API request per attempt, wrapped by
    a RelPrim reliability boundary that owns retry, timeout, and rate-limit
    behavior. The adapter holds no per-request mutable state, so ``generate``
    is safe for concurrent use with a client that supports concurrent requests.
    """

    _MODEL: str = "claude-opus-5"
    _OPERATION_NAME: str = "anthropic.messages.create"

    def __init__(
        self,
        *,
        max_tokens: int,
        timeout_seconds: float = 300.0,
        max_attempts: int = 3,
        max_rate_limit_wait_seconds: float = 30.0,
        api_key: str | None = None,
        client: AnthropicMessagesClient | None = None,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer.")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than 0.")
        if max_attempts < 1:
            raise ValueError("max_attempts must be greater than or equal to 1.")
        if max_rate_limit_wait_seconds < 0:
            raise ValueError("max_rate_limit_wait_seconds must be greater than or equal to 0.")

        self._max_tokens = max_tokens

        self._owned_client: AsyncAnthropic | None
        if client is None:
            owned = self._build_client(api_key)
            self._owned_client = owned
            self._create: _MessageCreate = owned.messages.create
        else:
            self._owned_client = None
            self._create = client.messages.create

        self._retry_policy = RetryPolicy(
            max_attempts=max_attempts,
            backoff=ExponentialBackoff(),
            retry_on=(ModelProviderTransientError, OperationTimeoutError),
        )
        self._timeout_policy = TimeoutPolicy(seconds=timeout_seconds)
        self._rate_limit_policy = RateLimitPolicy(
            rate_limit_on=(ModelProviderRateLimitError,),
            retry_after=_rate_limit_retry_after,
            max_wait_seconds=max_rate_limit_wait_seconds,
        )

    @staticmethod
    def _build_client(api_key: str | None) -> AsyncAnthropic:
        """Construct the provider-owned Anthropic client.

        SDK automatic retries are disabled (``max_retries=0``) so RelPrim is the
        only retry owner, and the transport request timeout is disabled
        (``timeout=None``) so RelPrim owns the request-attempt timeout boundary.
        Credential resolution follows the SDK's standard mechanism unless an
        explicit ``api_key`` is supplied.
        """

        return AsyncAnthropic(api_key=api_key, max_retries=0, timeout=None)

    def _build_request_kwargs(self, request: ModelRequest) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self._MODEL,
            "max_tokens": self._max_tokens,
            "messages": [{"role": "user", "content": request.prompt}],
        }
        if request.system_prompt is not None:
            kwargs["system"] = request.system_prompt
        return kwargs

    def _resilient_operation(
        self,
        request: ModelRequest,
    ) -> AsyncOperation[[], Message]:
        request_kwargs = self._build_request_kwargs(request)

        async def attempt() -> Message:
            try:
                return await self._create(**request_kwargs)
            except APIError as exc:
                raise _translate_anthropic_error(exc) from exc

        return (
            async_operation(self._OPERATION_NAME, attempt)
            .with_timeout(self._timeout_policy)
            .with_rate_limit(self._rate_limit_policy)
            .with_retry(self._retry_policy)
        )

    async def generate(self, request: ModelRequest) -> ModelResponse:
        """Generate a response for ``request`` via the Anthropic Messages API.

        On failure, exposes a normalized provider error while preserving
        exception chaining for diagnostics. Already-normalized provider errors
        are re-raised with their original cause intact; a RelPrim timeout
        becomes a provider timeout error; anything else defaults safely to the
        base provider error rather than being treated as retryable.
        """

        try:
            result = await self._resilient_operation(request).run()
        except OperationExecutionError as exc:
            cause = exc.cause
            if isinstance(cause, ModelProviderError):
                raise cause from cause.__cause__
            if isinstance(cause, OperationTimeoutError):
                raise ModelProviderTimeoutError("Anthropic request timed out.") from cause
            raise ModelProviderError("Anthropic request failed unexpectedly.") from cause

        return _normalize_response(result.value)

    async def aclose(self) -> None:
        """Close the provider-owned Anthropic client, if any.

        Injected clients are caller-owned and are never closed here.
        """

        if self._owned_client is not None:
            await self._owned_client.close()

    async def __aenter__(self) -> AnthropicProvider:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()
