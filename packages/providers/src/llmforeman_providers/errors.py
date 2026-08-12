"""Provider-independent failure contract for model providers.

Concrete provider adapters (e.g. Anthropic) translate their SDK-specific
exceptions into this small, provider-agnostic hierarchy at the adapter
boundary. Ordinary callers of :meth:`ModelProvider.generate` can therefore
distinguish failure categories without importing any provider SDK or the
underlying reliability library.

The hierarchy is deliberately minimal: it captures only the distinctions that
a real provider integration currently needs (permanent vs. transient, plus the
two transient sub-cases that carry actionable semantics). It intentionally
avoids HTTP-status-specific classes, numeric codes, and serialization
machinery.
"""

__all__ = [
    "ModelProviderError",
    "ModelProviderPermanentError",
    "ModelProviderRateLimitError",
    "ModelProviderTimeoutError",
    "ModelProviderTransientError",
]


class ModelProviderError(Exception):
    """Base class for all provider-independent generation failures.

    Raised (directly or via a subclass) from ``ModelProvider.generate`` when a
    provider cannot produce a response. Unknown or unclassifiable failures use
    this base type and are treated as non-retryable by default.
    """


class ModelProviderPermanentError(ModelProviderError):
    """A failure that will not succeed on retry.

    Represents caller/request faults such as invalid requests, authentication
    or permission problems, missing resources, and unprocessable input.
    """


class ModelProviderTransientError(ModelProviderError):
    """A failure that may succeed if the request is retried.

    Represents genuinely transient conditions such as network faults,
    conflicts, and server-side errors.
    """


class ModelProviderRateLimitError(ModelProviderTransientError):
    """A transient failure caused by provider rate limiting.

    Carries the provider-recommended delay (in seconds) when one was supplied
    and could be parsed safely; ``None`` when no usable value was available.
    """

    def __init__(
        self,
        message: str,
        *,
        retry_after_seconds: float | None = None,
    ) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class ModelProviderTimeoutError(ModelProviderTransientError):
    """A transient failure caused by an attempt exceeding its time budget."""
