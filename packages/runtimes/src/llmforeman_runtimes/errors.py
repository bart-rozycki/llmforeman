"""Runtime-independent failure contract for local model runtimes.

Concrete runtime adapters (e.g. Ollama) translate their SDK-specific and
transport exceptions into this small, runtime-agnostic hierarchy at the adapter
boundary. Ordinary callers of :meth:`ModelRuntime.generate` can therefore
distinguish failure categories without importing a runtime SDK or the
underlying reliability library.

The hierarchy is deliberately minimal: it captures only the distinctions a real
local runtime integration currently needs (permanent vs. transient, plus the
timeout sub-case that carries actionable semantics). It intentionally avoids
HTTP-status-specific classes, numeric codes, and serialization machinery.

This is a separate hierarchy from the cloud provider error contract in
``llmforeman_providers``: local runtimes and cloud providers are distinct
concepts whose failure semantics evolve independently and are not shared.
"""

__all__ = [
    "ModelRuntimeError",
    "ModelRuntimePermanentError",
    "ModelRuntimeStructuredOutputError",
    "ModelRuntimeTimeoutError",
    "ModelRuntimeTransientError",
]


class ModelRuntimeError(Exception):
    """Base class for all runtime-independent generation failures.

    Raised (directly or via a subclass) from ``ModelRuntime.generate`` when a
    runtime cannot produce a response. Unknown or unclassifiable failures use
    this base type and are treated as non-retryable by default.
    """


class ModelRuntimePermanentError(ModelRuntimeError):
    """A failure that will not succeed on retry.

    Represents caller/runtime faults such as invalid requests, a missing model,
    and responses that cannot be normalized into the runtime contract.
    """


class ModelRuntimeStructuredOutputError(ModelRuntimePermanentError):
    """A completed generation whose final output is not usable structured data.

    Raised when a runtime obtained a model response but its final ``response``
    text is not valid JSON for, or does not validate against, the requested
    Pydantic output type (including an empty or whitespace-only response). It is
    a permanent fault for the completed attempt: the model already produced its
    output, so re-running the same transport call is not a remedy and must not
    be retried. To avoid surfacing potentially sensitive model output, the raw
    response is never embedded in the message; the originating validation error
    is preserved via exception chaining instead.
    """


class ModelRuntimeTransientError(ModelRuntimeError):
    """A failure that may succeed if the request is retried.

    Represents genuinely transient conditions such as connection faults,
    transport interruptions, and server-side errors.
    """


class ModelRuntimeTimeoutError(ModelRuntimeTransientError):
    """A transient failure caused by an attempt exceeding its time budget."""
