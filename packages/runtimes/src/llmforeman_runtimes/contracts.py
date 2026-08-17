"""Runtime-agnostic local model generation contract.

Defines the minimal, runtime-independent types and interface that future local
inference runtime adapters (e.g. Ollama, MLX, llama.cpp) satisfy so that typed
application code can depend on::

    response = await runtime.generate(request)

without knowing which concrete local inference engine backs the call. This
module declares types and interface semantics only: it contains no runtime
clients, no external I/O, and no execution behavior.

A local *runtime* is deliberately distinct from a cloud *provider*
(``llmforeman_providers``). Even though the request/response shapes currently
resemble the provider contract, the two boundaries evolve independently and do
not reuse each other's types.
"""

from typing import Protocol

from pydantic import BaseModel, field_validator

from llmforeman_core import ModelUsage

__all__ = [
    "ModelRuntime",
    "RuntimeRequest",
    "RuntimeResponse",
    "StructuredModelRuntime",
    "StructuredRuntimeResponse",
]


class RuntimeRequest(BaseModel):
    """Runtime-independent input for a single local text-generation request.

    Represents only the input required for a simple local text-generation
    request. It deliberately carries no chat/message protocol, no model or
    runtime identity, and no inference configuration; those concerns are out of
    scope for this contract.
    """

    prompt: str
    system_prompt: str | None = None

    @field_validator("prompt")
    @classmethod
    def _reject_blank_prompt(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("prompt must not be empty or whitespace-only")
        return value

    @field_validator("system_prompt")
    @classmethod
    def _reject_blank_system_prompt(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("system_prompt must not be empty or whitespace-only")
        return value


class RuntimeResponse(BaseModel):
    """Runtime-independent result of a local text-generation request.

    Carries only the normalized textual output and the runtime-independent
    token usage. Raw runtime response objects, model identity, finish reasons,
    timing, and arbitrary runtime metadata are intentionally out of scope. An
    empty textual ``content`` is structurally valid: a runtime may legitimately
    produce an empty result in edge cases.
    """

    content: str
    usage: ModelUsage


class ModelRuntime(Protocol):
    """Typed, async runtime-agnostic generation contract.

    A structural interface that concrete local inference runtime adapters
    satisfy. It is stateless and holds no registry, configuration, model
    selection, retries, timeouts, or lifecycle hooks. It exists only so typed
    application code can depend on ``await runtime.generate(request)``.
    """

    async def generate(self, request: RuntimeRequest) -> RuntimeResponse:
        """Generate a response for ``request``.

        Asynchronous from day one because real local runtime interactions will
        involve process/local-server I/O, cancellation, and orchestration
        concurrency. This declaration defines the contract only and implements
        no execution behavior.
        """
        ...


class StructuredRuntimeResponse[T: BaseModel](BaseModel):
    """Runtime-independent result of a structured local-generation request.

    Generic over the caller-supplied Pydantic output type ``T`` so that the
    concrete requested type is preserved statically: requesting ``Foo`` yields a
    ``StructuredRuntimeResponse[Foo]`` whose ``output`` is a ``Foo``, not a bare
    ``BaseModel`` or ``dict[str, Any]``. A successful response always carries a
    fully validated ``output`` and normalized token ``usage``; partial,
    unparsed, or otherwise invalid results are not representable here. Runtimes
    that cannot produce a valid result fail through the runtime error boundary
    instead of returning this type. The raw runtime payload and the JSON Schema
    derived from ``T`` are adapter implementation details and are intentionally
    absent from this contract.
    """

    output: T
    usage: ModelUsage


class StructuredModelRuntime(Protocol):
    """Typed, async structured local-generation capability.

    An orthogonal structural capability, separate from ``ModelRuntime``: a
    caller that needs a validated typed result depends on this interface
    directly rather than probing whether a plain ``ModelRuntime`` happens to
    support a schema mode. It deliberately does not inherit from ``ModelRuntime``
    and does not require plain-text ``generate`` support; a concrete adapter may
    structurally satisfy both contracts. This mirrors the provider-side
    structured design without sharing its types. This declaration defines
    semantics only and performs no schema translation, runtime I/O, or JSON
    parsing.
    """

    async def generate_structured[T: BaseModel](
        self,
        request: RuntimeRequest,
        output_type: type[T],
    ) -> StructuredRuntimeResponse[T]:
        """Generate output conforming to the Pydantic ``output_type``.

        Reuses the existing text-generation ``RuntimeRequest`` for
        prompt/system input; the distinction from ``generate`` is solely the
        typed output contract. Returns a response whose ``output`` is a
        validated instance of ``output_type``, preserving that concrete type
        statically. Callers supply a Python type, never a raw JSON Schema.
        """
        ...
