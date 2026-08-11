"""Provider-agnostic model generation contract.

Defines the minimal, provider-independent types and interface that future
cloud provider adapters (e.g. Anthropic, OpenAI, Gemini) satisfy so that typed
application code can depend on::

    response = await provider.generate(request)

without knowing which concrete provider backs the call. This module declares
types and interface semantics only: it contains no provider SDKs, no external
I/O, and no execution behavior.
"""

from typing import Protocol

from pydantic import BaseModel, field_validator

from llmforeman_core import ModelUsage

__all__ = [
    "ModelProvider",
    "ModelRequest",
    "ModelResponse",
]


class ModelRequest(BaseModel):
    """Provider-independent input for a single text-generation request.

    Represents only the input required for the first simple text-generation
    request. It deliberately carries no chat/message protocol, no model or
    provider identity, and no provider configuration; those concerns are out of
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


class ModelResponse(BaseModel):
    """Provider-independent result of a text-generation request.

    Carries only the normalized textual output and the provider-independent
    token usage. Provider response objects, stop/finish reasons, tool calls,
    citations, and arbitrary provider metadata are intentionally out of scope.
    An empty textual ``content`` is structurally valid: a provider may
    legitimately produce an empty result in edge cases.
    """

    content: str
    usage: ModelUsage


class ModelProvider(Protocol):
    """Typed, async provider-agnostic generation contract.

    A structural interface that concrete cloud provider adapters satisfy. It is
    stateless and holds no registry, configuration, credentials, model
    selection, retries, timeouts, or lifecycle hooks. It exists only so typed
    application code can depend on ``await provider.generate(request)``.
    """

    async def generate(self, request: ModelRequest) -> ModelResponse:
        """Generate a response for ``request``.

        Asynchronous from day one because real provider interactions will
        involve network I/O, cancellation, and orchestration concurrency. This
        declaration defines the contract only and implements no execution
        behavior.
        """
        ...
