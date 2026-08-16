"""LLMForeman cloud provider integrations package.

Home of cloud LLM integrations. A cloud *provider* is distinct from a local
inference *runtime* (``llmforeman_runtimes``); the two boundaries are
deliberately kept separate.

Exposes the provider-agnostic generation contract, the provider-independent
error hierarchy, and the concrete Anthropic adapter.
"""

from importlib.metadata import version

from llmforeman_providers.anthropic import (
    AnthropicMessagesClient,
    AnthropicProvider,
)
from llmforeman_providers.contracts import (
    ModelProvider,
    ModelRequest,
    ModelResponse,
    StructuredModelProvider,
    StructuredModelResponse,
)
from llmforeman_providers.errors import (
    ModelProviderError,
    ModelProviderPermanentError,
    ModelProviderRateLimitError,
    ModelProviderTimeoutError,
    ModelProviderTransientError,
)
from llmforeman_providers.foreman import AnthropicForeman

__all__ = [
    "AnthropicForeman",
    "AnthropicMessagesClient",
    "AnthropicProvider",
    "ModelProvider",
    "ModelProviderError",
    "ModelProviderPermanentError",
    "ModelProviderRateLimitError",
    "ModelProviderTimeoutError",
    "ModelProviderTransientError",
    "ModelRequest",
    "ModelResponse",
    "StructuredModelProvider",
    "StructuredModelResponse",
    "__version__",
]

__version__: str = version("llmforeman-providers")
