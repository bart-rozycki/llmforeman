"""LLMForeman local inference runtime integrations package.

Home of local inference engine integrations (e.g. Ollama; future MLX,
llama.cpp). A local *runtime* is distinct from a cloud *provider*
(``llmforeman_providers``); the two boundaries are deliberately kept separate
and share no types, error classes, or reliability helpers.

Exposes the runtime-agnostic local model generation contract, its
runtime-independent error hierarchy, and the concrete Ollama runtime.
"""

from importlib.metadata import version

from llmforeman_runtimes.contracts import (
    ModelRuntime,
    RuntimeRequest,
    RuntimeResponse,
    StructuredModelRuntime,
    StructuredRuntimeResponse,
)
from llmforeman_runtimes.errors import (
    ModelRuntimeError,
    ModelRuntimePermanentError,
    ModelRuntimeTimeoutError,
    ModelRuntimeTransientError,
)
from llmforeman_runtimes.ollama import OllamaGenerateClient, OllamaRuntime

__all__ = [
    "ModelRuntime",
    "ModelRuntimeError",
    "ModelRuntimePermanentError",
    "ModelRuntimeTimeoutError",
    "ModelRuntimeTransientError",
    "OllamaGenerateClient",
    "OllamaRuntime",
    "RuntimeRequest",
    "RuntimeResponse",
    "StructuredModelRuntime",
    "StructuredRuntimeResponse",
    "__version__",
]

__version__: str = version("llmforeman-runtimes")
