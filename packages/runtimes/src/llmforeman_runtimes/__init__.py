"""LLMForeman local inference runtime integrations package.

Future home of local inference engine integrations (e.g. Ollama, MLX,
llama.cpp). A local *runtime* is distinct from a cloud *provider*
(``llmforeman_providers``); the two boundaries are deliberately kept separate.

Exposes the runtime-agnostic local model generation contract. No concrete
runtime clients are declared or implemented yet.
"""

from importlib.metadata import version

from llmforeman_runtimes.contracts import (
    ModelRuntime,
    RuntimeRequest,
    RuntimeResponse,
)

__all__ = [
    "ModelRuntime",
    "RuntimeRequest",
    "RuntimeResponse",
    "__version__",
]

__version__: str = version("llmforeman-runtimes")
