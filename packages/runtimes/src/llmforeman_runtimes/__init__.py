"""LLMForeman local inference runtime integrations package.

Future home of local inference engine integrations (e.g. Ollama, MLX,
llama.cpp). A local *runtime* is distinct from a cloud *provider*
(``llmforeman_providers``); the two boundaries are deliberately kept separate.

No runtime clients are declared or implemented yet.
"""

from importlib.metadata import version

__all__ = ["__version__"]

__version__: str = version("llmforeman-runtimes")
