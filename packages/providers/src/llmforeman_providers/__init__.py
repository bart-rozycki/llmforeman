"""LLMForeman cloud provider integrations package.

Future home of cloud LLM integrations (e.g. Anthropic, OpenAI, Gemini).
A cloud *provider* is distinct from a local inference *runtime*
(``llmforeman_runtimes``); the two boundaries are deliberately kept separate.

No provider SDKs are declared or implemented yet.
"""

from importlib.metadata import version

__all__ = ["__version__"]

__version__: str = version("llmforeman-providers")
