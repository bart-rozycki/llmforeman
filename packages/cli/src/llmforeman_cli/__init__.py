"""LLMForeman command-line interface package.

A thin user-facing entry point into the Python runtime. It must remain an
interface only and must not become a second backend or implement product
behavior. No real commands are implemented yet.
"""

from importlib.metadata import version

__all__ = ["__version__", "main"]

__version__: str = version("llmforeman-cli")


def main() -> int:
    """Entry point for the ``llmforeman`` console script.

    This is intentionally a no-op placeholder: it only proves the CLI package
    boundary and console-script wiring. Actual commands are out of scope for
    the foundational skeleton.
    """
    print(f"llmforeman {__version__}: no commands are implemented yet.")
    return 0
