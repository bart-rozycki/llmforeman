"""LLMForeman command-line interface package.

A thin user-facing entry point into the Python runtime. It is an interface and
executable composition root only; it must not become a second backend or
reimplement product/domain logic. The concrete ``run`` command composition
lives in :mod:`llmforeman_cli._cli`.
"""

from importlib.metadata import version

from llmforeman_cli._cli import main

__all__ = ["__version__", "main"]

__version__: str = version("llmforeman-cli")
