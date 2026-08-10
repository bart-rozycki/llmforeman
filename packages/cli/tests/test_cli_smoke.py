"""Smoke tests for the installed ``llmforeman_cli`` package."""

from importlib.metadata import version
from pathlib import Path

import llmforeman_cli


def test_package_is_importable_from_installed_location() -> None:
    module_file = Path(llmforeman_cli.__file__).resolve()
    assert "site-packages" in module_file.parts or "llmforeman_cli" in module_file.parts
    assert module_file.name == "__init__.py"


def test_version_matches_distribution_metadata() -> None:
    assert llmforeman_cli.__version__ == version("llmforeman-cli")


def test_entry_point_is_callable_and_returns_zero() -> None:
    assert llmforeman_cli.main() == 0
