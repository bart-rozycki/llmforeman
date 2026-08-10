"""Smoke tests for the installed ``llmforeman_providers`` package."""

from importlib.metadata import version
from pathlib import Path

import llmforeman_providers


def test_package_is_importable_from_installed_location() -> None:
    module_file = Path(llmforeman_providers.__file__).resolve()
    assert "site-packages" in module_file.parts or "llmforeman_providers" in module_file.parts
    assert module_file.name == "__init__.py"


def test_version_matches_distribution_metadata() -> None:
    assert llmforeman_providers.__version__ == version("llmforeman-providers")
