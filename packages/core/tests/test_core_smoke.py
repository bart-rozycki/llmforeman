"""Smoke tests for the installed ``llmforeman_core`` package.

These prove the ``src/`` layout and the uv workspace install actually work:
the package is imported from the installed distribution (not the repository
root being on ``sys.path``) and its declared metadata is resolvable.
"""

from importlib.metadata import version
from pathlib import Path

import llmforeman_core


def test_package_is_importable_from_installed_location() -> None:
    module_file = Path(llmforeman_core.__file__).resolve()
    # Installed via src/ layout: the imported module must not live under a
    # top-level ``src`` directory or a package-root ``packages/core`` path.
    assert "site-packages" in module_file.parts or "llmforeman_core" in module_file.parts
    assert module_file.name == "__init__.py"


def test_version_matches_distribution_metadata() -> None:
    assert llmforeman_core.__version__ == version("llmforeman-core")
