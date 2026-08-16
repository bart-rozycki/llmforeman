"""Workspace-level integration smoke tests.

Prove that every workspace Python package is installed into the same
environment and importable together, and that the coarse package boundaries
(provider != runtime) are represented by distinct distributions.
"""

from importlib.metadata import version


def test_all_workspace_packages_are_installed() -> None:
    for dist in (
        "llmforeman-core",
        "llmforeman-providers",
        "llmforeman-runtimes",
        "llmforeman-workspace",
        "llmforeman-cli",
    ):
        assert version(dist)


def test_all_workspace_packages_import_together() -> None:
    import llmforeman_cli
    import llmforeman_core
    import llmforeman_providers
    import llmforeman_runtimes
    import llmforeman_workspace

    # provider and runtime are distinct import roots (provider != runtime).
    assert llmforeman_providers.__name__ != llmforeman_runtimes.__name__
    assert llmforeman_core.__name__ != llmforeman_cli.__name__
    # workspace is a distinct infrastructure boundary from provider/runtime.
    assert llmforeman_workspace.__name__ != llmforeman_runtimes.__name__
    assert llmforeman_workspace.__name__ != llmforeman_providers.__name__
