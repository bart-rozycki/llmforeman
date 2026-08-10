# Contributing to LLMForeman

Thanks for your interest! LLMForeman is at an early foundational stage — this
document covers only what a contributor needs today.

## Environment setup

Install the prerequisites listed in the [README](./README.md#prerequisites)
(Python 3.13, `uv`, Node.js + npm, and Rust for desktop builds).

```sh
uv sync        # Python workspace + dev tooling
npm install    # desktop frontend workspace
```

## Quality commands

Run these before opening a pull request:

```sh
# Python
uv run pytest
uv run ruff check .
uv run mypy

# Desktop
npm run desktop:typecheck
npm run desktop:build
# Optional (requires Rust):
(cd apps/desktop/src-tauri && cargo check)
```

## Where the boundaries live

- `packages/core` — product/domain/orchestration runtime; must **not** depend on
  providers, runtimes, the CLI, or the desktop app.
- `packages/providers` — cloud LLM integrations.
- `packages/runtimes` — local inference engines. Keep distinct from providers
  (`provider != runtime`).
- `packages/cli` — a thin interface into the Python runtime, not a backend.
- `apps/desktop` — Tauri/React UI shell; the Rust layer is a narrow native
  bridge, not a second backend.

## Expectations

- Keep pull requests **focused**: a small, self-contained diff per change.
- Respect the boundaries above; do not merge provider/runtime concepts.
- Do not add speculative abstractions or future-feature stubs.
