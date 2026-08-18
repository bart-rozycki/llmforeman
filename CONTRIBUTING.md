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

## Live external integration tests (opt-in, paid)

`tests/integration/test_anthropic_foreman_live.py` is an end-to-end smoke test
that drives the real `AnthropicForeman` → `AnthropicProvider` planning path
against the **live Anthropic API**.

- It makes a **real, paid** Anthropic API request.
- It is **disabled by default**: ordinary `uv run pytest` never executes it (it
  is skipped), and no `ANTHROPIC_API_KEY` is required for a normal run.
- It runs only when you explicitly opt in with `LLMFOREMAN_RUN_LIVE_TESTS=1`.
  The API key is read only from `ANTHROPIC_API_KEY` in your environment; never
  commit a key, add it to `.env`, or place it in test data.

To run it manually:

```sh
LLMFOREMAN_RUN_LIVE_TESTS=1 \
ANTHROPIC_API_KEY='your-key-here' \
uv run pytest tests/integration/test_anthropic_foreman_live.py -s
```

If you enable live tests but `ANTHROPIC_API_KEY` is missing or blank, the test
fails immediately with a configuration error before any API request is made.

## Where the boundaries live

- `packages/core` — product/domain/worker-protocol runtime; must **not** depend on
  providers, runtimes, workspace, orchestration, the CLI, or the desktop app.
- `packages/workspace` — local coding workspace capabilities (repository
  search/read/write and command execution); depends only on `core`.
- `packages/orchestration` — application/composition layer that composes core
  worker semantics with workspace capabilities; depends only on `core` and
  `workspace`. Composition code belongs here, not in `workspace`.
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
