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

## Live local worker smoke test (opt-in, local model)

`tests/integration/test_local_coding_worker_live.py` drives the real
`OllamaRuntime` → `LocalCodingWorker` → `WorkspaceActionExecutor` loop against a
tiny disposable Git repository, using a real local model served by Ollama.

- It runs a **real local model** (no cloud/paid API) and needs `git`, ripgrep
  (`rg`), a running Ollama server, and the selected/default model installed.
- It is **disabled by default**: ordinary `uv run pytest` skips it and never
  contacts Ollama. It runs only with `LLMFOREMAN_RUN_LIVE_OLLAMA_TESTS=1`; once
  opted in, missing prerequisites (or a missing model) **fail** rather than
  skip. It never auto-starts Ollama and never pulls a model.
- Model-requested `run` commands go to a non-executing test-local runner, so no
  model-directed host command runs; writes still use the real secure
  `GitRepositoryFileWriter`, and generated code is only parsed (never executed).

To run it manually (optionally overriding the model with
`LLMFOREMAN_OLLAMA_MODEL`):

```sh
LLMFOREMAN_RUN_LIVE_OLLAMA_TESTS=1 \
uv run pytest tests/integration/test_local_coding_worker_live.py -s
```

## Where the boundaries live

- `packages/core` — product/domain/worker-protocol runtime; must **not** depend on
  providers, runtimes, workspace, orchestration, the CLI, or the desktop app.
- `packages/workspace` — local coding workspace capabilities (repository
  search/read/write and command execution); depends only on `core`.
- `packages/orchestration` — application/composition layer that composes core
  worker semantics with workspace capabilities (and, for the local coding-agent
  loop, a local inference `runtime`); depends only on `core`, `workspace`, and
  `runtimes`. It must not depend on `providers`, the CLI, or the desktop app.
  Composition code belongs here, not in `workspace`.
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
