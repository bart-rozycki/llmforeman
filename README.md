# LLMForeman

LLMForeman is a local-first AI software-engineering orchestration platform.

> **Early-stage monorepo.** This repository currently contains only the
> foundational skeleton — package boundaries, toolchain, and smoke tests.
> **No product functionality is implemented yet.**

## Repository layout

The repository is organized around a few coarse, deliberately separated
boundaries:

| Path                    | Boundary  | Responsibility                                                             |
| ----------------------- | --------- | -------------------------------------------------------------------------- |
| `packages/core`         | core      | Product/domain/orchestration runtime. Independent of providers & runtimes. |
| `packages/providers`    | providers | Cloud LLM integrations (Anthropic, OpenAI, Gemini, …).                     |
| `packages/runtimes`     | runtimes  | Local inference engines (Ollama, MLX, llama.cpp, …).                       |
| `packages/cli`          | CLI       | Thin entry point into the Python runtime.                                  |
| `apps/desktop`          | desktop   | Tauri 2 + React + TypeScript UI shell (narrow native bridge only).         |

An important architectural invariant: **`provider != runtime`**. A cloud model
provider and a local inference engine are distinct integration boundaries.

## Prerequisites

- [Python 3.13](https://www.python.org/) — project baseline.
- [uv](https://docs.astral.sh/uv/) — Python dependency/workspace manager.
- [Node.js](https://nodejs.org/) + npm — desktop frontend workspace.
- [Rust](https://www.rust-lang.org/tools/install) — only for building/validating
  the Tauri desktop app.

## Python: bootstrap & validation

The repository root is a `uv` workspace containing the four Python packages,
each using a `src/` layout and installed as an editable workspace member.

```sh
uv sync                 # create .venv and install all workspace packages + dev tools
uv run pytest           # run smoke tests against the installed packages
uv run ruff check .     # lint
uv run mypy             # static type check (strict)
```

Tooling (Ruff, mypy, pytest) is configured centrally in the root
`pyproject.toml`.

## Desktop: bootstrap & validation

The root uses **npm workspaces** to manage `apps/desktop`.

```sh
npm install                                 # install frontend workspace deps
npm run desktop:typecheck                   # TypeScript type-check
npm run desktop:build                       # Vite production build
npm run desktop:tauri build                 # full Tauri build (requires Rust)
```

## Status

This is the first public commit of the foundation. It intentionally contains
no LLM calls, orchestration, provider/runtime integrations, persistence, IPC,
or desktop product behavior.
