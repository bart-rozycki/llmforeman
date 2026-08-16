# LLMForeman

LLMForeman is a local-first AI software-engineering orchestration platform.

> **Early-stage and under active development.** The repository currently
> contains the foundational architecture plus initial vertical slices:
> Anthropic-backed Foreman planning, local Ollama runtime support,
> provider- and runtime-agnostic execution-domain models, and Git
> repository-context support. Full autonomous task execution and desktop
> product workflows are **not implemented yet**.

## Repository layout

The repository is organized around a few coarse, deliberately separated
boundaries:

| Path                 | Boundary  | Responsibility                                                                                        |
| -------------------- | --------- | ---------------------------------------------------------------------------------------------------- |
| `packages/core`      | core      | Product/domain/orchestration runtime: execution-domain models and the Foreman planning port. Independent of providers & runtimes. |
| `packages/providers` | providers | Cloud LLM integrations. Anthropic text/structured generation and Anthropic-backed Foreman planning implemented; OpenAI, Gemini, … planned. |
| `packages/runtimes`  | runtimes  | Local inference engines. Ollama runtime implemented; MLX, llama.cpp, … planned.                       |
| `packages/workspace` | workspace | Local coding workspace and repository-context infrastructure (Git repository-context loader).         |
| `packages/cli`       | CLI       | Thin command-line entry point over the Python runtime. Currently a no-op scaffold with no real commands. |
| `apps/desktop`       | desktop   | Tauri 2 + React + TypeScript UI shell (narrow native bridge only). No product workflows yet.          |

An important architectural invariant: **`provider != runtime`**. A cloud model
provider and a local inference engine are distinct integration boundaries.

## Prerequisites

- [Python 3.13](https://www.python.org/) — project baseline.
- [uv](https://docs.astral.sh/uv/) — Python dependency/workspace manager.
- [Node.js](https://nodejs.org/) + npm — desktop frontend workspace.
- [Rust](https://www.rust-lang.org/tools/install) — only for building/validating
  the Tauri desktop app.

## Python: bootstrap & validation

The repository root is a `uv` workspace containing the five Python packages,
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

LLMForeman is under active development and remains early-stage. Meaningful
foundational vertical slices exist today:

- provider- and runtime-agnostic execution-domain models, including the task
  lifecycle policy;
- a core-owned Foreman planning port with an Anthropic-backed implementation
  built on structured generation;
- local inference via an Ollama runtime adapter;
- a Git repository-context loader that produces normalized repository context.

Major product workflow pieces are **not implemented yet**, including end-to-end
autonomous task execution, local coding-worker execution, repository
search/exploration and file-editing tools, review loops, scheduling,
persistence-backed orchestration, the CLI command surface, and the desktop
product experience.
