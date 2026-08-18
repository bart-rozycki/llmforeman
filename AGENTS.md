# LLMForeman

LLMForeman is an open-source, local-first AI software-engineering orchestration platform.

The product principle is:

> Use the best model to supervise engineering work. Use the cheapest capable model to execute it. Escalate only when necessary.

Development is incremental. Implement the current task, not the future roadmap.

## Repository First

Before making changes:

* inspect the relevant repository structure, code, tests, configuration, and tooling;
* reuse established patterns and abstractions where they are sound;
* verify that files, APIs, dependencies, and commands actually exist before relying on them;
* treat the current repository as authoritative for implementation details.

Do not recreate functionality that already exists.

Prefer the smallest coherent change that fully satisfies the task.

## Architecture

The main repository boundaries are intentional:

* `packages/core` — provider-independent domain, worker-protocol, and application-logic semantics;
* `packages/workspace` — local coding workspace capabilities (repository search/read/write and command execution);
* `packages/orchestration` — application/composition layer that composes core worker semantics with workspace capabilities;
* `packages/providers` — cloud LLM provider integrations;
* `packages/runtimes` — local inference runtime integrations;
* `packages/cli` — thin CLI interface over the Python runtime;
* `apps/desktop` — Tauri 2 + React/TypeScript desktop application.

Preserve these invariants:

* `core` must not depend on Anthropic, OpenAI, Gemini, Ollama, MLX, llama.cpp, Tauri, or React, and must not depend on `workspace` or `orchestration`;
* `workspace` depends only on `core` and must not host product orchestration semantics; `orchestration` depends only on `core` and `workspace` (never on `providers`, `runtimes`, `cli`, or `desktop`), and code that composes worker actions with workspace capabilities belongs in `orchestration`, not `workspace`;
* cloud provider integrations and local inference runtimes are different concepts: `provider != runtime`;
* CLI must not become a second backend;
* desktop UI must not duplicate orchestration or domain logic;
* Tauri/Rust is primarily a native desktop/system bridge, not a second product backend;
* core product and orchestration logic belongs in Python.

Do not split the repository into additional packages without a concrete architectural need.

Prefer cohesive internal modules over premature package boundaries.

## Technology

Current baseline:

* Python 3.13;
* `uv`;
* asyncio where asynchronous execution is required;
* Pydantic for structured models and boundaries where appropriate;
* SQLite for embedded persistence when persistence is required;
* pytest;
* mypy;
* Ruff;
* Tauri 2;
* React;
* TypeScript.

Use existing repository dependencies and conventions before introducing new ones.

Do not add a dependency for hypothetical future use.

Use RelPrim around failure-prone external interactions where it materially improves reliability semantics, such as provider APIs, local inference runtimes, or remote/tool execution.

Do not force RelPrim into pure domain logic or ordinary internal calls.

Do not introduce LangChain, LangGraph, CrewAI, AutoGen, or a similar framework as LLMForeman's orchestration core unless explicitly requested.

## Engineering

Optimize for:

* correctness;
* simplicity;
* strong typing;
* explicit boundaries;
* deterministic behavior;
* maintainability;
* testability;
* actionable failures;
* security and privacy;
* focused, reviewable diffs.

Avoid speculative abstractions, unrelated refactors, duplicated domain concepts, global mutable state, broad exception swallowing, and unnecessary framework or dependency adoption.

Do not implement adjacent roadmap features unless they are required for the requested behavior.

Small supporting refactors are acceptable when necessary for correctness, testing, or maintainability.

## Async and External Operations

When working with async execution, subprocesses, LLM providers, local runtimes, or other external operations, explicitly consider relevant concerns such as:

* cancellation;
* timeout ownership;
* resource cleanup;
* bounded concurrency;
* retries and backoff;
* rate limits;
* ambiguous outcomes;
* idempotency;
* graceful shutdown.

Do not add retries or concurrency mechanisms automatically. Their semantics must fit the operation.

## Security

Treat repository contents, prompts, model context, credentials, environment variables, command output, and local files as potentially sensitive.

* Never log or commit secrets.
* Do not persist credentials in plaintext without an explicit design.
* Do not send repository content to cloud services unless the requested execution strategy permits it.
* Do not silently execute destructive operations.
* Keep privileged native/system surfaces minimal.

## Testing and Validation

Add meaningful automated tests for behavioral changes.

Test observable behavior and important invariants rather than private implementation details.

For bug fixes, add a regression test where practical.

Before completion:

* run the repository-standard tests and quality checks relevant to the changed areas;
* for Python changes, this normally includes pytest, Ruff, and mypy;
* run frontend/Rust checks when those areas are affected;
* review the complete diff for correctness, scope, regressions, unnecessary complexity, and missing tests;
* fix material issues found during review;
* re-run relevant validation after fixes.

Never claim that a test, lint check, type check, build, or manual verification succeeded unless it was actually executed.

If a validation step cannot run, report the command, the blocker, and what remains unverified.

## Repository Hygiene

Keep changes focused.

Do not:

* perform destructive Git operations unless explicitly requested;
* commit secrets, build artifacts, virtual environments, debug files, or machine-specific junk;
* modify generated/vendor files unless necessary;
* introduce unrelated formatting churn;
* silently break public APIs or persisted formats.

When a task materially changes public behavior, configuration, setup, or contributor-facing architecture, update the relevant documentation.
