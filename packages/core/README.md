# llmforeman-core

Provider- and runtime-agnostic product/domain/orchestration runtime for
LLMForeman.

This package intentionally contains no product logic yet — it only
establishes the `core` boundary. `core` must not depend on cloud providers
(`llmforeman-providers`), local inference runtimes (`llmforeman-runtimes`),
the CLI, or the desktop application.
