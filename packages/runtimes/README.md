# llmforeman-runtimes

Local inference runtime integrations for LLMForeman (e.g. Ollama, MLX,
llama.cpp).

Provides the runtime-agnostic generation contract (`RuntimeRequest`,
`RuntimeResponse`, `ModelRuntime`), a small runtime-independent error hierarchy
(`ModelRuntimeError` and its permanent/transient/timeout subclasses), and the
first concrete adapter, `OllamaRuntime`.

`OllamaRuntime` performs real non-streaming inference against a running Ollama
server via the official asynchronous `ollama.AsyncClient`. RelPrim is the sole
owner of retry and timeout semantics for the inference call; the SDK client is
created without a competing transport timeout. Model and host are runtime-instance
configuration (default model `qwen3.6:35b-a3b`), never part of `RuntimeRequest`.

A local *runtime* is intentionally distinct from a cloud *provider*
(`llmforeman-providers`); the two boundaries share no types, error classes, or
reliability helpers.
