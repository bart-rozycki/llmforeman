# llmforeman-runtimes

Local inference runtime integrations for LLMForeman (e.g. Ollama, MLX,
llama.cpp).

Provides the runtime-agnostic generation contract (`RuntimeRequest`,
`RuntimeResponse`, `ModelRuntime`), an orthogonal structured-output capability
(`StructuredModelRuntime` and the generic `StructuredRuntimeResponse[T]`), a
small runtime-independent error hierarchy (`ModelRuntimeError` and its
permanent/transient/timeout subclasses), and the first concrete adapter,
`OllamaRuntime`.

`ModelRuntime` covers plain text generation; `StructuredModelRuntime` covers
typed Pydantic output generation. They are separate, orthogonal capabilities:
`StructuredModelRuntime` does not inherit from `ModelRuntime`, and a caller that
needs a validated result depends on it directly. `StructuredRuntimeResponse[T]`
carries only the validated `output: T` (preserving the caller-supplied Pydantic
type) and the runtime-neutral `usage`; it exposes no raw payload and no JSON
Schema. Callers supply a Python `type[T]`, never a raw JSON Schema. Mirrors the
provider-side structured design without sharing its types.

`OllamaRuntime` performs real non-streaming inference against a running Ollama
server via the official asynchronous `ollama.AsyncClient`. RelPrim is the sole
owner of retry and timeout semantics for the inference call; the SDK client is
created without a competing transport timeout. Model and host are runtime-instance
configuration (default model `qwen3.6:35b-a3b`), never part of `RuntimeRequest`.

A local *runtime* is intentionally distinct from a cloud *provider*
(`llmforeman-providers`); the two boundaries share no types, error classes, or
reliability helpers.
