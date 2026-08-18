# llmforeman-cli

Thin command-line interface into the LLMForeman Python runtime.

The CLI is an interface and the executable composition root for the local
coding worker; it is not a parallel backend. It exposes the `llmforeman`
console entry point.

## Usage

Run the local coding worker against a repository:

```text
uv run llmforeman run --repo ~/Code/OpenSource/relprim \
  "Add validation for blank operation names and add tests"
```

`--repo` defaults to the current working directory, so this also works:

```text
uv run llmforeman run "Implement the requested change and verify it"
```

Options:

- `--repo PATH` — repository path (defaults to the current working directory);
- `--model MODEL` — Ollama model to use (defaults to the runtime's default model).

Every command the worker requests to run requires explicit interactive terminal
approval (`y`/`yes` to allow; anything else, an empty line, or EOF denies).
Running the worker requires a reachable Ollama server with the selected model
already available; the CLI never pulls or starts models. Approved commands are
executed with LLMForeman's own process permissions and are not sandboxed.
