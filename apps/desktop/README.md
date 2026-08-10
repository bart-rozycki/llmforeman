# @llmforeman/desktop

Minimal Tauri 2 + React + TypeScript desktop shell for LLMForeman.

This is a UI boundary only. Orchestration/domain behavior lives in the Python
runtime; the Tauri Rust layer (`src-tauri/`) is a narrow native bridge with no
product logic. No IPC, sidecar, or application state framework is defined yet.

## Commands

Run from the repository root using npm workspaces:

```sh
npm install                                   # install workspace deps
npm run -w @llmforeman/desktop typecheck      # TypeScript type-check
npm run -w @llmforeman/desktop build          # frontend production build
npm run -w @llmforeman/desktop tauri build    # full Tauri build (requires Rust)
```
