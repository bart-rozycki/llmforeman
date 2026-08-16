# llmforeman-workspace

Local coding workspace infrastructure boundary for LLMForeman.

This package owns infrastructure concerned with the local coding workspace
(repository/filesystem access, and in future Git operations, diffs, and
workspace isolation). It depends only on `llmforeman-core`; it must not depend
on cloud providers (`llmforeman-providers`) or local inference runtimes
(`llmforeman-runtimes`).

It currently defines a single typed async contract, `RepositoryContextLoader`,
for loading a core `RepositoryContext` from a local repository root. No
concrete loader is implemented yet.
