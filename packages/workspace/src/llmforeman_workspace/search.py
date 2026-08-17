"""Workspace-owned repository text-search result models.

These small typed data models describe the *result* of a repository text
search: an ordered list of single-line matches. They are workspace-owned rather
than core domain models because they currently describe the output of a
workspace/infrastructure operation, not a durable part of LLMForeman's domain
model. Nothing here performs a search, reads files, walks directories, invokes
Git or ripgrep, or interprets a query; a future concrete searcher owns all of
that runtime behavior.

``RepositorySearchMatch.path`` is a logical, repository-relative path suitable
for serialization and model context. The same OS-independent, filesystem-free
privacy invariant used by ``llmforeman_core.RepositoryFile`` is enforced here at
construction time (no absolute paths, no parent traversal, no NUL) so a match
can never leak an absolute checkout, working-directory, or home-directory path.
The validation is duplicated locally rather than promoting a private validator
into a shared abstraction: the two models are independent boundary results and
small local duplication is preferable to a premature cross-model coupling.
"""

from pathlib import PurePosixPath, PureWindowsPath

from pydantic import BaseModel, Field, field_validator

__all__ = [
    "RepositorySearchMatch",
    "RepositorySearchResult",
]


class RepositorySearchMatch(BaseModel):
    """One repository text-search match on a single line.

    Represents exactly a logical, repository-relative ``path``, a 1-based
    ``line_number``, and the matching ``line`` text. It carries no surrounding
    context, column information, highlighting, ranking, or engine metadata; a
    future concrete searcher owns production correctness and any richer output.
    """

    path: str
    line_number: int = Field(ge=1)
    line: str

    @field_validator("path")
    @classmethod
    def _validate_repository_relative_path(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("path must not be empty or whitespace-only")
        if "\x00" in value:
            raise ValueError("path must not contain NUL characters")
        # Cross-platform, filesystem-free absolute-path detection. Parse the
        # value as both POSIX and Windows pure paths so an absolute path is
        # rejected regardless of which OS runs this code.
        if PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute():
            raise ValueError("path must be repository-relative, not absolute")
        # Reject parent traversal in either separator convention without
        # resolving the path against the filesystem or the working directory.
        parts = PurePosixPath(value).parts + PureWindowsPath(value).parts
        if ".." in parts:
            raise ValueError("path must not contain parent traversal segments ('..')")
        return value


class RepositorySearchResult(BaseModel):
    """An ordered container of repository text-search matches.

    Holds only ``matches``. An empty list is a valid result: searching
    successfully and finding nothing is not an error and is never represented as
    ``None`` or an exception. The supplied order is preserved exactly; this model
    performs no sorting, ranking, de-duplication, or truncation. It intentionally
    carries no query echo, counts, timing, engine name, truncation flag, or
    warnings; the concrete searcher defines any such policy later.
    """

    matches: list[RepositorySearchMatch] = []
