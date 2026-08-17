"""Workspace-owned command-execution result model.

This small typed data model describes the *result* of running one command in a
local coding workspace: the exact argv that was executed, the process exit
code, and the captured standard-output and standard-error text. It is
workspace-owned rather than a core domain model because it currently describes
the output of a workspace/infrastructure operation, not a durable part of
LLMForeman's domain model. Nothing here spawns a process, invokes a shell,
parses command strings, reads the environment, or interprets output; a future
concrete ``WorkspaceCommandRunner`` owns all of that runtime behavior.

``CommandResult.command`` is an explicit argv ``list[str]`` (``command[0]`` is
the executable and ``command[1:]`` are its arguments). It is never a shell
command string: strings such as ``"|"``, ``">"``, ``"&&"``, or ``"$HOME"`` that
appear in argv are ordinary argument values carrying no shell semantics. The
model preserves argument order, contents, and the executable position exactly;
it performs no joining, shell-quoting, normalization, trimming, or reordering.
The only structural invariant enforced at construction time is that the command
contains at least one argv element (an executable) and that each argv entry is a
non-empty string. Whitespace-only argv entries remain valid because they can be
legitimate process arguments, so entries are never stripped.
"""

from pydantic import BaseModel, Field, field_validator

__all__ = [
    "CommandResult",
]


class CommandResult(BaseModel):
    """The result of running one argv command to completion in a workspace.

    Holds exactly the executed ``command`` (argv), the process ``exit_code``,
    and the captured ``stdout`` and ``stderr`` text. It carries no duration,
    timestamps, PID, signal, working directory, combined/merged output, or
    derived ``success`` flag; those are deliberately out of scope for this
    result and any richer semantics belong to a future concrete runner.

    A non-zero ``exit_code`` is a normal, valid result (for example a test tool
    reporting failures) and never implies an exception. ``exit_code`` is an
    unconstrained ``int``: negative values remain valid so a future concrete
    runner may use them to represent termination by signal. ``stdout`` and
    ``stderr`` may independently be empty (``""``); they are kept separate and
    never merged, and a non-empty ``stderr`` does not imply failure.
    """

    command: list[str] = Field(min_length=1)
    exit_code: int
    stdout: str
    stderr: str

    @field_validator("command")
    @classmethod
    def _validate_command_argv(cls, value: list[str]) -> list[str]:
        # A result must describe a command that had an executable to run; an
        # empty argv can never be an executed command. This structural invariant
        # belongs to the result object, not to the async Protocol (which cannot
        # validate at runtime).
        if not value:
            raise ValueError("command must contain at least one argv element")
        # Each argv entry must be a non-empty string: an empty string can never
        # be an executable or a meaningful argument. Whitespace-only entries are
        # intentionally left valid (they can be legitimate arguments), so no
        # entry is stripped or otherwise normalized.
        for entry in value:
            if entry == "":
                raise ValueError("command argv entries must be non-empty strings")
        return value
