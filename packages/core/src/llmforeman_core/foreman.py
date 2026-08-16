"""Core-owned Foreman planning capability.

Defines the first semantic application capability above the low-level
technical model abstractions that live in the integration packages. Those
lower layers describe technical text generation and local inference;
``Foreman`` instead describes a product/domain capability::

    engineering objective -> Foreman -> TaskPlan

This module declares interface semantics only: it owns no provider, runtime,
prompt, model, or reliability concepts, and implements no planning behavior.
Concrete Foreman adapters (defined outside core) may internally use those
technical abstractions, but that is an implementation detail the core port
must never depend on.
"""

from typing import Protocol

from llmforeman_core.models import RepositoryContext, TaskPlan

__all__ = [
    "Foreman",
    "ForemanPlanValidationError",
]


class ForemanPlanValidationError(Exception):
    """A generated plan is schema-valid but not a meaningful, consistent plan.

    Raised by concrete :class:`Foreman` implementations when a model returns a
    structurally well-formed planning result that nonetheless violates a
    plan-level semantic invariant (for example: an empty plan, duplicate task
    identifiers, a malformed identifier, a dependency on a task that does not
    exist, or a task that depends on itself).

    This is a semantic failure of the Foreman planning capability and is
    deliberately independent of any provider/runtime transport failure: it does
    not classify transient vs. permanent conditions and carries no provider
    identity. Provider-layer failures keep their own established error contract
    and are never translated into this type.
    """


class Foreman(Protocol):
    """Typed, async capability to turn an engineering objective into a plan.

    A structural interface that concrete Foreman adapters satisfy so typed
    application/orchestration code can depend on::

        plan = await foreman.create_plan(objective)

    without knowing which model provider or runtime, if any, backs the call.
    The contract is stateless and holds no provider/runtime identity, model
    selection, prompts, credentials, retries, timeouts, or lifecycle hooks.
    """

    async def create_plan(
        self,
        objective: str,
        repository_context: RepositoryContext | None = None,
    ) -> TaskPlan:
        """Produce a :class:`~llmforeman_core.models.TaskPlan` for ``objective``.

        ``objective`` is an engineering objective expressed as free text and is
        required to contain meaningful, non-whitespace content; blank or
        whitespace-only objectives (for example ``""``, ``"   "``, ``"\\t\\n"``)
        are not valid planning inputs. This is a precondition of the capability:
        because a ``Protocol`` declaration cannot itself enforce runtime
        validation, concrete implementations and application entry points are
        responsible for rejecting a blank objective before performing any work.
        A ``repository_context`` never makes an invalid objective valid.

        ``repository_context`` is optional already-prepared, normalized domain
        data (see :class:`~llmforeman_core.models.RepositoryContext`). When it is
        ``None`` (the default) planning proceeds from the objective alone, so
        existing objective-only call sites remain valid. When supplied, concrete
        implementations may include it as additional input for the planner; how
        the context was produced is out of scope for this port. This capability
        deliberately never accepts repository paths, filesystem handles, or any
        provider-specific request type: repository data enters only as this
        already-normalized core-owned model.

        Asynchronous from the outset because real planning will involve
        external model interactions and, later, repository/context operations.
        This declaration defines the contract only and implements no planning
        behavior.
        """
        ...
