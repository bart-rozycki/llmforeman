"""Behavioral tests for the core execution domain models."""

import pytest
from pydantic import ValidationError

from llmforeman_core import (
    AgentRole,
    Run,
    Task,
    TaskPlan,
    TaskStatus,
    can_transition,
)

# Independently-defined expected v0.1 lifecycle contract. This is deliberately
# written out by hand (not derived from production) so it can detect an
# incorrect production mapping.
_EXPECTED_LEGAL_TRANSITIONS: dict[TaskStatus, set[TaskStatus]] = {
    TaskStatus.TODO: {TaskStatus.IN_PROGRESS},
    TaskStatus.IN_PROGRESS: {
        TaskStatus.REVIEW,
        TaskStatus.BLOCKED,
        TaskStatus.FAILED,
    },
    TaskStatus.REVIEW: {
        TaskStatus.DONE,
        TaskStatus.IN_PROGRESS,
        TaskStatus.FAILED,
    },
    TaskStatus.BLOCKED: {TaskStatus.IN_PROGRESS},
    TaskStatus.DONE: set(),
    TaskStatus.FAILED: set(),
}


@pytest.mark.parametrize("current", list(TaskStatus))
@pytest.mark.parametrize("target", list(TaskStatus))
def test_full_transition_matrix(current: TaskStatus, target: TaskStatus) -> None:
    expected = target in _EXPECTED_LEGAL_TRANSITIONS[current]
    assert can_transition(current, target) is expected


def test_all_same_status_transitions_illegal() -> None:
    for status in TaskStatus:
        assert can_transition(status, status) is False


def test_done_has_no_outgoing_transitions() -> None:
    assert all(
        not can_transition(TaskStatus.DONE, target) for target in TaskStatus
    )


def test_failed_has_no_outgoing_transitions() -> None:
    assert all(
        not can_transition(TaskStatus.FAILED, target) for target in TaskStatus
    )


def test_task_can_transition_to_reflects_status() -> None:
    task = Task(id="TASK-001", title="Implement feature", description="...")
    assert task.can_transition_to(TaskStatus.IN_PROGRESS)
    assert not task.can_transition_to(TaskStatus.DONE)


def test_task_can_transition_to_follows_current_status() -> None:
    task = Task(
        id="TASK-001",
        title="Implement feature",
        description="...",
        status=TaskStatus.REVIEW,
    )
    assert task.can_transition_to(TaskStatus.DONE)
    assert task.can_transition_to(TaskStatus.IN_PROGRESS)
    assert task.can_transition_to(TaskStatus.FAILED)
    assert not task.can_transition_to(TaskStatus.TODO)
    assert not task.can_transition_to(TaskStatus.BLOCKED)
    assert not task.can_transition_to(TaskStatus.REVIEW)


def test_task_status_members() -> None:
    assert {status.value for status in TaskStatus} == {
        "TODO",
        "IN_PROGRESS",
        "REVIEW",
        "BLOCKED",
        "DONE",
        "FAILED",
    }


def test_agent_role_members() -> None:
    assert {role.value for role in AgentRole} == {
        "FOREMAN",
        "DEVELOPER",
        "TESTER",
        "REVIEWER",
    }


def test_minimal_task_can_be_created() -> None:
    task = Task(id="TASK-001", title="First task", description="First task description")
    assert task.id == "TASK-001"
    assert task.title == "First task"
    assert task.description == "First task description"


def test_task_defaults() -> None:
    task = Task(id="TASK-001", title="First task", description="desc")
    assert task.status is TaskStatus.TODO
    assert task.assigned_role is None
    assert task.dependencies == []


def test_default_dependencies_not_shared() -> None:
    first = Task(id="TASK-001", title="a", description="a")
    second = Task(id="TASK-002", title="b", description="b")
    first.dependencies.append("TASK-000")
    assert first.dependencies == ["TASK-000"]
    assert second.dependencies == []


def test_fully_populated_task() -> None:
    task = Task(
        id="TASK-002",
        title="Second task",
        description="Second task description",
        status=TaskStatus.IN_PROGRESS,
        assigned_role=AgentRole.DEVELOPER,
        dependencies=["TASK-001"],
    )
    assert task.status is TaskStatus.IN_PROGRESS
    assert task.assigned_role is AgentRole.DEVELOPER
    assert task.dependencies == ["TASK-001"]


@pytest.mark.parametrize("bad_id", ["", "   "])
def test_blank_task_id_rejected(bad_id: str) -> None:
    with pytest.raises(ValidationError):
        Task(id=bad_id, title="title", description="desc")


@pytest.mark.parametrize("bad_title", ["", "   "])
def test_blank_title_rejected(bad_title: str) -> None:
    with pytest.raises(ValidationError):
        Task(id="TASK-001", title=bad_title, description="desc")


def test_blank_dependency_rejected() -> None:
    with pytest.raises(ValidationError):
        Task(id="TASK-001", title="title", description="desc", dependencies=["   "])


def test_invalid_status_rejected() -> None:
    with pytest.raises(ValidationError):
        Task.model_validate(
            {"id": "TASK-001", "title": "title", "description": "desc", "status": "NOPE"}
        )


def test_invalid_role_rejected() -> None:
    with pytest.raises(ValidationError):
        Task.model_validate(
            {
                "id": "TASK-001",
                "title": "title",
                "description": "desc",
                "assigned_role": "MANAGER",
            }
        )


def test_task_plan_preserves_order() -> None:
    plan = TaskPlan(
        tasks=[
            Task(id="TASK-001", title="First task", description="First task description"),
            Task(
                id="TASK-002",
                title="Second task",
                description="Second task description",
                assigned_role=AgentRole.DEVELOPER,
                dependencies=["TASK-001"],
            ),
        ]
    )
    assert [task.id for task in plan.tasks] == ["TASK-001", "TASK-002"]


def test_run_contains_plan() -> None:
    plan = TaskPlan(
        tasks=[Task(id="TASK-001", title="First task", description="desc")]
    )
    run = Run(plan=plan)
    assert run.plan.tasks[0].id == "TASK-001"


def test_serialization_round_trip() -> None:
    run = Run(
        plan=TaskPlan(
            tasks=[
                Task(
                    id="TASK-001",
                    title="First task",
                    description="desc",
                    assigned_role=AgentRole.DEVELOPER,
                    dependencies=["TASK-000"],
                )
            ]
        )
    )
    dumped = run.model_dump()
    assert dumped == {
        "plan": {
            "tasks": [
                {
                    "id": "TASK-001",
                    "title": "First task",
                    "description": "desc",
                    "status": TaskStatus.TODO,
                    "assigned_role": AgentRole.DEVELOPER,
                    "dependencies": ["TASK-000"],
                }
            ]
        }
    }
    assert Run.model_validate(dumped) == run
