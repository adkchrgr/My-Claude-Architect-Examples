"""Tests for deterministic coordinator dispatch validation."""

from __future__ import annotations

from conftest import load_module

coordinator = load_module(
    "examples/narrow_task_decomposition/main.py",
    "task_decomposition_under_test",
)


def test_validate_dispatch_accepts_valid_input() -> None:
    subtask, complexity = coordinator.validate_dispatch(
        {"subtask": "Compare durability characteristics", "complexity": "complex"}
    )

    assert subtask == "Compare durability characteristics"
    assert complexity == "complex"


def test_validate_dispatch_rejects_empty_subtask() -> None:
    try:
        coordinator.validate_dispatch({"subtask": "   ", "complexity": "simple"})
    except ValueError as exc:
        assert "non-empty" in str(exc)
    else:
        raise AssertionError("Expected ValueError for empty subtask")


def test_validate_dispatch_rejects_unknown_complexity() -> None:
    try:
        coordinator.validate_dispatch({"subtask": "Check cost", "complexity": "medium"})
    except ValueError as exc:
        assert "simple" in str(exc)
        assert "complex" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown complexity")
