"""Tests for deterministic tool-dispatch behavior in the bounded-loop example."""

from __future__ import annotations

from conftest import load_module

loop_example = load_module(
    "examples/end_loop_correctly/main.py",
    "end_loop_example_under_test",
)


def test_known_tool_dispatches_successfully() -> None:
    result, is_error = loop_example.run_tool(
        "check_inventory",
        {"item": "widget-A"},
    )

    assert is_error is False
    assert "12 units" in result


def test_unknown_tool_returns_structured_error() -> None:
    result, is_error = loop_example.run_tool("does_not_exist", {})

    assert is_error is True
    assert "Unknown tool" in result


def test_invalid_order_quantity_returns_error() -> None:
    result, is_error = loop_example.run_tool(
        "place_order",
        {"item": "widget-A", "quantity": 0},
    )

    assert is_error is True
    assert "quantity must be at least 1" in result


def test_missing_required_argument_returns_error() -> None:
    result, is_error = loop_example.run_tool(
        "send_notification",
        {"recipient": "warehouse"},
    )

    assert is_error is True
    assert "Invalid arguments" in result
