"""Consistent display units for dashboard tables and metric cards."""

from __future__ import annotations

import math


def format_percentage(value: float, digits: int = 1) -> str:
    """Format a proportion as a percentage."""
    return f"{100.0 * value:.{digits}f}%"


def format_percentage_points(value: float, digits: int = 2) -> str:
    """Format a proportion difference as signed percentage points."""
    return f"{100.0 * value:+.{digits}f} pp"


def format_currency(value: float, digits: int = 0) -> str:
    """Format a dollar-valued quantity."""
    return f"${value:,.{digits}f}"


def format_number(value: float | int, digits: int = 0) -> str:
    """Format a finite number with separators."""
    if isinstance(value, float) and not math.isfinite(value):
        return "—"
    return f"{value:,.{digits}f}"
