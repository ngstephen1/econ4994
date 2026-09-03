"""Synthetic mortgage-application data-generating process."""

from fair_lending.simulation.generator import (
    OUTPUT_COLUMNS,
    generate_synthetic_data,
    save_synthetic_dataset,
)

__all__ = ["OUTPUT_COLUMNS", "generate_synthetic_data", "save_synthetic_dataset"]
