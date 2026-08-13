"""Stage 2c/3: training-example assembly and synthetic expansion."""

from .prepare import DatasetResult, prepare_dataset
from .synthetic import SyntheticResult, generate_synthetic

__all__ = ["DatasetResult", "prepare_dataset", "SyntheticResult", "generate_synthetic"]
