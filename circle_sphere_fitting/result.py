"""Data structures shared across fitting algorithms."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class FitResult:
    """Container for results returned by the fitting routines."""

    center: np.ndarray
    radius: float
    residuals: np.ndarray
    rmse: float
    iterations: int
    success: bool
    message: str = ""

    def as_dict(self) -> Dict[str, object]:
        """Return a JSON-serialisable dictionary of the result."""

        return {
            "center": self.center.tolist(),
            "radius": float(self.radius),
            "residuals": self.residuals.tolist(),
            "rmse": float(self.rmse),
            "iterations": int(self.iterations),
            "success": bool(self.success),
            "message": self.message,
        }