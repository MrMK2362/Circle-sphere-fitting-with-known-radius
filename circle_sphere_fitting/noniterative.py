"""Algebraic least-squares fitting when the radius is known."""
from __future__ import annotations

from typing import Optional

import numpy as np

from .result import FitResult


def fit_center_known_radius(
    points: np.ndarray,
    radius: float,
    *,
    weights: Optional[np.ndarray] = None,
) -> FitResult:
    """
    Estimate the centre via a single linear least-squares solve.

    Parameters
    ----------
    points:
        Array of shape (n_points, n_dim) containing sample points.
    radius:
        Known radius in the same units as the point coordinates.
    weights:
        Optional array of positive weights of length ``n_points``.
    """

    pts = np.asarray(points, dtype=float)

    n_points, n_dim = pts.shape

    w = None
    if weights is not None:
        w = np.asarray(weights, dtype=float)

    rhs = radius ** 2 - np.sum(pts * pts, axis=1)
    design = np.hstack((-2.0 * pts, np.ones((n_points, 1))))

    if w is not None:
        sqrt_w = np.sqrt(w)[:, None]
        design = design * sqrt_w
        rhs = rhs * sqrt_w[:, 0]

    lsq_solution, residuals, rank, sing_vals = np.linalg.lstsq(design, rhs, rcond=None)
    center = lsq_solution[:-1]

    residual_vector = np.linalg.norm(center - pts, axis=1) - radius
    if w is None:
        rmse = float(np.sqrt(np.mean(residual_vector ** 2)))
    else:
        rmse = float(np.sqrt(np.average(residual_vector ** 2, weights=w)))

    full_rank = rank == n_dim + 1
    message = "Solved weighted linear system" if full_rank else "Linear system was rank-deficient"

    return FitResult(
        center=center,
        radius=radius,
        residuals=residual_vector,
        rmse=rmse,
        iterations=1,
        success=full_rank,
        message=message,
    )


def fit_circle_known_radius(
    points: np.ndarray,
    radius: float,
    **kwargs,
) -> FitResult:
    """
    Convenience wrapper for 2D data.
    """

    pts = np.asarray(points, dtype=float)
    return fit_center_known_radius(pts, radius, **kwargs)


def fit_sphere_known_radius(
    points: np.ndarray,
    radius: float,
    **kwargs,
) -> FitResult:
    """
    Convenience wrapper for 3D data.
    """

    pts = np.asarray(points, dtype=float)
    return fit_center_known_radius(pts, radius, **kwargs)