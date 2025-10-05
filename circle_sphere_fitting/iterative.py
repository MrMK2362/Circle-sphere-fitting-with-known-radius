"""Gauss--Newton fitting when the radius is known."""
from __future__ import annotations

from typing import Optional

import numpy as np

from .result import FitResult


def fit_center_known_radius(
    points: np.ndarray,
    radius: float,
    *,
    initial_center: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
    max_iter: int = 50,
    tol: float = 1e-12,
) -> FitResult:
    """
    Estimate the centre of a circle/sphere using Gauss--Newton iterations.

    Parameters
    ----------
    points:
        Array of shape (n_points, n_dim) containing sample points.
    radius:
        Known radius in the same units as the point coordinates.
    initial_center:
        Optional initial guess for the centre. Defaults to the (weighted) centroid.
    weights:
        Optional array of positive weights of length ``n_points``.
    max_iter:
        Maximum number of Gauss--Newton iterations.
    tol:
        Convergence tolerance for both the parameter update and RMSE.
    """

    pts = np.asarray(points, dtype=float)
    
    n_points, n_dim = pts.shape

    w = None
    if weights is not None:
        w = np.asarray(weights, dtype=float)

    if initial_center is None:
        if w is None:
            center = np.mean(pts, axis=0)
        else:
            center = np.average(pts, axis=0, weights=w)
    else:
        center = np.asarray(initial_center, dtype=float)

    center = center.astype(float, copy=True)
    iterations = 0
    success = False
    message = "Maximum iterations reached without convergence"

    for iterations in range(1, max_iter + 1):
        diffs = center - pts
        dists = np.linalg.norm(diffs, axis=1)
        residuals = dists - radius

        if w is None:
            rmse = float(np.sqrt(np.mean(residuals ** 2)))
        else:
            rmse = float(np.sqrt(np.average(residuals ** 2, weights=w)))

        if rmse <= tol:
            success = True
            message = "Converged based on RMSE tolerance"
            break

        valid = dists > 1e-12
        if not np.any(valid):
            message = "All distances to the current centre are zero, cannot update further"
            break

        J = (diffs[valid] / dists[valid][:, None])
        r = residuals[valid]

        if w is not None:
            sqrt_w = np.sqrt(w[valid])[:, None]
            J = J * sqrt_w
            r = r * sqrt_w[:, 0]

        try:
            delta, *_ = np.linalg.lstsq(J, -r, rcond=None)
        except np.linalg.LinAlgError as exc:
            message = f"Least squares solve failed: {exc}"
            break

        center += delta

        if np.linalg.norm(delta) <= tol:
            success = True
            message = "Converged based on parameter update tolerance"
            break

    residuals = np.linalg.norm(center - pts, axis=1) - radius
    if w is None:
        rmse = float(np.sqrt(np.mean(residuals ** 2)))
    else:
        rmse = float(np.sqrt(np.average(residuals ** 2, weights=w)))

    return FitResult(
        center=center,
        radius=radius,
        residuals=residuals,
        rmse=rmse,
        iterations=iterations,
        success=success,
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