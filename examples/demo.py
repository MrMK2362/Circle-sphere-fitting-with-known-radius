"""Visual demonstration of iterative vs non-iterative fitting with known radius."""
from __future__ import annotations

from pathlib import Path
import sys

# Allow running the demo without installing the package first
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")  # ensure the demo works in headless environments
import matplotlib.pyplot as plt
import numpy as np

from circle_sphere_fitting import (
    fit_circle_known_radius_algebraic,
    fit_circle_known_radius_iterative,
    fit_sphere_known_radius_algebraic,
    fit_sphere_known_radius_iterative,
)


FIGURE_PATH = _PROJECT_ROOT / "examples" / "demo_output.png"


def simulate_circle(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, float]:
    true_center = np.array([1.0, -2.0])
    radius = 3.0

    angles = rng.uniform(0.0, 2 * np.pi, size=100)
    circle_points = true_center + radius * np.column_stack((np.cos(angles), np.sin(angles)))
    circle_points += rng.normal(0.0, 0.05, size=circle_points.shape)

    return circle_points, true_center, radius


def simulate_sphere(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, float]:
    true_center = np.array([0.6, -1.2, 2.2])
    radius = 2.3

    phi = rng.uniform(0.0, np.pi, size=500)
    theta = rng.uniform(0.0, 2 * np.pi, size=500)
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    sphere_points = true_center + radius * np.column_stack((x, y, z))
    sphere_points += rng.normal(0.0, 0.04, size=sphere_points.shape)

    return sphere_points, true_center, radius


def main() -> None:
    rng = np.random.default_rng(2025)

    circle_points, circle_true_center, circle_radius = simulate_circle(rng)
    sphere_points, sphere_true_center, sphere_radius = simulate_sphere(rng)

    circle_algebraic = fit_circle_known_radius_algebraic(circle_points, circle_radius)
    circle_iterative = fit_circle_known_radius_iterative(circle_points, circle_radius)

    sphere_algebraic = fit_sphere_known_radius_algebraic(sphere_points, sphere_radius)
    sphere_iterative = fit_sphere_known_radius_iterative(
        sphere_points,
        sphere_radius,
        initial_center=sphere_algebraic.center,
        max_iter=100,
        tol=1e-10,
    )

    print("Circle (2D):")
    print(f"  True centre            : {circle_true_center}")
    print(f"  Algebraic centre       : {circle_algebraic.center}")
    print(f"  Iterative centre       : {circle_iterative.center}")
    print(f"  Algebraic RMSE         : {circle_algebraic.rmse:.6f}")
    print(f"  Iterative RMSE         : {circle_iterative.rmse:.6f}\n")

    print("Sphere (3D):")
    print(f"  True centre            : {sphere_true_center}")
    print(f"  Algebraic centre       : {sphere_algebraic.center}")
    print(f"  Iterative centre       : {sphere_iterative.center}")
    print(f"  Algebraic RMSE         : {sphere_algebraic.rmse:.6f}")
    print(f"  Iterative RMSE         : {sphere_iterative.rmse:.6f}")
    print(f"  Iterative iterations   : {sphere_iterative.iterations}\n")

    fig = plt.figure(figsize=(12, 6))
    ax_circle = fig.add_subplot(1, 2, 1)
    ax_sphere = fig.add_subplot(1, 2, 2, projection="3d")

    # Circle plot
    ax_circle.scatter(circle_points[:, 0], circle_points[:, 1], s=10, alpha=0.6, label="Noisy samples")
    circle_patch = plt.Circle(circle_true_center, circle_radius, color="tab:gray", fill=False, linestyle="--")
    ax_circle.add_patch(circle_patch)
    ax_circle.scatter(*circle_true_center, color="black", marker="x", s=80, label="True centre")
    ax_circle.scatter(*circle_algebraic.center, color="tab:orange", marker="o", s=60, label="Algebraic centre")
    ax_circle.scatter(*circle_iterative.center, color="tab:blue", marker="^", s=60, label="Iterative centre")
    ax_circle.set_title("Circle fitting (radius known)")
    ax_circle.set_aspect("equal", adjustable="datalim")
    ax_circle.legend(loc="upper right")
    ax_circle.grid(True, linestyle=":", linewidth=0.5)

    # Sphere plot
    ax_sphere.scatter(
        sphere_points[:, 0],
        sphere_points[:, 1],
        sphere_points[:, 2],
        s=8,
        alpha=0.4,
        label="Noisy samples",
    )
    ax_sphere.scatter(*sphere_true_center, color="black", marker="x", s=80, label="True centre")
    ax_sphere.scatter(*sphere_algebraic.center, color="tab:orange", marker="o", s=60, label="Algebraic centre")
    ax_sphere.scatter(*sphere_iterative.center, color="tab:blue", marker="^", s=60, label="Iterative centre")
    ax_sphere.set_title("Sphere fitting (radius known)")
    ax_sphere.set_xlabel("x")
    ax_sphere.set_ylabel("y")
    ax_sphere.set_zlabel("z")
    ax_sphere.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=200, bbox_inches="tight")
    print(f"Saved visual comparison to {FIGURE_PATH}")


if __name__ == "__main__":
    main()