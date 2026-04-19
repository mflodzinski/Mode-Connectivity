import numpy as np
import matplotlib.pyplot as plt


def plot_mode_connectivity_geometry(seed=7):
    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------
    # 1) Generate 2 random points in the first quadrant
    # ------------------------------------------------------------
    A = rng.uniform(1.5, 8.0, size=2)
    B = rng.uniform(1.5, 8.0, size=2)

    # Avoid points being too close
    while np.linalg.norm(B - A) < 2.0:
        B = rng.uniform(1.5, 8.0, size=2)

    # ------------------------------------------------------------
    # 2) Midpoint = Init Point
    # ------------------------------------------------------------
    M = 0.5 * (A + B)

    # ------------------------------------------------------------
    # 3) Perpendicular direction to interpolation line AB
    #    If d = (dx, dy), then perpendicular is (-dy, dx)
    # ------------------------------------------------------------
    d = B - A
    perp = np.array([-d[1], d[0]], dtype=float)
    perp = perp / np.linalg.norm(perp)

    # Make the perpendicular line long enough for plotting
    perp_line_half_length = 3.5
    P1 = M - perp_line_half_length * perp
    P2 = M + perp_line_half_length * perp

    # ------------------------------------------------------------
    # 4) Pick an "optimized point" on the perpendicular line
    # ------------------------------------------------------------
    t_opt = 1.7
    O = M + t_opt * perp

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 9), dpi=140)

    # Background grid
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.grid(True, alpha=0.3)

    # Axes
    ax.axhline(0, color="black", linewidth=1.2)
    ax.axvline(0, color="black", linewidth=1.2)
    ax.set_xlabel(r"$\theta_1$", fontsize=14)
    ax.set_ylabel(r"$\theta_2$", fontsize=14)

    # Interpolation line between A and B
    ax.plot(
        [A[0], B[0]], [A[1], B[1]],
        linestyle="--", linewidth=2, label="Interpolation"
    )

    # Perpendicular symmetry plane
    ax.plot(
        [P1[0], P2[0]], [P1[1], P2[1]],
        linestyle=":", linewidth=2.5, label="Perpendicular Symmetry Plane"
    )

    # Colored line from Init Point to Optimized Point
    ax.annotate(
        "",
        xy=O, xytext=M,
        arrowprops=dict(arrowstyle="->", linewidth=3)
    )

    # Points
    ax.scatter(A[0], A[1], s=120, zorder=5)
    ax.scatter(B[0], B[1], s=120, zorder=5)
    ax.scatter(M[0], M[1], s=140, marker="D", facecolors="white", edgecolors="black", zorder=6)
    ax.scatter(O[0], O[1], s=140, marker="D", zorder=6)

    # Labels for points
    ax.text(A[0] + 0.15, A[1] + 0.15, "Network A", fontsize=12)
    ax.text(B[0] + 0.15, B[1] + 0.15, "Network B", fontsize=12)
    ax.text(M[0] + 0.15, M[1] - 0.35, "Init Point", fontsize=12)
    ax.text(O[0] + 0.15, O[1] + 0.15, "Optimized Point", fontsize=12)

    # Labels for lines
    interp_mid = 0.5 * (A + B)
    ax.text(
        interp_mid[0] + 0.2, interp_mid[1] + 0.3,
        "Interpolation", fontsize=12
    )

    perp_label_point = M + 1.2 * perp
    ax.text(
        perp_label_point[0] + 0.2, perp_label_point[1] + 0.2,
        "Perpendicular Symmetry Plane", fontsize=12
    )

    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Mode Connectivity Geometry", fontsize=16)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_mode_connectivity_geometry(42)