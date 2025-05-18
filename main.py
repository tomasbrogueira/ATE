# main.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

from base_functions import (
    simulate_full_grid,
    build_A_b_from_Y,
    filter_feasible_points,
    generate_random_rectangles,
    find_best_rectangle,
    compute_true_bounds
)

def plot_branch_currents(branch_currents, rates, m):
    for idx, (branch, current) in enumerate(branch_currents.items()):
        plt.figure()
        plt.plot(current, label="Current")
        plt.hlines(rates[idx], 0, m-1,
                   linestyles='--', label=f"Rating = {rates[idx]}")
        plt.title(f"Branch {branch} Current vs Time")
        plt.xlabel("Time step")
        plt.ylabel("Current")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

def plot_feasible_region(X_hist, lower_best, upper_best, dims=(0, 1)):
    points2d = X_hist[:, dims]
    hull = ConvexHull(points2d)
    verts = points2d[hull.vertices]

    plt.figure()
    plt.scatter(points2d[:,0], points2d[:,1], s=10, alpha=0.6, label="Feasible Points")
    plt.plot(
        np.append(verts[:,0], verts[0,0]),
        np.append(verts[:,1], verts[0,1]),
        lw=2, color='orange', label="Polytope Boundary"
    )
    rx = [lower_best[dims[0]], upper_best[dims[0]], upper_best[dims[0]],
          lower_best[dims[0]], lower_best[dims[0]]]
    ry = [lower_best[dims[1]], lower_best[dims[1]], upper_best[dims[1]],
          upper_best[dims[1]], lower_best[dims[1]]]
    plt.plot(rx, ry, lw=2, ls='--', color='red', label="Best Hyperrectangle")

    plt.title(f"2D Projection: Nodes {dims[0]} & {dims[1]}")
    plt.xlabel(f"Node {dims[0]} Injection")
    plt.ylabel(f"Node {dims[1]} Injection")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    # 1. Simulate network data
    m, seed = 500, 98
    Ii, branch_currents, Y, branch_list = simulate_full_grid(m=m, seed=seed)

    # 2. Define branch thermal ratings
    rates = [0.25, 1.0, 0.5]

    # 3. Build polytope A x <= b
    A, b = build_A_b_from_Y(Y, branch_list, rates)

    # 4. Filter historical injections
    X_hist, k = filter_feasible_points(Ii, A, b)
    print(f"Total simulated points: {m}")
    print(f"Feasible historical points: {k}")

    # 5. Empirical bounds
    emp_min = X_hist.min(axis=0)
    emp_max = X_hist.max(axis=0)

    # 6. True polytope bounds (with fallback)
    true_min, true_max = compute_true_bounds(A, b,
                                             empirical_min=emp_min,
                                             empirical_max=emp_max)

    # 7. Sample rectangles inside true bounds
    rectangles = generate_random_rectangles(A, b,
                                            true_min, true_max,
                                            n_rectangles=1000)
    print(f"Generated {len(rectangles)} candidate rectangles")

    # 8. Find best hyperrectangle
    best_idx, best_rect, best_count = find_best_rectangle(X_hist, rectangles)
    lower_best, upper_best = best_rect
    print(f"Best rectangle index: {best_idx}")
    print(f"Points inside best rectangle: {best_count}")
    print("Best lower bounds:", np.round(lower_best, 3))
    print("Best upper bounds:", np.round(upper_best, 3))

    # 9. Visualize
    plot_branch_currents(branch_currents, rates, m)
    plot_feasible_region(X_hist, lower_best, upper_best, dims=(0,1))

    return lower_best, upper_best, best_count

if __name__ == "__main__":
    main()
