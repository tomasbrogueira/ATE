import numpy as np
import itertools

from region_finder import (
    simulate_full_grid,
    build_A_b_from_Y,
    filter_feasible_points,
    compute_true_bounds,
    generate_rectangles_from_polytope,
    generate_random_rectangles,
    find_best_rectangle,
    filter_contained_rectangles
)
from hexagonal_test import (
    generate_hexagonal_prism_points,
    get_hexagonal_prism_polytope
)
from  plots import plot_time_series, plot_feasible_region, plot_feasible_region_3d



def analyze_network():
    """
    Simulates grid injections, filters feasible points, computes true bounds,
    and finds the best 3D axis-aligned hyperrectangle over the specified triangle sub-network.
    Ensures full containment of rectangles in the polytope before selecting the best.
    """
    # 1) Simulate the grid
    injections, branch_currents, Y, branch_list = simulate_full_grid(
        m=5000, seed=98
    )
    X_all = injections.T

    # 2) Build polytope constraints
    thermal_limits = [0.5, 1.0, 0.25]
    A, b = build_A_b_from_Y(Y, branch_list, thermal_limits)

    # 3) Filter feasible points
    X_hist, k = filter_feasible_points(injections, A, b)
    print(f"Simulated {X_all.shape[0]} points; {k} are feasible.")

    # 4) Compute true axis-aligned bounds
    true_lo, true_hi = compute_true_bounds(A, b)
    print("True variable bounds:")
    print(" Min:", np.round(true_lo, 3))
    print(" Max:", np.round(true_hi, 3))

    # --- Select triangle sub-network dims ---
    triangle_dims = [0, 1, 2]
    X_sub = X_hist[:, triangle_dims]
    A_sub = A[:, triangle_dims]
    b_sub = b

    # Compute empirical extents
    data_lo = X_sub.min(axis=0)
    data_hi = X_sub.max(axis=0)

    # Check boundedness
    bounded = np.all(np.isfinite(true_lo[triangle_dims]) & np.isfinite(true_hi[triangle_dims]))

    # 5) Generate rectangle candidates with full containment enforcement
    if bounded:
        # polytope-based enumeration
        I = np.eye(len(triangle_dims))
        A_box = np.vstack([A_sub, I, -I])
        b_box = np.concatenate([b_sub, data_hi, -data_lo])
        raw_candidates = generate_rectangles_from_polytope(
            A_box, b_box,
            n_rectangles=2000,
            dim=len(triangle_dims),
            random_state=98
        )
    else:
        # sample-based rectangle generation
        print("Sub-polytope unbounded; using historical-sample fallback.")
        raw_candidates = generate_random_rectangles(
            A_sub, b_sub,
            n_rectangles=2000,
            X_hist=X_sub
        )

    # enforce that all rectangle corners lie inside the sub-polytope
    candidates = filter_contained_rectangles(raw_candidates, A_sub, b_sub)
    if not candidates:
        print("No contained rectangle candidates generated.")
        return

    # 6) Pick best rectangle
    best_idx, (best_lo, best_hi), best_count = find_best_rectangle(
        X_sub, candidates
    )

    print("Best 3D hyperrectangle over nodes", triangle_dims)
    print(" Lower bounds:", np.round(best_lo, 3))
    print(" Upper bounds:", np.round(best_hi, 3))
    print(f"Covers {best_count} / {k} feasible points.")

    # 7) Visualize 3D and all 2D projections
    lo_vis = np.where(np.isfinite(true_lo), true_lo, X_hist.min(axis=0))
    hi_vis = np.where(np.isfinite(true_hi), true_hi, X_hist.max(axis=0))
    lo_vis[triangle_dims] = best_lo
    hi_vis[triangle_dims] = best_hi

    plot_feasible_region_3d(
        X_hist, lo_vis, hi_vis,
        dims_to_plot=triangle_dims
    )

    # plot every 2D projection
    for dims in itertools.combinations(triangle_dims, 2):
        i, j = dims
        X2 = X_hist[:, [i, j]]
        lo2 = best_lo[[triangle_dims.index(i), triangle_dims.index(j)]]
        hi2 = best_hi[[triangle_dims.index(i), triangle_dims.index(j)]]
        plot_feasible_region(X2, lo2, hi2, dims)


def test_hexagonal_prism(use_polytope_sampling=True,
                         num_points=500,
                         num_rectangles=5000,
                         seed=42):
    """
    Generate points in a hexagonal prism, then find and display
    the best-fitting axis-aligned rectangle fully contained in the prism.
    """
    # prism parameters
    side, z_min, z_max = 1.0, 0.0, 2.0

    print(f"\nHexagonal prism test: {num_points} pts, side={side}, z∈[{z_min},{z_max}]")
    data = generate_hexagonal_prism_points(num_points, side, z_min, z_max, seed=seed)

    # build true prism polytope
    A_p, b_p = get_hexagonal_prism_polytope(side, z_min, z_max)
    lo_p, hi_p = compute_true_bounds(A_p, b_p)
    print("Prism true bounds:\n Min:", np.round(lo_p,3), "\n Max:", np.round(hi_p,3))

    # sample rectangles
    if use_polytope_sampling:
        raw_rects = generate_rectangles_from_polytope(
            A_p, b_p,
            n_rectangles=num_rectangles,
            dim=A_p.shape[1],
            random_state=seed
        )
    else:
        raw_rects = generate_random_rectangles(
            A_p, b_p,
            n_rectangles=num_rectangles,
            X_hist=data
        )

    # filter for full containment
    prism_rects = filter_contained_rectangles(raw_rects, A_p, b_p)
    if not prism_rects:
        print("No contained prism rectangles generated.")
        return

    # select best
    idx_p, (lo_rp, hi_rp), count_p = find_best_rectangle(data, prism_rects)
    print(f"Best prism rectangle #{idx_p} covers {count_p} points.")
    print(" Prism rectangle bounds:\n Lower:", np.round(lo_rp,3), 
          "\n Upper:", np.round(hi_rp,3))

    # visualize
    if data.shape[1] >= 3:
        plot_feasible_region_3d(data, lo_rp, hi_rp)
    # plot every 2D projection
    for dims in itertools.combinations(range(data.shape[1]), 2):
        X2 = data[:, list(dims)]
        lo2 = lo_rp[list(dims)]
        hi2 = hi_rp[list(dims)]
        plot_feasible_region(X2, lo2, hi2, dims)

    return lo_rp, hi_rp, count_p


if __name__ == "__main__":
    test_hexagonal_prism()
    analyze_network()
