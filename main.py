import numpy as np
import itertools

from region_finder import (
    simulate_full_grid, build_A_b_from_Y, filter_feasible_points, calculate_axis_aligned_bounds,
    generate_rectangles_from_polytope, generate_random_rectangles, find_best_rectangle,
    filter_contained_rectangles
)
from hexagonal_test import generate_hexagonal_prism_points, get_hexagonal_prism_polytope
from plots import plot_time_series, plot_feasible_region, plot_feasible_region_3d


def kyte_grid(rates=[0.5, 1.0, 0.25],n_points=5000, n_rectangles=500):
    injections, branch_currents, Y, branch_list = simulate_full_grid(m=n_points, seed=98)
    X_all = injections.T
    
    # Visualize branch currents over time with their thermal limits
    plot_time_series(branch_currents, rates, "Branch Current")
    
    A, b = build_A_b_from_Y(Y, branch_list, rates)

    X_hist, k = filter_feasible_points(injections, A, b)
    print(f"Simulated {X_all.shape[0]} points; {k} are feasible.")

    true_lo, true_hi = calculate_axis_aligned_bounds(A, b)
    print("True variable bounds:")
    print(" Min:", np.round(true_lo, 3))
    print(" Max:", np.round(true_hi, 3))

    dims = [0, 1, 2]
    X_sub = X_hist[:, dims]
    A_sub = A[:, dims]
    b_sub = b

    data_lo = X_sub.min(axis=0)
    data_hi = X_sub.max(axis=0)

    bounded = np.all(np.isfinite(true_lo[dims]) & np.isfinite(true_hi[dims]))

    if bounded:
        I = np.eye(len(dims))
        A_box = np.vstack([A_sub, I, -I])
        b_box = np.concatenate([b_sub, data_hi, -data_lo])
        rects = generate_rectangles_from_polytope(
            A_box, b_box, n_rectangles=n_rectangles, dim=len(dims), random_state=98
        )
    else:
        print("Sub-polytope unbounded; using historical-sample fallback.")
        rects = generate_random_rectangles(A_sub, b_sub, n_rectangles=n_rectangles, X_hist=X_sub)

    candidates = filter_contained_rectangles(rects, A_sub, b_sub)
    if not candidates:
        print("No contained rectangle candidates generated.")
        return

    best_idx, (best_lo, best_hi), best_count = find_best_rectangle(X_sub, candidates)

    print("Best 3D hyperrectangle over nodes", dims)
    print(" Lower bounds:", np.round(best_lo, 3))
    print(" Upper bounds:", np.round(best_hi, 3))
    print(f"Covers {best_count} / {k} feasible points.")

    lo_vis = np.where(np.isfinite(true_lo), true_lo, X_hist.min(axis=0))
    hi_vis = np.where(np.isfinite(true_hi), true_hi, X_hist.max(axis=0))
    lo_vis[dims] = best_lo
    hi_vis[dims] = best_hi

    plot_feasible_region_3d(X_hist, lo_vis, hi_vis, dims_to_plot=dims)

    for pair in itertools.combinations(dims, 2):
        i, j = pair
        X2 = X_hist[:, [i, j]]
        lo2 = best_lo[[dims.index(i), dims.index(j)]]
        hi2 = best_hi[[dims.index(i), dims.index(j)]]
        plot_feasible_region(X2, lo2, hi2, pair)


def hexagonal_prism_grid(use_polytope_sampling=True, num_points=500, num_rectangles=500, seed=42):
    side, z_min, z_max = 1.0, 0.0, 2.0

    print(f"\nHexagonal prism test: {num_points} pts, side={side}, z∈[{z_min},{z_max}]")
    data = generate_hexagonal_prism_points(num_points, side, z_min, z_max, seed=seed)

    A_p, b_p = get_hexagonal_prism_polytope(side, z_min, z_max)
    prism_bounds_min, prism_bounds_max = calculate_axis_aligned_bounds(A_p, b_p)
    print("Prism true bounds:\n Min:", np.round(prism_bounds_min,3), "\n Max:", np.round(prism_bounds_max,3))

    if use_polytope_sampling:
        rectangle_candidates = generate_rectangles_from_polytope(
            A_p, b_p, n_rectangles=num_rectangles, dim=A_p.shape[1], random_state=seed
        )
    else:
        rectangle_candidates = generate_random_rectangles(
            A_p, b_p, n_rectangles=num_rectangles, X_hist=data
        )

    valid_rectangles = filter_contained_rectangles(rectangle_candidates, A_p, b_p)
    if not valid_rectangles:
        print("No contained prism rectangles generated.")
        return

    best_rect_idx, (best_lower, best_upper), points_covered = find_best_rectangle(data, valid_rectangles)
    print(f"Best prism rectangle #{best_rect_idx} covers {points_covered} points.")
    print(" Prism rectangle bounds:\n Lower:", np.round(best_lower,3), "\n Upper:", np.round(best_upper,3))

    if data.shape[1] >= 3:
        plot_feasible_region_3d(data, best_lower, best_upper)
    
    for projection_dims in itertools.combinations(range(data.shape[1]), 2):
        projected_data = data[:, list(projection_dims)]
        proj_lower = best_lower[list(projection_dims)]
        proj_upper = best_upper[list(projection_dims)]
        plot_feasible_region(projected_data, proj_lower, proj_upper, projection_dims)

    return best_lower, best_upper, points_covered


if __name__ == "__main__":
    hexagonal_prism_grid(use_polytope_sampling=True, num_points=5000, num_rectangles=500)
    kyte_grid(rates=[0.5, 1.0, 0.25], n_points=50000, n_rectangles=5000)
