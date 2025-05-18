import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

from region_finder import (
    simulate_full_grid,
    build_A_b_from_Y,
    filter_feasible_points,
    compute_true_bounds,
    generate_rectangles_from_polytope,
    generate_random_rectangles,
    find_best_rectangle
)
from hexagonal_test import (
    generate_hexagonal_prism_points,
    get_hexagonal_prism_polytope
)
from  plots import plot_feasible_region, plot_feasible_region_3d

def plot_time_series(data_series, thresholds, title_prefix):
    """
    Plot a time series with upper and lower threshold lines.
    """
    for idx, series in enumerate(data_series.values()):
        plt.figure()
        plt.plot(series, label="Value Over Time")
        plt.hlines(thresholds[idx], 0, len(series)-1, linestyles='--', colors='red',
                   label=f"+{thresholds[idx]} threshold")
        plt.hlines(-thresholds[idx], 0, len(series)-1, linestyles='--', colors='red',
                   label=f"-{thresholds[idx]} threshold")
        plt.title(f"{title_prefix} {list(data_series.keys())[idx]}")
        plt.xlabel("Step")
        plt.ylabel("Magnitude")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


def analyze_network(use_polytope_sampling=True,
                    total_samples=5000,
                    random_seed=98,
                    n_rectangles=2000):
    """
    Simulate network injections, find the best axis-aligned bounding rectangle,
    and display both 2D and 3D feasible regions.
    """
    # 1) simulate
    injections, branch_currents, Y, branches = simulate_full_grid(
        m=total_samples, seed=random_seed
    )

    # 2) build polytope Ax <= b
    thermal_limits = [0.5, 1.0, 0.25]
    A, b = build_A_b_from_Y(Y, branches, thermal_limits)

    # 3) historical feasibility (for evaluation/plotting)
    X_hist, k_hist = filter_feasible_points(injections, A, b)
    print(f"Simulated {total_samples} points; {k_hist} are feasible.")

    # 4) true axis-aligned bounds
    true_lo, true_hi = compute_true_bounds(A, b)
    print("True variable bounds:\n Min:", np.round(true_lo,3), "\n Max:", np.round(true_hi,3))

    # 5) sample rectangles
    if use_polytope_sampling:
        # sample from the true polytope via random LP directions
        candidates = generate_rectangles_from_polytope(
            A, b,
            n_rectangles=n_rectangles,
            dim=A.shape[1],
            random_state=random_seed
        )
    else:
        # legacy: sample from the historical feasible set
        candidates = generate_random_rectangles(
            A, b,
            n_rectangles=n_rectangles,
            X_hist=X_hist
        )

    # 6) pick best
    best_idx, (best_lo, best_hi), best_count = find_best_rectangle(
        X_hist, candidates
    )
    print(f"Best rectangle #{best_idx} contains {best_count} points.")
    print(" Best lower bounds:", np.round(best_lo,3))
    print(" Best upper bounds:", np.round(best_hi,3))

    # 7) plot
    plot_time_series(branch_currents, thermal_limits, label_prefix="Branch Current")
    if X_hist.size:
        if X_hist.shape[1] >= 3:
            plot_feasible_region_3d(X_hist, best_lo, best_hi, dims_to_plot=[0,1,2])
        plot_feasible_region(X_hist, best_lo, best_hi)
    else:
        print("No feasible data to visualize.")

    return best_lo, best_hi, best_count



def test_hexagonal_prism(use_polytope_sampling=True,
                         num_points=50000,
                         num_rectangles=5000,
                         seed=42):
    """
    Generate points in a hexagonal prism, then find and display
    the best-fitting axis-aligned rectangle.
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
        prism_rects = generate_rectangles_from_polytope(
            A_p, b_p,
            n_rectangles=num_rectangles,
            dim=A_p.shape[1],
            random_state=seed
        )
    else:
        prism_rects = generate_random_rectangles(
            A_p, b_p,
            n_rectangles=num_rectangles,
            X_hist=data
        )

    # select best
    idx_p, (lo_rp, hi_rp), count_p = find_best_rectangle(data, prism_rects)
    print(f"Best prism rectangle #{idx_p} covers {count_p} points.")
    print(" Prism rectangle bounds:\n Lower:", np.round(lo_rp,3), 
          "\n Upper:", np.round(hi_rp,3))

    # visualize
    if data.shape[1] >= 3:
        plot_feasible_region_3d(data, lo_rp, hi_rp)
    plot_feasible_region(data, lo_rp, hi_rp)

    return lo_rp, hi_rp, count_p



if __name__ == "__main__":
    test_hexagonal_prism()