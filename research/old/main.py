import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

from region_finder import (
    simulate_full_grid,
    build_A_b_from_Y,
    filter_feasible_points,
    compute_true_bounds,
    find_best_hyperrectangle_local_maxima,
    generate_random_rectangles,
    find_best_rectangle
)
from hexagonal_test import generate_hexagonal_prism_points, get_hexagonal_prism_polytope

def plot_branch_currents(branch_currents, rates, m):
    for idx, (branch, current) in enumerate(branch_currents.items()):
        plt.figure()
        plt.plot(current, label="Current")
        plt.hlines(rates[idx], 0, m-1, linestyles='--', colors='red', label=f"Max Rating = {rates[idx]}")
        plt.hlines(-rates[idx], 0, m-1, linestyles='--', colors='red', label=f"Min Rating = {-rates[idx]}")
        plt.title(f"Branch {branch} Current vs Time")
        plt.xlabel("Time step")
        plt.ylabel("Current")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

def plot_feasible_region(X_hist, lower, upper):
    n = X_hist.shape[1]
    if n < 2:
        print("Cannot plot feasible region for less than 2 dimensions.")
        return
    for i in range(n):
        for j in range(i + 1, n):
            pts = X_hist[:, [i, j]]
            if pts.shape[0] < 3:
                continue
            try:
                hull = ConvexHull(pts)
                verts = pts[hull.vertices]
            except Exception:
                verts = None
            plt.figure()
            plt.scatter(pts[:,0], pts[:,1], s=10, alpha=0.6, label="Feasible Points")
            if verts is not None:
                plt.plot(
                    np.append(verts[:,0], verts[0,0]),
                    np.append(verts[:,1], verts[0,1]),
                    lw=2, color='orange', label="Polytope Boundary"
                )
            rx = [lower[i], upper[i], upper[i], lower[i], lower[i]]
            ry = [lower[j], lower[j], upper[j], upper[j], lower[j]]
            plt.plot(rx, ry, lw=2, ls='--', color='red', label="Best Hyperrectangle Slice")
            plt.title(f"2D Projection: Nodes {i} & {j}")
            plt.xlabel(f"Node {i} Injection")
            plt.ylabel(f"Node {j} Injection")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

def plot_feasible_region_3d(X_hist, lower, upper, dims_to_plot=None):
    if dims_to_plot is None:
        if X_hist.shape[1] == 3:
            dims_to_plot = [0, 1, 2]
        else:
            print("Data is not 3-dimensional and no specific dimensions to plot were provided for 3D plot.")
            return
    if len(dims_to_plot) != 3:
        print("dims_to_plot must specify exactly three dimensions for a 3D plot.")
        return
    for dim_idx in dims_to_plot:
        if not (0 <= dim_idx < X_hist.shape[1]):
            print(f"Dimension index {dim_idx} is out of bounds for X_hist with {X_hist.shape[1]} dimensions.")
            return
    X_plot = X_hist[:, dims_to_plot]
    lower_plot = lower[dims_to_plot]
    upper_plot = upper[dims_to_plot]
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_plot[:, 0], X_plot[:, 1], X_plot[:, 2], s=10, alpha=0.4, label="Feasible Points")
    if X_plot.shape[0] >= 4:
        try:
            hull = ConvexHull(X_plot)
            for s in hull.simplices:
                s_aug = np.append(s, s[0])
                ax.plot(X_plot[s_aug, 0], X_plot[s_aug, 1], X_plot[s_aug, 2], "orange", lw=0.5, alpha=0.7)
            ax.plot([], [], [], color="orange", label="Polytope Boundary (Convex Hull)")
        except Exception:
            pass
    v = np.array([
        [lower_plot[0], lower_plot[1], lower_plot[2]],
        [upper_plot[0], lower_plot[1], lower_plot[2]],
        [upper_plot[0], upper_plot[1], lower_plot[2]],
        [lower_plot[0], upper_plot[1], lower_plot[2]],
        [lower_plot[0], lower_plot[1], upper_plot[2]],
        [upper_plot[0], lower_plot[1], upper_plot[2]],
        [upper_plot[0], upper_plot[1], upper_plot[2]],
        [lower_plot[0], upper_plot[1], upper_plot[2]]
    ])
    edges = [
        [v[0], v[1]], [v[1], v[2]], [v[2], v[3]], [v[3], v[0]],
        [v[4], v[5]], [v[5], v[6]], [v[6], v[7]], [v[7], v[4]],
        [v[0], v[4]], [v[1], v[5]], [v[2], v[6]], [v[3], v[7]]
    ]
    for edge in edges:
        ax.plot3D(*zip(*edge), color="red", linestyle='--', lw=2)
    ax.plot([], [], [], color="red", linestyle='--', lw=2, label="Best Hyperrectangle Slice")
    ax.set_xlabel(f"Node {dims_to_plot[0]} Injection")
    ax.set_ylabel(f"Node {dims_to_plot[1]} Injection")
    ax.set_zlabel(f"Node {dims_to_plot[2]} Injection")
    ax.set_title(f"3D Feasible Region (Nodes {dims_to_plot[0]},{dims_to_plot[1]},{dims_to_plot[2]})")
    ax.legend()
    plt.tight_layout()
    plt.show()

def main():
    USE_HEXAGONAL_PRISM_TEST = False
    USE_LOCAL_MAXIMA_SEARCH = False
    if USE_HEXAGONAL_PRISM_TEST:
        SIDE_LENGTH_PRISM = 1.0
        Z_MIN_PRISM = 0.0
        Z_MAX_PRISM = 2.0
        NUM_POINTS_PRISM = 5000
        SEED_PRISM = 42
        print('\n--- Using Hexagonal Prism Test Data ---')
        X_hist = generate_hexagonal_prism_points(NUM_POINTS_PRISM, SIDE_LENGTH_PRISM, Z_MIN_PRISM, Z_MAX_PRISM, seed=SEED_PRISM)
        m = X_hist.shape[0]
        k = m
        print(f'Total generated points for prism: {m}, Feasible: {k}')
        rates = None
        A, b = get_hexagonal_prism_polytope(SIDE_LENGTH_PRISM, Z_MIN_PRISM, Z_MAX_PRISM)
        print(f'Using prism polytope A: {A.shape}, b: {b.shape}')
        branch_currents = {}
    else:
        m, seed = 5000, 98
        Ii, branch_currents, Y, branch_list = simulate_full_grid(m=m, seed=seed)
        rates = [0.5, 1.0, 0.25]
        A, b = build_A_b_from_Y(Y, branch_list, rates)
        X_hist, k = filter_feasible_points(Ii, A, b)
        print(f"Total simulated points: {m}, Feasible: {k}")
    if USE_LOCAL_MAXIMA_SEARCH:
        true_min, true_max = compute_true_bounds(A, b)
        print("True bounds per dimension:")
        print("  min:", np.round(true_min, 3))
        print("  max:", np.round(true_max, 3))
        print("Searching for best hyperrectangle using local maxima search...")
        lower_best, upper_best, best_count = find_best_hyperrectangle_local_maxima(A, b, X_hist, n_anchors=20)
        if lower_best is None or upper_best is None or best_count < 0:
            print("No valid hyperrectangle found inside the polytope. Using axis-aligned bounding box of feasible points.")
            lower_best = np.min(X_hist, axis=0)
            upper_best = np.max(X_hist, axis=0)
            best_count = np.sum(np.all((X_hist >= lower_best) & (X_hist <= upper_best), axis=1))
        print(f"Points inside best rectangle: {best_count}")
        print("Best lower bounds:", np.round(lower_best, 3))
        print("Best upper bounds:", np.round(upper_best, 3))
        if not USE_HEXAGONAL_PRISM_TEST and branch_currents:
            plot_branch_currents(branch_currents, rates, m)
        if X_hist.shape[0] > 0:
            if X_hist.shape[1] >= 3:
                plot_feasible_region_3d(X_hist, lower_best, upper_best, dims_to_plot=[0, 1, 2])
            plot_feasible_region(X_hist, lower_best, upper_best)
        else:
            print("No feasible points to visualize.")
        return lower_best, upper_best, best_count
    else:
        print("Using random rectangles...")
        # Compute true polytope bounds for rectangle generation
        true_min, true_max = compute_true_bounds(A, b)
        # Generate rectangles inside the polytope using feasible points as anchors
        num_rectangles = 10000  # You can increase this for more thorough search
        rectangles = generate_random_rectangles(A, b, X_hist, n_rectangles=num_rectangles)
        # Find the rectangle with the most points inside
        best_idx, best_rectangle, best_count = find_best_rectangle(X_hist, rectangles)
        if best_rectangle is None:
            print("No valid rectangle found.")
            return
        print(f"Points inside best rectangle: {best_count}")
        print("Best rectangle bounds:")
        print("  lower:", np.round(best_rectangle[0], 3))
        print("  upper:", np.round(best_rectangle[1], 3))
        if X_hist.shape[0] > 0:
            plot_feasible_region(X_hist, best_rectangle[0], best_rectangle[1])
        else:
            print("No feasible points to visualize.")

if __name__ == "__main__":
    main()