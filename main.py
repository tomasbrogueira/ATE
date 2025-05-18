import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

from region_finder import (
    simulate_full_grid,
    build_A_b_from_Y,
    filter_feasible_points,
    compute_true_bounds,
    generate_random_rectangles,
    find_best_rectangle
)

def plot_branch_currents(branch_currents, rates, m):
    for idx, (branch, current) in enumerate(branch_currents.items()):
        plt.figure()
        plt.plot(current, label="Current")
        plt.hlines(rates[idx], 0, m-1, linestyles='--', colors='red',
                   label=f"Max Rating = {rates[idx]}")
        plt.hlines(-rates[idx], 0, m-1, linestyles='--', colors='red',
                   label=f"Min Rating = {-rates[idx]}")
        plt.title(f"Branch {branch} Current vs Time")
        plt.xlabel("Time step")
        plt.ylabel("Current")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

def plot_feasible_region(X_hist, lower, upper):
    num_dimensions = X_hist.shape[1]
    if num_dimensions < 2:
        print("Cannot plot feasible region for less than 2 dimensions.")
        return

    for i in range(num_dimensions):
        for j in range(i + 1, num_dimensions):
            dims = (i, j)
            pts = X_hist[:, dims]

            if pts.shape[0] < 3: # ConvexHull needs at least dim+1 points
                print(f"Not enough feasible points to plot for dimensions {dims}")
                continue
            
            try:
                hull = ConvexHull(pts)
                verts = pts[hull.vertices]
            except Exception as e:
                print(f"Could not compute Convex Hull for dimensions {dims}: {e}")
                # Fallback: plot points without hull if hull computation fails
                verts = None


            plt.figure()
            plt.scatter(pts[:,0], pts[:,1], s=10, alpha=0.6, label="Feasible Points")
            
            if verts is not None:
                plt.plot(
                    np.append(verts[:,0], verts[0,0]),
                    np.append(verts[:,1], verts[0,1]),
                    lw=2, color='orange', label="Polytope Boundary"
                )

            rx = [lower[dims[0]], upper[dims[0]], upper[dims[0]],
                  lower[dims[0]], lower[dims[0]]]
            ry = [lower[dims[1]], lower[dims[1]], upper[dims[1]],
                  upper[dims[1]], lower[dims[1]]]
            plt.plot(rx, ry, lw=2, ls='--', color='red', label="Best Hyperrectangle Slice")

            plt.title(f"2D Projection: Nodes {dims[0]} & {dims[1]}")
            plt.xlabel(f"Node {dims[0]} Injection")
            plt.ylabel(f"Node {dims[1]} Injection")
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
    elif len(dims_to_plot) != 3:
        print("dims_to_plot must specify exactly three dimensions for a 3D plot.")
        return
    
    for dim_idx in dims_to_plot:
        if not (0 <= dim_idx < X_hist.shape[1]):
            print(f"Dimension index {dim_idx} is out of bounds for X_hist with {X_hist.shape[1]} dimensions.")
            return

    X_plot = X_hist[:, dims_to_plot]
    lower_plot = lower[dims_to_plot]
    upper_plot = upper[dims_to_plot]

    if X_plot.shape[0] < 4: # ConvexHull in 3D needs at least 4 points
        print(f"Not enough feasible points (found {X_plot.shape[0]}) in the selected dimensions {dims_to_plot} to compute 3D Convex Hull. Need at least 4.")
        # We can still plot points and rectangle

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot feasible points
    ax.scatter(X_plot[:, 0], X_plot[:, 1], X_plot[:, 2], s=10, alpha=0.4, label="Feasible Points")

    # Plot Convex Hull of feasible points
    if X_plot.shape[0] >= 4:
        try:
            hull = ConvexHull(X_plot)
            # Draw the triangular faces of the hull
            for s in hull.simplices:
                s_aug = np.append(s, s[0]) # Close the triangle for plotting
                ax.plot(X_plot[s_aug, 0], X_plot[s_aug, 1], X_plot[s_aug, 2], "orange", lw=0.5, alpha=0.7)
            # Dummy plot for legend
            ax.plot([], [], [], color="orange", label="Polytope Boundary (Convex Hull)")
        except Exception as e:
            print(f"Could not compute or plot 3D Convex Hull for dimensions {dims_to_plot}: {e}")


    # Plot the best hyperrectangle (cuboid slice)
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
        [v[0], v[1]], [v[1], v[2]], [v[2], v[3]], [v[3], v[0]],  # Bottom face
        [v[4], v[5]], [v[5], v[6]], [v[6], v[7]], [v[7], v[4]],  # Top face
        [v[0], v[4]], [v[1], v[5]], [v[2], v[6]], [v[3], v[7]]   # Vertical edges
    ]

    for edge in edges:
        ax.plot3D(*zip(*edge), color="red", linestyle='--', lw=2)
    ax.plot([], [], [], color="red", linestyle='--', lw=2, label="Best Hyperrectangle Slice")


    ax.set_xlabel(f"Node {dims_to_plot[0]} Injection")
    ax.set_ylabel(f"Node {dims_to_plot[1]} Injection")
    ax.set_zlabel(f"Node {dims_to_plot[2]} Injection")
    ax.set_title(f"3D Feasible Region (Nodes {dims_to_plot[0]},{dims_to_plot[1]},{dims_to_plot[2]}) and Best Hyperrectangle Slice")
    ax.legend()
    plt.tight_layout()
    plt.show()

def main():
    # 1) Simulate network data
    m, seed = 5000, 98
    Ii, branch_currents, Y, branch_list = simulate_full_grid(m=m, seed=seed)

    # 2) Define branch thermal ratings
    # Aligned with grid_cap_regress.py: rate_12 (0-1), rate_13 (0-2), rate_23 (1-2)
    rates = [0.5, 1.0, 0.25]

    # 3) Build the polytope A x <= b
    A, b = build_A_b_from_Y(Y, branch_list, rates)

    # 4) Filter feasible historical injections
    X_hist, k = filter_feasible_points(Ii, A, b)
    print(f"Total simulated points: {m}, Feasible: {k}")

    # 5) Compute true axis-aligned bounds
    true_min, true_max = compute_true_bounds(A, b)
    print("True bounds per dimension:")
    print("  min:", np.round(true_min, 3))
    print("  max:", np.round(true_max, 3))

    # 6) Generate random hyperrectangles from pairs of feasible points
    n_rects = 5000
    rects = generate_random_rectangles(
        A, b,
        n_rectangles=n_rects,
        X_hist=X_hist
    )
    print(f"Generated {len(rects)} candidate rectangles")

    # 7) Pick the rectangle covering the most points
    best_idx, (lower_best, upper_best), best_count = find_best_rectangle(X_hist, rects)
    print(f"Best rectangle index: {best_idx}")
    print(f"Points inside best rectangle: {best_count}")
    print("Best lower bounds:", np.round(lower_best, 3))
    print("Best upper bounds:", np.round(upper_best, 3))

    # 8) Visualize results
    plot_branch_currents(branch_currents, rates, m)

    if X_hist.shape[0] > 0:
        if X_hist.shape[1] >= 3:
            # Assuming nodes 0, 1, and 2 form the triangle of interest
            plot_feasible_region_3d(X_hist, lower_best, upper_best, dims_to_plot=[0, 1, 2])
        plot_feasible_region(X_hist, lower_best, upper_best)
    else:
        print("No feasible points to visualize.")
    return lower_best, upper_best, best_count

if __name__ == "__main__":
    main()
