import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

def plot_feasible_region(X_hist, lower, upper, dims=(0,1)):
    """
    Plot a single 2D projection of the feasible region and the best hyperrectangle slice.
    X_hist: (n_points, 2) array of feasible points for the 2D projection.
    lower: (2,) array, lower bounds for the rectangle in the 2D projection.
    upper: (2,) array, upper bounds for the rectangle in the 2D projection.
    dims: tuple of two ints, indices of the original dimensions.
    """
    if X_hist.shape[1] != 2:
        print("plot_feasible_region expects a 2D array for X_hist (n_points, 2).")
        return

    pts = X_hist

    if pts.shape[0] < 3: # ConvexHull needs at least dim+1 points
        print(f"Not enough feasible points to plot for this 2D projection")
        verts = None
    else:
        try:
            hull = ConvexHull(pts)
            verts = pts[hull.vertices]
        except Exception as e:
            print(f"Could not compute Convex Hull for this 2D projection: {e}")
            verts = None

    plt.figure()
    plt.scatter(pts[:,0], pts[:,1], s=10, alpha=0.6, label="Feasible Points")

    if verts is not None:
        plt.plot(
            np.append(verts[:,0], verts[0,0]),
            np.append(verts[:,1], verts[0,1]),
            lw=2, color='orange', label="Polytope Boundary"
        )

    rx = [lower[0], upper[0], upper[0], lower[0], lower[0]]
    ry = [lower[1], lower[1], upper[1], upper[1], lower[1]]
    plt.plot(rx, ry, lw=2, ls='--', color='red', label="Best Hyperrectangle Slice")

    plt.xlabel(f"Node {dims[0]} Injection")
    plt.ylabel(f"Node {dims[1]} Injection")
    plt.title(f"2D Projection: Nodes {dims[0]} & {dims[1]}")
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
    ax.plot([], [], [], color="red", linestyle='--', lw=2, label="Best Hyperrectangle")


    ax.set_xlabel(f"Node {dims_to_plot[0]} Injection")
    ax.set_ylabel(f"Node {dims_to_plot[1]} Injection")
    ax.set_zlabel(f"Node {dims_to_plot[2]} Injection")
    ax.set_title(f"3D Feasible Region (Nodes {dims_to_plot[0]},{dims_to_plot[1]},{dims_to_plot[2]}) and Best Hyperrectangle Slice")
    ax.legend()
    plt.tight_layout()
    plt.show()

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