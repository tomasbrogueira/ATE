import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, HalfspaceIntersection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import Sequence, Tuple, List, Optional
from matplotlib import cm
import itertools


# ──────────────────────────────────────────────────────────────────────
# utility helpers
# ──────────────────────────────────────────────────────────────────────
def _rectangle_vertices(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Return the 2^n corner points of an axis‑aligned rectangle."""
    lo = np.asarray(lo)
    hi = np.asarray(hi)
    n  = len(lo)
    corners = []
    for mask in range(1 << n):
        c = np.where([(mask >> i) & 1 for i in range(n)], hi, lo)
        corners.append(c)
    return np.vstack(corners)


def _polytope_vertices(A: np.ndarray,
                       b: np.ndarray,
                       interior_pt: Optional[np.ndarray] = None
                       ) -> np.ndarray:
    """Vertices of a bounded polytope {x | A x ≤ b} (up to ≈ 6‑D)."""
    A = np.real(np.asarray(A, complex))
    b = np.real(np.asarray(b, complex))
    m, n = A.shape
    
    if interior_pt is None:
        from scipy.optimize import linprog
        eps = 1e-6
        try:
            res = linprog(np.zeros(n), A_ub=A, b_ub=b - eps, method='highs')
            if not res.success:
                if m > n:
                    points = []
                    for combo in itertools.combinations(range(m), n):
                        try:
                            A_sub = A[combo, :]
                            b_sub = b[combo]
                            if np.linalg.matrix_rank(A_sub) == n:
                                pt = np.linalg.solve(A_sub, b_sub)
                                if np.all(A @ pt <= b + 1e-10):
                                    points.append(pt)
                        except:
                            continue
                    if points:
                        interior_pt = np.mean(points, axis=0)
                    else:
                        raise RuntimeError("Could not find interior point")
                else:
                    raise RuntimeError("polytope seems empty or unbounded")
            else:
                interior_pt = res.x
        except Exception as e:
            interior_pt = np.zeros(n)
            
    try:
        half = np.hstack((-A, -b[:, None]))
        hs_int = HalfspaceIntersection(half, interior_pt)
        return hs_int.intersections
    except Exception as e:
        # Return a small hypercube as fallback
        unit_cube = np.array(list(itertools.product([-1, 1], repeat=n)))
        return unit_cube

def _project(points: np.ndarray, dims: Tuple[int, ...]) -> np.ndarray:
    """Select the given coordinates from every point row."""
    return points[:, dims]


# ──────────────────────────────────────────────────────────────────────
# main plotting routine
# ──────────────────────────────────────────────────────────────────────
def plot_region(
    dims: Tuple[int, ...] = (0, 1),
    *,
    poly_eq: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    pts: Optional[np.ndarray] = None,
    rects_from_pts: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
    rects_from_poly: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
    ax=None,
    show: bool = True,
    labels: dict | None = None,
    colours: dict | None = None,
    alphas: dict | None = None,
    view_elevation: float = 30,
    view_azimuth: float = 45,
    debug: bool = False,
    interactive: bool = True
):
    """
    Plot up to four kinds of geometric objects in the chosen projection.

    Parameters
    ----------
    dims            tuple of coordinate indices (len==2 or 3)
    poly_eq         (A, b)    – polytope from inequalities
    pts             (m, n)    – point cloud; convex hull will be drawn
    rects_from_pts  list[(lo, hi)]  – rectangles grown from points
    rects_from_poly list[(lo, hi)]  – rectangles valid by A,b
    ax              existing Axes or Axes3D (created if None)
    labels / colours / alphas
                    optional dicts to override legend text, colours,
                    transparency; keys: 'poly_eq','poly_hull',
                    'rect_pts','rect_poly'.
    view_elevation   float, default 30
                    Elevation angle for 3D plot viewing
    view_azimuth     float, default 45
                    Azimuth angle for 3D plot viewing
    interactive : bool, default True
        If True and is3d is True, creates an interactive plot that allows
        rotation and viewing from different angles.
    """
    # Default values for labels and colors
    labels  = labels  or dict(poly_eq="Polytope boundaries",
                              poly_hull="Convex Hull",
                              rect_pts="Best Hyperrectangle",
                              rect_poly="Polytope rectangle")
    
    colours = colours or dict(poly_eq="#3498DB",
                              poly_hull="orange",
                              rect_pts="red",
                              rect_poly="purple")
    
    alphas  = alphas  or dict(poly_eq=1.0,
                              poly_hull=0.7,
                              rect_pts=1.0,
                              rect_poly=1.0)

    is3d = len(dims) == 3
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d' if is3d else None)

    # 1. Draw the polytope from (A,b) if provided
    if poly_eq is not None:
        try:
            A, b = poly_eq
            V = _polytope_vertices(A, b)
            P = _project(V, dims)
            
            if is3d:
                hull = ConvexHull(P)
                # Draw only edges of the polytope
                added_label = False
                for simplex in hull.simplices:
                    for i, j in itertools.combinations(simplex, 2):
                        current_label = labels['poly_eq'] if not added_label else ""
                        ax.plot3D([P[i,0], P[j,0]], 
                                 [P[i,1], P[j,1]], 
                                 [P[i,2], P[j,2]],
                                 color=colours['poly_eq'], 
                                 linewidth=1.5,
                                 label=current_label)
                        added_label = True
            else:
                hull = ConvexHull(P)
                vertices = P[hull.vertices]
                vertices = np.vstack([vertices, vertices[0]])  # Close the loop
                ax.plot(vertices[:, 0], vertices[:, 1],
                       color=colours['poly_eq'], linewidth=1.5,
                       label=labels['poly_eq'])
        except Exception as e:
            pass

    # 2. Draw the convex hull of points
    if pts is not None and pts.shape[0] >= len(dims) + 1:
        H = ConvexHull(pts[:, dims]) 
        P = pts[:, dims]
        
        if is3d:
            # Draw the convex hull edges
            added_hull_label = False
            for simplex in H.simplices:
                s_aug = np.append(simplex, simplex[0])
                edges = [P[s_aug[i:i+2]] for i in range(len(s_aug)-1)]
                for edge in edges:
                    ax.plot3D(edge[:, 0], edge[:, 1], edge[:, 2],
                             color=colours['poly_hull'], 
                             linewidth=0.5,
                             alpha=0.7,
                             label=labels['poly_hull'] if not added_hull_label else "")
                    added_hull_label = True
            
            # Draw points as scatter
            ax.scatter(P[:, 0], P[:, 1], P[:, 2], 
                      c='#3498DB', s=10, alpha=0.4, label='Feasible Points')
        else:
            # Draw the convex hull outline for 2D
            vertices = P[H.vertices]
            vertices = np.vstack([vertices, vertices[0]])  # Close the loop
            ax.plot(vertices[:, 0], vertices[:, 1],
                   color=colours['poly_hull'], linewidth=1.5,
                   label=labels['poly_hull'])
            
            # Add point scatter for 2D
            ax.scatter(P[:, 0], P[:, 1], c='#3498DB', s=6, alpha=0.5)

    # Helper function for rectangles
    def _draw_rect(lo, hi, colour, alpha, lab):
        if is3d:
            # Create 8 vertices of the 3D rectangle
            v = np.array([
                [lo[0], lo[1], lo[2]],
                [hi[0], lo[1], lo[2]],
                [hi[0], hi[1], lo[2]],
                [lo[0], hi[1], lo[2]],
                [lo[0], lo[1], hi[2]],
                [hi[0], lo[1], hi[2]],
                [hi[0], hi[1], hi[2]],
                [lo[0], hi[1], hi[2]]
            ])
            
            # Define the 12 edges of a cuboid
            edges = [
                [v[0], v[1]], [v[1], v[2]], [v[2], v[3]], [v[3], v[0]],  # Bottom face
                [v[4], v[5]], [v[5], v[6]], [v[6], v[7]], [v[7], v[4]],  # Top face
                [v[0], v[4]], [v[1], v[5]], [v[2], v[6]], [v[3], v[7]]   # Vertical edges
            ]
            
            # Plot each edge with dashed lines
            added_rect_label = False
            for edge in edges:
                ax.plot3D(*zip(*edge), 
                         color=colour,
                         linestyle='--',
                         linewidth=2.0,
                         label=lab if not added_rect_label else "")
                added_rect_label = True
        else:
            # For 2D, draw rectangle edges
            P = _project(V, dims)
            hull = ConvexHull(P)
            vertices = P[hull.vertices]
            vertices = np.vstack([vertices, vertices[0]])  # Close the loop
            ax.plot(vertices[:, 0], vertices[:, 1], 
                   color=colour, linewidth=2.0, label=lab, linestyle='--')

    # 3. Draw rectangles from points
    if rects_from_pts:
        for k, (lo, hi) in enumerate(rects_from_pts):
            _draw_rect(lo, hi, colours['rect_pts'], alphas['rect_pts'], labels['rect_pts'])

    # 4. Draw rectangles from polytope equations
    if rects_from_poly:
        for k, (lo, hi) in enumerate(rects_from_poly):
            _draw_rect(lo, hi, colours['rect_poly'], alphas['rect_poly'], labels['rect_poly'])

    # Set up axis labels and appearance
    ax.set_xlabel(f"x[{dims[0]}]", fontsize=12)
    ax.set_ylabel(f"x[{dims[1]}]", fontsize=12)
    
    if is3d:
        ax.set_zlabel(f"x[{dims[2]}]", fontsize=12)
        ax.view_init(elev=view_elevation, azim=view_azimuth)
        
        # Clean up 3D appearance
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.grid(True)
        ax.set_box_aspect([1, 1, 1])
        ax.set_title("3D visualization (click and drag to rotate view)", fontsize=14)
    
    ax.legend(loc='best')
    
    if show:
        if is3d and interactive:
            plt.ion()
            plt.tight_layout()
            plt.draw()
            plt.pause(0.001)
            input("Press Enter to continue...\n")
        else:
            plt.tight_layout()
            plt.show()
    
    return ax


def plot_time_series(data_series, thresholds, title_prefix):
    """Plot a time series with upper and lower threshold lines."""
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