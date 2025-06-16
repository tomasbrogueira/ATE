import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, HalfspaceIntersection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import Sequence, Tuple, List, Optional


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
    """Vertices of a bounded polytope {x | A x ≤ b} (up to ≈ 6‑D)."""
    A, b = np.asarray(A, float), np.asarray(b, float)
    m, n = A.shape
    if interior_pt is None:
        # find a feasible interior point by LP  min 0  s.t. A x ≤ b − ε
        from scipy.optimize import linprog
        eps = 1e-6
        res = linprog(np.zeros(n), A_ub=A, b_ub=b - eps)
        if not res.success:
            raise RuntimeError("polytope seems empty or unbounded")
        interior_pt = res.x
    half = np.hstack((-A, -b[:, None]))
    hs_int = HalfspaceIntersection(half, interior_pt)
    return hs_int.intersections


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
    """

    labels  = labels  or dict(poly_eq="poly (A,b)",
                              poly_hull="poly (hull)",
                              rect_pts="rect‑pts",
                              rect_poly="rect‑poly")
    colours = colours or dict(poly_eq="#1f77b4",
                              poly_hull="#2ca02c",
                              rect_pts="#d62728",
                              rect_poly="#9467bd")
    alphas  = alphas  or dict(poly_eq=.18,
                              poly_hull=.10,
                              rect_pts=.08,
                              rect_poly=.08)

    is3d = len(dims) == 3
    if ax is None:
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection='3d' if is3d else None)

    # 1  polytope from (A,b) ------------------------------------------------
    if poly_eq is not None:
        A, b = poly_eq
        V    = _polytope_vertices(A, b)
        P    = _project(V, dims)
        if is3d:
            hull = ConvexHull(P)
            faces = [[P[v] for v in simplex] for simplex in hull.simplices]
            ax.add_collection3d(Poly3DCollection(
                faces, facecolor=colours['poly_eq'],
                alpha=alphas['poly_eq'], label=labels['poly_eq']))
        else:
            hull = ConvexHull(P)
            ax.fill(P[hull.vertices, 0], P[hull.vertices, 1],
                    color=colours['poly_eq'], alpha=alphas['poly_eq'],
                    label=labels['poly_eq'])

    # 2  convex hull of sample points --------------------------------------
    if pts is not None and pts.shape[0] >= len(dims) + 1:
        H = ConvexHull(pts[:, dims])
        P = pts[:, dims]
        if is3d:
            faces = [[P[v] for v in simplex] for simplex in H.simplices]
            ax.add_collection3d(Poly3DCollection(
                faces, facecolor=colours['poly_hull'],
                alpha=alphas['poly_hull'], label=labels['poly_hull']))
        else:
            ax.fill(P[H.vertices, 0], P[H.vertices, 1],
                    color=colours['poly_hull'], alpha=alphas['poly_hull'],
                    label=labels['poly_hull'])

    # rectangle helper ------------------------------------------------------
    def _draw_rect(lo, hi, colour, alpha, lab):
        V = _rectangle_vertices(lo, hi)
        P = _project(V, dims)
        if is3d:
            hull = ConvexHull(P)
            faces = [[P[v] for v in simplex] for simplex in hull.simplices]
            ax.add_collection3d(Poly3DCollection(
                faces, facecolor=colour, alpha=alpha, label=lab))
        else:
            hull = ConvexHull(P)
            ax.fill(P[hull.vertices, 0], P[hull.vertices, 1],
                    color=colour, alpha=alpha, label=lab)

    # 3  rectangles from points --------------------------------------------
    if rects_from_pts:
        for k, (lo, hi) in enumerate(rects_from_pts, 1):
            _draw_rect(lo, hi,
                       colours['rect_pts'], alphas['rect_pts'],
                       f"{labels['rect_pts']} {k}")

    # 4  rectangles from polytope equations ---------------------------------
    if rects_from_poly:
        for k, (lo, hi) in enumerate(rects_from_poly, 1):
            _draw_rect(lo, hi,
                       colours['rect_poly'], alphas['rect_poly'],
                       f\"{labels['rect_poly']} {k}\")

    # axis labels & legend --------------------------------------------------
    ax.set_xlabel(f"x[{dims[0]}]")
    ax.set_ylabel(f"x[{dims[1]}]")
    if is3d:
        ax.set_zlabel(f"x[{dims[2]}]")
    ax.legend(loc='best')
    if show:
        plt.tight_layout()
        plt.show()
    return ax
