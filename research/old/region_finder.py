import numpy as np
from scipy.optimize import linprog

# --------------------------------------------------------------------------
# 1) Simulate the grid and collect signed nodal injection history (Ii)
# --------------------------------------------------------------------------
def simulate_full_grid(m=500, seed=98):
    """
    Simulate the kite network and return:
      - Ii: (n_nodes × m) array of signed nodal injections over time
      - branch_currents: dict of branch currents (for debugging, not used here)
      - Y: (n_nodes × n_nodes) complex admittance matrix (slack removed)
      - branch_list: list of (i, j, yij) for later building A,b
    """
    j = 1j
    np.random.seed(seed)
    
    # Number of injection nodes (slack removed)
    n = 4
    
    # Initial complex injections: S = -(P + jQ)
    S = -np.array([0.224, 0.708, 1.572, 0.072]) * np.exp(j * 0.3176)
    I = np.conj(S).reshape(-1, 1)  # shape: (n, 1)
    
    # Branch admittances (complex)
    y12 = 1 - j * 10
    y13 = 2 * y12
    y23 = 3 - j * 20
    y34 = y23
    y45 = 2 * y12
    
    # Build the n×n admittance matrix Y (slack removed)
    Y = np.array([
        [y12+y13,   -y12,      -y13,      0],
        [  -y12,  y12+y23,     -y23,      0],
        [  -y13,     -y23, y13+y23+y34,  -y34],
        [     0,         0,      -y34, y34+y45]
    ], dtype=complex)
    
    # List of branches (for building A later):
    #   Each tuple = (node_i, node_j, branch_admittance)
    branch_list = [
        (0, 1, y12),  # branch between node 0 and node 1
        (0, 2, y13),  # branch between node 0 and node 2
        (1, 2, y23)   # branch between node 1 and node 2
    ]
    
    # AR(1) noise processes for injections:
    e4 = np.random.randn(n, m) * 0.25  # complex-part noise (but real here)
    e1 = np.random.randn(m) * 0.5      # real-part noise for node 1
    i1w = [I[0, 0].real]               # track real part of injection at node 1
    
    # Preallocate arrays
    Ii = np.zeros((n, m))   # signed nodal injection magnitudes (real*sign)
    i12 = np.zeros(m)
    i13 = np.zeros(m)
    i23 = np.zeros(m)
    
    # t = 0: compute nodal voltages and branch currents
    v = 1 + np.linalg.inv(Y) @ I[:, 0]
    for idx, (n1, n2, yij) in enumerate(branch_list):
        val = yij * (v[n1] - v[n2])
        [i12, i13, i23][idx][0] = np.abs(val) * np.sign(val.real)
    Ii[:, 0] = np.abs(I[:, 0]) * np.sign(I[:, 0].real)
    
    # t = 1..m-1: iterate AR(1) on injections, recalc voltages/currents
    for t in range(m-1):
        next_I = 0.65 * I[:, t:t+1] + e4[:, t:t+1]
        i1w.append(0.75 * i1w[-1] + e1[t])
        next_I[0, 0] = -i1w[-1] + j * next_I[0, 0].imag
        I = np.hstack((I, next_I))
        
        v = 1 + np.linalg.inv(Y) @ I[:, t+1]
        for idx, (n1, n2, yij) in enumerate(branch_list):
            val = yij * (v[n1] - v[n2])
            [i12, i13, i23][idx][t+1] = np.abs(val) * np.sign(val.real)
        
        Ii[:, t+1] = np.abs(I[:, t+1]) * np.sign(I[:, t+1].real)
    
    branch_currents = {'i12': i12, 'i13': i13, 'i23': i23}
    return Ii, branch_currents, Y, branch_list

# -----------------------------------------------------------------------------
# 2) Build A, b so that A x <= b enforces |a_k^T x| <= r_k for all branches
# -----------------------------------------------------------------------------
def build_A_b_from_Y(Y, branch_list, rates):
    """
    Y: (n x n) complex admittance matrix (slack removed)
    branch_list: list of (i,j,yij)
    rates: list of branch ratings [r1, r2, ...]
    Returns A (2m x n) and b (2m)
    """
    M = np.linalg.inv(Y)  # n×n complex
    rows = []
    bs = []
    for k, (i, j, yij) in enumerate(branch_list):
        a = np.real(yij * (M[i, :] - M[j, :]))
        rows.append(a)
        rows.append(-a)
        bs.append(rates[k])
        bs.append(rates[k])
    A = np.vstack(rows)
    b = np.array(bs, dtype=float)
    return A, b

# --------------------------------------------------------------------------
# 3) Filter historical points that are feasible and define allowed reasgion
# --------------------------------------------------------------------------
def filter_feasible_points(Ii, A, b):
    """
    Given Ii (n x m) historical injection points, keep only those x columns
    for which A @ x <= b holds. Returns a (k x n) array of feasible points.
    """
    X_all = Ii.T  # shape: (m, n)
    feas_mask = np.all(A.dot(X_all.T) <= b[:, None] + 1e-9, axis=0)
    X_hist = X_all[feas_mask]
    return X_hist, X_hist.shape[0]

def compute_true_bounds(A, b):
    """
    Compute axis-aligned bounds of {x | A x <= b} by LP per dimension.
    Returns:
      • min_vals, max_vals: ndarrays length n
    """
    n = A.shape[1]
    min_vals = np.full(n, -np.inf)
    max_vals = np.full(n, np.inf)
    bounds = [(None, None)] * n
    for j in range(n):
        # Only try to minimize/maximize if there is at least one constraint on x_j
        if not np.all(A[:, j] == 0):
            # Minimize x_j
            c = np.zeros(n)
            c[j] = 1
            res_min = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
            if res_min.success:
                min_vals[j] = res_min.x[j]
            elif res_min.status == 3:  # Problem is unbounded
                min_vals[j] = -np.inf
            else:
                raise ValueError(f"LP failed for min bound, dim {j}: {res_min.message}")

            # Maximize x_j (by minimizing -x_j)
            c[j] = -1
            res_max = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
            if res_max.success:
                max_vals[j] = res_max.x[j]
            elif res_max.status == 3:
                max_vals[j] = np.inf
            else:
                raise ValueError(f"LP failed for max bound, dim {j}: {res_max.message}")
    return min_vals, max_vals

# -----------------------------------------------------------------------------
# 4) Find best rectangle from random sampling
# -----------------------------------------------------------------------------

def rectangle_inside_polytope(A, b, lower, upper):
    """
    Returns True if all corners of the rectangle [lower, upper] are inside the polytope A x <= b.
    """
    from itertools import product
    n = len(lower)
    # Ensure lower <= upper for all dimensions
    lower = np.minimum(lower, upper)
    upper = np.maximum(lower, upper)
    # Generate all 2^n corners robustly
    corners = np.array(list(product(*zip(lower, upper))))
    # Ensure corners are shape (n,) for each
    for corner in corners:
        # Use float64 for numerical stability
        corner = np.array(corner, dtype=np.float64)
        if not np.all(A @ corner <= b + 1e-10):
            return False
    return True

def expand_rectangle_along_dim(A, b, lower, upper, dim, direction, step=0.01, max_iter=100):
    """
    Expand the rectangle along a single dimension (dim) in the given direction (+1 for upper, -1 for lower)
    as much as possible while keeping all corners inside the polytope.
    """
    n = len(lower)
    l, u = lower.copy(), upper.copy()
    # Start from current bound
    if direction == -1:
        bound = l[dim]
        limit = -np.inf
        # Try to expand towards true minimum
        for _ in range(max_iter):
            test = bound - step
            l[dim] = test
            if rectangle_inside_polytope(A, b, l, u):
                bound = test
            else:
                break
        l[dim] = bound
        return l[dim]
    else:
        bound = u[dim]
        limit = np.inf
        # Try to expand towards true maximum
        for _ in range(max_iter):
            test = bound + step
            u[dim] = test
            if rectangle_inside_polytope(A, b, l, u):
                bound = test
            else:
                break
        u[dim] = bound
        return u[dim]

def generate_random_rectangles(A, b, X_hist, n_rectangles=1000):
    """
    Generate rectangles by picking random pairs of feasible points and also by sampling random points inside the polytope.
    For each rectangle, expand each bound independently toward the polytope box as much as possible,
    keeping all corners inside the polytope.
    After expansion, always shrink slightly and check again.
    """
    if X_hist is None:
        raise ValueError("X_hist must be provided for rectangle generation")
    k, n = X_hist.shape
    rectangles = []
    volumes = []
    true_min, true_max = compute_true_bounds(A, b)
    rng = np.random.default_rng()

    n_hist_pairs = int(n_rectangles * 0.5)
    n_polytope_pairs = n_rectangles - n_hist_pairs

    # 1. Random pairs from X_hist
    for _ in range(n_hist_pairs):
        i, j = rng.integers(0, k, size=2)
        p, q = X_hist[i], X_hist[j]
        lower = np.minimum(p, q)
        upper = np.maximum(p, q)
        for d in range(n):
            lo, hi = true_min[d], lower[d]
            for _ in range(15):
                mid = (lo + hi) / 2
                test_lower = lower.copy()
                test_lower[d] = mid
                if rectangle_inside_polytope(A, b, test_lower, upper):
                    hi = mid
                else:
                    lo = mid
            lower[d] = hi
            lo, hi = upper[d], true_max[d]
            for _ in range(15):
                mid = (lo + hi) / 2
                test_upper = upper.copy()
                test_upper[d] = mid
                if rectangle_inside_polytope(A, b, lower, test_upper):
                    lo = mid
                else:
                    hi = mid
            upper[d] = lo
        # Always shrink slightly and check again
        center = (lower + upper) / 2
        shrink = 1 - 1e-6  # Make shrink factor slightly larger for more safety
        lower_shrunk = center - (center - lower) * shrink
        upper_shrunk = center + (upper - center) * shrink
        if rectangle_inside_polytope(A, b, lower_shrunk, upper_shrunk):
            volume = np.prod(np.maximum(upper_shrunk - lower_shrunk, 0))
            rectangles.append((lower_shrunk.copy(), upper_shrunk.copy()))
            volumes.append(volume)

    # 2. Random points inside the polytope (rejection sampling)
    finite_min = np.where(np.isfinite(true_min), true_min, np.min(X_hist, axis=0))
    finite_max = np.where(np.isfinite(true_max), true_max, np.max(X_hist, axis=0))
    polytope_points = []
    max_attempts = n_polytope_pairs * 10
    attempts = 0
    while len(polytope_points) < 2 * n_polytope_pairs and attempts < max_attempts:
        rand_pt = rng.uniform(finite_min, finite_max)
        if np.all(A @ rand_pt <= b + 1e-9):
            polytope_points.append(rand_pt)
        attempts += 1
    polytope_points = np.array(polytope_points)
    n_poly_pts = polytope_points.shape[0]
    for _ in range(n_polytope_pairs):
        if n_poly_pts < 2:
            break
        i, j = rng.integers(0, n_poly_pts, size=2)
        p, q = polytope_points[i], polytope_points[j]
        lower = np.minimum(p, q)
        upper = np.maximum(p, q)
        for d in range(n):
            lo, hi = true_min[d], lower[d]
            for _ in range(15):
                mid = (lo + hi) / 2
                test_lower = lower.copy()
                test_lower[d] = mid
                if rectangle_inside_polytope(A, b, test_lower, upper):
                    hi = mid
                else:
                    lo = mid
            lower[d] = hi
            lo, hi = upper[d], true_max[d]
            for _ in range(15):
                mid = (lo + hi) / 2
                test_upper = upper.copy()
                test_upper[d] = mid
                if rectangle_inside_polytope(A, b, lower, test_upper):
                    lo = mid
                else:
                    hi = mid
            upper[d] = lo
        center = (lower + upper) / 2
        shrink = 1 - 1e-6
        lower_shrunk = center - (center - lower) * shrink
        upper_shrunk = center + (upper - center) * shrink
        if rectangle_inside_polytope(A, b, lower_shrunk, upper_shrunk):
            volume = np.prod(np.maximum(upper_shrunk - lower_shrunk, 0))
            rectangles.append((lower_shrunk.copy(), upper_shrunk.copy()))
            volumes.append(volume)

    # Sort rectangles by volume (descending) and keep the top n_rectangles
    if volumes:
        idxs = np.argsort(volumes)[::-1]
        rectangles = [rectangles[i] for i in idxs[:n_rectangles]]
    return rectangles

def find_best_rectangle(X_points, rectangles):
    """
    From a list of (lower, upper) rectangles, count how many X_points lie inside,
    and return the index, rectangle, and count of the best one.
    """
    best_count = -1
    best_idx = None
    best_rect = None
    for idx, (lower, upper) in enumerate(rectangles):
        inside = np.all((X_points >= lower) & (X_points <= upper), axis=1)
        count = inside.sum()
        if count > best_count:
            best_count = count
            best_idx = idx
            best_rect = (lower, upper)
    return best_idx, best_rect, best_count


# Local maxima search for the largest axis-aligned rectangle

def find_max_volume_rectangle(A, b, X_hist, center=None, max_iter=10, tol=1e-6):
    """
    Find the largest axis-aligned rectangle (in volume) inside the polytope A x <= b,
    centered at 'center' (default: median of X_hist), by iteratively expanding each bound using LPs.
    Returns (lower, upper) bounds.
    """
    n = X_hist.shape[1]
    if center is None:
        center = np.median(X_hist, axis=0)
    lower = center.copy()
    upper = center.copy()
    # Start with a small epsilon box around the center
    eps = 1e-4
    lower = center - eps
    upper = center + eps

    for _ in range(max_iter):
        prev_lower = lower.copy()
        prev_upper = upper.copy()
        for d in range(n):
            # Minimize x_d with other coordinates bounded by current rectangle
            c = np.zeros(n)
            c[d] = 1
            bounds = [(lower[j], upper[j]) if j != d else (None, upper[d]) for j in range(n)]
            res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
            if res.success:
                lower[d] = res.fun
            # Maximize x_d with other coordinates bounded by current rectangle
            c[d] = -1
            bounds = [(lower[j], upper[j]) if j != d else (lower[d], None) for j in range(n)]
            res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
            if res.success:
                upper[d] = -res.fun
        # Check for convergence
        if np.allclose(lower, prev_lower, atol=tol) and np.allclose(upper, prev_upper, atol=tol):
            break
    return lower, upper

def get_data_polytope_bbox(X_hist, true_min, true_max, A, b):
    """
    Returns the largest axis-aligned rectangle (lower, upper) that is inside the polytope
    and contains as much of the data as possible.
    For box-constrained dimensions (e.g., z in a prism), always use the polytope bounds.
    For other dimensions, clips the data bounding box to the polytope bounds, then shrinks if needed to fit inside the polytope.
    """
    data_min = np.min(X_hist, axis=0)
    data_max = np.max(X_hist, axis=0)
    lower = np.where(np.isfinite(true_min), np.maximum(data_min, true_min), data_min)
    upper = np.where(np.isfinite(true_max), np.minimum(data_max, true_max), data_max)

    # For each dimension, if the polytope is a simple box (i.e., constraints are just x >= min, x <= max), use those bounds
    n = X_hist.shape[1]
    for d in range(n):
        # Check if there are only two constraints for this dimension: x_d <= max and -x_d <= -min
        col = A[:, d]
        is_box = (
            np.sum((col == 1) & (np.isclose(b, true_max[d], atol=1e-8))) == 1 and
            np.sum((col == -1) & (np.isclose(b, -true_min[d], atol=1e-8))) == 1
        )
        if is_box and np.isfinite(true_min[d]) and np.isfinite(true_max[d]):
            lower[d] = true_min[d]
            upper[d] = true_max[d]

    # Ensure the rectangle is inside the polytope (worst-case check)
    A_pos = np.clip(A, 0, None)
    A_neg = np.clip(A, None, 0)
    worst = A_pos.dot(upper) + A_neg.dot(lower)
    if np.all(worst <= b):
        return lower, upper

    # If not, shrink the rectangle towards its center until it fits
    center = (lower + upper) / 2
    shrink = 1.0
    for _ in range(30):
        test_lower = center - (center - lower) * shrink
        test_upper = center + (upper - center) * shrink
        # Always keep box-constrained dimensions at their bounds
        for d in range(n):
            col = A[:, d]
            is_box = (
                np.sum((col == 1) & (np.isclose(b, true_max[d], atol=1e-8))) == 1 and
                np.sum((col == -1) & (np.isclose(b, -true_min[d], atol=1e-8))) == 1
            )
            if is_box and np.isfinite(true_min[d]) and np.isfinite(true_max[d]):
                test_lower[d] = true_min[d]
                test_upper[d] = true_max[d]
        worst = A_pos.dot(test_upper) + A_neg.dot(test_lower)
        if np.all(worst <= b):
            return test_lower, test_upper
        shrink *= 0.9  # shrink towards center
    # As a last resort, return the center point as a degenerate rectangle
    return center, center

def find_best_hyperrectangle_local_maxima(A, b, X_hist, n_anchors=20):
    """
    Find several local maxima of hyperrectangle volume by expanding from diverse anchor points.
    Always includes the axis-aligned bounding box of the feasible points (clipped and shrunk to fit polytope).
    For each, count the number of feasible points inside, and return the rectangle with the most points.
    Returns: lower_best, upper_best, best_count
    """
    k, n = X_hist.shape
    anchors = [np.median(X_hist, axis=0), X_hist.min(axis=0), X_hist.max(axis=0)]
    rng = np.random.default_rng()
    for _ in range(max(0, n_anchors - 3)):
        anchors.append(X_hist[rng.integers(0, k)])

    true_min, true_max = compute_true_bounds(A, b)

    # Always include the axis-aligned bounding box of the data (clipped and shrunk to fit polytope)
    rectangles = []
    lower_box, upper_box = get_data_polytope_bbox(X_hist, true_min, true_max, A, b)
    rectangles.append((lower_box, upper_box))

    # Try local maxima from anchor points
    A_pos = np.clip(A, 0, None)
    A_neg = np.clip(A, None, 0)
    for anchor in anchors:
        lower, upper = find_max_volume_rectangle(A, b, X_hist, center=anchor)
        worst = A_pos.dot(upper) + A_neg.dot(lower)
        if not np.all(worst <= b):
            center = (lower + upper) / 2
            shrink = 1.0
            for _ in range(30):
                test_lower = center - (center - lower) * shrink
                test_upper = center + (upper - center) * shrink
                worst = A_pos.dot(test_upper) + A_neg.dot(test_lower)
                if np.all(worst <= b):
                    lower = test_lower
                    upper = test_upper
                    break
                shrink *= 0.9
            else:
                continue
        rectangles.append((lower, upper))

    # Now, select the rectangle with the most points inside
    best_count = -1
    lower_best = None
    upper_best = None
    for lower, upper in rectangles:
        inside = np.all((X_hist >= lower) & (X_hist <= upper), axis=1)
        count = inside.sum()
        if count > best_count:
            best_count = count
            lower_best = lower
            upper_best = upper
    return lower_best, upper_best, best_count
