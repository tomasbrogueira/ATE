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
    min_vals = np.zeros(n)
    max_vals = np.zeros(n)
    bounds = [(None, None)] * n
    for j in range(n):
        # Minimize x_j
        c = np.zeros(n)
        c[j] = 1
        res_min = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
        
        if res_min.success:
            min_vals[j] = res_min.fun
        elif res_min.status == 3:  # Problem is unbounded
            min_vals[j] = -np.inf
        else:
            # Other failure (infeasible, iteration limit, etc.)
            raise ValueError(f"LP failed for min bound, dim {j}: {res_min.message}")

        # Maximize x_j (by minimizing -x_j)
        c[j] = -1 # c is already array of zeros, just set c[j]
        res_max = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')

        if res_max.success:
            max_vals[j] = -res_max.fun # Negate because we minimized -x_j
        elif res_max.status == 3:  # Problem is unbounded
            max_vals[j] = np.inf   # Maximization leads to +inf if unbounded
        else:
            # Other failure
            raise ValueError(f"LP failed for max bound, dim {j}: {res_max.message}")
            
    return min_vals, max_vals

# -----------------------------------------------------------------------------
# 4) Find best rectangle from random sampling
# -----------------------------------------------------------------------------

def generate_rectangles_from_polytope(A, b, n_rectangles, dim, random_state=None):
    """
    Create n_rectangles axis-aligned boxes by picking random directions w in R^dim,
    then finding the min/max points of the polytope along w.
    Returns a list of (lo, hi) pairs.
    """
    rng = np.random.default_rng(random_state)
    rectangles = []
    for _ in range(n_rectangles):
        # pick a random direction on the unit sphere
        w = rng.normal(size=dim)
        w /= np.linalg.norm(w)
        # minimize w^T x
        res_min = linprog(c=w, A_ub=A, b_ub=b, bounds=[(None, None)]*dim)
        # maximize w^T x  ⇔ minimize −w^T x
        res_max = linprog(c=-w, A_ub=A, b_ub=b, bounds=[(None, None)]*dim)
        if res_min.success and res_max.success:
            lo = res_min.x
            hi = res_max.x
            rectangles.append((np.minimum(lo, hi), np.maximum(lo, hi)))
    return rectangles

def generate_random_rectangles(A, b, n_rectangles=10000, X_hist=None):
    """
    Generate candidate axis-aligned rectangles by sampling pairs of feasible points.
    Each rectangle is defined by two points p,q in X_hist:
      lower = min(p,q), upper = max(p,q).
    Only keep rectangles fully inside the polytope.
    Args:
      • A, b: polytope constraints
      • n_rectangles: desired number of candidates
      • X_hist: ndarray (k, n) of feasible points
    Returns:
      • rectangles: list of (lower, upper)
    """
    if X_hist is None:
        raise ValueError("X_hist must be provided for rectangle generation")
    rectangles = []
    A_pos = np.clip(A, 0, None)
    A_neg = np.clip(A, None, 0)
    k, n = X_hist.shape
    attempts = 0
    max_attempts = n_rectangles * 50
    while len(rectangles) < n_rectangles and attempts < max_attempts:
        attempts += 1
        i, j = np.random.randint(0, k, size=2)
        p, q = X_hist[i], X_hist[j]
        lower = np.minimum(p, q)
        upper = np.maximum(p, q)
        # worst-case check: a^T x <= b for all x in rectangle
        worst = A_pos.dot(upper) + A_neg.dot(lower)
        if np.all(worst <= b):
            rectangles.append((lower, upper))
    return rectangles


def find_best_rectangle(X_points, rectangles):
    """
    From a list of (lower, upper) rectangles, count how many X_points lie inside,
    and return the index, rectangle, and count of the best one.
    Args:
      • X_points: ndarray (k, n)
      • rectangles: list of (lower, upper)
    Returns:
      • best_idx: int
      • best_rect: tuple (lower, upper)
      • best_count: int
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