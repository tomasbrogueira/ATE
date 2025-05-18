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
# 3) Filter historical points that are feasible (no branch violation)
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

# -----------------------------------------------------------------------------
# 4) Generate random axis-aligned hyperrectangles that respect A x <= b
# -----------------------------------------------------------------------------
def compute_true_bounds(A, b, empirical_min=None, empirical_max=None):
    """
    Compute axis-aligned bounds of {x | A x <= b} via LP per dimension.
    Falls back to empirical_min/empirical_max if unbounded.

    Args:
        A (ndarray): constraint matrix
        b (ndarray): RHS vector
        empirical_min (ndarray): fallback mins
        empirical_max (ndarray): fallback maxs

    Returns:
        min_vals (ndarray), max_vals (ndarray)
    """
    n = A.shape[1]
    min_vals = np.zeros(n)
    max_vals = np.zeros(n)
    bounds = [(None, None)] * n
    for j in range(n):
        c = np.zeros(n)
        c[j] = 1
        res_min = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
        res_max = linprog(-c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
        if res_min.success and res_max.success:
            min_vals[j] = res_min.fun
            max_vals[j] = -res_max.fun
        else:
            if empirical_min is not None and empirical_max is not None:
                min_vals[j] = empirical_min[j]
                max_vals[j] = empirical_max[j]
            else:
                raise ValueError(f"Unbounded dimension {j} and no empirical fallback provided")
    return min_vals, max_vals
def rect_inside_polytope(A, b, lower, upper, tol=1e-9):
    """
    Check whether the rectangle [lower, upper] lies fully inside {x: A x <= b}.
    Equivalent to checking max_{x in rect} A[k,:] x <= b[k] for each k.
    That max is sum_j A[k,j] * (upper[j] if A[k,j]>=0 else lower[j]).
    """
    # Compute worst-case A x over the rectangle:
    worst = A.clip(min=0).dot(upper) + A.clip(max=0).dot(lower)
    return np.all(worst <= b + tol)

def generate_random_rectangles(A, b, min_vals, max_vals, n_rectangles=1000, rng=None):
    """
    Randomly generate n_rectangles axis-aligned hyperrectangles inside the polytope {A x <= b}.
    We sample lower and upper within [min_vals, max_vals] and accept only those that lie inside.
    Returns a list of (lower, upper) pairs.
    """
    if rng is None:
        rng = np.random.default_rng()
    n = len(min_vals)
    total_range = max_vals - min_vals

    rectangles = []
    attempts = 0
    cap = n_rectangles * 20
    while len(rectangles) < n_rectangles and attempts < cap:
        # Random lower corner in unconditional box:
        frac = rng.uniform(0, 1, size=n)
        lower = min_vals + frac * total_range
        # Random upper corner coords:
        upper = np.zeros(n)
        for i in range(n):
            upper[i] = rng.uniform(lower[i], max_vals[i])
        # Check if rectangle is inside A x <= b:
        if rect_inside_polytope(A, b, lower, upper):
            rectangles.append((lower, upper))
        attempts += 1
    return rectangles

def count_points_in_rectangle(X_points, lower, upper):
    """
    Count how many rows x in X_points (shape k×n) satisfy 
    lower[i] <= x[i] <= upper[i] for all i.
    """
    mask = np.ones(X_points.shape[0], dtype=bool)
    for i in range(X_points.shape[1]):
        mask &= (X_points[:, i] >= lower[i]) & (X_points[:, i] <= upper[i])
    return int(np.sum(mask))

def find_best_rectangle(X_points, rectangles):
    """
    Among a list of (lower, upper) rectangles, return the one 
    that contains the most historical feasible points.
    """
    best_count = -1
    best_idx = -1
    best_rect = None
    for idx, (lower, upper) in enumerate(rectangles):
        cnt = count_points_in_rectangle(X_points, lower, upper)
        if cnt > best_count:
            best_count = cnt
            best_idx = idx
            best_rect = (lower, upper)
    return best_idx, best_rect, best_count


