import numpy as np
from scipy.optimize import linprog
import itertools

def simulate_full_grid(m=500, seed=98):
    j = 1j
    np.random.seed(seed)
    
    n = 4
    S = -np.array([0.224, 0.708, 1.572, 0.072]) * np.exp(j * 0.3176)
    I = np.conj(S).reshape(-1, 1)
    
    # Branch admittances for power grid
    y12 = 1 - j * 10
    y13 = 2 * y12
    y23 = 3 - j * 20
    y34 = y23
    y45 = 2 * y12
    
    Y = np.array([
        [y12+y13,   -y12,      -y13,      0],
        [  -y12,  y12+y23,     -y23,      0],
        [  -y13,     -y23, y13+y23+y34,  -y34],
        [     0,         0,      -y34, y34+y45]
    ], dtype=complex)
    
    branch_list = [
        (0, 1, y12),
        (0, 2, y13),
        (1, 2, y23)
    ]
    
    # AR(1) noise processes
    e4 = np.random.randn(n, m) * 0.25
    e1 = np.random.randn(m) * 0.5
    i1w = [I[0, 0].real]
    
    Ii = np.zeros((n, m))
    i12 = np.zeros(m)
    i13 = np.zeros(m)
    i23 = np.zeros(m)
    
    # Calculate initial state
    v = 1 + np.linalg.inv(Y) @ I[:, 0]
    for idx, (n1, n2, yij) in enumerate(branch_list):
        val = yij * (v[n1] - v[n2])
        [i12, i13, i23][idx][0] = np.abs(val) * np.sign(val.real)
    Ii[:, 0] = np.abs(I[:, 0]) * np.sign(I[:, 0].real)
    
    # Iterate over time steps
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

def build_A_b_from_Y(Y, branch_list, rates):
    # Convert network model to linear constraints |a^T x| <= rate
    M = np.linalg.inv(Y)
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

def filter_feasible_points(Ii, A, b):
    X_all = Ii.T
    feas_mask = np.all(A.dot(X_all.T) <= b[:, None] + 1e-9, axis=0)
    X_hist = X_all[feas_mask]
    return X_hist, X_hist.shape[0]

def calculate_axis_aligned_bounds(A, b):
    # Find min/max bounds for each dimension
    n = A.shape[1]
    min_vals = np.zeros(n)
    max_vals = np.zeros(n)
    bounds = [(None, None)] * n
    for j in range(n):
        c = np.zeros(n)
        c[j] = 1
        res_min = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
        
        if res_min.success:
            min_vals[j] = res_min.fun
        elif res_min.status == 3:
            min_vals[j] = -np.inf
        else:
            raise ValueError(f"LP failed for min bound, dim {j}: {res_min.message}")

        c[j] = -1
        res_max = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')

        if res_max.success:
            max_vals[j] = -res_max.fun
        elif res_max.status == 3:
            max_vals[j] = np.inf
        else:
            raise ValueError(f"LP failed for max bound, dim {j}: {res_max.message}")
            
    return min_vals, max_vals

def generate_rectangles_from_polytope(A, b, n_rectangles, dim, random_state=None):
    # Generate rectangles by finding extreme points along random directions
    rng = np.random.default_rng(random_state)
    rectangles = []
    for _ in range(n_rectangles):
        w = rng.normal(size=dim)
        w /= np.linalg.norm(w)
        res_min = linprog(c=w, A_ub=A, b_ub=b, bounds=[(None, None)]*dim)
        res_max = linprog(c=-w, A_ub=A, b_ub=b, bounds=[(None, None)]*dim)
        if res_min.success and res_max.success:
            lo = res_min.x
            hi = res_max.x
            rectangles.append((np.minimum(lo, hi), np.maximum(lo, hi)))
    return rectangles

def generate_random_rectangles(A, b, n_rectangles=10000, X_hist=None):
    if X_hist is None:
        raise ValueError("X_hist must be provided for rectangle generation")
    rectangles = []
    A_pos = np.clip(A, 0, None)  # Extract positive coefficients
    A_neg = np.clip(A, None, 0)  # Extract negative coefficients
    k, n = X_hist.shape
    attempts = 0
    max_attempts = n_rectangles * 50
    while len(rectangles) < n_rectangles and attempts < max_attempts:
        attempts += 1
        i, j = np.random.randint(0, k, size=2)
        p, q = X_hist[i], X_hist[j]
        lower = np.minimum(p, q)
        upper = np.maximum(p, q)
        # Check if rectangle is fully contained in polytope using worst-case analysis
        worst = A_pos.dot(upper) + A_neg.dot(lower)
        if np.all(worst <= b):
            rectangles.append((lower, upper))
    return rectangles

def filter_contained_rectangles(candidates, A, b):
    # Keep only rectangles whose every vertex satisfies constraints
    filtered = []
    for lo, hi in candidates:
        valid = True
        for vertex in itertools.product(*zip(lo, hi)):
            if np.any(A.dot(vertex) > b + 1e-8):
                valid = False
                break
        if valid:
            filtered.append((lo, hi))
    return filtered

def find_best_rectangle(X_points, rectangles):
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