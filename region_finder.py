import numpy as np
from scipy.optimize import linprog
import itertools

def simulate_full_grid(m=500, seed=98):
    j = 1j
    np.random.seed(seed)
    
    S = -np.array([0.224, 0.708, 1.572, 0.072, 0.0]) * np.exp(j * 0.3176)
    I = np.conj(S).reshape(-1, 1)
    
    y12 = 1 - j * 10
    y13 = 2 * y12
    y23 = 3 - j * 20
    y34 = y23
    y45 = 2 * y12
    
    Y = np.array([
        [y12+y13,   -y12,      -y13,      0,        0],
        [  -y12,  y12+y23,     -y23,      0,        0],
        [  -y13,     -y23, y13+y23+y34,  -y34,      0],
        [     0,         0,      -y34, y34+y45,   -y45],
        [     0,         0,         0,    -y45,    y45]
    ], dtype=complex)
    
    branch_list = [
        (0, 1, y12),
        (0, 2, y13),
        (1, 2, y23),
        (2, 3, y34),
        (3, 4, y45)
    ]
    
    e4 = np.random.randn(5, m) * 0.25
    e1 = np.random.randn(m) * 0.5
    i1w = [I[0, 0].real]
    
    Ii = np.zeros((4, m))
    i12 = np.zeros(m)
    i13 = np.zeros(m)
    i23 = np.zeros(m)
    i34 = np.zeros(m)
    i45 = np.zeros(m)
    
    Y_reduced = np.delete(np.delete(Y, 4, axis=0), 4, axis=1)
    I_reduced = I[:4]
    
    v_reduced = 1 + np.linalg.pinv(Y_reduced) @ I_reduced[:, 0]
    v = np.append(v_reduced, 0.0)
    
    for idx, (n1, n2, yij) in enumerate(branch_list):
        val = yij * (v[n1] - v[n2])
        if idx == 0:
            i12[0] = np.abs(val) * np.sign(val.real)
        elif idx == 1:
            i13[0] = np.abs(val) * np.sign(val.real)
        elif idx == 2:
            i23[0] = np.abs(val) * np.sign(val.real)
        elif idx == 3:
            i34[0] = np.abs(val) * np.sign(val.real)
        elif idx == 4:
            i45[0] = np.abs(val) * np.sign(val.real)
    
    Ii[:, 0] = np.abs(I_reduced[:, 0]) * np.sign(I_reduced[:, 0].real)
    
    for t in range(m-1):
        next_I = 0.65 * I[:, t:t+1] + e4[:, t:t+1]
        i1w.append(0.75 * i1w[-1] + e1[t])
        next_I[0, 0] = -i1w[-1] + j * next_I[0, 0].imag
        I = np.hstack((I, next_I))
        
        I_reduced_t = I[:4, t+1]
        v_reduced = 1 + np.linalg.pinv(Y_reduced) @ I_reduced_t
        v = np.append(v_reduced, 0.0)
        
        for idx, (n1, n2, yij) in enumerate(branch_list):
            val = yij * (v[n1] - v[n2])
            if idx == 0:
                i12[t+1] = np.abs(val) * np.sign(val.real)
            elif idx == 1:
                i13[t+1] = np.abs(val) * np.sign(val.real)
            elif idx == 2:
                i23[t+1] = np.abs(val) * np.sign(val.real)
            elif idx == 3:
                i34[t+1] = np.abs(val) * np.sign(val.real)
            elif idx == 4:
                i45[t+1] = np.abs(val) * np.sign(val.real)
        
        Ii[:, t+1] = np.abs(I_reduced_t) * np.sign(I_reduced_t.real)

    branch_currents = {'i12': i12, 'i13': i13, 'i23': i23, 'i34': i34, 'i45': i45}
    return Ii, branch_currents, Y_reduced, branch_list

def build_A_b_from_Y(Y, branch_list, rates):
    """
    Build constraint matrix A and bounds b from branch admittances and rate limits.
    """
    if len(rates) != len(branch_list):
        raise ValueError(f"Expected {len(branch_list)} rates, got {len(rates)}")
    
    n = Y.shape[0]  # Number of nodes
    rows, bs = [], []
    
    for k, (i, j, yij) in enumerate(branch_list):
        # For a branch current from node i to node j
        a = np.zeros(n)
        
        if j == 4:
            a[i] = np.real(yij)
        else:
            a[i] = np.real(yij)
            a[j] = -np.real(yij)
        
        # Add constraints for both directions of current flow
        rows.append(a)
        rows.append(-a)
        bs.append(rates[k])
        bs.append(rates[k])
    
    A = np.vstack(rows)
    b = np.array(bs)
    
    return A, b

def filter_feasible_points(Ii, A, b, tol=1e-6):
    X_all = Ii.T
    feas_mask = np.all(A.dot(X_all.T) <= b[:, None] + tol, axis=0)
    X_hist = X_all[feas_mask]
    return X_hist, X_hist.shape[0]

def calculate_axis_aligned_bounds(A, b):
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

def calculate_focused_bounds(A, b, dims_of_interest):
    n = A.shape[1]
    min_vals = np.zeros(len(dims_of_interest))
    max_vals = np.zeros(len(dims_of_interest))
    
    for i, dim_idx in enumerate(dims_of_interest):
        # Create objective vector that only cares about this dimension
        c = np.zeros(n)
        c[dim_idx] = 1
        
        # Find minimum bound
        res_min = linprog(c, A_ub=A, b_ub=b, method='highs')
        if res_min.success:
            min_vals[i] = res_min.fun
        else:
            min_vals[i] = -np.inf
            
        # Find maximum bound
        res_max = linprog(-c, A_ub=A, b_ub=b, method='highs')
        if res_max.success:
            max_vals[i] = -res_max.fun
        else:
            max_vals[i] = np.inf
            
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
    """
    Generate rectangles from historical data points
    """
    if X_hist is None or X_hist.shape[0] == 0:
        print("No feasible points provided for rectangle generation.")
        return []
    
    if X_hist.shape[0] == 1:
        # Only one feasible point - create a small box around it
        p = X_hist[0]
        margin = 0.01
        return [(p - margin, p + margin)]
    
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
        
        # Check if rectangle is valid
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

def generate_improved_rectangles(A, b, X_hist, n_rectangles=1000):
    if X_hist.shape[0] == 0:
        print("No feasible points provided for improved rectangle generation.")
        return []
    
    rectangles = []
    k, n = X_hist.shape
    A_pos = np.clip(A, 0, None)
    A_neg = np.clip(A, None, 0)
    
    # Small boxes around individual points
    for i in range(min(k, 50)):
        p = X_hist[i]
        margin = 0.01
        lower = p - margin
        upper = p + margin

        worst = A_pos.dot(upper) + A_neg.dot(lower)
        if np.all(worst <= b):
            rectangles.append((lower, upper))

    # Grow boxes in all dimensions
    for _ in range(min(100, n_rectangles)):
        if len(rectangles) == 0:
            break
            
        # Pick a random starting rectangle
        idx = np.random.randint(0, len(rectangles))
        lo, hi = rectangles[idx]
        
        # Try growing in a random dimension
        dim = np.random.randint(0, n)
        factor = 1.2  # Growth factor
        
        # Try growing upper bound
        new_hi = hi.copy()
        new_hi[dim] = lo[dim] + (hi[dim] - lo[dim]) * factor
        
        # Check if still valid
        worst = A_pos.dot(new_hi) + A_neg.dot(lo)
        if np.all(worst <= b):
            rectangles.append((lo.copy(), new_hi))

    # Create rectangles between feasible point pairs
    attempts = 0
    while len(rectangles) < n_rectangles and attempts < n_rectangles * 10:
        attempts += 1
        if k < 2:
            break
            
        i, j = np.random.randint(0, k, size=2)
        p, q = X_hist[i], X_hist[j]
        lower = np.minimum(p, q)
        upper = np.maximum(p, q)
        
        # Check validity before adding
        worst = A_pos.dot(upper) + A_neg.dot(lower)
        if np.all(worst <= b):
            rectangles.append((lower, upper))
    
    return rectangles