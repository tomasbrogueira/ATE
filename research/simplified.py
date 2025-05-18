import numpy as np
from scipy.optimize import linprog
from sklearn.linear_model import LinearRegression, Ridge, Lasso, HuberRegressor
import pulp

# Simulate grid and build X, y
# X = [signed nodal injections (4), bias]
# y = max branch violation

def simulate_grid(m=500, seed=98, rates=(0.25, 1.0, 0.5)):
    j = 1j
    np.random.seed(seed)

    # initial nodal injections
    S = -np.array([0.224, 0.708, 1.572, 0.072]) * np.exp(j*0.3176)
    I = np.conj(S).reshape(-1, 1)  # shape (4,1)

    # line admittances
    y12 = 1 - j*10
    y13 = 2 * y12
    y23 = 3 - j*20
    y34 = y23
    y45 = 2 * y12

    # build 4x4 admittance matrix (node 5 slack removed)
    Y = np.array([
        [y12+y13,   -y12,      -y13,      0],
        [  -y12,  y12+y23,     -y23,      0],
        [  -y13,     -y23, y13+y23+y34,  -y34],
        [     0,         0,      -y34, y34+y45]
    ], dtype=complex)

    # AR(1) noise processes
    e4 = np.random.randn(4, m) * 0.25
    e1 = np.random.randn(m) * 0.5
    i1w = [I[0, 0].real]

    # preallocate arrays
    Ii = np.zeros((4, m))  # nodal signed magnitudes
    i12 = np.zeros(m)
    i13 = np.zeros(m)
    i23 = np.zeros(m)

    # time 0 voltages and currents
    v = 1 + np.linalg.inv(Y) @ I[:, 0]
    for idx, (n1, n2) in enumerate(((0,1), (0,2), (1,2))):
        val = ([y12, y13, y23][idx]) * (v[n1] - v[n2])
        [i12, i13, i23][idx][0] = abs(val) * np.sign(val.real)
    Ii[:, 0] = abs(I[:, 0]) * np.sign(I[:, 0].real)

    # simulate for t=1..m-1
    for t in range(m-1):
        # update injections
        I = np.hstack((I, 0.65*I[:, t:t+1] + e4[:, t:t+1]))
        # update i1
        i1w.append(0.75 * i1w[-1] + e1[t])
        I[0, t+1] = -i1w[-1] + j * I[0, t+1].imag

        # nodal voltages
        v = 1 + np.linalg.inv(Y) @ I[:, t+1]

        # branch currents
        for idx, (n1, n2) in enumerate(((0,1), (0,2), (1,2))):
            val = ([y12, y13, y23][idx]) * (v[n1] - v[n2])
            [i12, i13, i23][idx][t+1] = abs(val) * np.sign(val.real)

        # nodal signed magnitudes
        Ii[:, t+1] = abs(I[:, t+1]) * np.sign(I[:, t+1].real)

    # build dataset of violations
    X, y = [], []
    for t in range(m):
        vio = max(i12[t] - rates[0], i13[t] - rates[1], i23[t] - rates[2])
        if vio > 0:
            X.append(np.hstack((Ii[:, t], 1)))
            y.append(vio)

    return np.array(X), np.array(y)

# find largest axis-aligned hypercube inside Ax <= b

def find_box_maximum_coverage_with_polytope(X, A, b):
    """
    Inputs:
      - X: array of shape (k, n) of historical data points
      - A: array of shape (m, n) defining the polytope constraints A x <= b
      - b: array of shape (m,)

    Returns:
      - L_opt: array of length n (lower bounds)
      - U_opt: array of length n (upper bounds)
      - z_opt: array of length k of binaries indicating coverage
      - count: total covered points
    """
    X = np.asarray(X)
    A = np.asarray(A)
    b = np.asarray(b).ravel()
    k, n = X.shape
    m, _ = A.shape

    # Big-M per dimension = data span
    M = X.max(axis=0) - X.min(axis=0)
    M[M == 0] = 1.0  # avoid zeros

    # Define MILP
    prob = pulp.LpProblem("max_covered_points_in_polytope_box", pulp.LpMaximize)

    # Decision vars
    L = [pulp.LpVariable(f"L_{j}", lowBound=None, upBound=None) for j in range(n)]
    U = [pulp.LpVariable(f"U_{j}", lowBound=None, upBound=None) for j in range(n)]
    z = [pulp.LpVariable(f"z_{i}", cat=pulp.LpBinary) for i in range(k)]

    # Objective: maximize number of covered points
    prob += pulp.lpSum(z)

    # 1) Box non-degeneracy
    for j in range(n):
        prob += L[j] <= U[j]

    # 2) Big-M inclusion constraints
    for i in range(k):
        for j in range(n):
            # if z[i]=1 => L[j] <= X[i,j] <= U[j]
            prob += L[j] <= X[i, j] + M[j] * (1 - z[i])
            prob += U[j] >= X[i, j] - M[j] * (1 - z[i])

    # 3) Ensure [L, U] ⊆ { x | A x <= b }
    #    For each face i:
    #       max_{x∈[L,U]} A_i x = sum_{A_ij>=0} A_ij U_j + sum_{A_ij<0} A_ij L_j
    for i in range(m):
        row = A[i, :]
        pos = row >= 0
        neg = ~pos
        expr = pulp.lpSum(row[j] * U[j]   for j in np.where(pos)[0]) + \
               pulp.lpSum(row[j] * L[j]   for j in np.where(neg)[0])
        prob += expr <= b[i]

    # Solve
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    # Extract
    L_opt = np.array([v.value() for v in L], dtype=float)
    U_opt = np.array([v.value() for v in U], dtype=float)
    z_opt = np.array([v.value() for v in z], dtype=int)
    count = int(z_opt.sum())

    return L_opt, U_opt, z_opt, count

if __name__ == '__main__':
    # use same rates as MATLAB
    rates = (0.25, 1.0, 0.5)
    X, y = simulate_grid(rates=rates)
    print('X,y shapes:', X.shape, y.shape)

    # fit OLS
    model = LinearRegression(fit_intercept=False)
    model.fit(X, y)
    print('beta:', np.round(model.coef_, 8))

    # placeholder A,b for hypercube
    A = np.vstack([np.eye(5), -np.eye(5)])
    b = np.ones(10) * 0.5
    lb, ub, _, _ = find_max_hypercube(A, b)
    print('bounds:', lb, ub)
