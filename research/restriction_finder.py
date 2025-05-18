import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, HuberRegressor, LogisticRegression
from sklearn.svm import SVC
from scipy.optimize import linprog
import pulp

# -----------------------------------------------------------------------------
# 1) Métodos para análise de violações (X, y obtidos do processo de simulação)
# -----------------------------------------------------------------------------

def apply_ols(X, y):
    """
    Regressão Linear por Mínimos Quadrados Ordinários
    Retorna o modelo e coeficientes.
    """
    model = LinearRegression(fit_intercept=False)
    model.fit(X, y.ravel())
    return model, model.coef_


def apply_ridge(X, y, alpha=1.0):
    """
    Regressão Ridge (regularização L2)
    `alpha` controla a força da regularização.
    """
    model = Ridge(alpha=alpha, fit_intercept=False)
    model.fit(X, y.ravel())
    return model, model.coef_


def apply_lasso(X, y, alpha=0.1):
    """
    Regressão Lasso (regularização L1)
    `alpha` controla a força da regularização.
    """
    model = Lasso(alpha=alpha, fit_intercept=False, max_iter=10000)
    model.fit(X, y.ravel())
    return model, model.coef_


def apply_huber(X, y, epsilon=1.35):
    """
    Regressão robusta de Huber (insensível a outliers)
    `epsilon` define a sensibilidade.
    """
    model = HuberRegressor(epsilon=epsilon, fit_intercept=False)
    model.fit(X, y.ravel())
    return model, model.coef_


def apply_svm(X, y, kernel='linear', C=1.0):
    """
    Classificador SVM binário: classifica pontos seguros vs violadores
    Converte y em classes 0 (sem violação) e 1 (violações).
    Retorna o modelo treinado.
    """
    y_cls = (y.ravel() > 0).astype(int)
    model = SVC(kernel=kernel, C=C)
    model.fit(X, y_cls)
    return model

# -----------------------------------------------------------------------------
# 2) Construção do maior hipercubo alinhado aos eixos dentro de um politopo
#    definido por Ax ≤ b
# -----------------------------------------------------------------------------

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

# --------------------------- Example ---------------------------
if __name__ == "__main__":
    # Suppose regression gave you A x <= b:
    A = np.array([[ 1, 0],
                  [-1, 0],
                  [ 0, 1],
                  [ 0,-1]])
    b = np.array([2, 2, 1.5, 1.5])

    # Historical points
    X = np.random.rand(30, 2) * 3

    L, U, z, cnt = find_box_maximum_coverage_with_polytope(X, A, b)
    print(f"Covered {cnt}/{X.shape[0]} points")
    print("L:", L)
    print("U:", U)