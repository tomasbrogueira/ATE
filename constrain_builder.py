"""
Reusable helpers to construct A, b for
   • branch‑current limits  |I_ij| ≤ rate_ij
   • nodal‑voltage limits   v_min ≤ v ≤ v_max
from historical data.
"""
from __future__ import annotations
import numpy as np
from numpy.linalg import inv, norm
from typing import Literal, Tuple
from scipy.optimize import linprog
from sklearn.linear_model import LinearRegression


# ---------------------------------------------------------------------
# 1.  Low‑level branch‑fitting utilities
# ---------------------------------------------------------------------

def _linf_branch_fit(
    X: np.ndarray,
    I_col: np.ndarray,
    safety_factor: float = 1.05,
) -> Tuple[np.ndarray, float]:
    """
    Min–max (Chebyshev) regression for *one* branch current.

    Returns
    -------
    a     : (n,) slope vector
    delta : inflated worst‑case residual  Δ = λ · max|error|
    """
    m, n = X.shape
    # LP variables  z = [a_1 … a_n,  t]^T
    A_ub = np.vstack((np.hstack((+X, -np.ones((m, 1)))),
                      np.hstack((-X, -np.ones((m, 1))))))
    b_ub = np.hstack((+I_col, -I_col))
    c    = np.zeros(n + 1)
    c[-1] = 1.0
    bounds = [(None, None)] * n + [(0.0, None)]

    res = linprog(c, A_ub=A_ub, b_ub=b_ub,
                  bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"L∞ LP failed: {res.message}")

    a     = res.x[:-1]
    delta = safety_factor * res.x[-1]
    return a, delta


def _ols_branch_fit(
    X: np.ndarray,
    I_col: np.ndarray,
    *,
    margin: Literal["max", "sigma", "quant"] = "max",
    z: float = 2.58,
    q: float = 0.995,
    inflate: float = 1.05,
) -> Tuple[np.ndarray, float]:
    """
    Ordinary least‑squares slope + explicit safety margin.

    margin = "max"   → Δ = inflate · max|residual|
           = "sigma" → Δ = inflate · z · σ
           = "quant" → Δ = inflate · q‑quantile(|residual|)
    """
    reg = LinearRegression(fit_intercept=False).fit(X, I_col)
    a   = reg.coef_
    r   = I_col - X @ a

    if margin == "max":
        delta = inflate * np.abs(r).max()
    elif margin == "sigma":
        delta = inflate * z * r.std(ddof=1)
    elif margin == "quant":
        delta = inflate * np.quantile(np.abs(r), q)
    else:
        raise ValueError("margin must be 'max', 'sigma', or 'quant'")
    return a, delta


# ---------------------------------------------------------------------
# 2.  Branch‑current inequality factory
# ---------------------------------------------------------------------

def build_branch_current_constraints(
    X: np.ndarray,
    I: np.ndarray,
    rates: np.ndarray,
    *,
    method: Literal["linf", "ols"] = "linf",
    verbose: bool = False,
    **kw,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create  2·b  inequalities for |I_ij| ≤ rate_ij.

    Extra kwargs (kw) are forwarded to the chosen fitter:
        • linf : safety_factor
        • ols  : margin / z / q / inflate
    """
    X   = np.asarray(X, float)
    I   = np.asarray(I, float)
    rates = np.asarray(rates, float)

    m, n = X.shape
    b_br = I.shape[1]
    if rates.shape[0] != b_br:
        raise ValueError("rates size mismatch")

    A_rows, b_rows = [], []
    fit_fun = _linf_branch_fit if method == "linf" else _ols_branch_fit

    for j in range(b_br):
        a, delta = fit_fun(X,  I[:, j], **kw)

        if verbose:
            tag  = "L∞" if method == "linf" else "OLS"
            print(f"[{tag}  br {j:3d}]  Δ = {delta:.4g}")

        A_rows += [ +a,  -a ]
        b_rows += [ rates[j] - delta,  rates[j] - delta ]

    return np.vstack(A_rows), np.array(b_rows)


# ---------------------------------------------------------------------
# 3.  Voltage‑limit inequality factory
# ---------------------------------------------------------------------

def _reduced_Y_inverse(Y_full: np.ndarray, slack_idx: int = -1) -> np.ndarray:
    """
    Remove the slack row/column and return  (Y_red)^‑1.
    Assumes Y is nonsingular after slack removal.
    """
    mask = np.ones(Y_full.shape[0], bool)
    mask[slack_idx] = False
    Y_red = Y_full[np.ix_(mask, mask)]
    return inv(Y_red)


def build_voltage_constraints(
    Y_full: np.ndarray,
    v_min: np.ndarray | None,
    v_max: np.ndarray | None,
    *,
    slack_idx: int = -1,
    safety_factor: float = 1.02,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Two inequalities per non‑slack bus enforcing v_min ≤ v ≤ v_max.
    """
    if v_min is None or v_max is None:
        return np.empty((0, Y_full.shape[0]-1)), np.empty(0)

    v_min = np.asarray(v_min, float)
    v_max = np.asarray(v_max, float)
    S     = _reduced_Y_inverse(Y_full, slack_idx)      # sensitivity

    n = S.shape[0]
    if v_min.shape != (n,) or v_max.shape != (n,):
        raise ValueError("v_min/v_max must match #non‑slack buses")

    A_rows, b_rows = [], []
    for k in range(n):
        row = S[k]
        A_rows.append(+row)
        b_rows.append((v_max[k] - 1.0) / safety_factor)

        A_rows.append(-row)
        b_rows.append((1.0 - v_min[k]) / safety_factor)

    return np.vstack(A_rows), np.array(b_rows)


# ---------------------------------------------------------------------
# 4.  High‑level wrapper  — now with full docstring
# ---------------------------------------------------------------------
def build_A_b_full(
    X: np.ndarray,
    I: np.ndarray,
    rates: np.ndarray,
    Y_full: np.ndarray,
    *,
    v_min: np.ndarray | None = None,
    v_max: np.ndarray | None = None,
    current_method: Literal["linf", "ols"] = "linf",
    current_kwargs: dict | None = None,
    slack_idx: int = -1,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct a complete set of linear safety constraints
    for a power‑grid operating point **x** (vector of non‑slack injections).

    The resulting polytope is

        { x ∈ ℝⁿ :  A  x  ≤  b } ,

    where rows come from

        • branch‑current limits   |I_ij(x)| ≤ rates[j]      (2·b rows)
        • nodal‑voltage limits    v_min ≤ v(x) ≤ v_max      (2·n rows, optional)

    Parameters
    ----------
    X : ndarray, shape (m, n)
        Historical injection matrix.  Each row is a non‑slack injection vector
        xᵏ  whose dimension n equals the number of controllable buses.

    I : ndarray, shape (m, b)
        Historical signed branch currents corresponding to X.
        Column order must match ``rates``.

    rates : array‑like, shape (b,)
        Positive ampacity (thermal) limits, one per monitored branch.

    Y_full : ndarray, shape (N, N)
        Full complex nodal admittance matrix of the network, including the
        slack bus.  The routine removes the slack row/column internally to
        obtain  Y_red and its inverse for voltage sensitivities.

    v_min, v_max : ndarray, shape (n,), optional
        Per‑bus voltage lower / upper bounds (pu) for the *non‑slack* buses.
        If either is ``None`` the corresponding half‑spaces are omitted.

    current_method : {'linf', 'ols'}, default 'linf'
        Technique used to estimate the branch‑current sensitivity vectors:
            'linf'  – Chebyshev (min‑max) regression, tightest deterministic
            'ols'   – ordinary least‑squares + explicit safety margin

    current_kwargs : dict, optional
        Extra keyword arguments forwarded to the chosen current‑fit routine.
        * For 'linf':  ``safety_factor=float``   (λ ≥ 1, default 1.05)
        * For 'ols' :  ``margin={'max','sigma','quant'}``
                       ``inflate=float``         (λ ≥ 1)
                       ``z=float`` or ``q=float`` (tail parameters)

    slack_idx : int, default -1
        Index of the slack bus in ``Y_full`` (negative values are Pythonic).

    verbose : bool, default False
        If True, prints per‑branch residuals / margins while fitting.

    Returns
    -------
    A : ndarray, shape (2·b + 2·n_v, n)
        Stacked constraint matrix where n_v = n if both v_min & v_max given,
        else n_v = 0.

    b : ndarray, shape (A.shape[0],)
        Right‑hand‑side vector already tightened by the data‑driven margins
        (branch currents) and by the fixed voltage safety factor (default
        1.02 inside ``build_voltage_constraints``).

    Notes
    -----
    * For 'linf' the branch‑current constraints are satisfied by **all**
      historical samples; the optional ``safety_factor`` λ > 1 enlarges the
      margin for unseen operating points.

    * For 'ols' the quality of the bound depends on the chosen margin rule:
      ``margin='max'`` is deterministic on the training set,
      ``'sigma'`` or ``'quant'`` give probabilistic control.

    * The extra rows added for voltage limits increase the total constraint
      count by 2·n but have negligible computational impact because typically
      b ≫ n.
    """
    current_kwargs = {} if current_kwargs is None else current_kwargs
    A_I, b_I = build_branch_current_constraints(
        X, I, rates,
        method=current_method,
        verbose=verbose,
        **current_kwargs
    )

    A_V, b_V = build_voltage_constraints(
        Y_full, v_min, v_max,
        slack_idx=slack_idx
    )

    A = np.vstack((A_I, A_V))
    b = np.hstack((b_I, b_V))
    return A, b



# ---------------------------------------------------------------------
# 5.  Convenience wrappers reproducing earlier functions
# ---------------------------------------------------------------------

def build_A_b_linf(
    X, I, rates,
    Y_full=None, v_min=None, v_max=None,
    **kw
):
    return build_A_b_full(
        X, I, rates,
        Y_full if Y_full is not None else np.eye(X.shape[1]+1),  # dummy
        v_min=v_min, v_max=v_max,
        current_method="linf",
        current_kwargs=kw
    )


def build_A_b_ols(
    X, I, rates,
    Y_full=None, v_min=None, v_max=None,
    **kw
):
    """OLS + margin for currents; voltages optional."""
    return build_A_b_full(
        X, I, rates,
        Y_full if Y_full is not None else np.eye(X.shape[1]+1),
        v_min=v_min, v_max=v_max,
        current_method="ols",
        current_kwargs=kw
    )
