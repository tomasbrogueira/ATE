
# Safe-Region Finder (Python port)

This project ports the supervisor’s `Grid_Cap_Regress.m` to Python. It simulates stochastic **node-level injections**, builds linear branch-current limits and finds one **axis-aligned hyper-rectangle** (independent range per node) that is guaranteed to lie inside those limits.

---
## Repo layout
```
.
├─ region_finder.py      # sampling, constraints, three rectangle generators
├─ main.py               # drives the 5-node “Kyte” case and a 3-D hexagon toy set
├─ hexagonal_test.py     # hexagonal-prism point cloud + analytic inequalities
├─ plots.py              # 2-D / 3-D plots
├─ Grid_Cap_Regress.m    # original MATLAB reference
├─ sampling_error_bound.md  # derivation of the O(1/m) miss–probability
└─ README.md             # this file
```

---
## Problem

**Inputs**  
* `n` nodes with real current injections collected in a vector \(x∈ℝⁿ\)  
* Linear branch-current limits expressed as a half-space system \(A x ≤ b\)

**Feasible set**  
\[𝒫 = \{x ∈ ℝⁿ \mid A x ≤ b\}.\]

**Goal**  
Pick lower/upper vectors \(ℓ,u∈ℝⁿ\) such that the axis-aligned box  
\[ℛ(ℓ,u)=\{x \mid ℓ ≤ x ≤ u\}\]  
lies entirely inside 𝒫.  The intervals \([ℓ_i,u_i]\) then give **independent limits per node**.

**Why boxes?**  On-line enforcement is trivial: each control centre checks a single inequality \(ℓ_i ≤ x_i ≤ u_i\) without solving coupled flows.

**Hardness**  Maximising \(\mathrm{vol}(ℛ)\) is NP-hard for \(n ≥ 3\), so an exact solver is impractical beyond toy grids.

**Heuristic used in code**  
1. Draw many candidate boxes via three generators (polytope directions, point pairs, local growth).  
2. Keep only those whose **every vertex** satisfies \(A v ≤ b\).  
3. Score candidates by the number of sampled feasible points they enclose; return the top scorer.

**Safety guarantee**  Because all vertices pass the test, the returned box is *provably contained* in 𝒫 even if the sampling missed extreme regions.

---
## Implemented box generators
1. **Polytope directions** – `generate_rectangles_from_polytope`
2. **Point pairs** – `generate_random_rectangles`
3. **Local growth** – `generate_improved_rectangles`

Each candidate is filtered by `filter_contained_rectangles`; `find_best_rectangle` selects the one covering most feasible samples.

---
## Test cases provided
* **Kyte 5-node grid** – `simulate_full_grid` (realistic)
* **Hexagonal prism** – `hexagonal_test.py` (analytic inequalities)

---
## Quick start
```bash
python main.py        # runs both demos and opens plots
```
To change sample sizes or rate limits, edit the two function calls at the bottom of *main.py*.

---
## Future work
- **Skip the `A,b` step?**  Build a box only from the `m` Monte‑Carlo samples and rely on probability.  See `sampling_error_bound.md` for the \(O(1/m)\) miss–probability argument.  A helper `estimate_leakage_probability(m, trials)` could measure the curve.

