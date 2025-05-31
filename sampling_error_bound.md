# Sampling‑error bounds for the three rectangle rules

---

## Shared context

Safe region (volume rescaled to 1)

$$
  \mathcal P=\{x\in\mathbb R^{n}\mid A x\le b\},\quad \lambda(\mathcal P)=1.
$$

Monte‑Carlo cloud

$$
  x^{(1)},\dots,x^{(m)}\;\overset{\text{i.i.d.}}{\sim}\; \operatorname{Unif}(\mathcal P).
$$

A rule $\mathsf R$ maps the sample cloud to **one or more** axis‑aligned boxes. Error event

$$
  E=\{\exists\,\text{built box }\mathcal R\text{ with }\mathcal R\not\subset\mathcal P\}.
$$

---

## 1  Point‑pair rule — *k random pairs*

### Construction

Pick *k* unordered index pairs $(i,j)$ without replacement. For each pair build

$$
  \mathcal R_{ij}=\bigl[\min(x^{(i)},x^{(j)}),\,\max(x^{(i)},x^{(j)})\bigr].
$$


### Proof outline

1. **One pair is bad with constant probability.**  Choose a thin boundary slab
   $S_\varepsilon=\{x\mid n^T x>b_{\max}-\varepsilon\}$.  Its volume is
   $\lambda(S_\varepsilon)=c\varepsilon$.  If *both* points of a pair fall in
   that slab, the top‑corner of the box violates the same inequality; call this
   event $B$.  Taking $\varepsilon$ small but fixed gives
   $\Pr(B)=c_0>0.$
2. **Many pairs make failure near‑certain.**  For weakly‑dependent draws the
   chance all *k* boxes are safe is $(1-c_0)^k$, hence the heuristic bound
   $\Pr(E)=1-(1-c_0)^k.$
   A rigorous union bound gives
   $\Pr(E)\;\ge\;k c_0-\tbinom{k}{2}c_0^2\;\ge\;(kc_0/2)\wedge1.$
   For $k\gg1$ the probability is therefore **very close to 1**.

### Result

$$
  \boxed{\Pr(E)\approx1-(1-c_0)^k\;\text{ → 1 exponentially in }k.}
$$

---

## 2  Local‑growth rule — isotropic factor *r*

### Construction

Pick a seed sample $p$ and grow a cube centred at $p$; at each step multiply
all half‑widths by $r>1$ until a second sample touches every one of the $2n$ faces.
Repeat for as many seeds as desired.

### Detailed derivation

* **1‑D order statistic.**  For uniform $[0,1]$ samples, the expected minimum is
  $\mathbb E[X_{(1)}]=1/(m+1)$.  After scaling each axis of $\mathcal P$ to unit
  length the same bound applies to every coordinate gap $\delta_i$.
* **Sliver volume before inflation.**  Unsafe region sits in $2n$ hyper‑slabs of
  thickness $\delta_i$, so
  $\mathbb E[\lambda(\text{sliver})]\le2n\,\tfrac{1}{m+1}=\tfrac{C_0}{m}.$
* **After inflation.**  Multiplying half‑widths by $r$ multiplies each
  $\delta_i$ by $r$ ⇒ volume by $r$.
* **Markov.**  Using $\Pr[Z>0]\le\mathbb E[Z]$ for $Z=\lambda(\text{sliver})$ gives
  $\Pr(E)\le \dfrac{C_0 r}{m}=\dfrac{C r}{m}.$

### Result

$$
  \boxed{\Pr(E)\le\dfrac{C r}{m}}\qquad (C\text{ depends on }n,\;\mathcal P).
$$


---

## 3  LP‑direction rule — inequalities enforced

### Construction

Choose random $w$; solve
$\min_{A x\le b} w^T x,\quad \max_{A x\le b} w^T x,$
then bound component‑wise between those two optima.

### Containment proof

All vertices inherit the *feasible* property $A x\le b$ from the LP optima and
are re‑checked by `filter_contained_rectangles`.

### Result

$$
  \boxed{\Pr(E)=0}\qquad\text{(sampling does not affect safety).}
$$
