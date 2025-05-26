
# Why the miss–probability decays like $O(1/m)$

Let $m$ be the number of i.i.d. *uniform* samples $x^{(1)},\dots,x^{(m)}$ drawn inside the true safe polytope
\[\mathcal P = \{x \mid A x \le b\}.\]
A candidate axis-aligned box
\[\mathcal R(\ell,u)=\{x \mid \ell \le x \le u\}\]
is built by taking coordinate-wise minima/maxima of those samples.  We want the probability that the box actually leaks outside $\mathcal P$.

---
## 1  Error event

\[E_m = \bigl\{\exists x \in \mathcal R : Ax > b\bigr\}.\]
Define the *danger sliver*
\[\mathcal S = \mathcal R \setminus \mathcal P.\]

---
## 2  All $m$ points avoided $\mathcal S$

Because every sample landed in $\mathcal P$, none landed in $\mathcal S$.  With $\lambda(\cdot)$ the Lebesgue volume and $\lambda(\mathcal P)=1$ (rescale if needed), one sample misses $\mathcal S$ with probability $1-\lambda(\mathcal S)$.  Independence gives
$$(*)\qquad\Pr[E_m] = \bigl(1-\lambda(\mathcal S)\bigr)^m \le e^{-m\,\lambda(\mathcal S)}.$$

---
## 3  Expected size of the missed sliver

For each coordinate $i$,
\[\ell_i = \min_k x_i^{(k)},\quad u_i = \max_k x_i^{(k)}.\]
In 1‑D the expected gap between the true bound and the sample min (or max) is $1/(m+1)$. Extending to $n$ dimensions and union‑bounding over $2n$ faces yields
\[\mathbb E[\lambda(\mathcal S)] \le \tfrac{C}{m},\quad C=2n\cdot \text{(edge factor)}.\]

---
## 4  Plugging into $(*)$

Apply Markov’s inequality to $\lambda(\mathcal S)$:
\[\Pr[E_m] = \Pr[\lambda(\mathcal S)>0] \le \frac{\mathbb E[\lambda(\mathcal S)]}{\epsilon} \le \frac{C}{m\,\epsilon}.\]
Choosing $\epsilon=1$ gives
\[\boxed{\Pr[E_m] \le \tfrac{C}{m}}.\]

---
## 5  Interpretation

*Each extra sample trims roughly a $1/m$‑sized slice off every face of the box, so the leftover unsafe volume—and hence the miss‑probability—shrinks at the same $1/m$ rate.*

The constant $C$ hides geometry‑specific factors.  If sampling is not uniform, the rate can change (faster with corner‑biased sampling, slower if heavily skewed).  Uniform i.i.d. sampling is the reference case.
