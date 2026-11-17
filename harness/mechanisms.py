"""Correlated-noise mechanisms for the step-20 mechanism arm (v3 §4.3).

Uniform abstraction: every mechanism adds noise in a "strategy space"

    W = G X + sigma Z,       Z iid rows,  sigma = z_mech * C,

and the adversary's per-round update stream is  X_hat = X + sigma * M_noise Z:

  mechanism | strategy G (for sensitivity)      | M_noise (stream noise map)
  ----------|-----------------------------------|---------------------------
  fedavg    | I (T x T)                         | I
  bandmf    | C_s banded lower-tri (T x T)      | C_s^{-1}
  blt       | C_s Toeplitz lower-tri (T x T)    | C_s^{-1}
  tree      | node-round incidence Pn (V x T)   | D @ P_prefix  (T x V)

ACCOUNTING (min-separation deterministic participation, gate G7's fixed
adjacency): a client participating in rounds S contributes to strategy rows
through the columns G[:, r], r in S; with nonnegative G the worst case is all
contributions aligned, giving EXACT sensitivity

    sens(S) = C * || sum_{r in S} G[:, r] ||_2.

The whole mechanism is then ONE Gaussian event with effective noise
multiplier z_mech / ||sum_{r in S} G[:, r]||, so pinning eps is a single
plain-Gaussian calibration (no composition). For fedavg this reduces to the
familiar z_mech / sqrt(k) (k-fold Gaussian composition); NOTE this is the
min-separation certificate, deliberately different from step 17's
Poisson-subsampled certificate -- never mix the two in one figure.

Strategies here are representative hand-parameterized Toeplitz matrices
(validated instrument-side in step 21); the jax_privacy-optimized matrices
are a drop-in replacement for the camera-ready mechanism arm.
"""

from dataclasses import dataclass

import numpy as np
import torch
from scipy.linalg import solve_triangular

from .accountant import calibrate_noise_multiplier
from .auditor_whitened import (banded_toeplitz_strategy, blt_toeplitz_strategy,
                               tree_prefix_incidence)


def tree_node_incidence(T: int) -> np.ndarray:
    """Node-round incidence Pn (V x T): Pn[v, r] = 1 iff round r+1 lies in
    node v's dyadic interval. Uses the same node indexing as
    tree_prefix_incidence (nodes appearing in prefix decompositions)."""
    # rebuild the node dict exactly as tree_prefix_incidence does
    nodes = {}

    def node_id(start, length):
        if (start, length) not in nodes:
            nodes[(start, length)] = len(nodes)
        return nodes[(start, length)]

    for t in range(1, T + 1):
        start, rest = 1, t
        while rest > 0:
            length = 1
            while length * 2 <= rest and (start - 1) % (length * 2) == 0:
                length *= 2
            node_id(start, length)
            start += length
            rest -= length
    Pn = np.zeros((len(nodes), T))
    for (start, length), v in nodes.items():
        Pn[v, start - 1: start - 1 + length] = 1.0
    return Pn


@dataclass
class Mechanism:
    name: str
    G: np.ndarray          # strategy/incidence for sensitivity (rows x T)
    M_noise: np.ndarray    # update-stream noise map (T x noise_dim)
    Sigma_unit: np.ndarray  # M_noise @ M_noise.T -- projection-noise cov / (zC/m)^2

    def sensitivity_norm(self, schedule) -> float:
        """|| sum_{r in S} G[:, r] ||_2 (nonnegative G -> exact aligned worst case)."""
        return float(np.linalg.norm(self.G[:, list(schedule)].sum(axis=1)))

    def calibrate(self, target_eps: float, schedule, delta: float) -> float:
        """z_mech s.t. the min-sep certificate for `schedule` equals target_eps.
        Single plain-Gaussian event: nm_required * sensitivity_norm."""
        nm = calibrate_noise_multiplier(target_eps, 1.0, 1, delta, "rdp")
        return nm * self.sensitivity_norm(schedule)

    def noise_stream(self, T: int, d: int, z: float, C: float,
                     seed: int) -> torch.Tensor:
        """Pre-generated (T, d) update-stream noise, already scaled by z*C."""
        g = np.random.default_rng(seed)
        Z = g.standard_normal((self.M_noise.shape[1], d))
        N = self.M_noise @ Z
        return torch.tensor(z * C * N, dtype=torch.float32)


def _guard(m: "Mechanism") -> "Mechanism":
    # The sensitivity certificate assumes an aligned worst case, which is exact
    # only for nonnegative strategies; signed G (e.g. blt with rho < 0) breaks
    # the certificate BEFORE it breaks any downstream theory.
    assert (m.G >= -1e-12).all(), (
        f"{m.name}: strategy matrix has negative entries; the aligned-adversary "
        f"sensitivity certificate is invalid for signed G")
    return m


# ---------------------------------------------------------------------------
# production-style strategies (added for the refocused paper; the hand-set
# "bandmf" / "blt" above are left untouched so earlier results stay valid)
# ---------------------------------------------------------------------------
def toeplitz_from_column(col: np.ndarray) -> np.ndarray:
    """Lower-triangular Toeplitz matrix with the given first column."""
    T = len(col)
    C = np.zeros((T, T))
    for i in range(T):
        C[i, : i + 1] = col[: i + 1][::-1]
    return C


def sqrt_prefix_column(T: int) -> np.ndarray:
    """First column of the square root of the prefix-sum matrix: the
    coefficients of (1 - x)^(-1/2), r_0 = 1, r_j = r_{j-1} (2j - 1) / (2j).
    This is the closed-form dense Toeplitz factorization (C @ C = A) used,
    e.g., by Agrawal et al. ("Let's Ask Gauss", 2026) for DP-MF-FTRL."""
    r = np.ones(T)
    for j in range(1, T):
        r[j] = r[j - 1] * (2 * j - 1) / (2 * j)
    return r


def prefix_error_objective(col: np.ndarray, T: int, k: int) -> float:
    """Squared error of the prefix sums per unit of certificate: for strategy
    C = toeplitz(col), ||C e_S||^2 * ||A C^{-1}||_F^2 / T with A the prefix-sum
    workload and S the min-separation schedule with k participations. This is
    the quantity matrix-factorization mechanisms are optimized for; it is
    invariant to the scale of C."""
    C = toeplitz_from_column(col)
    A = np.tril(np.ones((T, T)))
    B = solve_triangular(C, A.T, lower=True, trans="T").T      # A C^{-1}
    e = np.zeros(T)
    e[minsep_schedule(k, T)] = 1.0
    return float(np.sum((C @ e) ** 2) * np.sum(B ** 2) / T)


def production_style_column(name: str, T: int, band: int = 10,
                            k: int = 10) -> np.ndarray:
    """First column of a production-style Toeplitz strategy.

    sqrtmf     : closed form (no optimization).
    bandmf_opt : `band`-banded, nonnegative coefficients chosen to minimize
                 the prefix error at a fixed certificate for k min-separated
                 participations (the BandMF design objective, restricted to
                 the Toeplitz family).
    blt_opt    : single-buffer BLT column [1, theta, theta*rho, ...] with
                 (theta, rho) chosen for the same objective.
    """
    from scipy.optimize import minimize

    if name == "sqrtmf":
        return sqrt_prefix_column(T)
    if name == "bandmf_opt":
        def col_of(x):
            c = np.zeros(T)
            c[0] = 1.0
            c[1:band] = x
            return c
        x0 = sqrt_prefix_column(T)[1:band]
        res = minimize(lambda x: prefix_error_objective(col_of(x), T, k), x0,
                       method="L-BFGS-B", bounds=[(0.0, 2.0)] * (band - 1))
        return col_of(res.x)
    if name == "blt_opt":
        def col_of(x):
            theta, rho = x
            return np.concatenate([[1.0], theta * rho ** np.arange(T - 1)])
        best = None
        for x0 in [(0.4, 0.7), (0.5, 0.9), (0.3, 0.5), (0.6, 0.95)]:
            res = minimize(lambda x: prefix_error_objective(col_of(x), T, k), x0,
                           method="L-BFGS-B", bounds=[(0.0, 1.5), (0.0, 0.999)])
            if best is None or res.fun < best.fun:
                best = res
        return col_of(best.x)
    raise ValueError(name)


def make_mechanism(name: str, T: int, band: int = 10,
                   blt_theta: float = 0.4, blt_rho: float = 0.7) -> Mechanism:
    """`band` applies to bandmf only (inert for blt: shape is set by
    blt_theta/blt_rho)."""
    if name == "fedavg":
        I = np.eye(T)
        return _guard(Mechanism(name, G=I, M_noise=I, Sigma_unit=I))
    if name == "bandmf":
        C_s = banded_toeplitz_strategy(T, bandwidth=band)
        Minv = solve_triangular(C_s, np.eye(T), lower=True)
        return _guard(Mechanism(name, G=C_s, M_noise=Minv, Sigma_unit=Minv @ Minv.T))
    if name == "blt":
        C_s = blt_toeplitz_strategy(T, theta=blt_theta, rho=blt_rho)
        Minv = solve_triangular(C_s, np.eye(T), lower=True)
        return _guard(Mechanism(name, G=C_s, M_noise=Minv, Sigma_unit=Minv @ Minv.T))
    if name in ("sqrtmf", "bandmf_opt", "blt_opt"):
        C_s = toeplitz_from_column(production_style_column(name, T, band))
        Minv = solve_triangular(C_s, np.eye(T), lower=True)
        return _guard(Mechanism(name, G=C_s, M_noise=Minv, Sigma_unit=Minv @ Minv.T))
    if name == "tree":
        P = tree_prefix_incidence(T)          # (T, V) prefix decomposition
        Pn = tree_node_incidence(T)           # (V, T) node membership
        D = np.eye(T) - np.eye(T, k=-1)       # prefix -> increment difference
        M = D @ P                              # (T, V) increment noise map
        return _guard(Mechanism(name, G=Pn, M_noise=M, Sigma_unit=M @ M.T))
    raise ValueError(name)


def minsep_schedule(k: int, T: int, offset: int = 0):
    """k rounds spaced T//k apart (min-separation participation)."""
    stride = T // k
    return [(offset + i * stride) % T for i in range(k)]
