"""Whitened one-shot auditing statistic for correlated-noise mechanisms (Thm 2).

Matrix-factorization DP releases an update stream whose per-round noise is
correlated with KNOWN covariance (per canary-projection coordinate):

    y = x + n,     n ~ N(0, Sigma),   Sigma known to the auditor.

The auditor whitens with the Cholesky factor L (Sigma = L L^T) and applies the
matched filter for the canary's expected mean-shift template t (e.g. its
participation indicator e_S in increment coordinates, or A e_S in prefix
coordinates):

    stat = < L^{-1} t / ||L^{-1} t||,  L^{-1} y >

Under the null the noise component of the statistic is EXACTLY N(0, 1)-scaled
(Thm 2a); the deflection ||L^{-1} t|| is maximal among linear statistics of y
(Thm 2c, Cauchy-Schwarz). DP-FedAvg (Sigma = sigma^2 I) recovers the existing
participation-sum statistic; the binary tree in prefix coordinates recovers
the ancestor-sum statistic of "Let's Ask Gauss" (Rmk 6.1).

Everything here operates on per-canary PROJECTION vectors (length T), never on
d-dimensional iterates: O(T^2) per canary (Rmk 6.2).
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from scipy.linalg import cholesky, solve_triangular

from .auditor import gdp_epsilon


# ---------------------------------------------------------------------------
# noise-covariance builders (per unit projection coordinate)
# ---------------------------------------------------------------------------
def cov_iid(T: int, sigma: float) -> np.ndarray:
    """DP-FedAvg: independent per-round noise."""
    return sigma ** 2 * np.eye(T)


def cov_from_strategy(C_s: np.ndarray, sigma: float) -> np.ndarray:
    """MF mechanism with invertible lower-triangular strategy C_s:
    update-stream noise = sigma * C_s^{-1} Z  ->  Sigma = sigma^2 C_s^{-1} C_s^{-T}."""
    T = C_s.shape[0]
    Cinv = solve_triangular(C_s, np.eye(T), lower=True)
    return sigma ** 2 * (Cinv @ Cinv.T)


def banded_toeplitz_strategy(T: int, bandwidth: int, decay: float = 0.6) -> np.ndarray:
    """A valid (invertible, lower-tri, unit-diagonal) banded Toeplitz strategy.
    For instrument VALIDATION any such C_s works; step 20 swaps in the
    jax_privacy-optimized matrices for the real mechanism arm."""
    col = np.zeros(T)
    col[:bandwidth] = decay ** np.arange(bandwidth)
    C = np.zeros((T, T))
    for i in range(T):
        C[i, max(0, i - bandwidth + 1): i + 1] = col[: min(bandwidth, i + 1)][::-1]
    return C


def blt_toeplitz_strategy(T: int, theta: float = 0.4, rho: float = 0.7) -> np.ndarray:
    """BLT-style full lower-tri Toeplitz strategy: first column
    [1, theta, theta*rho, theta*rho^2, ...] (single-buffer parameterization)."""
    col = np.concatenate([[1.0], theta * rho ** np.arange(T - 1)])
    C = np.zeros((T, T))
    for i in range(T):
        C[i, : i + 1] = col[: i + 1][::-1]
    return C


def tree_prefix_incidence(T: int) -> np.ndarray:
    """Binary-tree aggregation in PREFIX coordinates. Row t (0-indexed round
    t+1) marks the dyadic nodes whose noises sum into the released prefix
    [1..t+1] (the standard greedy dyadic decomposition). Returns (T, V)."""
    nodes = {}  # (start, length) -> column index

    def node_id(start, length):
        if (start, length) not in nodes:
            nodes[(start, length)] = len(nodes)
        return nodes[(start, length)]

    rows = []
    for t in range(1, T + 1):
        cover, start = [], 1
        rest = t
        while rest > 0:
            length = 1
            while length * 2 <= rest and (start - 1) % (length * 2) == 0:
                length *= 2
            cover.append(node_id(start, length))
            start += length
            rest -= length
        rows.append(cover)
    P = np.zeros((T, len(nodes)))
    for t, cover in enumerate(rows):
        P[t, cover] = 1.0
    return P


def cov_tree_prefix(T: int, sigma_node: float) -> np.ndarray:
    P = tree_prefix_incidence(T)
    return sigma_node ** 2 * (P @ P.T)


# ---------------------------------------------------------------------------
# the whitened audit
# ---------------------------------------------------------------------------
@dataclass
class WhitenedAuditResult:
    mu_hat: float
    epsilon_hat: float
    signal: float
    null_mean: float
    null_std: float
    stats_ins: np.ndarray
    stats_nul: np.ndarray
    analytic_mu: Optional[float] = None


def matched_filter_stats(Y: np.ndarray, templates: List[np.ndarray],
                         L: np.ndarray) -> np.ndarray:
    """stat_i = < L^{-1} t_i / ||L^{-1} t_i||, L^{-1} y_i >.
    Y: (m, T) per-canary projection vectors; templates: list of (T,) shifts."""
    out = np.empty(Y.shape[0])
    for i in range(Y.shape[0]):
        w = solve_triangular(L, Y[i], lower=True)
        v = solve_triangular(L, templates[i], lower=True)
        out[i] = float(v @ w) / (np.linalg.norm(v) + 1e-12)
    return out


def audit_whitened(Y_ins: np.ndarray, Y_nul: np.ndarray,
                   templates_ins: List[np.ndarray], Sigma: np.ndarray,
                   delta: float,
                   signal_scale: Optional[float] = None) -> WhitenedAuditResult:
    """One-shot audit with the whitened matched filter.

    Held-out canaries are scored with the inserted templates (cycled), exactly
    as in `audit_all_iterates`, so the empirical null matches in law.
    `signal_scale`: expected per-round mean shift (e.g. C/m); if given, the
    analytic deflection  signal_scale * ||L^{-1} t||  is reported for the
    theory-vs-measurement check (Thm 2b).
    """
    L = cholesky(Sigma, lower=True)
    m = len(templates_ins)
    stats_ins = matched_filter_stats(Y_ins, templates_ins, L)
    templates_nul = [templates_ins[j % m] for j in range(Y_nul.shape[0])]
    stats_nul = matched_filter_stats(Y_nul, templates_nul, L)

    null_mean = float(stats_nul.mean())
    null_std = float(stats_nul.std(ddof=1))
    signal = float(stats_ins.mean() - null_mean)
    mu_hat = max(0.0, signal / (null_std + 1e-12))

    analytic = None
    if signal_scale is not None:
        defl = [signal_scale * np.linalg.norm(
            solve_triangular(L, t, lower=True)) for t in templates_ins]
        analytic = float(np.mean(defl))

    return WhitenedAuditResult(
        mu_hat=mu_hat, epsilon_hat=gdp_epsilon(mu_hat, delta), signal=signal,
        null_mean=null_mean, null_std=null_std,
        stats_ins=stats_ins, stats_nul=stats_nul, analytic_mu=analytic)


def audit_all_iterates_whitened(res, canaries, delta: float,
                                Sigma_unit: np.ndarray) -> WhitenedAuditResult:
    """Whitened all-iterates audit on a real SimResult (step-20 adapter).

    Extracts per-round per-canary projections exactly as
    `auditor.audit_all_iterates` does, then applies the matched filter with
    the mechanism's unit noise covariance. The audit statistic mu is
    scale-invariant (empirical null), so Sigma_unit = M_noise M_noise^T
    suffices -- no (zC/m)^2 factor needed.
    """
    import torch

    if res.iterates is None:
        raise ValueError("whitened audit needs record_iterates=True")
    updates, prev = [], res.init_vec.detach().cpu()
    for it in res.iterates:
        cur = it.detach().cpu()
        updates.append(cur - prev)
        prev = cur
    U = torch.stack(updates)                                   # (T, d)
    T = U.shape[0]

    def unit(x):
        return x / (x.norm(dim=-1, keepdim=True) + 1e-12)

    Y_ins = (unit(canaries.directions.detach().cpu()) @ U.T).numpy()
    Y_nul = (unit(canaries.held_out.detach().cpu()) @ U.T).numpy()
    templates = [np.isin(np.arange(T), S).astype(float)
                 for S in canaries.participation]
    return audit_whitened(Y_ins, Y_nul, templates, Sigma_unit, delta)


def audit_unwhitened(Y_ins: np.ndarray, Y_nul: np.ndarray,
                     templates_ins: List[np.ndarray],
                     delta: float) -> WhitenedAuditResult:
    """The existing participation-sum statistic <t, y> (mismatched filter
    under correlated noise) -- the comparison baseline for Thm 2c."""
    m = len(templates_ins)
    stats_ins = np.array([float(templates_ins[i] @ Y_ins[i])
                          for i in range(Y_ins.shape[0])])
    stats_nul = np.array([float(templates_ins[j % m] @ Y_nul[j])
                          for j in range(Y_nul.shape[0])])
    null_mean = float(stats_nul.mean())
    null_std = float(stats_nul.std(ddof=1))
    signal = float(stats_ins.mean() - null_mean)
    mu_hat = max(0.0, signal / (null_std + 1e-12))
    return WhitenedAuditResult(
        mu_hat=mu_hat, epsilon_hat=gdp_epsilon(mu_hat, delta), signal=signal,
        null_mean=null_mean, null_std=null_std,
        stats_ins=stats_ins, stats_nul=stats_nul)
