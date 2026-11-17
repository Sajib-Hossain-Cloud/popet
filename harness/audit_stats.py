"""Audit statistics computed from stored per-round projections.

Every audit of the released updates is a function of the per-round
projections of the update on each canary direction. Storing those
projections once per run lets any statistic, and its uncertainty, be
recomputed without retraining.

Inputs (as written by experiments/step46 and step47):
  P : (T, m_ins)   projection of round t's update on inserted direction i
  Q : (T, m_null)  the same for held-out directions
  unorm : (T,)     norm of round t's update
  schedules : list of participation sets, one per inserted direction

Statistics (per direction; held-out directions reuse the inserted schedules,
cycled, exactly as harness.auditor does):
  final    : sum over all rounds                       (final-model audit)
  max      : max over rounds of the per-round cosine   (Andrew et al., ICLR'24)
  sum      : sum over the canary's participation rounds
  single   : whitened matched filter on the canary's FIRST round only
             (the single-participation likelihood-ratio score of Agrawal et
             al., "Let's Ask Gauss", 2026)
  whitened : whitened matched filter on all participation rounds (this work)
"""

from typing import Dict, List, Tuple

import numpy as np
from scipy.linalg import cholesky, solve_triangular

AUDITS = ("final", "max", "sum", "single", "whitened")


def per_direction_stats(P, Q, unorm, schedules, Sigma_unit) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    T, m = P.shape
    n = Q.shape[1]
    sch_ins = [list(s) for s in schedules]
    sch_nul = [sch_ins[j % m] for j in range(n)]
    L = cholesky(np.asarray(Sigma_unit, dtype=np.float64), lower=True)

    def template(S, first_only=False):
        t = np.zeros(T)
        t[[S[0]] if first_only else S] = 1.0
        return t

    def mf(y, t):
        w = solve_triangular(L, y, lower=True)
        v = solve_triangular(L, t, lower=True)
        return float(v @ w) / (np.linalg.norm(v) + 1e-12)

    un = np.asarray(unorm, dtype=np.float64)[:, None] + 1e-12
    out = dict(
        final=(P.sum(0), Q.sum(0)),
        max=((P / un).max(0), (Q / un).max(0)),
        sum=(np.array([P[sch_ins[i], i].sum() for i in range(m)]),
             np.array([Q[sch_nul[j], j].sum() for j in range(n)])),
        single=(np.array([mf(P[:, i], template(sch_ins[i], True)) for i in range(m)]),
                np.array([mf(Q[:, j], template(sch_nul[j], True)) for j in range(n)])),
        whitened=(np.array([mf(P[:, i], template(sch_ins[i])) for i in range(m)]),
                  np.array([mf(Q[:, j], template(sch_nul[j])) for j in range(n)])),
    )
    return out


def mu_hat(ins, nul) -> float:
    sd = float(np.std(nul, ddof=1))
    return max(0.0, float(np.mean(ins) - np.mean(nul)) / (sd + 1e-12))


def mu_with_bound(ins, nul, rng, n_boot: int = 2000, alpha: float = 0.05):
    """mu-hat and a one-sided (1 - alpha) bootstrap LOWER bound, resampling
    inserted and held-out directions independently."""
    ins, nul = np.asarray(ins), np.asarray(nul)
    est = mu_hat(ins, nul)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        boots[b] = mu_hat(ins[rng.integers(0, len(ins), len(ins))],
                          nul[rng.integers(0, len(nul), len(nul))])
    return est, float(np.quantile(boots, alpha))


# ---------------------------------------------------------------------------
# Exact epsilon between two Gaussians with DIFFERENT variances (Andrew et al.,
# ICLR 2024, Appendix A). Their Algorithms 2 and 3 fit one Gaussian to the
# statistics of observed canaries and one to those of unobserved canaries
# and report this epsilon; the equal-variance (GDP) conversion used by
# `mu_hat` is the special case s1 == s0.
def _log_region_prob(a, b, c, mu, s):
    """log Pr[a X^2 + b X + c > 0] for X ~ N(mu, s^2)."""
    from scipy.special import log_ndtr, logsumexp
    if abs(a) < 1e-14:
        if abs(b) < 1e-14:
            return 0.0 if c > 0 else -np.inf
        r = -c / b
        return float(log_ndtr((mu - r) / s) if b > 0 else log_ndtr((r - mu) / s))
    disc = b * b - 4 * a * c
    if disc <= 0:
        return 0.0 if a > 0 else -np.inf
    sq = np.sqrt(disc)
    r1, r2 = sorted(((-b - sq) / (2 * a), (-b + sq) / (2 * a)))
    lo, hi = log_ndtr((r1 - mu) / s), log_ndtr((mu - r2) / s)   # X<r1, X>r2
    if a > 0:
        return float(logsumexp([lo, hi]))
    # a < 0: r1 < X < r2  ->  1 - Pr[X<r1] - Pr[X>r2]
    p = -np.expm1(logsumexp([lo, hi]))
    return float(np.log(p)) if p > 0 else -np.inf


def _hockey_stick(eps, mu1, s1, mu2, s2):
    """delta_eps(P1 || P2) = Pr_P1[f > eps] - e^eps Pr_P2[f > eps],
    f = log p1/p2 = a x^2 + b x + c."""
    a = 0.5 * (1 / s2 ** 2 - 1 / s1 ** 2)
    b = mu1 / s1 ** 2 - mu2 / s2 ** 2
    c = 0.5 * (mu2 ** 2 / s2 ** 2 - mu1 ** 2 / s1 ** 2) + np.log(s2) - np.log(s1)
    lp1 = _log_region_prob(a, b, c - eps, mu1, s1)
    lp2 = _log_region_prob(a, b, c - eps, mu2, s2)
    if lp1 == -np.inf:
        return 0.0
    return max(0.0, float(np.exp(lp1) - np.exp(min(700.0, eps + lp2))))


def eps_two_gaussians(mu0, s0, mu1, s1, delta, eps_max=200.0):
    """Smallest eps with delta_eps(P0||P1) <= delta and delta_eps(P1||P0) <= delta
    for P0 = N(mu0, s0^2) (unobserved canaries), P1 = N(mu1, s1^2) (observed)."""
    def d(eps):
        return max(_hockey_stick(eps, mu0, s0, mu1, s1), _hockey_stick(eps, mu1, s1, mu0, s0))
    if d(0.0) <= delta:
        return 0.0
    if d(eps_max) > delta:
        return float(eps_max)
    lo, hi = 0.0, eps_max
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if d(mid) > delta:
            lo = mid
        else:
            hi = mid
    return float(hi)


def eps_fitted(ins, nul, delta) -> float:
    """Andrew et al.'s reading: fit a Gaussian to each group, exact eps."""
    ins, nul = np.asarray(ins, float), np.asarray(nul, float)
    return eps_two_gaussians(float(nul.mean()), float(nul.std(ddof=1)),
                             float(ins.mean()), float(ins.std(ddof=1)), delta)
