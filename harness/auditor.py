"""One-shot empirical privacy auditor for DP-FedAvg (Andrew & Kairouz, ICLR'24).

Idea: insert canary clients with fixed RANDOM update directions. In high
dimension, a random direction is near-orthogonal to everything it did not
influence, so an unlearned canary's projection onto the final model is ~0
(noise only). A canary that participated leaves a signal component along its
direction. Comparing inserted-canary projections to held-out (never-inserted)
projections estimates the Gaussian-DP parameter mu, which converts to an
empirical (epsilon, delta) -- all from ONE training run, no retraining.

Threat models:
  - final-model  : project Delta = theta_T - theta_0 (realistic; hidden state).
  - all-iterates : combine per-round projections (worst case; sees each update).

We implement final-model here; all-iterates is added in Step 4 for the control.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
from scipy.stats import norm

from .fedavg import CanarySet, SimResult


# ---------------------------------------------------------------------------
# canary construction
# ---------------------------------------------------------------------------
def make_canaries(
    dim: int,
    num_inserted: int,
    num_held_out: int,
    clip_norm: float,
    rounds: int,
    participate_rounds: Optional[int] = None,
    spread: bool = False,
    seed: int = 0,
) -> CanarySet:
    """Random unit directions scaled to `clip_norm` (valid clipped updates).

    `participate_rounds`: how many rounds each inserted canary joins.
      None  -> every round (worst case, matches Andrew & Kairouz).
      k     -> k rounds each (sparse participation; for RQ2/RQ3).
    `spread=True`: each canary gets k RANDOM distinct rounds (mirrors real
    clients checking in at arbitrary times -- required for the single-epoch
    benchmark control). False -> the first k rounds (order-invariant for the
    final-model statistic, so equivalent there).
    """
    g = torch.Generator().manual_seed(seed)
    total = num_inserted + num_held_out
    dirs = torch.randn(total, dim, generator=g)
    dirs = dirs / dirs.norm(dim=1, keepdim=True) * clip_norm

    inserted = dirs[:num_inserted]
    held_out = dirs[num_inserted:]

    k = rounds if participate_rounds is None else min(participate_rounds, rounds)
    if spread:
        participation = [
            sorted(torch.randperm(rounds, generator=g)[:k].tolist())
            for _ in range(num_inserted)
        ]
    else:
        participation = [list(range(k)) for _ in range(num_inserted)]
    return CanarySet(directions=inserted, participation=participation, held_out=held_out)


def make_mixed_canaries(
    dim: int,
    n_worst: int,
    n_realistic: int,
    k_realistic: int,
    num_held_out: int,
    clip_norm: float,
    rounds: int,
    seed: int = 0,
):
    """Two canary groups in ONE run:
      - worst-case group: participates every round (max auditor power,
        matches Andrew & Kairouz's worst-case protocol),
      - realistic group: participates in k_realistic rounds (mirrors a typical
        q-sampled client, so its eps is comparable to the theoretical (eps,delta)).

    Returns (CanarySet, slice_worst, slice_realistic). Audit each group with
    `subset(canaries, sl)`.
    """
    g = torch.Generator().manual_seed(seed)
    total = n_worst + n_realistic + num_held_out
    dirs = torch.randn(total, dim, generator=g)
    dirs = dirs / dirs.norm(dim=1, keepdim=True) * clip_norm

    n_ins = n_worst + n_realistic
    k = min(k_realistic, rounds)
    participation = [list(range(rounds)) for _ in range(n_worst)] + [
        list(range(k)) for _ in range(n_realistic)
    ]
    cs = CanarySet(directions=dirs[:n_ins], participation=participation,
                   held_out=dirs[n_ins:])
    return cs, slice(0, n_worst), slice(n_worst, n_ins)


def subset(canaries: CanarySet, sl: slice) -> CanarySet:
    """View of one canary group (shares the held-out null set)."""
    return CanarySet(directions=canaries.directions[sl],
                     participation=canaries.participation[sl],
                     held_out=canaries.held_out)


# ---------------------------------------------------------------------------
# Gaussian-DP (mu-GDP) -> (epsilon, delta)
# ---------------------------------------------------------------------------
def gdp_epsilon(mu: float, delta: float) -> float:
    """epsilon of a mu-GDP mechanism at a given delta.

    delta(eps) = Phi(-eps/mu + mu/2) - e^eps * Phi(-eps/mu - mu/2), decreasing
    in eps; invert by bisection. mu<=0 -> epsilon 0.
    """
    if mu <= 1e-9:
        return 0.0

    def delta_of_eps(eps: float) -> float:
        return norm.cdf(-eps / mu + mu / 2.0) - np.exp(eps) * norm.cdf(-eps / mu - mu / 2.0)

    if delta_of_eps(0.0) <= delta:
        return 0.0
    lo, hi = 0.0, 1.0
    while delta_of_eps(hi) > delta and hi < 1e6:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if delta_of_eps(mid) > delta:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# the audit
# ---------------------------------------------------------------------------
@dataclass
class AuditResult:
    mu_hat: float
    epsilon_hat: float
    signal: float          # mean inserted projection - mean null projection
    null_mean: float
    null_std: float
    null_cosine_rms: float  # RMS cosine of held-out canaries with Delta (near-orthogonality check)


def _unit(x: torch.Tensor) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + 1e-12)


def audit_scalar(stat_ins, stat_nul, delta: float) -> AuditResult:
    """Generic audit from per-canary scalar statistics (inserted vs null).
    Used by the weak (loss-based data-canary) adversary in RQ3."""
    stat_ins = np.asarray(stat_ins, dtype=np.float64)
    stat_nul = np.asarray(stat_nul, dtype=np.float64)
    null_mean = float(stat_nul.mean())
    null_std = float(stat_nul.std(ddof=1))
    signal = float(stat_ins.mean() - null_mean)
    mu_hat = max(0.0, signal / (null_std + 1e-12))
    return AuditResult(mu_hat=mu_hat, epsilon_hat=gdp_epsilon(mu_hat, delta),
                       signal=signal, null_mean=null_mean, null_std=null_std,
                       null_cosine_rms=float("nan"))


def audit_final_model(res: SimResult, canaries: CanarySet, delta: float) -> AuditResult:
    """Empirical (mu, epsilon) from the final model only."""
    Delta = (res.final_vec - res.init_vec).detach().cpu()
    ins = canaries.directions.detach().cpu()
    nul = canaries.held_out.detach().cpu()

    # projection of Delta onto each canary's unit direction
    proj_ins = (_unit(ins) @ Delta).numpy()
    proj_nul = (_unit(nul) @ Delta).numpy()

    null_mean = float(proj_nul.mean())
    null_std = float(proj_nul.std(ddof=1))
    signal = float(proj_ins.mean() - null_mean)
    mu_hat = max(0.0, signal / (null_std + 1e-12))
    eps_hat = gdp_epsilon(mu_hat, delta)

    # near-orthogonality sanity: cosine of held-out canaries with Delta
    cos_nul = (_unit(nul) @ _unit(Delta)).numpy()
    null_cosine_rms = float(np.sqrt((cos_nul ** 2).mean()))

    return AuditResult(
        mu_hat=mu_hat, epsilon_hat=eps_hat, signal=signal,
        null_mean=null_mean, null_std=null_std, null_cosine_rms=null_cosine_rms,
    )


def audit_all_iterates(res: SimResult, canaries: CanarySet, delta: float) -> AuditResult:
    """Empirical (mu, epsilon) for the ALL-ITERATES adversary.

    Sees each round's update separately. For a canary, the optimal statistic
    sums its projection over the rounds it participates in (other rounds are
    noise-only). Held-out canaries use the same schedule so they estimate the
    null. Theory: mu ~ sqrt(k)/z (k = participation rounds) vs the final-model
    mu ~ k/(sqrt(T)*z) -- their ratio sqrt(T/k) IS the hidden-state gap.
    """
    if res.iterates is None:
        raise ValueError("all-iterates audit needs record_iterates=True")

    # per-round update_r = theta_r - theta_{r-1}
    updates = []
    prev = res.init_vec.detach().cpu()
    for it in res.iterates:
        cur = it.detach().cpu()
        updates.append(cur - prev)
        prev = cur
    U = torch.stack(updates)  # (T, d)

    ins = _unit(canaries.directions.detach().cpu())   # (m, d)
    nul = _unit(canaries.held_out.detach().cpu())      # (m_null, d)
    # <update_r, g> for every (round, canary)
    proj_ins = ins @ U.T   # (m, T)
    proj_nul = nul @ U.T   # (m_null, T)

    # sum over each canary's OWN participation schedule; held-out canaries
    # reuse the inserted schedules (cycled) so the null matches in length
    m = len(canaries.participation)
    stat_ins = np.array([
        proj_ins[i, canaries.participation[i]].sum().item() for i in range(m)
    ])
    stat_nul = np.array([
        proj_nul[j, canaries.participation[j % m]].sum().item()
        for j in range(nul.shape[0])
    ])

    null_mean = float(stat_nul.mean())
    null_std = float(stat_nul.std(ddof=1))
    signal = float(stat_ins.mean() - null_mean)
    mu_hat = max(0.0, signal / (null_std + 1e-12))
    return AuditResult(
        mu_hat=mu_hat, epsilon_hat=gdp_epsilon(mu_hat, delta), signal=signal,
        null_mean=null_mean, null_std=null_std, null_cosine_rms=float("nan"),
    )


def audit_all_iterates_max(res: SimResult, canaries: CanarySet, delta: float) -> AuditResult:
    """All-iterates statistic of Andrew et al. (ICLR'24, Sec. 5): "the maximum
    over rounds of the angle between the canary and the model delta at that
    round". Per direction, the statistic is max_t cos(v, u_t) with
    u_t = theta_t - theta_{t-1}; held-out directions give the null of the max.

    This is the PUBLISHED all-iterates audit. `audit_all_iterates` above is a
    different statistic (the sum over the canary's participation rounds, the
    matched filter under independent noise); the two must not be conflated.
    """
    if res.iterates is None:
        raise ValueError("all-iterates audit needs record_iterates=True")
    ins = _unit(canaries.directions.detach().cpu())
    nul = _unit(canaries.held_out.detach().cpu())
    best_ins = torch.full((ins.shape[0],), -float("inf"))
    best_nul = torch.full((nul.shape[0],), -float("inf"))
    prev = res.init_vec.detach().cpu()
    for it in res.iterates:
        cur = it.detach().cpu()
        u = _unit(cur - prev)
        prev = cur
        best_ins = torch.maximum(best_ins, ins @ u)
        best_nul = torch.maximum(best_nul, nul @ u)
    return audit_scalar(best_ins.numpy(), best_nul.numpy(), delta)
