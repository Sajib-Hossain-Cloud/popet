"""Theoretical client-level (epsilon, delta) accounting for DP-FedAvg.

The DP-FedAvg mechanism, per round, is the *sampled Gaussian mechanism*:
  - client (Poisson) subsampling with rate q = cohort_size / num_clients,
  - Gaussian noise with multiplier z (= noise stddev on the summed clipped
    deltas divided by the per-client sensitivity C; note z is independent of C).
Composed over T rounds. Neighboring relation = add/remove ONE client => this is
user-level (client-level) DP, matching the auditor's threat model.

We wrap Google's `dp_accounting` and expose:
  - theoretical_epsilon(...)        : (z, q, T, delta) -> epsilon   [RDP and PLD]
  - calibrate_noise_multiplier(...) : target_eps -> z   (for fixed-eps sweeps)

Calibration is the crux of RQ1: to hold (eps, delta) fixed while varying local
epochs / clip / LR / cohort size, we re-solve z for each config.
"""

import logging
from typing import List

from dp_accounting import dp_event, pld, rdp

# dp_accounting emits benign "failed to converge" warnings for very low RDP
# orders at small noise; those orders are never the binding epsilon and are
# correctly excluded. Silence to keep experiment output readable.
logging.getLogger("absl").setLevel(logging.ERROR)

# A wide order grid so RDP stays tight across the z range we sweep.
DEFAULT_ORDERS: List[float] = (
    [1.1, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0]
    + list(range(9, 65))
    + [80.0, 100.0, 128.0, 200.0, 256.0, 400.0, 512.0]
)


def _accountant(kind: str):
    if kind == "rdp":
        return rdp.RdpAccountant(orders=DEFAULT_ORDERS)
    if kind == "pld":
        return pld.PLDAccountant()
    raise ValueError(f"unknown accountant kind: {kind!r}")


def theoretical_epsilon(
    noise_multiplier: float,
    sampling_prob: float,
    rounds: int,
    delta: float,
    kind: str = "rdp",
) -> float:
    """Client-level epsilon for T rounds of Poisson-subsampled Gaussian."""
    acc = _accountant(kind)
    event = dp_event.PoissonSampledDpEvent(
        sampling_prob, dp_event.GaussianDpEvent(noise_multiplier)
    )
    acc.compose(event, rounds)
    return acc.get_epsilon(delta)


def calibrate_noise_multiplier(
    target_epsilon: float,
    sampling_prob: float,
    rounds: int,
    delta: float,
    kind: str = "rdp",
    lo: float = 0.05,
    hi: float = 200.0,
    tol: float = 1e-3,
    max_iter: int = 100,
) -> float:
    """Smallest z achieving <= target_epsilon (epsilon is decreasing in z).

    Binary search; expands `hi` if the initial bracket is too tight.
    """
    def eps(z: float) -> float:
        return theoretical_epsilon(z, sampling_prob, rounds, delta, kind)

    # Ensure hi is private enough (eps(hi) <= target); expand if not.
    it = 0
    while eps(hi) > target_epsilon and it < 20:
        hi *= 2.0
        it += 1
    if eps(lo) <= target_epsilon:
        return lo  # even lo is private enough

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        e = eps(mid)
        if abs(e - target_epsilon) < tol:
            return mid
        if e > target_epsilon:   # not private enough -> need more noise
            lo = mid
        else:                    # too much noise -> can reduce
            hi = mid
    return 0.5 * (lo + hi)


def sampling_prob(cohort_size: int, num_clients: int) -> float:
    return cohort_size / num_clients
