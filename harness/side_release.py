"""Audits of a release with a DP side channel (model stream + clipped count).

Setting: one canary client participating in ONE round (k = 1). In that round
the release is the noised update sum AND the noised centred clipped count
(harness.fedavg.AdaptiveClip). Per canary, in normalised coordinates,

    a = <v_hat, update> * m / (server_lr * C_t)   ~ N(1[present], z_Delta^2 (1+gamma))
    c = count_noised - count_honest               ~ N(1/2 [present], sigma_b^2)

so the canary pair is a two-coordinate Gaussian shift with

    mu_model = 1 / z_Delta,   mu_count = 1 / (2 sigma_b),
    mu_joint^2 = mu_model^2 + mu_count^2 = 1 / z^2          (gamma = 0),

z being the certified joint noise multiplier. `lam = mu_count^2 / mu_joint^2`
is the share of the squared deflection carried by the count channel:

    z_Delta = z / sqrt(1 - lam),      sigma_b = z / (2 sqrt(lam)).

Three auditors, all standardised by an empirical null exactly as
harness.auditor does:
  model  : the canonical canary-direction statistic `a` (reads mu_model);
  count  : `c` alone (white-box: the auditor subtracts the honest count);
  joint  : the matched filter a / z_Delta^2 + (1/2) c / sigma_b^2 (Thm 2 on the
           two-coordinate release; reads mu_joint).
`*_raw` variants use the released count without the honest baseline -- what an
output-level auditor sees; the honest count's round-to-round variance then
acts as extra noise, and the joint filter uses plug-in null scales.
"""

from dataclasses import dataclass
from typing import List

import numpy as np

from .auditor import audit_scalar


def split_noise(z: float, lam: float):
    """(z_Delta, sigma_b) spending share `lam` of the squared deflection on the
    count channel at joint multiplier z. lam = 0 -> (z, inf): no side release."""
    if not 0.0 <= lam < 1.0:
        raise ValueError("lam must be in [0, 1)")
    z_delta = z / np.sqrt(1.0 - lam)
    sigma_b = np.inf if lam == 0.0 else z / (2.0 * np.sqrt(lam))
    return float(z_delta), float(sigma_b)


def joint_multiplier(z_delta: float, sigma_b: float) -> float:
    """Effective z of the joint release: 1/z^2 = 1/z_Delta^2 + 1/(2 sigma_b)^2."""
    return float((z_delta ** -2 + (2.0 * sigma_b) ** -2) ** -0.5)


def count_share(z_delta: float, sigma_b: float) -> float:
    """lam = share of the squared deflection in the count channel."""
    mc2 = (2.0 * sigma_b) ** -2
    return float(mc2 / (z_delta ** -2 + mc2))


@dataclass
class Streams:
    a_ins: np.ndarray        # (n_ins,)  normalised model projection, own round
    a_nul: np.ndarray        # (n_nul,)  held-out direction at its paired null round
    c_ins: np.ndarray        # white-box count residual at the canary's round
    c_nul: np.ndarray
    craw_ins: np.ndarray     # released centred count (no honest baseline)
    craw_nul: np.ndarray
    rounds_ins: np.ndarray
    rounds_nul: np.ndarray


def extract_streams(res, canaries, null_rounds: List[int]) -> Streams:
    """Per-canary (model, count) coordinates from a k=1 adaptive-clip run.

    Each inserted canary is scored at its own round; held-out direction j is
    scored at null_rounds[j % len(null_rounds)], a round with NO canary, so the
    empirical null is correct for both channels."""
    if res.iterates is None or res.clip_log is None:
        raise ValueError("needs record_iterates=True and adaptive_clip")
    log = res.clip_log
    m, slr = log["cohort_size"], log["server_lr"]
    C = np.asarray(log["C"], dtype=np.float64)
    cn = np.asarray(log["count_noised"], dtype=np.float64)
    ch = np.asarray(log["count_honest"], dtype=np.float64)
    ncan = np.asarray(log["n_canaries"])

    for S in canaries.participation:
        if len(S) != 1:
            raise ValueError("extract_streams assumes k = 1 canaries")
    r_ins = np.array([S[0] for S in canaries.participation])
    if (ncan[r_ins] != 1).any():
        raise ValueError("expected exactly one canary per inserted round")
    n_nul = canaries.held_out.shape[0]
    r_nul = np.array([null_rounds[j % len(null_rounds)] for j in range(n_nul)])
    if (ncan[r_nul] != 0).any():
        raise ValueError("null rounds must contain no canary")

    def unit(x):
        return x / (x.norm(dim=-1, keepdim=True) + 1e-12)

    dev = res.final_vec.device
    ins = unit(canaries.directions.to(dev))
    nul = unit(canaries.held_out.to(dev))
    a_ins = np.empty(len(r_ins))
    a_nul = np.empty(n_nul)
    prev = res.init_vec
    for t, it in enumerate(res.iterates):
        upd = it - prev
        prev = it
        scale = m / (slr * C[t])
        sel = np.nonzero(r_ins == t)[0]
        if sel.size:
            a_ins[sel] = (ins[sel] @ upd).cpu().double().numpy() * scale
        sel = np.nonzero(r_nul == t)[0]
        if sel.size:
            a_nul[sel] = (nul[sel] @ upd).cpu().double().numpy() * scale

    resid = cn - ch
    return Streams(a_ins=a_ins, a_nul=a_nul,
                   c_ins=resid[r_ins], c_nul=resid[r_nul],
                   craw_ins=cn[r_ins], craw_nul=cn[r_nul],
                   rounds_ins=r_ins, rounds_nul=r_nul)


def joint_stat(a, c, z_delta: float, sigma_b: float):
    """Likelihood-ratio statistic of the two-coordinate Gaussian shift
    (1, 1/2) with noise (z_Delta, sigma_b)."""
    return np.asarray(a) / z_delta ** 2 + 0.5 * np.asarray(c) / sigma_b ** 2


def audit_streams(st: Streams, z_delta: float, sigma_b: float, delta: float):
    """dict name -> AuditResult for model / count / joint (+ raw-count variants)."""
    out = dict(model=audit_scalar(st.a_ins, st.a_nul, delta))
    if np.isfinite(sigma_b):
        out["count"] = audit_scalar(st.c_ins, st.c_nul, delta)
        out["joint"] = audit_scalar(
            joint_stat(st.a_ins, st.c_ins, z_delta, sigma_b),
            joint_stat(st.a_nul, st.c_nul, z_delta, sigma_b), delta)
        out["count_raw"] = audit_scalar(st.craw_ins, st.craw_nul, delta)
        out["joint_raw"] = audit_scalar(*joint_raw_stats(st), delta)
    return out


def joint_raw_stats(st: Streams):
    """Output-level joint statistic: same linear form as `joint_stat`, but with
    plug-in null scales. The released count still contains the honest clients'
    centred count, whose round-to-round variance an output-level auditor
    cannot subtract, so the nominal sigma_b would mis-weight it."""
    sa = float(np.std(st.a_nul, ddof=1))
    sc = float(np.std(st.craw_nul, ddof=1))
    return (joint_stat(st.a_ins, st.craw_ins, sa, sc),
            joint_stat(st.a_nul, st.craw_nul, sa, sc))
