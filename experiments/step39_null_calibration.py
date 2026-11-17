"""step39 -- Gaussianity and Type-I calibration of the audit's null channel.

Theorem 2's exact null and its most-powerful-among-all-tests claim rest on
the projected release stream being Gaussian under the null.  The (A3) test
(step17_a3) checks the *covariance* structure -- cross-round independence for
DP-FedAvg -- but not Gaussianity itself.  This script checks Gaussianity and
finite-sample Type-I calibration directly, on the stored per-round
projections of the held-out (never-submitted) canary directions in
results/projections/*.npz, which are the primitive the null statistic is
built from.

For every file the `null` array is (T, m_null): per-round projections of
the release increment onto m_null held-out unit directions.  Three null
statistics are formed exactly as the auditor forms them and standardized
with the null's own sample mean and standard deviation, which is how the
audit computes mu-hat:

  perround  each per-round projection on its own          (T * m_null values)
  sum_T     the T-round sum, i.e. the projection of theta_T - theta_0: the
            final-model null statistic.  A held-out direction has no
            participation rounds, so the all-iterates null statistic
            coincides with it.                              (m_null values)

For correlated-noise files the per-round variance differs across rounds by
construction (that is what the strategy matrix does), so a pool
standardized once per file mixes Gaussians of different scale and reads as
heavy-tailed even when every round is exactly Gaussian.  Those files are
therefore also reported with each round standardized by its own
cross-canary mean and sd (`perround_roundwise`).

Reported per pool: skew, excess kurtosis, KS distance/p against N(0,1),
Anderson-Darling A^2 with its 5%/1% critical values, and the empirical
mass beyond the nominal two-sided and one-sided 5% and 1% thresholds.  Also
a per-file Shapiro-Wilk on the all-iterates sums, and the per-round pool
split into independent-noise (DP-FedAvg) and correlated-noise (tree,
BandMF, BLT) files, since the whitened statistic of Theorem 2 is a linear
map of the latter.

Writes results/step39_null_calibration.json.
"""
import glob
import json
import os
import warnings

import numpy as np
from scipy import stats

warnings.filterwarnings("ignore")
T = 50
FILES = sorted(glob.glob("results/projections/*.npz"))
OUT = "results/step39_null_calibration.json"


def rounds_first(a):
    """Return the null array as (T, m)."""
    if a.ndim == 1:
        return a[:, None]
    return a if a.shape[0] == T else a.T


def standardize(x):
    x = np.asarray(x, float)
    return (x - x.mean()) / x.std(ddof=1)


def describe(z):
    z = np.asarray(z, float)
    ks = stats.kstest(z, "norm")
    ad = stats.anderson(z, "norm")
    return {
        "n": int(z.size),
        "skew": float(stats.skew(z)),
        "excess_kurtosis": float(stats.kurtosis(z)),
        "ks_D": float(ks.statistic),
        "ks_p": float(ks.pvalue),
        "anderson_A2": float(ad.statistic),
        "anderson_crit_5pct": float(ad.critical_values[2]),
        "anderson_crit_1pct": float(ad.critical_values[4]),
        "tail_two_sided": {
            "nominal_0.05": float(np.mean(np.abs(z) > 1.959964)),
            "nominal_0.01": float(np.mean(np.abs(z) > 2.575829)),
            "nominal_0.001": float(np.mean(np.abs(z) > 3.290527)),
            "nominal_0.0001": float(np.mean(np.abs(z) > 3.890592)),
        },
        "tail_one_sided_upper": {
            "nominal_0.05": float(np.mean(z > 1.644854)),
            "nominal_0.01": float(np.mean(z > 2.326348)),
        },
    }


def noise_class(basename):
    b = basename.lower()
    if b.startswith("mech_"):
        for m in ("tree", "bandmf", "blt"):
            if f"_{m}_" in b:
                return "correlated", m
        return "independent", "fedavg"
    return "independent", "fedavg"


pools = {"perround": [], "sum_T": []}
# Self-standardization by the pooled sample's own mean/sd bounds the standardized
# values and thins the tails mechanically.  As a control, each
# file's null canaries are split into two disjoint halves and half A is
# standardized with the moments of half B.
pools_heldout = {"perround": [], "sum_T": []}
by_noise = {"independent": [], "correlated": []}
by_noise_roundwise = {"independent": [], "correlated": []}
per_file = []
groups = {}
m_null_values = set()

excluded = []
for f in FILES:
    b = os.path.basename(f)
    d = np.load(f)
    n = rounds_first(d["null"].astype(float))          # (T, m)
    # One file (pythia-70m tree, T=10, m_null=10) is neither the
    # T used anywhere else nor a sample large enough to standardize; exclude it
    # and say so in the artifact rather than pooling it silently.
    if n.shape[0] != T or n.shape[1] < 60:
        excluded.append({"file": b, "T": int(n.shape[0]), "m_null": int(n.shape[1])})
        continue
    m_null_values.add(int(n.shape[1]))
    cls, mech = noise_class(b)
    groups.setdefault(mech, 0)
    groups[mech] += 1

    sum_T = n.sum(axis=0)                                # final-model null statistic
    pools["sum_T"].append(standardize(sum_T))
    zr = standardize(n).ravel()
    pools["perround"].append(zr)
    half = n.shape[1] // 2
    A, B = n[:, :half], n[:, half:]
    pools_heldout["perround"].append(((A - B.mean()) / B.std(ddof=1)).ravel())
    sA, sB = A.sum(axis=0), B.sum(axis=0)
    pools_heldout["sum_T"].append((sA - sB.mean()) / sB.std(ddof=1))
    by_noise[cls].append(zr)
    # each round standardized across its own m_null canaries
    zrw = ((n - n.mean(axis=1, keepdims=True)) / n.std(axis=1, ddof=1, keepdims=True)).ravel()
    by_noise_roundwise[cls].append(zrw)
    per_file.append({
        "file": b, "mechanism": mech, "m_null": int(n.shape[1]),
        "shapiro_p_sum_T": float(stats.shapiro(sum_T)[1]) if 3 <= sum_T.size <= 5000 else None,
    })

out = {
    "note": ("Gaussianity and Type-I calibration of the null channel, computed "
             "from the stored per-round projections of held-out canary "
             "directions. Standardization uses the null's own sample mean and "
             "sd, as the audit does. Complements the (A3) autocorrelation test, "
             "which checks covariance structure, not Gaussianity."),
    "n_files": len(FILES),
    "T": T,
    "files_by_mechanism": groups,
    "m_null_values": sorted(m_null_values),
    "pooled": {k: describe(np.concatenate(v)) for k, v in pools.items()},
    "pooled_heldout_sd": {k: describe(np.concatenate(v)) for k, v in pools_heldout.items()},
    "excluded_files": excluded,
    "perround_by_noise_class": {k: describe(np.concatenate(v))
                                for k, v in by_noise.items() if v},
    "perround_roundwise_by_noise_class": {k: describe(np.concatenate(v))
                                          for k, v in by_noise_roundwise.items() if v},
    "per_file_shapiro_sum_T": {
        "n_files": sum(p["shapiro_p_sum_T"] is not None for p in per_file),
        "fraction_p_below_0.05": float(np.mean(
            [p["shapiro_p_sum_T"] < 0.05 for p in per_file if p["shapiro_p_sum_T"] is not None])),
        "min_p": float(min(p["shapiro_p_sum_T"] for p in per_file if p["shapiro_p_sum_T"] is not None)),
    },
    "per_file": per_file,
}
os.makedirs("results", exist_ok=True)
json.dump(out, open(OUT, "w"), indent=1)

print(f"files {len(FILES)} ({len(excluded)} excluded: {excluded})  by mechanism {groups}  m_null {sorted(m_null_values)}")
for k in ("perround", "sum_T"):
    r = out["pooled_heldout_sd"][k]
    print(f"[{k:8s} held-out-SD] n={r['n']:>7d} skew {r['skew']:+.3f} kurt {r['excess_kurtosis']:+.3f} "
          f"KS p={r['ks_p']:.2f} A2={r['anderson_A2']:.3f} |z|>1.96: {r['tail_two_sided']['nominal_0.05']:.4f} "
          f">2.58: {r['tail_two_sided']['nominal_0.01']:.4f} >3.29: {r['tail_two_sided']['nominal_0.001']:.5f} >3.89: {r['tail_two_sided']['nominal_0.0001']:.6f}")
for k in ("perround", "sum_T"):
    r = out["pooled"][k]
    print(f"[{k:8s}] n={r['n']:>7d} skew {r['skew']:+.3f} kurt {r['excess_kurtosis']:+.3f} "
          f"KS p={r['ks_p']:.2f} A2={r['anderson_A2']:.3f}(5%:{r['anderson_crit_5pct']:.3f}) "
          f"|z|>1.96: {r['tail_two_sided']['nominal_0.05']:.4f} >2.58: {r['tail_two_sided']['nominal_0.01']:.4f} "
          f">3.29: {r['tail_two_sided']['nominal_0.001']:.5f} >3.89: {r['tail_two_sided']['nominal_0.0001']:.6f}")
for key, lab in (("perround_by_noise_class", "file-std"), ("perround_roundwise_by_noise_class", "round-std")):
    for k, r in out[key].items():
        print(f"[perround {lab:9s}/{k:11s}] n={r['n']:>7d} skew {r['skew']:+.3f} kurt {r['excess_kurtosis']:+.3f} "
              f"KS p={r['ks_p']:.2f} A2={r['anderson_A2']:.3f} |z|>1.96: {r['tail_two_sided']['nominal_0.05']:.4f} |z|>2.576: {r['tail_two_sided']['nominal_0.01']:.4f}")
s = out["per_file_shapiro_sum_T"]
print(f"per-file Shapiro (sum_T): {s['n_files']} files, frac p<0.05 = {s['fraction_p_below_0.05']:.3f}, min p = {s['min_p']:.3f}")
print("m_null by mechanism:", {m: sorted({p['m_null'] for p in per_file if p['mechanism']==m}) for m in groups})
print("wrote", OUT)
