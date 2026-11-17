"""step42 -- the whitened matched filter under a misspecified covariance.

The whitened audit whitens with the mechanism covariance
Sigma taken from the public strategy matrix, but the realized null covariance
is sigma^2 (Sigma + gamma I) under (A3') and deployed strategy matrices differ
from the representative Toeplitz instances audited here.  Two facts make the
instrument deployable rather than fragile, and this script quantifies the
second:

  Level.  The audit standardizes mu_hat by the empirical held-out null, so the
  false-alarm rate is inherited from that standardization whatever the true
  covariance; misspecification costs power only (no computation needed).

  Power.  Whitening with Sigma' when the truth is Sigma gives the linear
  statistic a = Sigma'^{-1} e_S with deflection
      D(Sigma'; Sigma) = (a' e_S) / sqrt(a' Sigma a),
  against the optimum D*(Sigma) = sqrt(e_S' Sigma^{-1} e_S) and the
  unwhitened all-iterates sum D_cos(Sigma) = k / sqrt(e_S' Sigma e_S).
  loss = D/D* <= 1 by Cauchy-Schwarz; realized gain = D/D_cos.

Sweeps: (i) crosstalk gamma (in units of the per-round unit variance) with
whitening by Sigma; (ii) wrong BLT (theta, rho) and wrong BandMF bandwidth /
decay with the truth at the audited parameters.  Also solves for the gamma at
which the tree k=10, T=100 realized/predicted gain equals the observed 0.79
(3.34 / 4.21 in step22), as a consistency check, not an attribution.

Writes results/step42_mismatched_whitening.json.
"""
import json, math
import numpy as np
from harness.mechanisms import make_mechanism, minsep_schedule

def e_S(T, k, offset):
    v = np.zeros(T); v[list(minsep_schedule(k, T, offset))] = 1.0; return v

def deflections(Sig_true, Sig_white, t):
    a = np.linalg.solve(Sig_white, t)
    D = float(a @ t / math.sqrt(a @ Sig_true @ a))
    Dstar = float(math.sqrt(t @ np.linalg.solve(Sig_true, t)))
    Dcos = float(t.sum() / math.sqrt(t @ Sig_true @ t))
    return D, Dstar, Dcos

def avg_over_offsets(T, k, Sig_true, Sig_white, offsets=None):
    """offsets=None averages over all min-separation offsets; offsets=[0] is the
    single-offset convention of step25/step22 that produced the paper's
    predicted gains (e.g. tree k=10, T=100: 4.21)."""
    offs = range(T // k) if offsets is None else offsets
    vals = [deflections(Sig_true, Sig_white, e_S(T, k, o)) for o in offs]
    D, Ds, Dc = map(np.mean, zip(*vals))
    return {"loss": D / Ds, "gain_realized": D / Dc, "gain_optimal": Ds / Dc}

out = {"note": __doc__.strip(), "gamma_sweep": {}, "parameter_sweep": {}, "tree_gamma_for_observed_gain": {}}
GAMMAS = [0.0, 0.03, 0.1, 0.3, 1.0, 3.0]
for T, ks in ((100, (10, 100)), (50, (5,))):
    for mech in ("tree", "bandmf", "blt"):
        m = make_mechanism(mech, T); S = m.Sigma_unit; I = np.eye(T)
        for k in ks:
            key = f"{mech}_T{T}_k{k}"; rows = []
            for g in GAMMAS:
                r = avg_over_offsets(T, k, S + g * I, S)          # whiten with Sigma, truth Sigma+gamma I
                pred = avg_over_offsets(T, k, S, S)["gain_optimal"]
                r0 = avg_over_offsets(T, k, S + g * I, S, offsets=[0])
                pred0 = avg_over_offsets(T, k, S, S, offsets=[0])["gain_optimal"]
                rows.append({"gamma": g, **{kk: float(v) for kk, v in r.items()}, "gain_predicted_gamma0": float(pred),
                             "realized_over_predicted": float(r["gain_realized"] / pred),
                             "offset0": {"loss": float(r0["loss"]), "gain_realized": float(r0["gain_realized"]),
                                         "gain_predicted_gamma0": float(pred0), "realized_over_predicted": float(r0["gain_realized"] / pred0)}})
            out["gamma_sweep"][key] = rows
            print(f"{key:18s} " + "  ".join(f"g={r['gamma']:<4}: loss {r['loss']:.3f} gain {r['gain_realized']:.2f}/{r['gain_predicted_gamma0']:.2f}" for r in rows))
# parameter misspecification (T=100, k=10): truth = audited parameters
T = 100; k = 10
truth = {"blt": make_mechanism("blt", T).Sigma_unit, "bandmf": make_mechanism("bandmf", T).Sigma_unit}
variants = {
    "blt": {f"theta={th},rho={rh}": make_mechanism("blt", T, blt_theta=th, blt_rho=rh).Sigma_unit
            for th in (0.32, 0.4, 0.48) for rh in (0.56, 0.7, 0.84)},
    "bandmf": {f"band={b}": make_mechanism("bandmf", T, band=b).Sigma_unit for b in (5, 10, 20)},
}
for mech, vs in variants.items():
    out["parameter_sweep"][mech] = {}
    for lab, Sw in vs.items():
        r = avg_over_offsets(T, k, truth[mech], Sw)
        out["parameter_sweep"][mech][lab] = {kk: float(v) for kk, v in r.items()}
        print(f"{mech:7s} whiten with {lab:20s}: loss {r['loss']:.3f}  gain {r['gain_realized']:.2f} (optimal {r['gain_optimal']:.2f})")
# gamma consistent with the observed tree k=10 realized/predicted ratio
S = make_mechanism("tree", 100).Sigma_unit; I = np.eye(100); target = 3.3447 / 4.2144
res = {}
def step22_gain(Sig_true, Sig_white, T=100, k=10):
    """step22's pooling: mean over offsets of the whitened deflection, times
    sqrt(mean over offsets of e'Sigma e), over k -- a ratio of means."""
    Ds, vm = [], []
    for o in range(T // k):
        t = e_S(T, k, o); a = np.linalg.solve(Sig_white, t)
        Ds.append(float(a @ t / math.sqrt(a @ Sig_true @ a))); vm.append(float(t @ Sig_true @ t))
    return float(np.mean(Ds)) * math.sqrt(float(np.mean(vm))) / k
for conv in ("step22_ratio_of_means", "offset0", "mean_of_ratios"):
    def gain(g):
        if conv == "step22_ratio_of_means": return step22_gain(S + g * I, S)
        offs = [0] if conv == "offset0" else None
        return avg_over_offsets(100, 10, S + g * I, S, offsets=offs)["gain_realized"]
    pred = gain(0.0); lo, hi = 0.0, 10.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if gain(mid) / pred > target else (lo, mid)
    res[conv] = {"predicted_gain_gamma0": float(pred), "gamma_consistent": (lo + hi) / 2,
                 "realized_over_predicted_at_gamma": {str(g): float(gain(g) / pred) for g in (0.003, 0.01, 0.03, 0.1)}}
    print(f"tree k=10 T=100 [{conv:22s}]: predicted gain {pred:.3f}; observed/predicted {target:.3f} reproduced by gamma = {(lo+hi)/2:.4f}")
out["tree_gamma_for_observed_gain"] = {"observed_gain": 3.3447, "predicted_gain_step22": 4.2144, "ratio": target, **res,
                                        "note": "gamma (per-round unit-variance units) at which whitening by Sigma against truth Sigma+gamma I reproduces the observed ratio; a consistency value, not an estimate of gamma"}
# at the gamma that reproduces the tree k=10 realization, what would the other k=10 cells lose?
g_tree = res["step22_ratio_of_means"]["gamma_consistent"]
out["other_cells_at_tree_gamma"] = {"gamma": g_tree}
for mech in ("tree", "bandmf", "blt"):
    Sm = make_mechanism(mech, 100).Sigma_unit; Im = np.eye(100)
    r0 = avg_over_offsets(100, 10, Sm, Sm)["gain_realized"]; r1 = avg_over_offsets(100, 10, Sm + g_tree * Im, Sm)
    out["other_cells_at_tree_gamma"][f"{mech}_k10"] = {"loss": r1["loss"], "realized_over_predicted": r1["gain_realized"] / r0}
    print(f"at gamma={g_tree:.4f}: {mech:6s} k=10 loss {r1['loss']:.3f}, realized/predicted {r1['gain_realized']/r0:.3f}")
json.dump(out, open("results/step42_mismatched_whitening.json", "w"), indent=1)
print("wrote results/step42_mismatched_whitening.json")
