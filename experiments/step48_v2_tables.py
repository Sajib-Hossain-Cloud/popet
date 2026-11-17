"""Step 48 -- tables for the refocused draft, from stored projections only.

Trains nothing. For every run of the mechanism comparison (step 46 and its
production-matrix and StackOverflow variants) the five audits of
harness.audit_stats are recomputed from the stored per-round projections,
so that all audits of a run use exactly the same released updates.

Outputs
  results/step48_v2_summary.json        per-run readings, per-cell summaries,
                                        predictions, null-distribution checks
  paper/PoPETS/generated/*.tex          table bodies that main_v2.tex \\input's

Per cell (mechanism, k) the summary holds, for each audit: the mean
deflection mu-hat and audited epsilon over runs, a 95% bootstrap interval
over seeds, and the number of runs in which the audit of all updates reads
below the final-model audit of the same run (the inconsistent reading).

Predictions use no training run: closed forms for final, sum, single and
whitened (averaged over the participation offsets the canaries cycle
through) and a Monte-Carlo evaluation of the max audit under the same
signal model, with the per-round update norm taken as noise-dominated.

Run:  .venv/bin/python -m experiments.step48_v2_tables
"""

import json
import os
import warnings

import numpy as np
from scipy import stats
from scipy.linalg import cholesky, solve_triangular

from harness.accountant import calibrate_noise_multiplier
from harness.audit_stats import AUDITS, eps_fitted, mu_hat, per_direction_stats
from harness.auditor import gdp_epsilon
from harness.mechanisms import make_mechanism, minsep_schedule

warnings.filterwarnings("ignore")

SOURCES = [   # (tag, result json, projection dir)
    ("emnist", "results/step46_mechanism_arm_emnist.json", "results/projections_step46"),
    ("emnist_production", "results/step46_mechanism_arm_emnist_production.json",
     "results/projections_step46_production"),
    ("stackoverflow", "results/step46_mechanism_arm_stackoverflow.json",
     "results/projections_step46_so"),
    # participation sweep (E = 1, C = 0.1, five seeds), run on CPU
    ("emnist_ksweep", "results/step46_mechanism_arm_emnist_ksweep.json",
     "results/projections_step46_ksweep"),
    # two stricter certificates (k = 10, E = 1, C = 0.1, five seeds), run on CPU
    ("emnist_eps4", "results/step46_mechanism_arm_emnist_eps4.json",
     "results/projections_step46_eps4"),
    ("emnist_eps8", "results/step46_mechanism_arm_emnist_eps8.json",
     "results/projections_step46_eps8"),
]
OUT = "results/step48_v2_summary.json"
GEN = "paper/PoPETS/generated"
NAMES = {"fedavg": "DP-FedAvg", "tree": "Tree aggregation",
         "sqrtmf": "Square-root factorization", "bandmf_opt": "BandMF, optimized",
         "blt_opt": "BLT, optimized", "bandmf": "BandMF, hand-set",
         "blt": "BLT, hand-set"}
ORDER = ["fedavg", "tree", "sqrtmf", "bandmf_opt", "blt_opt", "bandmf", "blt"]
_MECH = {}


def mech(name, T):
    if (name, T) not in _MECH:
        _MECH[(name, T)] = make_mechanism(name, T)
    return _MECH[(name, T)]


def schedules_of(z):
    s = z["schedules"]
    if s.dtype.kind in "US":
        return json.loads(str(s))
    return [list(map(int, r)) for r in s]


# ---------------------------------------------------------------------------
# predictions
def predict(name, T, k, nm, n_mc=200_000, seed=0):
    """Deflection of each audit under the signal model, averaged over the
    participation offsets; z from offset 0 (as the runs calibrate)."""
    M = mech(name, T)
    S = M.Sigma_unit
    z = nm * M.sensitivity_norm(minsep_schedule(k, T))
    Sinv = np.linalg.inv(S)
    L = cholesky(S, lower=True)
    one = np.ones(T)
    sd_t = np.sqrt(np.diag(S))
    rng = np.random.default_rng(seed)
    offs = range(max(1, T // k))
    vals = {a: [] for a in AUDITS}
    g = rng.standard_normal((n_mc // len(offs) + 1, T)) @ L.T      # unit-noise paths
    null_max = (g / sd_t).max(1)
    for off in offs:
        sch = minsep_schedule(k, T, offset=off)
        e = np.zeros(T)
        e[sch] = 1
        e1 = np.zeros(T)
        e1[sch[0]] = 1
        vals["final"].append(k / (z * np.sqrt(one @ S @ one)))
        vals["sum"].append(k / (z * np.sqrt(e @ S @ e)))
        vals["whitened"].append(np.sqrt(e @ Sinv @ e) / z)
        vals["single"].append((e @ Sinv @ e1) / (z * np.sqrt(e1 @ Sinv @ e1)))
        ins_max = ((e / z + g) / sd_t).max(1)
        vals["max"].append((ins_max.mean() - null_max.mean()) / null_max.std(ddof=1))
    return {a: float(np.mean(v)) for a, v in vals.items()}


# ---------------------------------------------------------------------------
def describe(pool):
    """Null statistics of one audit, one array per run (already standardized).

    Runs that share a seed share canary directions and noise, so pooled
    goodness-of-fit p-values would be anti-conservative. We therefore report
    quantities that stay unbiased under such dependence: pooled moments and
    tail masses, and the share of runs in which a per-run test rejects."""
    x = np.concatenate(pool)
    sw = [stats.shapiro(v)[1] < 0.05 for v in pool]
    ad = [stats.anderson(v, "norm") for v in pool]
    return dict(n=int(x.size), runs=len(pool), skew=float(stats.skew(x)),
                kurt=float(stats.kurtosis(x)),
                shapiro_reject=float(np.mean(sw)),
                anderson_reject=float(np.mean([a.statistic > a.critical_values[2] for a in ad])),
                upper=[float(np.mean(x > c)) for c in (1.644854, 2.326348, 3.090232)])


def boot_ci(per_seed, n_boot=4000, seed=0):
    """95% interval of the mean over runs, resampling seeds."""
    rng = np.random.default_rng(seed)
    keys = sorted(per_seed)
    if len(keys) < 2:
        return [float("nan"), float("nan")]
    means = []
    for _ in range(n_boot):
        pick = rng.choice(len(keys), len(keys))
        means.append(np.mean(np.concatenate([per_seed[keys[i]] for i in pick])))
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def null_widening(Q, st, Sigma_unit):
    """How much wider the null of the whitened and of the final-model
    statistic is than Equation (model) says, given the noise of single rounds.

    Under the model the whitened series L^-1 q of a held-out direction is
    white with one variance s2, the whitened statistic has variance s2 and
    the final-model statistic has variance s2 * 1'Sigma 1. s2 is measured
    from the whitened per-round values, so it already contains whatever
    honest clients add to a single round. A ratio above one therefore
    measures correlation ACROSS rounds that the model leaves out (honest
    updates that keep pointing the same way)."""
    Q = np.asarray(Q, dtype=np.float64)
    Sig = np.asarray(Sigma_unit, dtype=np.float64)
    L = cholesky(Sig, lower=True)
    W = solve_triangular(L, Q - Q.mean(1, keepdims=True), lower=True)
    s2 = float((W ** 2).mean())
    one = np.ones(Q.shape[0])
    return (float(st["whitened"][1].var(ddof=1) / s2),
            float(st["final"][1].var(ddof=1) / (s2 * (one @ Sig @ one))))


def analyse(tag, path, proj_dir):
    with open(path) as f:
        d = json.load(f)
    meta = d["meta"]
    T, delta = meta["T"], meta["delta"]
    runs, nullpool, nullpool_ho = [], {a: [] for a in AUDITS}, {a: [] for a in AUDITS}
    for r in d["rows"]:
        if r.get("diverged"):
            continue
        f = os.path.join(proj_dir, r["proj"] + ".npz")
        if not os.path.exists(f):
            continue
        z = np.load(f, allow_pickle=False)
        st = per_direction_stats(z["ins"], z["null"], z["update_norm"],
                                 schedules_of(z), mech(r["mechanism"], T).Sigma_unit)
        run = {k: r[k] for k in ("mechanism", "k", "E", "C", "seed", "acc", "z") if k in r}
        run["widen_whitened"], run["widen_final"] = null_widening(
            z["null"], st, mech(r["mechanism"], T).Sigma_unit)
        for a in AUDITS:
            ins, nul = st[a]
            run[f"muclip_{a}"] = mu_hat(ins, nul)        # as the audit reports it
            # not truncated at zero: unbiased, used for means and predictions
            run[f"mu_{a}"] = float((ins.mean() - nul.mean()) / nul.std(ddof=1))
            run[f"eps_{a}"] = gdp_epsilon(run[f"muclip_{a}"], delta)
            # conversion of Andrew et al.: one Gaussian per group, exact eps
            run[f"epsfit_{a}"] = eps_fitted(ins, nul, delta)
            nullpool[a].append((nul - nul.mean()) / nul.std(ddof=1))
            h = len(nul) // 2
            nullpool_ho[a].append((nul[:h] - nul[h:].mean()) / nul[h:].std(ddof=1))
        runs.append(run)
        # the script that trained must agree with this recomputation
        for a in ("final", "max", "sum", "whitened"):
            if f"mu_{a}" in r and abs(r[f"mu_{a}"] - run[f"muclip_{a}"]) > 5e-3:
                print(f"  [warn] {r['proj']} {a}: stored {r[f'mu_{a}']:.4f} "
                      f"recomputed {run[f'muclip_{a}']:.4f}")

    nm = meta["mu_cert"] ** -1
    cells = []
    for m in ORDER:
        for k in sorted({r["k"] for r in runs}):
            rs = [r for r in runs if r["mechanism"] == m and r["k"] == k]
            if not rs:
                continue
            cell = dict(mechanism=m, k=k, n=len(rs),
                        seeds=len({r["seed"] for r in rs}), pred=predict(m, T, k, nm))
            for a in AUDITS:
                for q in ("mu", "eps", "epsfit"):
                    v = np.array([r[f"{q}_{a}"] for r in rs])
                    cell[f"{q}_{a}"] = float(v.mean())
                    cell[f"{q}_{a}_ci"] = boot_ci(
                        {s: np.array([r[f"{q}_{a}"] for r in rs if r["seed"] == s])
                         for s in {r["seed"] for r in rs}})
                if a in ("final", "whitened"):
                    cell[f"widen_{a}"] = float(np.mean([r[f"widen_{a}"] for r in rs]))
                if a != "final":
                    cell[f"below_final_{a}"] = int(sum(
                        r[f"mu_{a}"] < r["mu_final"] - 1e-6 for r in rs))
                    cell[f"below_final_fit_{a}"] = int(sum(
                        r[f"epsfit_{a}"] < r["epsfit_final"] - 1e-6 for r in rs))
            accs = [r["acc"] for r in rs if "acc" in r]
            cell["acc"] = float(np.mean(accs)) if accs else float("nan")
            cells.append(cell)
    null = {a: dict(own=describe(nullpool[a]), heldout=describe(nullpool_ho[a]))
            for a in AUDITS} if runs else {}
    return dict(meta={k: meta[k] for k in ("T", "delta", "mu_cert", "eps_cert", "N")
                      if k in meta} | dict(n_runs=len(runs)),
                runs=runs, cells=cells, null=null, _pools=nullpool)


# ---------------------------------------------------------------------------
def fmt_main_rows(cells, cert_mu, bold=True):
    """Body of the main table: eps-hat and mu-hat of five audits per cell."""
    lines, last_k = [], None
    for c in sorted(cells, key=lambda c: (c["k"], ORDER.index(c["mechanism"]))):
        if last_k is not None and c["k"] != last_k:
            lines.append(r"\addlinespace")
        last_k = c["k"]

        def cellstr(q, a, nd):
            s = f"{c[f'{q}_{a}']:.{nd}f}"
            inconsistent = a != "final" and c[f"mu_{a}"] < c["mu_final"] - 0.005
            return rf"\textbf{{{s}}}" if (bold and inconsistent) else s
        eps = " & ".join(cellstr("eps", a, 1) for a in AUDITS)
        mu = " & ".join(cellstr("mu", a, 2) for a in AUDITS)
        rec = 100 * c["mu_whitened"] / c["pred"]["whitened"]   # mean certificate of the canaries
        lines.append(f"{NAMES[c['mechanism']]} & {c['k']} & {eps} & {mu} & {rec:.0f}\\% \\\\")
    return "\n".join(lines) + "\n\\bottomrule\n"


def fmt_null_rows(null):
    lab = {"final": "final-model audit", "max": "max audit", "sum": "sum audit",
           "single": "whitened audit, single round", "whitened": "whitened audit"}
    lines = []
    for a in AUDITS:
        r = null[a]["own"]
        lines.append(
            f"{lab[a]} & {r['runs']} & ${r['skew']:+.2f}$ & ${r['kurt']:+.2f}$"
            f" & {r['shapiro_reject']:.2f} & {r['anderson_reject']:.2f}"
            f" & {r['upper'][0]:.3f} & {r['upper'][1]:.4f} & {r['upper'][2]:.4f} \\\\")
    return "\n".join(lines) + "\n\\bottomrule\n"


def fmt_fit_rows(cells):
    """Appendix table: the same runs under the two-Gaussian conversion."""
    lines, last_k = [], None
    for c in sorted(cells, key=lambda c: (c["k"], ORDER.index(c["mechanism"]))):
        if last_k is not None and c["k"] != last_k:
            lines.append(r"\addlinespace")
        last_k = c["k"]
        eps = " & ".join(f"{c[f'epsfit_{a}']:.1f}" for a in AUDITS)
        cnt = " & ".join(f"{c[f'below_final_{a}']}/{c[f'below_final_fit_{a}']}"
                         for a in ("max", "sum", "whitened"))
        lines.append(f"{NAMES[c['mechanism']]} & {c['k']} & {c['n']} & {eps} & {cnt} \\\\")
    return "\n".join(lines) + "\n\\bottomrule\n"


def fmt_sweep_rows(cells):
    """Appendix table for a sweep: deflections of the five audits, the share
    of the canaries' certificate that the whitened audit recovers, and the
    share that the measured widening of its null implies."""
    lines, last_m = [], None
    for c in sorted(cells, key=lambda c: (ORDER.index(c["mechanism"]), c["k"])):
        if last_m is not None and c["mechanism"] != last_m:
            lines.append(r"\addlinespace")
        last_m = c["mechanism"]
        mu = " & ".join(
            (rf"\textbf{{{c[f'mu_{a}']:.2f}}}" if a != "final" and c[f"mu_{a}"] < c["mu_final"] - 0.005
             else f"{c[f'mu_{a}']:.2f}") for a in AUDITS)
        pred = " & ".join(f"{c['pred'][a]:.2f}" for a in ("final", "max", "whitened"))
        rec = 100 * c["mu_whitened"] / c["pred"]["whitened"]
        imp = 100 / np.sqrt(c["widen_whitened"])
        lines.append(f"{NAMES[c['mechanism']]} & {c['k']} & {mu} & {pred} & {rec:.0f}\\% & {imp:.0f}\\% \\\\")
    return "\n".join(lines) + "\n\\bottomrule\n"


def fmt_cert_rows(out, tags=("emnist_eps4", "emnist_eps8")):
    """Appendix table: the comparison at k = 10 under other certificates."""
    lines = []
    for t in tags:
        if t not in out or not out[t]["cells"]:
            continue
        if lines:
            lines.append(r"\addlinespace")
        meta = out[t]["meta"]
        for i, c in enumerate(sorted(out[t]["cells"], key=lambda c: ORDER.index(c["mechanism"]))):
            mu = " & ".join(
                (rf"\textbf{{{c[f'mu_{a}']:.2f}}}" if a != "final" and c[f"mu_{a}"] < c["mu_final"] - 0.005
                 else f"{c[f'mu_{a}']:.2f}") for a in AUDITS)
            lab = (rf"${meta['eps_cert']:.1f}$ & ${meta['mu_cert']:.2f}$" if i == 0 else " & ")
            rec = 100 * c["mu_whitened"] / c["pred"]["whitened"]
            cnt = " & ".join(f"{c[f'below_final_{a}']}" for a in ("max", "sum", "whitened"))
            lines.append(f"{lab} & {NAMES[c['mechanism']]} & {mu} & {c['pred']['final']:.2f} & "
                         f"{c['pred']['whitened']:.2f} & {rec:.0f}\\% & {cnt} \\\\")
    return "\n".join(lines) + "\n\\bottomrule\n"


def fmt_ci_rows(cells):
    """Appendix table: mean deflection of each audit with a 95% percentile
    bootstrap interval that resamples seeds (the independent unit)."""
    lines, last_k = [], None
    for c in sorted(cells, key=lambda c: (c["k"], ORDER.index(c["mechanism"]))):
        if last_k is not None and c["k"] != last_k:
            lines.append(r"\addlinespace")
        last_k = c["k"]
        cols = " & ".join(
            f"{c[f'mu_{a}']:.2f} [{c[f'mu_{a}_ci'][0]:.2f}, {c[f'mu_{a}_ci'][1]:.2f}]"
            for a in AUDITS)
        lines.append(f"{NAMES[c['mechanism']]} & {c['k']} & {cols} \\\\")
    return "\n".join(lines) + "\n\\bottomrule\n"


def fmt_predicted_rows(T=100, k=10, mu_cert=None):
    """Body of the prediction table: all seven mechanisms, five audits."""
    nm = 1.0 / mu_cert
    lines = []
    for m in ORDER:
        p = predict(m, T, k, nm)
        cells = []
        for a in AUDITS:
            v = f"{p[a]:.2f}"
            cells.append(rf"\textbf{{{v}}}" if (a != "final" and p[a] < p["final"] - 0.02) else v)
        if m == "bandmf":
            lines.append(r"\addlinespace")
        lines.append(f"{NAMES[m]} & " + " & ".join(cells) + " \\\\")
    return "\n".join(lines) + "\n\\bottomrule\n"


# ---------------------------------------------------------------------------
# fault injection (step 47)
FAULT_PATH = "results/step47_fault_injection_emnist.json"
FAULT_NAMES = {"none": "no fault", "noise0.8": r"noise $\times0.8$",
               "noise0.5": r"noise $\times0.5$", "noise0.25": r"noise $\times0.25$",
               "unclipped1.25": r"norm $\times1.25$",
               "unclipped1.5": r"norm $\times1.5$",
               "unclipped2": r"norm $\times2$",
               "gap9": "every 9 rounds", "gap5": "every 5 rounds"}
MILD = ("noise0.8", "unclipped1.25")


def analyse_faults(path=FAULT_PATH):
    """Detection rates under two rules, from the stored one-sided 95% lower
    bounds: (cert) the bound exceeds the certificate; (pred) the bound
    exceeds the audit's own predicted reading for the CORRECT mechanism and
    the DECLARED participation, which is known before training."""
    with open(path) as f:
        d = json.load(f)
    meta = d["meta"]
    T, k, cert = meta["T"], meta["k"], meta["mu_cert"]
    rows = [r for r in d["rows"] if not r.get("diverged")]
    pred = {m: predict(m, T, k, 1.0 / cert) for m in {r["mechanism"] for r in rows}}
    for r in rows:
        for a in AUDITS:
            r[f"cert_{a}"] = bool(r[f"lo_{a}"] > cert)
            r[f"predrule_{a}"] = bool(r[f"lo_{a}"] > pred[r["mechanism"]][a])
    faults = [f for f in meta["faults"] if any(r["fault"] == f for r in rows)]
    mechs = [m for m in ORDER if any(r["mechanism"] == m for r in rows)]

    def cell(rs):
        c = dict(n=len(rs), true_ratio=float(np.mean([r["true_mu"] for r in rs]) / cert))
        for a in AUDITS:
            c[f"mu_{a}"] = float(np.mean([r[f"mu_{a}"] for r in rs]))
            c[f"cert_{a}"] = int(sum(r[f"cert_{a}"] for r in rs))
            c[f"pred_{a}"] = int(sum(r[f"predrule_{a}"] for r in rs))
        return c
    pooled = {f: cell([r for r in rows if r["fault"] == f]) for f in faults}
    per = {(m, f): cell([r for r in rows if r["fault"] == f and r["mechanism"] == m])
           for m in mechs for f in faults
           if any(r["fault"] == f and r["mechanism"] == m for r in rows)}
    groups = dict(
        faulty=cell([r for r in rows if r["fault"] != "none"]),
        corr=cell([r for r in rows if r["fault"] != "none" and r["mechanism"] != "fedavg"]),
        mild=cell([r for r in rows if r["fault"] in MILD]),
        mildcorr=cell([r for r in rows if r["fault"] in MILD and r["mechanism"] != "fedavg"]))
    return dict(meta=dict(T=T, k=k, mu_cert=cert, n_runs=len(rows), mechanisms=mechs,
                          faults=faults, seeds=len({r["seed"] for r in rows})),
                groups=groups, pooled=pooled, per={f"{m}|{f}": c for (m, f), c in per.items()}, pred=pred)


def fmt_fault_rows(F):
    """Main-text table: one row per fault, pooled over mechanisms; for every
    audit the number of runs that detect, certificate rule / prediction rule."""
    lines = []
    for f in F["meta"]["faults"]:
        c = F["pooled"][f]
        cells = " & ".join(f"{c[f'cert_{a}']}/{c[f'pred_{a}']}" for a in AUDITS)
        lines.append(f"{FAULT_NAMES[f]} & {c['true_ratio']:.2f} & {cells} \\\\")
        if f == "none":
            lines.append(r"\addlinespace")
    return "\n".join(lines) + "\n\\bottomrule\n"


def fmt_fault_rows_full(F):
    lines = []
    for m in F["meta"]["mechanisms"]:
        for f in F["meta"]["faults"]:
            c = F["per"].get(f"{m}|{f}")
            if c is None:
                continue
            mu = " & ".join(f"{c[f'mu_{a}']:.2f}" for a in AUDITS)
            det = " & ".join(f"{c[f'cert_{a}']}/{c[f'pred_{a}']}" for a in AUDITS)
            lines.append(f"{NAMES[m] if f == F['meta']['faults'][0] else ''} & {FAULT_NAMES[f]} & "
                         f"{c['true_ratio']:.2f} & {mu} & {det} \\\\")
        if m != F["meta"]["mechanisms"][-1]:
            lines.append(r"\addlinespace")
    return "\n".join(lines) + "\n\\bottomrule\n"


def fault_numbers(F, put):
    put("flt-runs", F["meta"]["n_runs"])
    put("flt-seeds", F["meta"]["seeds"])
    for f, c in F["pooled"].items():
        key = f.replace(".", "p")
        put(f"flt-{key}-n", c["n"])
        put(f"flt-{key}-true", f"{c['true_ratio']:.2f}")
        for a in AUDITS:
            put(f"flt-{key}-cert-{a}", c[f"cert_{a}"])
            put(f"flt-{key}-pred-{a}", c[f"pred_{a}"])
            put(f"flt-{key}-mu-{a}", f"{c[f'mu_{a}']:.2f}")
    for g, c in F["groups"].items():
        put(f"flt-{g}-n", c["n"])
        for a in AUDITS:
            put(f"flt-{g}-cert-{a}", c[f"cert_{a}"])
            put(f"flt-{g}-pred-{a}", c[f"pred_{a}"])


def write_numbers(out, F=None):
    """Every number the text cites, as \\rv{key} macros, so that the prose
    cannot drift from the result files. Keys: <task>-<mech>-<k>-<what>, with
    task em (Federated EMNIST) or so (StackOverflow) and the mechanism name
    without underscores."""
    lines = ["%% generated by experiments/step48_v2_tables.py -- do not edit",
             r"\makeatletter"]

    def put(key, val):
        lines.append(rf"\expandafter\def\csname res@{key}\endcsname{{{val}}}")

    for tag, res in out.items():
        task = {"emnist": "em", "emnist_production": "em", "stackoverflow": "so",
                "emnist_ksweep": "ks", "emnist_eps4": "ca", "emnist_eps8": "cb"}[tag]
        cert = res["meta"]["mu_cert"]
        for c in res["cells"]:
            base = f"{task}-{c['mechanism'].replace('_', '')}-{c['k']}"
            put(f"{base}-n", c["n"])
            put(f"{base}-seeds", c["seeds"])
            for a in AUDITS:
                put(f"{base}-eps-{a}", f"{c[f'eps_{a}']:.1f}")
                put(f"{base}-fit-{a}", f"{c[f'epsfit_{a}']:.1f}")
                put(f"{base}-mu-{a}", f"{c[f'mu_{a}']:.2f}")
                put(f"{base}-pred-{a}", f"{c['pred'][a]:.2f}")
                put(f"{base}-rec-{a}", f"{100 * c[f'mu_{a}'] / c['pred']['whitened']:.0f}")
                if a != "final":
                    put(f"{base}-below-{a}", c[f"below_final_{a}"])
                    put(f"{base}-belowfit-{a}", c[f"below_final_fit_{a}"])
            # whitened minus final, tested over seeds (runs of one seed share
            # canaries and noise, so the seed is the independent unit)
            rs = [r for r in res["runs"] if r["mechanism"] == c["mechanism"] and r["k"] == c["k"]]
            per = np.array([np.mean([r["mu_whitened"] - r["mu_final"] for r in rs if r["seed"] == sd])
                            for sd in sorted({r["seed"] for r in rs})])
            put(f"{base}-whdiff", f"{per.mean():+.2f}".replace("+", "").replace("-", "$-$"))
            if len(per) > 2 and per.std() > 1e-9:
                put(f"{base}-whdiff-p", f"{stats.ttest_1samp(per, 0.0).pvalue:.2f}")
            put(f"{base}-widen-whitened", f"{100 * (c['widen_whitened'] - 1):.0f}")
            put(f"{base}-widen-final", f"{100 * (c['widen_final'] - 1):.0f}")
            put(f"{base}-impl-whitened", f"{100 / np.sqrt(c['widen_whitened']):.0f}")
            put(f"{base}-acc", f"{100 * c['acc']:.0f}")
            if rs and "z" in rs[0]:                      # calibrated noise multiplier
                put(f"{base}-z", f"{rs[0]['z']:.2f}")
            put(f"{base}-gain", f"{c['mu_whitened'] / c['mu_sum']:.2f}")
            put(f"{base}-predgain", f"{c['pred']['whitened'] / c['pred']['sum']:.2f}")
    for task, tags in (("em", ("emnist", "emnist_production")), ("so", ("stackoverflow",)),
                       ("ks", ("emnist_ksweep",)), ("ca", ("emnist_eps4",)),
                       ("cb", ("emnist_eps8",)), ("cc", ("emnist_eps4", "emnist_eps8")),
                       # every EMNIST run at the main certificate
                       ("ea", ("emnist", "emnist_production", "emnist_ksweep"))):
        cells = [c for t in tags if t in out for c in out[t]["cells"]]
        runs = [r for t in tags if t in out for r in out[t]["runs"]]
        if not cells:
            continue
        cert = out[tags[0]]["meta"]["mu_cert"]
        put(f"{task}-runs", len(runs))
        put(f"{task}-cert-mu", f"{cert:.2f}")
        if out[tags[0]]["meta"].get("eps_cert"):
            put(f"{task}-cert-eps", f"{out[tags[0]]['meta']['eps_cert']:.1f}")
        put(f"{task}-seeds", len({r["seed"] for r in runs}))
        put(f"{task}-cells", len(cells))
        for a in AUDITS:
            rec = [100 * c[f"mu_{a}"] / c["pred"]["whitened"] for c in cells]
            put(f"{task}-rec-{a}-min", f"{min(rec):.0f}")
            put(f"{task}-rec-{a}-max", f"{max(rec):.0f}")
            err = [abs(c[f"mu_{a}"] - c["pred"][a]) for c in cells]
            put(f"{task}-prederr-{a}-mean", f"{np.mean(err):.2f}")
            put(f"{task}-prederr-{a}-max", f"{max(err):.2f}")
            if a != "final":
                put(f"{task}-below-{a}", sum(c[f"below_final_{a}"] for c in cells))
                put(f"{task}-belowfit-{a}", sum(c[f"below_final_fit_{a}"] for c in cells))
                # cells whose MEAN reading is below the final-model mean
                put(f"{task}-cellsbelow-{a}", sum(c[f"mu_{a}"] < c["mu_final"] - 0.05 for c in cells))
                put(f"{task}-cellsbelowfit-{a}", sum(c[f"epsfit_{a}"] < c["epsfit_final"] - 0.05 for c in cells))
        corr = [c for c in cells if c["mechanism"] != "fedavg"]
        put(f"{task}-corr-n", sum(c["n"] for c in corr))
        for a in AUDITS[1:]:
            put(f"{task}-corr-below-{a}", sum(c[f"below_final_{a}"] for c in corr))
        # widening of the whitened null, and how well it accounts for the shortfall
        wid = [100 * (c["widen_whitened"] - 1) for c in cells]
        put(f"{task}-widen-whitened-min", f"{min(wid):.0f}")
        put(f"{task}-widen-whitened-max", f"{max(wid):.0f}")
        gap = [abs(100 / np.sqrt(c["widen_whitened"]) - 100 * c["mu_whitened"] / c["pred"]["whitened"])
               for c in cells]
        put(f"{task}-implerr-mean", f"{np.mean(gap):.0f}")
        put(f"{task}-implerr-max", f"{max(gap):.0f}")
        allerr = [abs(c[f"mu_{a}"] - c["pred"][a]) for c in cells for a in AUDITS]
        put(f"{task}-prederr-mean", f"{np.mean(allerr):.2f}")
        put(f"{task}-prederr-max", f"{max(allerr):.2f}")
        # largest shortfall of the whitened mean against the final-model mean
        put(f"{task}-wh-shortfall", f"{max(0.0, max(c['mu_final'] - c['mu_whitened'] for c in cells)):.2f}")
        put(f"{task}-wh-shortfall-pct", f"{max(0.0, max(100 * (1 - c['eps_whitened'] / c['eps_final']) for c in cells)):.1f}")
        # correct mechanisms that the two-Gaussian conversion puts above the certificate
        epsc = out[tags[0]]["meta"].get("eps_cert")
        if epsc:
            put(f"{task}-fit-above-cert-whitened",
                sum(r["epsfit_whitened"] > epsc for r in runs))
            put(f"{task}-gdp-above-cert-whitened",
                sum(r["eps_whitened"] > epsc for r in runs))
        cacc = [np.mean([r["acc"] for r in runs if r["mechanism"] == c["mechanism"]
                         and r["k"] == c["k"] and "acc" in r]) for c in cells]
        put(f"{task}-cellacc-min", f"{100 * min(cacc):.0f}")
        put(f"{task}-cellacc-max", f"{100 * max(cacc):.0f}")
        acc = [r["acc"] for r in runs if "acc" in r]
        if acc:
            put(f"{task}-acc-min", f"{100 * min(acc):.0f}")
            put(f"{task}-acc-max", f"{100 * max(acc):.0f}")
    if F is not None:
        fault_numbers(F, put)
    lines.append(r"\makeatother")
    with open(os.path.join(GEN, "numbers.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    return len(lines) - 3


def main():
    os.makedirs(GEN, exist_ok=True)
    out = {}
    for tag, path, proj in SOURCES:
        if not os.path.exists(path):
            print(f"[skip] {tag}: {path} not there yet")
            continue
        print(f"[{tag}] {path}")
        out[tag] = res = analyse(tag, path, proj)
        cert = res["meta"]["mu_cert"]
        print(f"  runs {res['meta']['n_runs']}  certificate mu={cert:.3f} eps={res['meta'].get('eps_cert', float('nan')):.2f}")
        print(f"  {'mech':>10} {'k':>4} {'n':>3} | " + " ".join(f"{a:>17}" for a in AUDITS)
              + " | below final: max sum single wh")
        for c in res["cells"]:
            print(f"  {c['mechanism']:>10} {c['k']:>4} {c['n']:>3} | " + " ".join(
                f"{c[f'mu_{a}']:4.2f}({c['pred'][a]:4.2f}) e{c[f'eps_{a}']:5.1f}" for a in AUDITS)
                + " | " + " ".join(f"{c[f'below_final_{a}']:>3}" for a in AUDITS[1:])
                + " | fit: " + " ".join(f"{c[f'epsfit_{a}']:5.1f}" for a in AUDITS)
                + " below " + " ".join(f"{c[f'below_final_fit_{a}']:>3}" for a in AUDITS[1:]))
        with open(os.path.join(GEN, f"tab_main_{tag}.tex"), "w") as f:
            f.write(fmt_main_rows(res["cells"], cert))
        if res["null"]:
            with open(os.path.join(GEN, f"tab_null_{tag}.tex"), "w") as f:
                f.write(fmt_null_rows(res["null"]))
            for a in AUDITS:
                for key in ("own", "heldout"):
                    r = res["null"][a][key]
                    print(f"  null[{a:>8} {key:>7}] runs {r['runs']} skew {r['skew']:+.3f} kurt {r['kurt']:+.2f} "
                          f"reject SW {r['shapiro_reject']:.2f} AD {r['anderson_reject']:.2f} "
                          f"upper tails {[round(u, 4) for u in r['upper']]}")
        with open(os.path.join(GEN, f"tab_fit_{tag}.tex"), "w") as f:
            f.write(fmt_fit_rows(res["cells"]))
    # null statistics pooled over all Federated EMNIST runs
    pools = {a: [v for t in ("emnist", "emnist_production") if t in out
                 for v in out[t]["_pools"][a]] for a in AUDITS}
    if pools[AUDITS[0]]:
        with open(os.path.join(GEN, "tab_null_emnist.tex"), "w") as f:
            f.write(fmt_null_rows({a: dict(own=describe(pools[a])) for a in AUDITS}))
    for t in out:
        out[t].pop("_pools", None)
    # Federated EMNIST: hand-set and optimized strategies in one table
    em = [c for t in ("emnist", "emnist_production") if t in out for c in out[t]["cells"]]
    if em:
        for name, fn in (("tab_main.tex", lambda c: fmt_main_rows(c, out["emnist"]["meta"]["mu_cert"])),
                         ("tab_fit.tex", fmt_fit_rows), ("tab_ci.tex", fmt_ci_rows)):
            with open(os.path.join(GEN, name), "w") as f:
                f.write(fn(em))
    for t in ("emnist_ksweep", "emnist_eps4", "emnist_eps8", "stackoverflow"):
        if t in out and out[t]["cells"]:
            with open(os.path.join(GEN, f"tab_sweep_{t}.tex"), "w") as f:
                f.write(fmt_sweep_rows(out[t]["cells"]))
    if "emnist_eps4" in out or "emnist_eps8" in out:
        with open(os.path.join(GEN, "tab_cert.tex"), "w") as f:
            f.write(fmt_cert_rows(out))
    if "emnist" in out:
        with open(os.path.join(GEN, "tab_predicted.tex"), "w") as f:
            f.write(fmt_predicted_rows(mu_cert=out["emnist"]["meta"]["mu_cert"]))
    F = None
    if os.path.exists(FAULT_PATH):
        F = out["faults"] = analyse_faults()
        with open(os.path.join(GEN, "tab_faults.tex"), "w") as f:
            f.write(fmt_fault_rows(F))
        with open(os.path.join(GEN, "tab_faults_full.tex"), "w") as f:
            f.write(fmt_fault_rows_full(F))
        print(f"[faults] {F['meta']['n_runs']} runs, {F['meta']['seeds']} seeds; "
              "detections certificate rule / prediction rule")
        for fl, c in F["pooled"].items():
            print(f"  {fl:>14} true x{c['true_ratio']:.2f} n={c['n']:>3} | " + "  ".join(
                f"{a[:5]} {c[f'cert_{a}']:>2}/{c[f'pred_{a}']:>2}" for a in AUDITS))
    n = write_numbers({k: v for k, v in out.items() if k != "faults"}, F)
    print(f"wrote {n} numbers to {GEN}/numbers.tex")
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", OUT, "and table bodies in", GEN)


if __name__ == "__main__":
    main()
