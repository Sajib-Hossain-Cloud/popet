"""Step 26: PRE-REGISTERED Arm G forecast (Pythia-1B, Qwen2.5-0.5B).

Built ONLY from the 70M/160M/410M decoder ladder already on disk, BEFORE any
Arm G cell runs. Emits a forecast with three explicitly separated parts, per
the preregistration agreement:

  1. STRUCTURAL prediction: H approximately follows the 1/2-law (the
     scale-stable structural invariant). Tight band.
  2. LEVEL prediction: eps_final need NOT be monotonic in scale; predicted
     as a range from the observed cross-scale spread, WIDENED for
     extrapolation (Pythia-1B is one rung past the ladder; Qwen-0.5B is a
     different family, so its level band is deliberately wide / low-confidence).
  3. DECISION CRITERIA: what outcome counts as CONFIRM / ATTENUATE / FAIL for
     structure, and CONFIRM / MISS for level. Fixed here, never altered after
     results are seen.

The file is written to results/prereg/ and a sha256 of its exact bytes is
written alongside as the git-independent integrity anchor. The user then
commits both and records the commit hash; only then does Arm G run.
"""

import datetime
import hashlib
import json
import math
import os

import numpy as np

T = 50
LADDER = [("70M", "results/step17_decomposition_pythia-70m.json"),
          ("160M", "results/step17_decomposition_pythia-160m.json"),
          ("410M", "results/step17_decomposition_pythia-410m.json")]
TARGETS = [("pythia-1b", "Pythia-1B", "same family, one rung past ladder"),
           ("qwen2.5-0.5b", "Qwen2.5-0.5B", "different family, cross-family "
            "extrapolation -- low confidence on level")]
KS = [1, 5]
OUTDIR = "results/prereg"


def scale_levels(k):
    """Per-scale mean log eps_final at (k, C=0.1, eta=0.1), finite & >0."""
    out = {}
    for name, f in LADDER:
        rows = [r for r in json.load(open(f))["rows"]
                if not r.get("diverged") and r["k"] == k
                and abs(r.get("C", 0.1) - 0.1) < 1e-9
                and abs(r.get("lr", 0.1) - 0.1) < 1e-9
                and r["eps_final"] > 1e-6]
        if rows:
            out[name] = float(np.mean([math.log(r["eps_final"]) for r in rows]))
    return out


def main():
    now = datetime.datetime.now(datetime.UTC).isoformat()
    fc = dict(
        meta=dict(
            registered_utc=now,
            built_from=[f for _, f in LADDER],
            protocol="Arm G preregistration; forecast fixed before any "
                     "Pythia-1B/Qwen cell runs; never altered post-hoc",
            arm_G_cells="k in {1,5} x 2 seeds, C=0.1, eta=0.1, T=50, "
                        "per target model",
        ),
        structural={}, level={}, decision_criteria={})

    # ---- structural: the 1/2-law, scale-stable invariant ----
    for k in KS:
        h_law = 0.5 * math.log(T / k)
        fc["structural"][f"k{k}"] = dict(
            H_pred=round(h_law, 3),
            confirm_band=[round(h_law - 0.5, 3), round(h_law + 0.5, 3)],
            note="H = log(mu_allit/mu_final); decoder ladder gave H within "
                 "~0.3 nats of this at 70/160/410M")

    # ---- level: non-monotone; range from observed spread, widened ----
    for tkey, tname, tnote in TARGETS:
        cross_family = "qwen" in tkey
        for k in KS:
            lv = scale_levels(k)
            vals = list(lv.values())
            mean_log = float(np.mean(vals))
            spread = float(np.std(vals)) if len(vals) > 1 else 0.5
            # extrapolation widening: +/-2 sd, plus extra half-nat off-ladder,
            # plus a full nat more for cross-family (Qwen)
            half = 2 * spread + 0.5 + (1.0 if cross_family else 0.0)
            lo, hi = mean_log - half, mean_log + half
            fc["level"].setdefault(tkey, {})[f"k{k}"] = dict(
                observed_ladder_log_eps={n: round(v, 3) for n, v in lv.items()},
                point_log_eps=round(mean_log, 3),
                point_eps=round(math.exp(mean_log), 2),
                ci_log_eps=[round(lo, 3), round(hi, 3)],
                ci_eps=[round(math.exp(lo), 2), round(math.exp(hi), 2)],
                confidence="low (cross-family)" if cross_family else "moderate",
                note=f"{tname}: {tnote}; level explicitly NOT assumed "
                     "monotone in scale")

    # ---- decision criteria (fixed) ----
    fc["decision_criteria"] = dict(
        structural=dict(
            CONFIRM="per-arm fitted slope beta in [0.35, 0.65] AND both "
                    "k-point H within confirm_band",
            ATTENUATE="beta in [0.20, 0.35) (LSTM-like deformation) with H "
                      "monotone in log(T/k)",
            FAIL="beta < 0.20 or non-monotone H or H outside 2x the "
                 "confirm_band"),
        level=dict(
            CONFIRM="observed mean log eps_final inside ci_log_eps for that "
                    "target and k",
            MISS="outside the CI; report as a systematic level deviation "
                 "(itself a boundary-of-validity result), NOT re-fit"),
        overall="Arm G is UPSIDE, not foundation: the paper's claims stand on "
                "the framework + the in-hand 70/160/410M triptych regardless "
                "of this forecast's outcome. A clean non-monotone level with "
                "honest CIs is an acceptable and interesting result.")

    os.makedirs(OUTDIR, exist_ok=True)
    path = f"{OUTDIR}/armG_forecast.json"
    body = json.dumps(fc, indent=1, sort_keys=True)
    with open(path, "w") as f:
        f.write(body)
    digest = hashlib.sha256(body.encode()).hexdigest()
    with open(f"{OUTDIR}/armG_forecast.sha256", "w") as f:
        f.write(f"{digest}  armG_forecast.json\n")

    print(f"registered_utc: {now}")
    print(f"sha256: {digest}")
    print(f"wrote {path} (+ .sha256)\n")
    print("STRUCTURAL (1/2-law, all targets):")
    for k in KS:
        s = fc["structural"][f"k{k}"]
        print(f"  k={k}: H_pred={s['H_pred']}  confirm {s['confirm_band']}")
    print("LEVEL (eps_final point [CI]):")
    for tkey, tname, _ in TARGETS:
        for k in KS:
            l = fc["level"][tkey][f"k{k}"]
            print(f"  {tname:14} k={k}: {l['point_eps']} "
                  f"{l['ci_eps']}  ({l['confidence']})")


if __name__ == "__main__":
    main()
