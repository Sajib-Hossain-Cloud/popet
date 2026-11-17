"""Step 29: PRE-REGISTERED Arm H forecast (GPT-2-124M, third decoder family).

Same construction as step26 (Arm G), frozen before any GPT-2 cell runs:
structural prediction from the 1/2-law; level range from the Pythia
70M/160M/410M ladder with the identical cross-family widening used for Qwen.
Honest disclosure recorded in meta: at registration time the Arm G (Qwen)
outcome IS known (both axes confirmed). The recipe below is nonetheless
byte-identical to step26's: nothing about it was tuned on the Qwen result,
and the ladder inputs are the same files. GPT-2 runs on a separate machine
(M4) via run_armH_gpt2_m4.sh, which verifies this file's sha256 before
training.
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
TARGET = ("gpt2-124m", "GPT-2-124M",
          "different family AND different architecture generation (learned "
          "positional embeddings, pre-LN GPT-2 block, fused Conv1D QKV); "
          "cross-family extrapolation -- low confidence on level")
KS = [1, 5]
OUTDIR = "results/prereg"


def scale_levels(k):
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
    tkey, tname, tnote = TARGET
    fc = dict(
        meta=dict(
            registered_utc=now,
            built_from=[f for _, f in LADDER],
            protocol="Arm H preregistration; forecast fixed before any "
                     "GPT-2-124M cell runs; never altered post-hoc; recipe "
                     "byte-identical to step26 (Arm G)",
            disclosure="At registration the Arm G (Qwen2.5-0.5B) outcome is "
                       "known: structural and level CONFIRM. The Arm H "
                       "recipe, bands, and criteria are copied from step26 "
                       "unchanged, not tuned on that outcome.",
            arm_H_cells="k in {1,5} x 2 seeds, C=0.1, eta=0.1, T=50, N=500, "
                        "E=1, LoRA r=8, seq_len=48, StackOverflow corpus, "
                        "server damping 0.2, fp32, device=auto (MPS on M4)",
        ),
        structural={}, level={}, decision_criteria={})

    for k in KS:
        h_law = 0.5 * math.log(T / k)
        fc["structural"][f"k{k}"] = dict(
            H_pred=round(h_law, 3),
            confirm_band=[round(h_law - 0.5, 3), round(h_law + 0.5, 3)],
            note="H = log(mu_allit/mu_final); decoder ladder gave H within "
                 "~0.3 nats of this at 70/160/410M; Qwen landed within 0.12")

    for k in KS:
        lv = scale_levels(k)
        vals = list(lv.values())
        mean_log = float(np.mean(vals))
        spread = float(np.std(vals)) if len(vals) > 1 else 0.5
        # identical widening to step26's cross-family case (Qwen):
        # +/- 2 sd + 0.5 off-ladder + 1.0 cross-family
        half = 2 * spread + 0.5 + 1.0
        lo, hi = mean_log - half, mean_log + half
        fc["level"].setdefault(tkey, {})[f"k{k}"] = dict(
            observed_ladder_log_eps={n: round(v, 3) for n, v in lv.items()},
            point_log_eps=round(mean_log, 3),
            point_eps=round(math.exp(mean_log), 2),
            ci_log_eps=[round(lo, 3), round(hi, 3)],
            ci_eps=[round(math.exp(lo), 2), round(math.exp(hi), 2)],
            confidence="low (cross-family)",
            note=f"{tname}: {tnote}; level explicitly NOT assumed monotone "
                 "in scale")

    fc["decision_criteria"] = dict(
        structural=dict(
            CONFIRM="per-arm fitted slope beta in [0.35, 0.65] AND both "
                    "k-point H within confirm_band",
            ATTENUATE="beta in [0.20, 0.35) (LSTM-like deformation) with H "
                      "monotone in log(T/k)",
            FAIL="beta < 0.20 or non-monotone H or H outside 2x the "
                 "confirm_band"),
        level=dict(
            CONFIRM="observed mean log eps_final inside ci_log_eps for that k",
            MISS="outside the CI; report as a systematic level deviation "
                 "(itself a boundary-of-validity result), NOT re-fit"),
        overall="Arm H is UPSIDE, not foundation: with Arm G confirmed the "
                "pre-registration record is 1/1; this arm makes it 2/2 or "
                "an honestly reported miss. The paper's claims stand "
                "regardless of this forecast's outcome.")

    os.makedirs(OUTDIR, exist_ok=True)
    path = f"{OUTDIR}/armH_gpt2_forecast.json"
    body = json.dumps(fc, indent=1, sort_keys=True)
    with open(path, "w") as f:
        f.write(body)
    digest = hashlib.sha256(body.encode()).hexdigest()
    with open(f"{OUTDIR}/armH_gpt2_forecast.sha256", "w") as f:
        f.write(f"{digest}  armH_gpt2_forecast.json\n")

    print(f"registered_utc: {now}")
    print(f"sha256: {digest}")
    print(f"wrote {path} (+ .sha256)\n")
    print("STRUCTURAL (1/2-law):")
    for k in KS:
        s = fc["structural"][f"k{k}"]
        print(f"  k={k}: H_pred={s['H_pred']}  confirm {s['confirm_band']}")
    print("LEVEL (eps_final point [CI]):")
    for k in KS:
        l = fc["level"][tkey][f"k{k}"]
        print(f"  {tname} k={k}: {l['point_eps']} {l['ci_eps']} "
              f"({l['confidence']})")


if __name__ == "__main__":
    main()
