"""Step 47 -- FAULT INJECTION: which audit notices a broken mechanism?

An audit is used in practice to find implementation errors: if the audited
leakage exceeds what the certificate allows, something is wrong. An audit
that reads far below the certificate on a CORRECT mechanism has no room to
do this: a fault that doubles the leakage still leaves its reading below
the certificate. This experiment measures that directly.

Each mechanism is calibrated to the same certificate (mu_cert = 2.75) and
then broken in one of three ways. The operator still claims mu_cert.

  noise   x f : the noise actually added is f times the calibrated noise
                (f = 0.8, 0.5, 0.25).  True deflection: mu_cert / f.
  unclipped x c : the canary's update is not clipped; it submits norm c*C
                (c = 1.25, 1.5, 2).    True deflection: c * mu_cert.
  gap g       : participation limits are not enforced; the canary returns
                every g rounds instead of every T/k (g = 9: one round early,
                as in the gap drift reported for a production system;
                g = 5: twice as often, 2k participations).
                True deflection: ||G e_S'|| / z for the actual rounds S'.

For every run the five audits of harness.audit_stats are computed from the
same released updates, each with a one-sided 95% bootstrap lower bound on
its deflection. An audit DETECTS the fault when that lower bound exceeds
mu_cert (the certificate rule). Condition `none` (no fault) measures the
false-alarm rate. experiments/step48_v2_tables.py additionally evaluates
the prediction rule from the stored bounds: the lower bound exceeds the
audit's own predicted reading for the correct mechanism.

Smoke:  RL_SEEDS=1 RL_MECHANISMS=tree RL_FAULTS=none,noise0.5 \
        python -m experiments.step47_fault_injection
Full:   RL_DEVICE=mps RL_SEEDS=5 python -m experiments.step47_fault_injection
"""

import itertools
import json
import os
import time

import numpy as np
import torch

from harness.audit_stats import AUDITS, mu_with_bound, per_direction_stats
from harness.auditor import make_canaries
from harness.fedavg import DPFedAvg, pick_device
from harness.mechanisms import make_mechanism, minsep_schedule
from harness.tasks import make_image_benchmark_task

DEVICE = os.environ.get("RL_DEVICE", "auto")
SEEDS = list(range(int(os.environ.get("RL_SEEDS", "5"))))
T = int(os.environ.get("RL_ROUNDS", "100"))
K = int(os.environ.get("RL_K", "10"))
TARGET_EPS = float(os.environ.get("RL_TARGET_EPS", "16"))
N_CLIENTS = int(os.environ.get("RL_N_CLIENTS", "200"))
MECHS = os.environ.get("RL_MECHANISMS", "fedavg,tree,bandmf_opt,blt_opt").split(",")
FAULTS = os.environ.get(
    "RL_FAULTS",
    "none,noise0.8,noise0.5,noise0.25,unclipped1.25,unclipped1.5,unclipped2,gap9,gap5").split(",")
E = int(os.environ.get("RL_E", "1"))
C = float(os.environ.get("RL_C", "0.1"))
ETA = float(os.environ.get("RL_ETA", "0.5"))
N_INS = int(os.environ.get("RL_N_INS", "30"))
N_NULL = int(os.environ.get("RL_N_NULL", "300"))
DELTA = 1e-5
RESUME = os.environ.get("RL_RESUME", "1") == "1"
OUT = os.environ.get("RL_OUT", "results/step47_fault_injection_emnist.json")
PROJ_DIR = os.environ.get("RL_PROJ_DIR", "results/projections_step47")
CANARY_SEED = 10_007


def minsep_cohorts(k, T, N):
    stride = T // k
    groups = [[] for _ in range(stride)]
    for i in range(N):
        groups[i % stride].append(i)
    return [groups[r % stride] for r in range(T)]


def fault_spec(name):
    """-> (noise_factor, canary_scale, gap or None)"""
    if name == "none":
        return 1.0, 1.0, None
    if name.startswith("noise"):
        return float(name[5:]), 1.0, None
    if name.startswith("unclipped"):
        return 1.0, float(name[9:]), None
    if name.startswith("gap"):
        return 1.0, 1.0, int(name[3:])
    raise ValueError(name)


def schedule(j, gap):
    """Participation rounds of canary j: declared min-separation schedule, or
    (fault) every `gap` rounds starting at the canary's offset."""
    if gap is None:
        return minsep_schedule(K, T, offset=j % (T // K))
    return list(range(j % gap, T, gap))


def save(payload):
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    tmp = OUT + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=1)
    os.replace(tmp, OUT)


def main():
    dev = pick_device(DEVICE)
    torch.manual_seed(0)
    task = make_image_benchmark_task(dataset="emnist", num_clients=N_CLIENTS,
                                     samples_per_client=150,
                                     non_iid_alpha=0.3, seed=0)
    cohorts = minsep_cohorts(K, T, N_CLIENTS)
    mechs, zs = {}, {}
    for m in MECHS:
        mechs[m] = make_mechanism(m, T)
        zs[m] = mechs[m].calibrate(TARGET_EPS, minsep_schedule(K, T), DELTA)
    g0 = mechs[MECHS[0]].sensitivity_norm(minsep_schedule(K, T))
    mu_cert = g0 / zs[MECHS[0]]
    meta = dict(T=T, k=K, N=N_CLIENTS, E=E, C=C, eta=ETA, delta=DELTA,
                mu_cert=mu_cert, mechanisms=MECHS, faults=FAULTS, n_ins=N_INS,
                n_null=N_NULL, device=str(dev), torch=torch.__version__,
                detection_rule="one-sided 95% bootstrap lower bound > mu_cert")
    rows, done = [], set()
    if RESUME and os.path.exists(OUT):
        with open(OUT) as f:
            rows = json.load(f).get("rows", [])
        done = {(r["mechanism"], r["fault"], r["seed"]) for r in rows}
        print(f"[resume] {len(rows)} rows already in {OUT}")
    payload = dict(meta=meta, rows=rows)

    print(f"device={dev} EMNIST T={T} k={K} certificate mu={mu_cert:.3f}")
    print(f"{'mech':>10} {'fault':>12} {'s':>2} | {'true':>5} | "
          + " ".join(f"{a:>14}" for a in AUDITS) + "   (mu-hat [lower], * = detects)")
    t0 = time.time()
    for m, fault, s in itertools.product(MECHS, FAULTS, SEEDS):
        if (m, fault, s) in done:
            continue
        nf, cs, gap = fault_spec(fault)
        mech, z = mechs[m], zs[m]
        can = make_canaries(dim=task.dim, num_inserted=N_INS, num_held_out=N_NULL,
                            clip_norm=C, rounds=T, participate_rounds=K,
                            seed=CANARY_SEED + s)
        can.participation = [schedule(j, gap) for j in range(N_INS)]
        if cs != 1.0:                      # the server fails to clip the canary
            can.directions = can.directions * cs
        noise = mech.noise_stream(T, task.dim, z, C, seed=10_000 * s + K) * nf
        sim = DPFedAvg(task, clip_norm=C, noise_multiplier=z, server_lr=1.0,
                       local_lr=ETA, local_epochs=E, device=DEVICE, seed=s)
        tc = time.time()
        res = sim.run(rounds=T, cohort_size=len(cohorts[0]), add_noise=True,
                      record_iterates=True, canaries=can, noise_stream=noise,
                      fixed_cohorts=cohorts)
        if not torch.isfinite(res.final_vec).all():
            rows.append(dict(mechanism=m, fault=fault, seed=s, diverged=True))
            save(payload)
            continue

        def unit(x):
            return x / (x.norm(dim=-1, keepdim=True) + 1e-12)
        dirs = unit(can.directions.to(res.final_vec.device))
        held = unit(can.held_out.to(res.final_vec.device))
        prev, P, Q, U = res.init_vec, [], [], []
        for it in res.iterates:
            upd = it - prev
            prev = it
            P.append((dirs @ upd).cpu().numpy())
            Q.append((held @ upd).cpu().numpy())
            U.append(float(upd.norm()))
        P, Q, U = np.stack(P), np.stack(Q), np.array(U)
        os.makedirs(PROJ_DIR, exist_ok=True)
        tag = f"{m}_{fault}_s{s}"
        np.savez(os.path.join(PROJ_DIR, tag + ".npz"), ins=P.astype(np.float32),
                 null=Q.astype(np.float32), update_norm=U,
                 schedules=json.dumps(can.participation))

        # true deflection of the faulty mechanism for these canaries
        true_mu = float(np.mean([mech.sensitivity_norm(S) for S in can.participation])
                        * cs / (z * nf))
        st = per_direction_stats(P, Q, U, can.participation, mech.Sigma_unit)
        rng = np.random.default_rng(1000 + s)
        row = dict(mechanism=m, fault=fault, seed=s, z=z, true_mu=true_mu,
                   proj=tag, elapsed_s=time.time() - tc)
        cells = []
        for a in AUDITS:
            est, lo = mu_with_bound(*st[a], rng)
            row[f"mu_{a}"], row[f"lo_{a}"] = est, lo
            row[f"detect_{a}"] = bool(lo > mu_cert)
            cells.append(f"{est:5.2f} [{lo:5.2f}]{'*' if lo > mu_cert else ' '}")
        rows.append(row)
        save(payload)
        print(f"{m:>10} {fault:>12} {s:>2} | {true_mu:>5.2f} | " + " ".join(
            f"{c:>14}" for c in cells), flush=True)

    print(f"\n[{(time.time() - t0) / 60:.1f} min] {len(rows)} rows -> {OUT}\n")
    print("detection rate over seeds (share of runs whose lower bound exceeds "
          f"the certificate mu={mu_cert:.2f})")
    print(f"{'mech':>10} {'fault':>12} {'true':>5} | "
          + " ".join(f"{a:>9}" for a in AUDITS))
    summary = []
    for m, fault in itertools.product(MECHS, FAULTS):
        rs = [r for r in rows if r["mechanism"] == m and r["fault"] == fault
              and not r.get("diverged")]
        if not rs:
            continue
        cell = dict(mechanism=m, fault=fault, n=len(rs),
                    true_mu=float(np.mean([r["true_mu"] for r in rs])))
        for a in AUDITS:
            cell[f"detect_{a}"] = float(np.mean([r[f"detect_{a}"] for r in rs]))
            cell[f"mu_{a}"] = float(np.mean([r[f"mu_{a}"] for r in rs]))
        summary.append(cell)
        print(f"{m:>10} {fault:>12} {cell['true_mu']:>5.2f} | " + " ".join(
            f"{cell[f'detect_{a}']:>4.2f}({cell[f'mu_{a}']:>4.2f})" for a in AUDITS))
    payload["summary"] = summary
    save(payload)


if __name__ == "__main__":
    main()
