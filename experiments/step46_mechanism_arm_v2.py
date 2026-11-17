"""Step 46 -- MECHANISM ARM, v2: four all-iterates statistics, five seeds.

Same protocol as step 20 (EMNIST, min-separation participation, every
mechanism calibrated to the same single-event certificate), with three
changes that the refocused paper needs:

  1. The PUBLISHED all-iterates audit is added. Andrew et al. (ICLR'24) use
     the maximum over rounds of the cosine between the canary and that
     round's model delta. Step 20 reported only the participation-sum
     statistic, which is NOT theirs. Each run now yields
         final   : projection of theta_T - theta_0            (final model)
         max     : max over rounds of the per-round cosine     (Andrew et al.)
         sum     : sum over the canary's participation rounds  (matched filter
                   under independent noise)
         whitened: matched filter after whitening by Sigma     (this work)
  2. Five seeds per cell instead of three, and the client split is seeded
     (harness.tasks draws it from the global RNG).
  3. Per-round projections of inserted and held-out directions are stored
     for every run, so the null-distribution checks can be run on this arm.

Smoke:  RL_SEEDS=1 RL_MECHANISMS=tree RL_K_LEVELS=10 RL_E_GRID=1 RL_C_GRID=0.1 \
        python -m experiments.step46_mechanism_arm_v2
Full:   RL_DEVICE=mps RL_SEEDS=5 python -m experiments.step46_mechanism_arm_v2
"""

import itertools
import json
import os
import time

import numpy as np
import torch

from harness.accountant import theoretical_epsilon
from harness.auditor import (audit_all_iterates, audit_all_iterates_max,
                             audit_final_model, gdp_epsilon, make_canaries)
from harness.auditor_whitened import audit_all_iterates_whitened
from harness.fedavg import DPFedAvg, evaluate_accuracy, pick_device
from harness.mechanisms import make_mechanism, minsep_schedule
from harness.tasks import make_image_benchmark_task, make_stackoverflow_task

DEVICE = os.environ.get("RL_DEVICE", "auto")
DATASET = os.environ.get("RL_DATASET", "emnist")
SEEDS = list(range(int(os.environ.get("RL_SEEDS", "5"))))
T = int(os.environ.get("RL_ROUNDS", "100"))
TARGET_EPS = float(os.environ.get("RL_TARGET_EPS", "16"))
IS_SO = DATASET == "stackoverflow"
N_CLIENTS = int(os.environ.get("RL_N_CLIENTS", "500" if IS_SO else "200"))
SPC = int(os.environ.get("RL_SPC", "150"))
BAND = int(os.environ.get("RL_BAND", "10"))
MECHS = os.environ.get("RL_MECHANISMS", "fedavg,tree,bandmf,blt").split(",")
K_LEVELS = [int(x) for x in os.environ.get("RL_K_LEVELS", "10,100").split(",")]
E_GRID = [int(x) for x in os.environ.get("RL_E_GRID", "1,5").split(",")]
C_GRID = [float(x) for x in os.environ.get("RL_C_GRID", "0.1,1.0").split(",")]
ETA = float(os.environ.get("RL_ETA", "0.5"))
N_INS = int(os.environ.get("RL_N_INS", "30"))
N_NULL = int(os.environ.get("RL_N_NULL", "60" if IS_SO else "300"))
DELTA = N_CLIENTS ** -1.1 if IS_SO else 1e-5
RESUME = os.environ.get("RL_RESUME", "1") == "1"
# Stop after this many new runs and exit with code 3 ("more to do"). The LSTM
# leaks MPS memory across runs (run 3 of a single process ran out of memory on
# a 16 GB machine), so the LSTM arm is driven one run per process:
#   until RL_MAX_RUNS=1 ... python -m experiments.step46_mechanism_arm_v2; [ $? -ne 3 ]; do :; done
MAX_RUNS = int(os.environ.get("RL_MAX_RUNS", "0"))
OUT = os.environ.get("RL_OUT", f"results/step46_mechanism_arm_{DATASET}.json")
PROJ_DIR = os.environ.get("RL_PROJ_DIR", "results/projections_step46")
CANARY_SEED = 10_007


def minsep_cohorts(k, T, N):
    stride = T // k
    groups = [[] for _ in range(stride)]
    for i in range(N):
        groups[i % stride].append(i)
    return [groups[r % stride] for r in range(T)]


def noise_stream(mech, d, z, C, seed, chunk=200_000):
    """Update-stream noise, already scaled by z*C. Small models use the
    harness generator unchanged (identical draws to step 20). Large models
    are generated in float32 coordinate chunks, because the float64
    (noise_dim x d) array of the harness generator does not fit in memory at
    d ~ 4M."""
    if d <= 1_000_000:
        return mech.noise_stream(T, d, z, C, seed)
    g = np.random.default_rng(seed)
    M = mech.M_noise.astype(np.float32)
    out = torch.empty((M.shape[0], d), dtype=torch.float32)
    for a in range(0, d, chunk):
        b = min(d, a + chunk)
        Z = g.standard_normal((M.shape[1], b - a), dtype=np.float32)
        out[:, a:b] = torch.from_numpy(z * C * (M @ Z))
    return out


def save(payload):
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    tmp = OUT + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=1)
    os.replace(tmp, OUT)


def main():
    dev = pick_device(DEVICE)
    torch.manual_seed(0)        # the client split uses the global RNG
    if IS_SO:      # next-word LSTM on real StackOverflow users (d ~ 4.1M)
        task = make_stackoverflow_task(num_clients=N_CLIENTS, seed=0)
    else:
        task = make_image_benchmark_task(dataset=DATASET, num_clients=N_CLIENTS,
                                         samples_per_client=SPC,
                                         non_iid_alpha=0.3, seed=0)
    mechs = {m: make_mechanism(m, T, band=BAND) for m in MECHS}
    calib = {}
    for m, k in itertools.product(MECHS, K_LEVELS):
        sched = minsep_schedule(k, T)
        z = mechs[m].calibrate(TARGET_EPS, sched, DELTA)
        g = mechs[m].sensitivity_norm(sched)
        calib[(m, k)] = (z, g, theoretical_epsilon(z / g, 1.0, 1, DELTA, "rdp"))
    mu_cert = calib[(MECHS[0], K_LEVELS[0])][1] / calib[(MECHS[0], K_LEVELS[0])][0]
    meta = dict(dataset=DATASET, N=N_CLIENTS, T=T, target_eps=TARGET_EPS,
                delta=DELTA, device=str(dev), adjacency="minsep", band=BAND,
                mechanisms=MECHS, k_levels=K_LEVELS, E_grid=E_GRID,
                C_grid=C_GRID, eta=ETA, n_ins=N_INS, n_null=N_NULL,
                mu_cert=mu_cert, eps_cert=gdp_epsilon(mu_cert, DELTA),
                torch=torch.__version__,
                calibration={f"{m}_{k}": dict(z=v[0], sens_norm=v[1], eps=v[2])
                             for (m, k), v in calib.items()})
    rows, done = [], set()
    if RESUME and os.path.exists(OUT):
        with open(OUT) as f:
            rows = json.load(f).get("rows", [])
        done = {(r["mechanism"], r["k"], r["E"], r["C"], r["seed"]) for r in rows}
        print(f"[resume] {len(rows)} rows already in {OUT}")
    payload = dict(meta=meta, rows=rows)

    print(f"device={dev} {DATASET} dim={task.dim} T={T} certificate "
          f"mu={mu_cert:.3f} eps={meta['eps_cert']:.2f}")
    print(f"{'mech':>7} {'k':>4} {'E':>2} {'C':>4} {'s':>2} | mu: "
          f"{'final':>6} {'max':>6} {'sum':>6} {'white':>6} | eps: "
          f"{'final':>6} {'max':>6} {'sum':>6} {'white':>6} | {'acc':>5} {'min':>4}")
    t0 = time.time()
    new_runs = 0
    for (m, k, E, C) in itertools.product(MECHS, K_LEVELS, E_GRID, C_GRID):
        z, gnorm, _ = calib[(m, k)]
        cohorts = minsep_cohorts(k, T, N_CLIENTS)
        for s in SEEDS:
            if (m, k, E, C, s) in done:
                continue
            tc = time.time()
            can = make_canaries(dim=task.dim, num_inserted=N_INS,
                                num_held_out=N_NULL, clip_norm=C, rounds=T,
                                participate_rounds=k, seed=CANARY_SEED + s)
            can.participation = [minsep_schedule(k, T, offset=j % (T // k))
                                 for j in range(N_INS)]
            noise = noise_stream(mechs[m], task.dim, z, C, seed=10_000 * s + k)
            sim = DPFedAvg(task, clip_norm=C, noise_multiplier=z, server_lr=1.0,
                           local_lr=ETA, local_epochs=E, device=DEVICE, seed=s)
            res = sim.run(rounds=T, cohort_size=len(cohorts[0]), add_noise=True,
                          record_iterates=True, canaries=can,
                          noise_stream=noise, fixed_cohorts=cohorts)
            base = dict(mechanism=m, k=k, E=E, C=C, lr=ETA, seed=s,
                        cohort=len(cohorts[0]), z=z, sens_norm=gnorm)
            if not torch.isfinite(res.final_vec).all():
                rows.append(dict(base, diverged=True, elapsed_s=time.time() - tc))
                save(payload)
                print(f"{m:>7} {k:>4} {E:>2} {C:>4} {s:>2} | DIVERGED", flush=True)
                continue

            # per-round projections (in units of the clip norm) and update norms
            os.makedirs(PROJ_DIR, exist_ok=True)
            dirs = can.directions.to(res.final_vec.device)
            held = can.held_out.to(res.final_vec.device)
            prev, P, Q, U = res.init_vec, [], [], []
            for it in res.iterates:
                upd = it - prev
                prev = it
                P.append(((dirs @ upd) / C).cpu().numpy())
                Q.append(((held @ upd) / C).cpu().numpy())
                U.append(float(upd.norm()))
            tag = f"{DATASET}_{m}_k{k}_E{E}_C{C}_s{s}"
            np.savez(os.path.join(PROJ_DIR, tag + ".npz"),
                     ins=np.stack(P).astype(np.float32),
                     null=np.stack(Q).astype(np.float32),
                     update_norm=np.array(U),
                     schedules=json.dumps(can.participation))

            fin = audit_final_model(res, can, DELTA)
            mx = audit_all_iterates_max(res, can, DELTA)
            sm = audit_all_iterates(res, can, DELTA)
            wh = audit_all_iterates_whitened(res, can, DELTA, mechs[m].Sigma_unit)
            del noise, dirs, held
            try:
                acc = evaluate_accuracy(task, res.final_vec, device=dev)
            except RuntimeError as err:       # MPS out of memory: sanity metric on CPU
                print(f"[accuracy on cpu] {str(err)[:80]}", flush=True)
                if dev.type == "mps":
                    torch.mps.empty_cache()
                acc = evaluate_accuracy(task, res.final_vec.cpu(),
                                        device=torch.device("cpu"))
            rows.append(dict(
                base, mu_final=fin.mu_hat, mu_max=mx.mu_hat, mu_sum=sm.mu_hat,
                mu_whitened=wh.mu_hat, eps_final=fin.epsilon_hat,
                eps_max=mx.epsilon_hat, eps_sum=sm.epsilon_hat,
                eps_whitened=wh.epsilon_hat, acc=acc, proj=tag,
                elapsed_s=time.time() - tc))
            save(payload)
            print(f"{m:>7} {k:>4} {E:>2} {C:>4} {s:>2} |     "
                  f"{fin.mu_hat:>6.2f} {mx.mu_hat:>6.2f} {sm.mu_hat:>6.2f} "
                  f"{wh.mu_hat:>6.2f} |      {fin.epsilon_hat:>6.2f} "
                  f"{mx.epsilon_hat:>6.2f} {sm.epsilon_hat:>6.2f} "
                  f"{wh.epsilon_hat:>6.2f} | {acc:>5.3f} "
                  f"{(time.time() - tc) / 60:>4.1f}", flush=True)
            new_runs += 1
            if MAX_RUNS and new_runs >= MAX_RUNS:
                total = len(MECHS) * len(K_LEVELS) * len(E_GRID) * len(C_GRID) * len(SEEDS)
                if len(rows) < total:
                    print(f"[max runs] {len(rows)}/{total} done; exit 3", flush=True)
                    raise SystemExit(3)

    print(f"\n[{(time.time() - t0) / 60:.1f} min] {len(rows)} rows -> {OUT}\n")
    print("means per cell (mu; share of runs in which the all-iterates "
          "statistic reads BELOW the final-model statistic)")
    print(f"{'mech':>7} {'k':>4} {'n':>3} | {'final':>6} {'max':>6} {'sum':>6} "
          f"{'white':>6} | below final: {'max':>5} {'sum':>5} {'white':>5}")
    for m, k in itertools.product(MECHS, K_LEVELS):
        rs = [r for r in rows if r["mechanism"] == m and r["k"] == k
              and not r.get("diverged")]
        if not rs:
            continue
        g = lambda f: np.mean([r[f] for r in rs])
        b = lambda f: np.mean([r[f] < r["mu_final"] for r in rs])
        print(f"{m:>7} {k:>4} {len(rs):>3} | {g('mu_final'):>6.2f} "
              f"{g('mu_max'):>6.2f} {g('mu_sum'):>6.2f} {g('mu_whitened'):>6.2f} | "
              f"            {b('mu_max'):>5.2f} {b('mu_sum'):>5.2f} "
              f"{b('mu_whitened'):>5.2f}")


if __name__ == "__main__":
    main()
