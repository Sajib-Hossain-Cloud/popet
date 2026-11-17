"""Step 20 -- MECHANISM ARM (v3 §4.3, contribution C3).

First empirical audit of the deployed correlated-noise family in FL:
same EMNIST task, PINNED certificate, min-separation deterministic
participation (fixed adjacency, gate G7), across

    M in {fedavg, tree, bandmf, blt}  x  k in {10, 100}  x  C x E.

Per cell, one training run yields eps_final, eps_allit (unwhitened baseline),
eps_allit_wh (Thm 2 whitened matched filter), mu's, acc.

Estimands:
  (i)   H(k,T;M): does the hidden-state gap keep/deviate-from the 1/2-law
        under correlated noise (Prop 1 correction)?
  (ii)  P ordering stability across M (Prop 2 prediction).
  (iii) whitening payoff eps_allit_wh vs eps_allit per M (C4 on real runs).

ADJACENCY NOTE (G7): the certificate here is the min-separation single-
Gaussian bound (see harness/mechanisms.py) -- deliberately different from
step 17's Poisson-subsampled certificate. fedavg cells under BOTH appear only
in the appendix dual-report, never in one figure.

Gate G6: fedavg cells must qualitatively reproduce the step-17 pattern
(final < allit; gap ~ sqrt(T/k)) before BandMF/BLT cells are trusted.

Smoke:  RL_SEEDS=1 RL_MECHANISMS=fedavg RL_K_LEVELS=10 RL_E_GRID=1 \
        RL_C_GRID=0.1 python -m experiments.step20_mechanism_arm
Full:   RL_DEVICE=mps RL_SEEDS=3 python -m experiments.step20_mechanism_arm
"""

import itertools
import json
import os
import time

import numpy as np
import torch

from harness.accountant import theoretical_epsilon
from harness.auditor import audit_all_iterates, audit_final_model, make_canaries
from harness.auditor_whitened import audit_all_iterates_whitened
from harness.fedavg import DPFedAvg, evaluate_accuracy, pick_device
from harness.mechanisms import make_mechanism, minsep_schedule
from harness.tasks import make_image_benchmark_task

DEVICE = os.environ.get("RL_DEVICE", "auto")
DATASET = os.environ.get("RL_DATASET", "emnist")
SEEDS = list(range(int(os.environ.get("RL_SEEDS", "3"))))
T = int(os.environ.get("RL_ROUNDS", "100"))
TARGET_EPS = float(os.environ.get("RL_TARGET_EPS", "16"))
N_CLIENTS = int(os.environ.get("RL_N_CLIENTS", "200"))
SPC = int(os.environ.get("RL_SPC", "150"))
BAND = int(os.environ.get("RL_BAND", "10"))

MECHS = os.environ.get("RL_MECHANISMS", "fedavg,tree,bandmf,blt").split(",")
K_LEVELS = [int(x) for x in os.environ.get("RL_K_LEVELS", "10,100").split(",")]
E_GRID = [int(x) for x in os.environ.get("RL_E_GRID", "1,5").split(",")]
C_GRID = [float(x) for x in os.environ.get("RL_C_GRID", "0.1,1.0").split(",")]
ETA = float(os.environ.get("RL_ETA", "0.5"))
RESUME = os.environ.get("RL_RESUME", "1") == "1"

# decoder-LoRA arms (see step17 for the same conventions)
IS_DECODER = DATASET.startswith(("pythia", "qwen"))
MAX_ELEMENTS = int(os.environ.get("RL_MAX_ELEMENTS", "400"))
SEQ_LEN = int(os.environ.get("RL_SEQ_LEN", "48"))
ADAPTER_R = int(os.environ.get("RL_ADAPTER_R", "8"))
SERVER_LR = float(os.environ.get("RL_SERVER_LR",
                                 "0.2" if IS_DECODER else "1.0"))
SAVE_PROJ = os.environ.get("RL_SAVE_PROJ", "1" if IS_DECODER else "0") == "1"

N_INS = int(os.environ.get("RL_N_INS", "30"))
N_NULL = int(os.environ.get("RL_N_NULL", "60" if IS_DECODER else "300"))
DELTA = N_CLIENTS ** -1.1 if IS_DECODER else 1e-5
OUT = os.environ.get("RL_OUT", f"results/step20_mechanism_arm_{DATASET}.json")


def minsep_cohorts(k, T, N):
    """Deterministic min-separation participation: clients split into T//k
    groups; group g serves rounds {g, g + T//k, ...}. Every client
    participates in exactly k rounds, min-separated by T//k."""
    stride = T // k
    groups = [[] for _ in range(stride)]
    for i in range(N):
        groups[i % stride].append(i)
    return [groups[r % stride] for r in range(T)]


def load_existing():
    if RESUME and os.path.exists(OUT):
        with open(OUT) as f:
            saved = json.load(f)
        rows = saved.get("rows", [])
        done = {(r["mechanism"], r["k"], r["E"], r["C"], r["seed"]) for r in rows}
        print(f"[resume] {len(rows)} cells already in {OUT}\n")
        return rows, done
    return [], set()


def save(meta, rows):
    os.makedirs("results", exist_ok=True)
    tmp = OUT + ".tmp"
    with open(tmp, "w") as f:
        json.dump(dict(meta=meta, rows=rows), f, indent=1)
    os.replace(tmp, OUT)


def make_task():
    if IS_DECODER:
        from experiments.step17_decomposition import DECODER_HF_NAMES
        from harness.decoder_task import (get_decoder_model,
                                          make_decoder_task, reset_adapters)
        base = DATASET.replace("-reddit", "")
        source = "reddit" if DATASET.endswith("-reddit") else "stackoverflow"
        hf_name = DECODER_HF_NAMES[base]
        t = make_decoder_task(model_name=hf_name, num_clients=N_CLIENTS,
                              max_elements_per_client=MAX_ELEMENTS,
                              seq_len=SEQ_LEN, adapter_r=ADAPTER_R,
                              seed=0, amp=False, source=source)
        t._decoder_reset = (
            lambda s, _n=hf_name, _r=ADAPTER_R:
            reset_adapters(get_decoder_model(_n, adapter_r=_r, amp=False), s))
        return t
    return make_image_benchmark_task(dataset=DATASET, num_clients=N_CLIENTS,
                                     samples_per_client=SPC,
                                     non_iid_alpha=0.3, seed=0)


def main():
    dev = pick_device(DEVICE)
    task = make_task()
    configs = list(itertools.product(MECHS, K_LEVELS, E_GRID, C_GRID))
    print(f"device={dev}  {DATASET} dim={task.dim}  N={N_CLIENTS} T={T}  "
          f"target_eps={TARGET_EPS} (min-sep certificate)  delta={DELTA:.0e}")
    print(f"grid: M={MECHS} x k={K_LEVELS} x E={E_GRID} x C={C_GRID} "
          f"(eta={ETA}) = {len(configs)} configs x {len(SEEDS)} seeds\n")

    # mechanisms + per-(M, k) calibration (schedule-dependent sensitivity)
    mechs = {m: make_mechanism(m, T, band=BAND) for m in MECHS}
    calib = {}
    for m, k in itertools.product(MECHS, K_LEVELS):
        sched = minsep_schedule(k, T)
        z = mechs[m].calibrate(TARGET_EPS, sched, DELTA)
        gnorm = mechs[m].sensitivity_norm(sched)
        eps_chk = theoretical_epsilon(z / gnorm, 1.0, 1, DELTA, "rdp")
        calib[(m, k)] = (z, gnorm, eps_chk)
        print(f"  {m:>7} k={k:>4}: sens_norm={gnorm:7.3f}  z={z:.4f}  "
              f"-> eps={eps_chk:.3f}")
    print()

    meta = dict(dataset=DATASET, N=N_CLIENTS, T=T, target_eps=TARGET_EPS,
                delta=DELTA, device=str(dev), adjacency="minsep",
                band=BAND, mechanisms=MECHS, k_levels=K_LEVELS,
                E_grid=E_GRID, C_grid=C_GRID, eta=ETA,
                n_ins=N_INS, n_null=N_NULL,
                calibration={f"{m}_{k}": dict(z=v[0], sens_norm=v[1],
                                              eps=v[2])
                             for (m, k), v in calib.items()})
    rows, done = load_existing()

    print(f"{'mech':>7} {'k':>4} {'E':>3} {'C':>5} {'seed':>4} | "
          f"{'eps_final':>9} {'eps_allit':>9} {'eps_wh':>8} {'wh_gain':>7} "
          f"{'gap':>6} {'acc':>6} {'mins':>5}")
    t0 = time.time()
    for (m, k, E, C) in configs:
        z, gnorm, _ = calib[(m, k)]
        cohorts = minsep_cohorts(k, T, N_CLIENTS)
        cohort_size = len(cohorts[0])
        mech = mechs[m]
        for s in SEEDS:
            if (m, k, E, C, s) in done:
                continue
            tc = time.time()
            if hasattr(task, "_decoder_reset"):
                task._decoder_reset(s)
            # canaries on min-sep schedules (offsets cycle over canaries)
            can = make_canaries(dim=task.dim, num_inserted=N_INS,
                                num_held_out=N_NULL, clip_norm=C, rounds=T,
                                participate_rounds=k, seed=s)
            can.participation = [minsep_schedule(k, T, offset=j % (T // k))
                                 for j in range(N_INS)]
            noise = mech.noise_stream(T, task.dim, z, C, seed=10_000 * s + k)

            sim = DPFedAvg(task, clip_norm=C, noise_multiplier=z,
                           server_lr=SERVER_LR, local_lr=ETA, local_epochs=E,
                           device=DEVICE, seed=s)
            res = sim.run(rounds=T, cohort_size=cohort_size, add_noise=True,
                          record_iterates=True, canaries=can,
                          noise_stream=noise, fixed_cohorts=cohorts)

            if not torch.isfinite(res.final_vec).all():
                rows.append(dict(mechanism=m, k=k, E=E, C=C, lr=ETA, seed=s,
                                 cohort=cohort_size, z=z, sens_norm=gnorm,
                                 eps_theory=TARGET_EPS, eps_final=-1.0,
                                 mu_final=0.0, eps_allit=-1.0, mu_allit=0.0,
                                 eps_allit_wh=-1.0, mu_allit_wh=0.0,
                                 whitening_gain_mu=0.0, hidden_gap_mu=0.0,
                                 acc=None, diverged=True,
                                 elapsed_s=time.time() - tc))
                save(meta, rows)
                print(f"{m:>7} {k:>4} {E:>3} {C:>5.1f} {s:>4} | DIVERGED "
                      f"-- recorded, continuing", flush=True)
                continue

            if SAVE_PROJ and res.iterates is not None:
                # per-round ins+null projections: offset-robustness (Thm 3d)
                # and checkpoint-granularity analyses consume these
                os.makedirs("results/projections", exist_ok=True)
                dirs = can.directions.to(res.final_vec.device)
                held = can.held_out.to(res.final_vec.device)
                prev_it, P, Q = res.init_vec, [], []
                for it in res.iterates:
                    upd = it - prev_it
                    P.append(((dirs @ upd) / C).cpu().numpy())
                    Q.append(((held @ upd) / C).cpu().numpy())
                    prev_it = it
                np.savez(f"results/projections/mech_{DATASET}_{m}_T{T}_k{k}"
                         f"_C{C}_s{s}.npz",
                         ins=np.stack(P).astype(np.float32),
                         null=np.stack(Q).astype(np.float32))

            fin = audit_final_model(res, can, DELTA)
            allit = audit_all_iterates(res, can, DELTA)
            wh = audit_all_iterates_whitened(res, can, DELTA, mech.Sigma_unit)
            acc = evaluate_accuracy(task, res.final_vec, device=dev)

            gap = allit.mu_hat / max(fin.mu_hat, 1e-9)
            wh_gain = wh.mu_hat / max(allit.mu_hat, 1e-9)
            rows.append(dict(
                mechanism=m, k=k, E=E, C=C, lr=ETA, seed=s,
                cohort=cohort_size, z=z, sens_norm=gnorm,
                eps_theory=TARGET_EPS,
                eps_final=fin.epsilon_hat, mu_final=fin.mu_hat,
                eps_allit=allit.epsilon_hat, mu_allit=allit.mu_hat,
                eps_allit_wh=wh.epsilon_hat, mu_allit_wh=wh.mu_hat,
                whitening_gain_mu=wh_gain, hidden_gap_mu=gap,
                acc=acc, elapsed_s=time.time() - tc))
            save(meta, rows)
            print(f"{m:>7} {k:>4} {E:>3} {C:>5.1f} {s:>4} | "
                  f"{fin.epsilon_hat:>9.3f} {allit.epsilon_hat:>9.3f} "
                  f"{wh.epsilon_hat:>8.3f} {wh_gain:>7.2f} {gap:>6.2f} "
                  f"{acc:>6.3f} {(time.time() - tc) / 60:>5.1f}", flush=True)

    print(f"\n[{(time.time() - t0) / 60:.1f} min this session] "
          f"{len(rows)} cells -> {OUT}")

    # quick per-mechanism readout (step 18-style fit comes later)
    print("\nper-mechanism summary (mean over cells):")
    for m in MECHS:
        sub = [r for r in rows if r["mechanism"] == m]
        if not sub:
            continue
        for k in K_LEVELS:
            sk = [r for r in sub if r["k"] == k]
            if not sk:
                continue
            print(f"  {m:>7} k={k:>4}: gap={np.mean([r['hidden_gap_mu'] for r in sk]):5.2f} "
                  f"(sqrt(T/k)={np.sqrt(T / k):4.2f})  "
                  f"wh_gain={np.mean([r['whitening_gain_mu'] for r in sk]):5.2f}  "
                  f"eps_final={np.mean([r['eps_final'] for r in sk]):6.2f}")


if __name__ == "__main__":
    main()
