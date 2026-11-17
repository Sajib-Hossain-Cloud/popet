"""Step 45 -- COUPLED-RUN RETENTION PROBE: a direct test of assumption (A1).

Proposition 1 (the H law) rests on (A1): each participation adds a mean shift
C/m along the canary direction, and that shift is still there, unchanged, in
the final model. Then mu_final = k / (z sqrt(T)) whatever the round the
canary participated in. If training instead contracts (or amplifies) a
canary's contribution after it is injected, a canary placed late reads
differently from one placed early and the law acquires a timing term.

This is measured almost noise-free with two COUPLED runs: identical seed,
cohorts and noise draws, one with canaries and one without. Every random
draw is shared, so

    D_s = theta_s(with canaries) - theta_s(without)

contains no direct noise: only the canaries' injections and how honest
training responds to them. With one k = 1 canary direction v_i injected at
round t_i, the retention

    R[s, i] = <v_i, D_s> * m / (server_lr * C)

equals 1 for s >= t_i under (A1) and 0 before. (The same estimate from one
run would carry the full noise z sqrt(T) per canary; the coupled difference
leaves only cross-canary crosstalk of order sqrt(n_canaries / d).)

Reported per cell (E, C) and seed:
  * retention by AGE a = s - t_i (how much of an injection survives a rounds);
  * final-model retention by INSERTION ROUND, and the implied mu_final by
    decile of insertion round (retention / final-model null scale);
  * the share of ||D_T||^2 lying along the canary directions, i.e. how much
    of the canaries' effect on the final model is outside the directions the
    audit projects on.

D is not a "mean shift": it is the paired counterfactual displacement under
common randomness, and for a nonlinear model it contains canary-noise
interaction (a canary changes the state, which changes later honest
updates). That is exactly the part (A1) assumes away.

Smoke:  RL_SEEDS=1 RL_E_GRID=1 RL_C_GRID=0.1 python -m experiments.step45_retention
Full:   RL_SEEDS=5 python -m experiments.step45_retention
"""

import itertools
import json
import os
import time

import numpy as np
import torch

from harness.accountant import calibrate_noise_multiplier
from harness.auditor import make_canaries
from harness.fedavg import DPFedAvg, evaluate_accuracy, pick_device
from harness.tasks import make_image_benchmark_task

DEVICE = os.environ.get("RL_DEVICE", "cpu")   # CPU: deterministic by contract
DATASET = os.environ.get("RL_DATASET", "emnist")
SEEDS = list(range(int(os.environ.get("RL_SEEDS", "5"))))
T = int(os.environ.get("RL_ROUNDS", "100"))
TARGET_EPS = float(os.environ.get("RL_TARGET_EPS", "16"))
N_CLIENTS = int(os.environ.get("RL_N_CLIENTS", "200"))
SPC = int(os.environ.get("RL_SPC", "150"))
K_HONEST = int(os.environ.get("RL_K_HONEST", "10"))
PER_ROUND = int(os.environ.get("RL_PER_ROUND", "3"))      # canaries per round
E_GRID = [int(x) for x in os.environ.get("RL_E_GRID", "1,5").split(",")]
C_GRID = [float(x) for x in os.environ.get("RL_C_GRID", "0.1,1.0").split(",")]
ETA = float(os.environ.get("RL_ETA", "0.5"))
N_NULL = int(os.environ.get("RL_N_NULL", "300"))
DELTA = 1e-5
RESUME = os.environ.get("RL_RESUME", "1") == "1"
OUT = os.environ.get("RL_OUT", f"results/step45_retention_{DATASET}.json")
NPZ_DIR = os.environ.get("RL_NPZ_DIR", "results/retention")
CANARY_SEED = 10_007      # never the simulator's seed (see step 44)
AGES = [0, 1, 2, 5, 10, 25, 50, 75, 99]


def minsep_cohorts(k, T, N):
    stride = T // k
    groups = [[] for _ in range(stride)]
    for i in range(N):
        groups[i % stride].append(i)
    return [groups[r % stride] for r in range(T)]


def save(payload):
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    tmp = OUT + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=1)
    os.replace(tmp, OUT)


def run_cell(task, dev, z, E, C, seed, cohorts):
    m = len(cohorts[0])
    n_ins = PER_ROUND * T
    can = make_canaries(dim=task.dim, num_inserted=n_ins, num_held_out=N_NULL,
                        clip_norm=C, rounds=T, participate_rounds=1,
                        seed=CANARY_SEED + seed)
    t_ins = np.repeat(np.arange(T), PER_ROUND)
    can.participation = [[int(t)] for t in t_ins]

    def sim():
        return DPFedAvg(task, clip_norm=C, noise_multiplier=z, server_lr=1.0,
                        local_lr=ETA, local_epochs=E, device=DEVICE, seed=seed)

    tc = time.time()
    base = sim().run(rounds=T, cohort_size=m, record_iterates=True,
                     fixed_cohorts=cohorts)
    with_c = sim().run(rounds=T, cohort_size=m, record_iterates=True,
                       canaries=can, fixed_cohorts=cohorts)
    if not (torch.isfinite(base.final_vec).all()
            and torch.isfinite(with_c.final_vec).all()):
        return dict(E=E, C=C, seed=seed, diverged=True)
    if not torch.equal(base.init_vec, with_c.init_vec):
        raise RuntimeError("coupled runs do not share theta_0")

    def unit(x):
        return x / (x.norm(dim=-1, keepdim=True) + 1e-12)

    d_ = base.final_vec.device
    V = unit(can.directions.to(d_))                 # (n_ins, d)
    W = unit(can.held_out.to(d_))                   # (n_null, d)
    scale = m / (1.0 * C)                           # server_lr = 1

    R = np.empty((T, n_ins))                        # retention matrix
    Dnorm = np.empty(T)
    for s in range(T):
        D = with_c.iterates[s] - base.iterates[s]
        R[s] = (V @ D).cpu().double().numpy() * scale
        Dnorm[s] = float(D.norm()) * scale
    D_T = with_c.final_vec - base.final_vec
    along = float(((V @ D_T) ** 2).sum()) * scale ** 2      # V near-orthonormal
    total = float((D_T ** 2).sum()) * scale ** 2

    # final-model null scale of the audited (with-canary) run, same units
    nul = (W @ (with_c.final_vec - with_c.init_vec)).cpu().double().numpy() * scale
    null_std = float(nul.std(ddof=1))
    # held-out directions in the coupled difference: the crosstalk floor
    xt = (W @ D_T).cpu().double().numpy() * scale
    xtalk_std = float(xt.std(ddof=1))

    # retention by age: mean over canaries of R[t_i + a, i]
    by_age = {}
    for a in range(T):
        ok = t_ins + a <= T - 1
        by_age[a] = float(R[(t_ins + a)[ok], np.nonzero(ok)[0]].mean())
    r_final = R[T - 1]                              # per canary, by insertion round
    r_by_round = r_final.reshape(T, PER_ROUND).mean(axis=1)
    pre = float(np.mean([R[s, t_ins > s].mean() for s in range(T - 1)]))

    dec = [r_final[(t_ins >= lo) & (t_ins < lo + T // 10)].mean()
           for lo in range(0, T, T // 10)]
    slope = float(np.polyfit(t_ins / (T - 1), r_final, 1)[0])

    os.makedirs(NPZ_DIR, exist_ok=True)
    tag = f"{DATASET}_E{E}_C{C}_s{seed}"
    np.savez(os.path.join(NPZ_DIR, tag + ".npz"),
             r_by_round=r_by_round, by_age=np.array([by_age[a] for a in range(T)]),
             Dnorm=Dnorm, t_ins=t_ins, r_final=r_final)

    return dict(
        E=E, C=C, seed=seed, z=z, cohort=m, n_canaries=n_ins,
        retention_age={str(a): by_age[a] for a in AGES if a < T},
        retention_final_mean=float(r_final.mean()),
        retention_final_decile=[float(x) for x in dec],
        retention_before_injection=pre,
        slope_final_vs_round=slope,              # retention(last) - retention(first)
        null_std_final=null_std, null_std_law=float(z * np.sqrt(T)),
        mu_final_decile=[float(x / null_std) for x in dec],
        mu_final_law=float(1.0 / (z * np.sqrt(T))),
        xtalk_std=xtalk_std,
        energy_along_canaries=along / total,
        Dnorm_final=float(Dnorm[-1]), Dnorm_linear=float(np.sqrt(n_ins)),
        acc_base=evaluate_accuracy(task, base.final_vec, device=dev),
        acc_canary=evaluate_accuracy(task, with_c.final_vec, device=dev),
        npz=tag, elapsed_s=time.time() - tc)


def main():
    dev = pick_device(DEVICE)
    z = calibrate_noise_multiplier(TARGET_EPS, 1.0, 1, DELTA, "rdp")
    torch.manual_seed(0)          # client split uses the global RNG (see step 44)
    task = make_image_benchmark_task(dataset=DATASET, num_clients=N_CLIENTS,
                                     samples_per_client=SPC,
                                     non_iid_alpha=0.3, seed=0)
    cohorts = minsep_cohorts(K_HONEST, T, N_CLIENTS)
    meta = dict(dataset=DATASET, N=N_CLIENTS, T=T, z=z, target_eps=TARGET_EPS,
                delta=DELTA, device=str(dev), k_canary=1, per_round=PER_ROUND,
                k_honest=K_HONEST, E_grid=E_GRID, C_grid=C_GRID, eta=ETA,
                n_null=N_NULL, torch=torch.__version__)
    rows, done = [], set()
    if RESUME and os.path.exists(OUT):
        with open(OUT) as f:
            rows = json.load(f).get("rows", [])
        done = {(r["E"], r["C"], r["seed"]) for r in rows}
        print(f"[resume] {len(rows)} rows already in {OUT}")
    payload = dict(meta=meta, rows=rows)

    print(f"{DATASET} dim={task.dim} device={dev} cohort={len(cohorts[0])} "
          f"T={T} z={z:.4f}  law: retention=1 at every age, "
          f"mu_final={1 / (z * np.sqrt(T)):.3f}\n")
    print(f"{'E':>2} {'C':>4} {'s':>2} | retention at age "
          + " ".join(f"{a:>5}" for a in AGES)
          + f" | {'early':>6} {'late':>6} {'slope':>6} {'along':>6} {'xtalk':>6}")
    for E, C, s in itertools.product(E_GRID, C_GRID, SEEDS):
        if (E, C, s) in done:
            continue
        r = run_cell(task, dev, z, E, C, s, cohorts)
        rows.append(r)
        save(payload)
        if r.get("diverged"):
            print(f"{E:>2} {C:>4} {s:>2} | DIVERGED")
            continue
        print(f"{E:>2} {C:>4} {s:>2} |                  "
              + " ".join(f"{r['retention_age'][str(a)]:>5.2f}" for a in AGES)
              + f" | {r['retention_final_decile'][0]:>6.2f} "
              f"{r['retention_final_decile'][-1]:>6.2f} "
              f"{r['slope_final_vs_round']:>6.2f} "
              f"{r['energy_along_canaries']:>6.2f} {r['xtalk_std']:>6.3f}",
              flush=True)

    # pooled over seeds, with a seed-cluster bootstrap (Convention I)
    rng = np.random.default_rng(0)
    summary = []
    for E, C in itertools.product(E_GRID, C_GRID):
        rs = [r for r in rows if (r["E"], r["C"]) == (E, C) and not r.get("diverged")]
        if not rs:
            continue

        def stat(sub):
            dec = np.mean([r["retention_final_decile"] for r in sub], axis=0)
            mu = np.mean([r["mu_final_decile"] for r in sub], axis=0)
            return dict(early=dec[0], late=dec[-1], slope=np.mean(
                [r["slope_final_vs_round"] for r in sub]),
                mean=np.mean([r["retention_final_mean"] for r in sub]),
                mu_ratio=mu[-1] / mu[0] if mu[0] > 0 else np.nan)
        est = stat(rs)
        cell = dict(E=E, C=C, n_seeds=len(rs),
                    retention_final_decile=[float(x) for x in np.mean(
                        [r["retention_final_decile"] for r in rs], axis=0)],
                    mu_final_decile=[float(x) for x in np.mean(
                        [r["mu_final_decile"] for r in rs], axis=0)],
                    mu_final_law=rs[0]["mu_final_law"],
                    retention_age={a: float(np.mean(
                        [r["retention_age"][a] for r in rs]))
                        for a in rs[0]["retention_age"]},
                    energy_along_canaries=float(np.mean(
                        [r["energy_along_canaries"] for r in rs])),
                    null_std_ratio=float(np.mean(
                        [r["null_std_final"] / r["null_std_law"] for r in rs])),
                    **{k: float(v) for k, v in est.items()})
        if len(rs) >= 4:
            boots = [stat([rs[i] for i in rng.integers(0, len(rs), len(rs))])
                     for _ in range(2000)]
            for k in est:
                v = [b[k] for b in boots]
                cell[k + "_ci"] = [float(np.nanpercentile(v, 2.5)),
                                   float(np.nanpercentile(v, 97.5))]
        summary.append(cell)
    payload["summary"] = summary
    save(payload)

    print("\npooled over seeds (95% seed-bootstrap CI)   law: retention = 1, "
          "no dependence on insertion round")
    for c in summary:
        ci = lambda k: (f" [{c[k + '_ci'][0]:.2f}, {c[k + '_ci'][1]:.2f}]"
                        if k + "_ci" in c else "")
        print(f"  E={c['E']} C={c['C']}: final retention mean {c['mean']:.2f}{ci('mean')}"
              f"  first decile {c['early']:.2f}{ci('early')}"
              f"  last decile {c['late']:.2f}{ci('late')}"
              f"  slope {c['slope']:.2f}{ci('slope')}"
              f"  energy along canaries {c['energy_along_canaries']:.2f}"
              f"  null sd / law {c['null_std_ratio']:.2f}")


if __name__ == "__main__":
    main()
