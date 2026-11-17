"""Step 44 -- THE HIDDEN GAUSSIAN COORDINATE, REALISED (Open Problem 1).

Open Problem 1 names the candidate construction for Theorem 1's two worlds
inside the Gaussian class: an independent hidden Gaussian coordinate with
mu_h^2 = mu_det^2 - mu_obs^2. A published mechanism supplies exactly that
coordinate: the DP clipped count released by adaptive clipping (Andrew et
al., NeurIPS'21). One client shifts the centred count by 1/2, so a round's
joint release is one Gaussian mechanism with

    1/z^2 = 1/z_Delta^2 + 1/(2 sigma_b)^2,

and the canonical canary-direction audit, which never looks at the count,
reads sqrt(1 - lam) of the certified deflection, lam = z^2 / (4 sigma_b^2).

Three parts, written to one JSON:

  A  analytic   exact mu / eps of the model-only, count-only and joint
                audits over lam, and the lam implied by the published
                default sigma_b = m/20 at several cohort sizes. No training.
  B  rung-0     Monte Carlo on the two-coordinate Gaussian channel itself
                (no model): the three estimators against their exact values.
  C  training   EMNIST, canaries with k = 1 (one canary per even round, odd
                rounds are the null), certificate pinned at the joint
                multiplier z, sweep lam x {diagnostic (clip lr 0), adaptive
                (clip lr 0.2)}. Per run: model-only, count, joint audits
                (white-box and raw count), final-model audit, accuracy.
                `worldA` rows re-run the diagnostic cells WITHOUT the side
                release at the same z_Delta and check the model is
                bit-identical: same audit statistics, different true leakage.

Certificate: the canary sits in one fixed round, so its exact certificate is
the single Gaussian event mu_det = 1/z (as in step 20, min-separation
adjacency with k = 1). z is calibrated so the RDP label of that event is
RL_TARGET_EPS.

Smoke:  RL_SEEDS=1 RL_LAMS=0,0.5 RL_E_GRID=1 RL_C_GRID=0.1 \
        python -m experiments.step44_side_release
Full:   RL_SEEDS=5 python -m experiments.step44_side_release
"""

import hashlib
import itertools
import json
import os
import time

import numpy as np
import torch

from harness.accountant import calibrate_noise_multiplier
from harness.auditor import (audit_final_model, audit_scalar, gdp_epsilon,
                             make_canaries)
from harness.fedavg import AdaptiveClip, DPFedAvg, evaluate_accuracy, pick_device
from harness.side_release import (Streams, audit_streams, extract_streams,
                                  joint_raw_stats, joint_stat, split_noise)
from harness.tasks import make_image_benchmark_task

DEVICE = os.environ.get("RL_DEVICE", "cpu")   # CPU: deterministic by contract
DATASET = os.environ.get("RL_DATASET", "emnist")
SEEDS = list(range(int(os.environ.get("RL_SEEDS", "5"))))
T = int(os.environ.get("RL_ROUNDS", "100"))
TARGET_EPS = float(os.environ.get("RL_TARGET_EPS", "16"))
N_CLIENTS = int(os.environ.get("RL_N_CLIENTS", "200"))
SPC = int(os.environ.get("RL_SPC", "150"))
K_HONEST = int(os.environ.get("RL_K_HONEST", "10"))       # honest participation
LAMS = [float(x) for x in os.environ.get("RL_LAMS", "0,0.25,0.5,0.75").split(",")]
CLIP_LRS = [float(x) for x in os.environ.get("RL_CLIP_LRS", "0,0.2").split(",")]
E_GRID = [int(x) for x in os.environ.get("RL_E_GRID", "1,5").split(",")]
C_GRID = [float(x) for x in os.environ.get("RL_C_GRID", "0.1,1.0").split(",")]
ETA = float(os.environ.get("RL_ETA", "0.5"))
QUANTILE = float(os.environ.get("RL_QUANTILE", "0.5"))
N_NULL = int(os.environ.get("RL_N_NULL", "300"))
DELTA = 1e-5
RESUME = os.environ.get("RL_RESUME", "1") == "1"
TRAIN = os.environ.get("RL_TRAIN", "1") == "1"
MC_TRIALS = int(os.environ.get("RL_MC_TRIALS", "200000"))
OUT = os.environ.get("RL_OUT", f"results/step44_side_release_{DATASET}.json")
STATS_DIR = os.environ.get("RL_STATS_DIR", "results/side_release")

# Canary directions must NOT share the simulator's seed: with fixed cohorts the
# simulator's first draw is round-0 noise, which would then be the very same
# normals as canary 0's direction (a perfectly aligned noise vector).
CANARY_SEED = 10_007

INS_ROUNDS = list(range(0, T, 2))     # one canary in each even round
NULL_ROUNDS = list(range(1, T, 2))    # odd rounds carry no canary


def minsep_cohorts(k, T, N):
    stride = T // k
    groups = [[] for _ in range(stride)]
    for i in range(N):
        groups[i % stride].append(i)
    return [groups[r % stride] for r in range(T)]


def vec_hash(v: torch.Tensor) -> str:
    return hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Part A -- exact values
# ---------------------------------------------------------------------------
def part_a(z):
    mu_det = 1.0 / z
    eps_det = gdp_epsilon(mu_det, DELTA)
    sweep = []
    for lam in sorted(set(LAMS + [0.05, 0.1, 0.9])):
        z_d, s_b = split_noise(z, lam)
        mu_m, mu_c = np.sqrt(1 - lam) * mu_det, np.sqrt(lam) * mu_det
        sweep.append(dict(
            lam=lam, z_delta=z_d, sigma_b=None if np.isinf(s_b) else s_b,
            mu_model=mu_m, mu_count=mu_c, mu_joint=mu_det,
            eps_model=gdp_epsilon(mu_m, DELTA),
            eps_count=gdp_epsilon(mu_c, DELTA), eps_joint=eps_det,
            underread_nats=float(np.log(eps_det)
                                 - np.log(max(gdp_epsilon(mu_m, DELTA), 1e-12)))))
    # the published default sigma_b = m/20: which lam does it imply?
    defaults = []
    for z_joint, m in itertools.product([z, 0.5, 1.0, 2.0],
                                        [20, 100, 1000, 10000]):
        s_b = m / 20.0
        lam = z_joint ** 2 / (4 * s_b ** 2)
        if lam >= 1:
            defaults.append(dict(z=z_joint, cohort=m, sigma_b=s_b, lam=lam,
                                 feasible=False))
            continue
        mu = 1.0 / z_joint
        e_j = gdp_epsilon(mu, DELTA)
        e_m = gdp_epsilon(np.sqrt(1 - lam) * mu, DELTA)
        defaults.append(dict(z=z_joint, cohort=m, sigma_b=s_b, lam=lam,
                             feasible=True, eps_joint=e_j, eps_model=e_m,
                             underread_nats=float(np.log(e_j) - np.log(e_m))))
    return dict(z=z, mu_det=mu_det, eps_det=eps_det, sweep=sweep,
                published_default=defaults)


# ---------------------------------------------------------------------------
# Part B -- rung-0 Monte Carlo on the channel itself
# ---------------------------------------------------------------------------
def part_b(z):
    rng = np.random.default_rng(0)
    rows = []
    for lam in LAMS:
        if lam == 0:
            continue
        z_d, s_b = split_noise(z, lam)
        n = MC_TRIALS
        a_i, a_n = 1 + z_d * rng.standard_normal(n), z_d * rng.standard_normal(n)
        c_i, c_n = 0.5 + s_b * rng.standard_normal(n), s_b * rng.standard_normal(n)
        au = dict(model=audit_scalar(a_i, a_n, DELTA),
                  count=audit_scalar(c_i, c_n, DELTA),
                  joint=audit_scalar(joint_stat(a_i, c_i, z_d, s_b),
                                     joint_stat(a_n, c_n, z_d, s_b), DELTA))
        exact = dict(model=np.sqrt(1 - lam) / z, count=np.sqrt(lam) / z,
                     joint=1 / z)
        rows.append(dict(lam=lam, trials=n,
                         **{f"mu_{k}": v.mu_hat for k, v in au.items()},
                         **{f"mu_{k}_exact": float(v) for k, v in exact.items()}))
    return rows


# ---------------------------------------------------------------------------
# Part C -- training
# ---------------------------------------------------------------------------
def load_existing():
    if RESUME and os.path.exists(OUT):
        with open(OUT) as f:
            saved = json.load(f)
        rows = saved.get("rows", [])
        done = {(r["mode"], r["E"], r["C0"], r["lam"], r["clip_lr"], r["seed"])
                for r in rows}
        print(f"[resume] {len(rows)} rows already in {OUT}\n")
        return rows, done
    return [], set()


def save(payload):
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    tmp = OUT + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=1)
    os.replace(tmp, OUT)


def run_cell(task, dev, z, E, C0, lam, clip_lr, seed, cohorts, mode):
    z_d, s_b = split_noise(z, lam)
    can = make_canaries(dim=task.dim, num_inserted=len(INS_ROUNDS),
                        num_held_out=N_NULL, clip_norm=C0, rounds=T,
                        participate_rounds=1, seed=CANARY_SEED + seed)
    can.participation = [[r] for r in INS_ROUNDS]
    sim = DPFedAvg(task, clip_norm=C0, noise_multiplier=z_d, server_lr=1.0,
                   local_lr=ETA, local_epochs=E, device=DEVICE, seed=seed)
    tc = time.time()

    if mode == "worldA":            # same z_Delta, NO side release at all
        res = sim.run(rounds=T, cohort_size=len(cohorts[0]), add_noise=True,
                      record_iterates=False, canaries=can, fixed_cohorts=cohorts)
        return dict(mode=mode, E=E, C0=C0, lam=lam, clip_lr=clip_lr, seed=seed,
                    z=z, z_delta=z_d, sigma_b=None,
                    final_hash=vec_hash(res.final_vec),
                    mu_final=audit_final_model(res, can, DELTA).mu_hat,
                    elapsed_s=time.time() - tc)

    # lam = 0 has no side release; a huge finite sigma_b keeps the code path
    # identical while making the count channel carry (numerically) nothing
    ac = AdaptiveClip(sigma_b=s_b if np.isfinite(s_b) else 1e9,
                      target_quantile=QUANTILE, lr=clip_lr)
    res = sim.run(rounds=T, cohort_size=len(cohorts[0]), add_noise=True,
                  record_iterates=True, canaries=can, fixed_cohorts=cohorts,
                  adaptive_clip=ac)
    if not torch.isfinite(res.final_vec).all():
        return dict(mode=mode, E=E, C0=C0, lam=lam, clip_lr=clip_lr, seed=seed,
                    z=z, z_delta=z_d, sigma_b=s_b if np.isfinite(s_b) else None,
                    diverged=True, elapsed_s=time.time() - tc)

    st = extract_streams(res, can, NULL_ROUNDS)
    au = audit_streams(st, z_d, s_b, DELTA)
    fin = audit_final_model(res, can, DELTA)
    acc = evaluate_accuracy(task, res.final_vec, device=dev)

    os.makedirs(STATS_DIR, exist_ok=True)
    tag = f"{DATASET}_E{E}_C{C0}_lam{lam}_lr{clip_lr}_s{seed}"
    np.savez(os.path.join(STATS_DIR, tag + ".npz"),
             a_ins=st.a_ins, a_nul=st.a_nul, c_ins=st.c_ins, c_nul=st.c_nul,
             craw_ins=st.craw_ins, craw_nul=st.craw_nul,
             rounds_ins=st.rounds_ins, rounds_nul=st.rounds_nul,
             C=np.asarray(res.clip_log["C"]))

    row = dict(mode=mode, E=E, C0=C0, lam=lam, clip_lr=clip_lr, seed=seed,
               z=z, z_delta=z_d, sigma_b=s_b if np.isfinite(s_b) else None,
               mu_model_exact=float(np.sqrt(1 - lam) / z),
               mu_count_exact=float(np.sqrt(lam) / z), mu_joint_exact=1 / z,
               mu_final=fin.mu_hat, eps_final=fin.epsilon_hat,
               # crosstalk: model-null variance over its noise-only value
               gamma_hat=float(au["model"].null_std ** 2 / z_d ** 2 - 1),
               C_last=float(res.clip_log["C"][-1]),
               frac_clipped=float(np.mean([l.frac_clipped for l in res.logs])),
               acc=acc, final_hash=vec_hash(res.final_vec), stats=tag,
               elapsed_s=time.time() - tc)
    for name, a in au.items():
        row[f"mu_{name}"] = a.mu_hat
        row[f"eps_{name}"] = a.epsilon_hat
    return row


def fast_mu(ins, nul):
    """Same mu-hat as audit_scalar, without the epsilon bisection (the seed
    bootstrap calls this ~1e5 times)."""
    sd = float(np.std(nul, ddof=1))
    return max(0.0, float(np.mean(ins) - np.mean(nul)) / (sd + 1e-12))


def pooled(rows, z):
    """Pool the per-canary statistics over seeds per cell; seed-cluster
    bootstrap (Convention I) for the pooled mu."""
    out = []
    key = lambda r: (r["E"], r["C0"], r["clip_lr"], r["lam"])
    cells = sorted({key(r) for r in rows if r["mode"] == "sweep"
                    and not r.get("diverged")})
    rng = np.random.default_rng(0)
    for (E, C0, lr, lam) in cells:
        rs = [r for r in rows if r["mode"] == "sweep" and key(r) == (E, C0, lr, lam)
              and not r.get("diverged")]
        z_d, s_b = split_noise(z, lam)
        per_seed = []
        for r in rs:
            d = np.load(os.path.join(STATS_DIR, r["stats"] + ".npz"))
            per_seed.append({k: d[k] for k in d.files})

        def mu_of(seed_idx, which):
            def cat(k):
                return np.concatenate([per_seed[i][k] for i in seed_idx])
            if which == "model":
                return fast_mu(cat("a_ins"), cat("a_nul"))
            if which == "count":
                return fast_mu(cat("c_ins"), cat("c_nul"))
            if which == "count_raw":
                return fast_mu(cat("craw_ins"), cat("craw_nul"))
            if which == "joint":
                return fast_mu(
                    joint_stat(cat("a_ins"), cat("c_ins"), z_d, s_b),
                    joint_stat(cat("a_nul"), cat("c_nul"), z_d, s_b))
            st = Streams(**{k: cat(k) for k in per_seed[0] if k != "C"})  # joint_raw
            return fast_mu(*joint_raw_stats(st))

        names = ["model"] + ([] if lam == 0 else
                             ["count", "joint", "count_raw", "joint_raw"])
        cell = dict(E=E, C0=C0, clip_lr=lr, lam=lam, n_seeds=len(rs),
                    acc=float(np.mean([r["acc"] for r in rs])),
                    gamma_hat=float(np.mean([r["gamma_hat"] for r in rs])),
                    mu_final=float(np.mean([r["mu_final"] for r in rs])),
                    mu_model_exact=float(np.sqrt(1 - lam) / z),
                    mu_joint_exact=1 / z)
        n = len(per_seed)
        for nm in names:
            est = mu_of(range(n), nm)
            cell[f"mu_{nm}"] = est
            cell[f"eps_{nm}"] = gdp_epsilon(est, DELTA)
            if n >= 4:
                boot = [mu_of(rng.integers(0, n, n), nm) for _ in range(1000)]
                cell[f"mu_{nm}_ci"] = [float(np.percentile(boot, 2.5)),
                                       float(np.percentile(boot, 97.5))]
        out.append(cell)

    # adaptive vs diagnostic clipping share every noise draw, so the normalised
    # model statistic can differ only through honest-client crosstalk. Report
    # that paired difference against the statistic's own null scale.
    ks = []
    for (E, C0, lr, lam) in cells:
        if lr == 0 or (E, C0, 0.0, lam) not in cells:
            continue
        diffs, scale = [], []
        for r in rows:
            if (r["mode"] != "sweep" or r.get("diverged")
                    or key(r) != (E, C0, lr, lam)):
                continue
            twin = [x for x in rows if x["mode"] == "sweep"
                    and not x.get("diverged") and x["seed"] == r["seed"]
                    and key(x) == (E, C0, 0.0, lam)]
            if not twin:
                continue
            da = np.load(os.path.join(STATS_DIR, r["stats"] + ".npz"))
            dd = np.load(os.path.join(STATS_DIR, twin[0]["stats"] + ".npz"))
            diffs.append(np.concatenate([da["a_ins"] - dd["a_ins"],
                                         da["a_nul"] - dd["a_nul"]]))
            scale.append(dd["a_nul"].std(ddof=1))
        if diffs:
            d = np.concatenate(diffs)
            ks.append(dict(E=E, C0=C0, lam=lam, clip_lr=lr, n=int(d.size),
                           rms_paired_diff=float(np.sqrt((d ** 2).mean())),
                           null_std=float(np.mean(scale)),
                           rel=float(np.sqrt((d ** 2).mean()) / np.mean(scale))))
    return out, ks


def main():
    dev = pick_device(DEVICE)
    z = calibrate_noise_multiplier(TARGET_EPS, 1.0, 1, DELTA, "rdp")
    A = part_a(z)
    print(f"joint multiplier z={z:.4f}  mu_det={A['mu_det']:.3f}  "
          f"exact eps_det={A['eps_det']:.2f} (RDP label {TARGET_EPS})\n")
    print("Part A -- exact readings over lam")
    print(f"{'lam':>5} {'z_Delta':>8} {'sigma_b':>8} | {'mu_model':>8} "
          f"{'eps_model':>9} {'eps_joint':>9} {'underread':>9}")
    for r in A["sweep"]:
        sb = "inf" if r["sigma_b"] is None else f"{r['sigma_b']:.3f}"
        print(f"{r['lam']:>5.2f} {r['z_delta']:>8.3f} {sb:>8} | "
              f"{r['mu_model']:>8.3f} {r['eps_model']:>9.2f} "
              f"{r['eps_joint']:>9.2f} {r['underread_nats']:>8.2f}n")
    print("\npublished default sigma_b = m/20:")
    for r in A["published_default"]:
        if r["feasible"]:
            print(f"  z={r['z']:.3f} cohort={r['cohort']:>6}: lam={r['lam']:.2e}  "
                  f"eps_model={r['eps_model']:.3f} vs eps_joint={r['eps_joint']:.3f}"
                  f"  ({r['underread_nats']:.4f} nats)")
        else:
            print(f"  z={r['z']:.3f} cohort={r['cohort']:>6}: lam={r['lam']:.2f} "
                  f">= 1 (default infeasible at this z)")

    B = part_b(z)
    print("\nPart B -- rung-0 Monte Carlo (no model)")
    for r in B:
        print(f"  lam={r['lam']:.2f}: model {r['mu_model']:.3f} "
              f"(exact {r['mu_model_exact']:.3f})  count {r['mu_count']:.3f} "
              f"({r['mu_count_exact']:.3f})  joint {r['mu_joint']:.3f} "
              f"({r['mu_joint_exact']:.3f})")

    meta = dict(dataset=DATASET, N=N_CLIENTS, T=T, target_eps=TARGET_EPS,
                delta=DELTA, device=str(dev), z=z, k_canary=1,
                k_honest=K_HONEST, lams=LAMS, clip_lrs=CLIP_LRS, E_grid=E_GRID,
                C_grid=C_GRID, eta=ETA, quantile=QUANTILE,
                n_ins=len(INS_ROUNDS), n_null=N_NULL,
                torch=torch.__version__)
    rows, done = load_existing()
    payload = dict(meta=meta, analytic=A, rung0=B, rows=rows)
    save(payload)
    if not TRAIN:
        return

    # harness.tasks._dirichlet_partition draws from the GLOBAL torch RNG, so the
    # client split differs per process unless it is seeded here. Seeding makes
    # every run of this script reproducible from (seed, config) alone.
    torch.manual_seed(0)
    task = make_image_benchmark_task(dataset=DATASET, num_clients=N_CLIENTS,
                                     samples_per_client=SPC,
                                     non_iid_alpha=0.3, seed=0)
    cohorts = minsep_cohorts(K_HONEST, T, N_CLIENTS)
    print(f"\nPart C -- {DATASET} dim={task.dim} device={dev} "
          f"cohort={len(cohorts[0])} T={T}")
    print(f"{'mode':>6} {'E':>2} {'C0':>4} {'lam':>5} {'lr':>4} {'s':>2} | "
          f"{'model':>6} {'(exact)':>7} {'count':>6} {'joint':>6} {'(exact)':>7} "
          f"{'j_raw':>6} {'final':>6} {'acc':>5} {'C_T':>6}")
    t0 = time.time()
    for E, C0 in itertools.product(E_GRID, C_GRID):
        for lam, lr, s in itertools.product(LAMS, CLIP_LRS, SEEDS):
            if ("sweep", E, C0, lam, lr, s) in done:
                continue
            if lam == 0 and lr > 0:
                continue    # no count release -> nothing to adapt on
            r = run_cell(task, dev, z, E, C0, lam, lr, s, cohorts, "sweep")
            rows.append(r)
            save(payload)
            if r.get("diverged"):
                print(f"{'sweep':>6} {E:>2} {C0:>4} {lam:>5.2f} {lr:>4} {s:>2} | DIVERGED")
                continue
            g = lambda k: f"{r[k]:>6.2f}" if k in r else f"{'-':>6}"
            print(f"{'sweep':>6} {E:>2} {C0:>4} {lam:>5.2f} {lr:>4} {s:>2} | "
                  f"{r['mu_model']:>6.2f} {r['mu_model_exact']:>7.2f} "
                  f"{g('mu_count')} {g('mu_joint')} {r['mu_joint_exact']:>7.2f} "
                  f"{g('mu_joint_raw')} {r['mu_final']:>6.2f} {r['acc']:>5.3f} "
                  f"{r['C_last']:>6.3f}", flush=True)
        # two-world check: no side release, same z_Delta -> identical model
        if 0.0 in CLIP_LRS:
            for lam in [l for l in LAMS if l > 0]:
                s = SEEDS[0]
                if ("worldA", E, C0, lam, 0.0, s) in done:
                    continue
                r = run_cell(task, dev, z, E, C0, lam, 0.0, s, cohorts, "worldA")
                twin = [x for x in rows if x["mode"] == "sweep"
                        and (x["E"], x["C0"], x["lam"], x["clip_lr"], x["seed"])
                        == (E, C0, lam, 0.0, s)]
                r["bit_identical_to_side_release_run"] = bool(
                    twin and twin[0].get("final_hash") == r["final_hash"])
                rows.append(r)
                save(payload)
                print(f"{'worldA':>6} {E:>2} {C0:>4} {lam:>5.2f} {0.0:>4} {s:>2} | "
                      f"model bit-identical to the side-release run: "
                      f"{r['bit_identical_to_side_release_run']}", flush=True)

    payload["pooled"], payload["adaptive_vs_diagnostic"] = pooled(rows, z)
    save(payload)
    print(f"\n[{(time.time() - t0) / 60:.1f} min] {len(rows)} rows -> {OUT}\n")
    print("pooled over seeds (mu; 95% seed-bootstrap CI where >= 4 seeds)")
    print(f"{'E':>2} {'C0':>4} {'lr':>4} {'lam':>5} | {'model':>6} {'exact':>6} "
          f"{'joint':>6} {'exact':>6} {'joint_raw':>9} {'eps_model':>9} "
          f"{'eps_joint':>9} {'acc':>5}")
    for c in payload["pooled"]:
        j = f"{c['mu_joint']:>6.2f}" if "mu_joint" in c else f"{'-':>6}"
        jr = f"{c['mu_joint_raw']:>9.2f}" if "mu_joint_raw" in c else f"{'-':>9}"
        ej = f"{c['eps_joint']:>9.2f}" if "eps_joint" in c else f"{'-':>9}"
        print(f"{c['E']:>2} {c['C0']:>4} {c['clip_lr']:>4} {c['lam']:>5.2f} | "
              f"{c['mu_model']:>6.2f} {c['mu_model_exact']:>6.2f} {j} "
              f"{c['mu_joint_exact']:>6.2f} {jr} {c['eps_model']:>9.2f} {ej} "
              f"{c['acc']:>5.3f}")
    for k in payload["adaptive_vs_diagnostic"]:
        print(f"adaptive vs fixed clip, model statistic  E={k['E']} C0={k['C0']} "
              f"lam={k['lam']:.2f}: RMS paired diff {k['rms_paired_diff']:.4f} "
              f"= {100 * k['rel']:.1f}% of the null sd")

if __name__ == "__main__":
    main()
