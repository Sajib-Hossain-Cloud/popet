"""Step 17 -- UNIFIED DECOMPOSITION FACTORIAL (idea_decomposition_v3.md, Phase 2/3).

One script produces every observable the H/P/A/R decomposition regression
(step 18) needs, cell by cell, in ONE training run per cell:

  - eps_final, eps_allit  (realistic k-round canaries; H and P terms)
  - mu_final, mu_allit    (raw GDP statistics; H law uses their ratio)
  - eps_weak              (loss-based data-canary adversary; A-term bound,
                           subset of cells only -- see WEAK_CELL below)
  - eps_theory_rdp/pld    (certificate PINNED at RL_TARGET_EPS by
                           recalibrating z per (q, T, delta) -- the
                           subsampling-amplification confound control)
  - acc, z, q, k, theta

Factorial (defaults; every axis env-overridable):
  k (canary/client participation rounds) x E x C x eta  ->  4x2x2x2 = 32 configs.
  Cohort is derived from k so real clients' expected participation matches the
  canaries': cohort = k*N/T  (q = k/T).

Held-out prediction protocol (v3 C2, PRE-REGISTERED here, not post-hoc):
  the 8 corner configs {k in {kmin,kmax}} x {C in {Cmin,Cmax}} x
  {(E_lo, eta_hi), (E_hi, eta_lo)} are flagged holdout=true. Step 18 must fit
  H+P on holdout=false rows and report out-of-sample error on holdout=true rows.
  The split is by CONFIG, never by seed.

Resume: results JSON is (re)written after every cell; completed (config, seed)
cells are skipped on restart when RL_RESUME=1 (default).

Smoke test (one cell, ~2 min, gate: row has all fields and eps_theory ~= 16):
  RL_DEVICE=mps RL_DATASET=emnist RL_SEEDS=1 \
    RL_K_LEVELS=10 RL_E_GRID=1 RL_C_GRID=0.1 RL_ETA_GRID=0.5 \
    python -m experiments.step17_decomposition

Full run: see idea_decomposition_v3.md Appendix / v2 Phase 3.
"""

import itertools
import json
import os
import time

import numpy as np
import torch

from harness.accountant import calibrate_noise_multiplier, theoretical_epsilon
from harness.auditor import (audit_all_iterates, audit_final_model,
                             audit_scalar, make_canaries)
from harness.fedavg import DPFedAvg, evaluate_accuracy, pick_device

DEVICE = os.environ.get("RL_DEVICE", "auto")
DATASET = os.environ.get("RL_DATASET", "emnist")
SEEDS = list(range(int(os.environ.get("RL_SEEDS", "3"))))
T = int(os.environ.get("RL_ROUNDS", "100"))
TARGET_EPS = float(os.environ.get("RL_TARGET_EPS", "16"))
N_CLIENTS = int(os.environ.get("RL_N_CLIENTS",
                               "2000" if DATASET == "stackoverflow" else "200"))
SPC = int(os.environ.get("RL_SPC", "150"))

K_LEVELS = [int(x) for x in os.environ.get("RL_K_LEVELS", "5,10,40,100").split(",")]
E_GRID = [int(x) for x in os.environ.get("RL_E_GRID", "1,5").split(",")]
C_GRID = [float(x) for x in os.environ.get("RL_C_GRID", "0.1,1.0").split(",")]
ETA_GRID = [float(x) for x in os.environ.get("RL_ETA_GRID", "0.1,0.5").split(",")]
RESUME = os.environ.get("RL_RESUME", "1") == "1"

# Decoder-LoRA arms (Pythia/Qwen on the StackOverflow corpus) share the
# StackOverflow conventions: same data, decoder architecture.
IS_DECODER = DATASET.startswith(("pythia", "qwen", "gpt2"))
IS_SO_DATA = DATASET == "stackoverflow" or IS_DECODER

# 4M-dim StackOverflow canaries are ~16 MB each -> keep null set small there
N_INS = int(os.environ.get("RL_N_INS", "30"))
N_NULL = int(os.environ.get("RL_N_NULL", "60" if IS_SO_DATA else "300"))
# delta convention: 1e-5 for image benchmarks, N^-1.1 for StackOverflow
# (Andrew-Kairouz convention; keep the two straight in the writeup).
DELTA = N_CLIENTS ** -1.1 if IS_SO_DATA else 1e-5

# decoder-arm knobs (inert for other datasets)
MAX_ELEMENTS = int(os.environ.get("RL_MAX_ELEMENTS", "400"))
SEQ_LEN = int(os.environ.get("RL_SEQ_LEN", "48"))
ADAPTER_R = int(os.environ.get("RL_ADAPTER_R", "8"))
SAVE_PROJ = os.environ.get("RL_SAVE_PROJ", "1" if IS_DECODER else "0") == "1"
# Server-side damping for decoder arms: scaling the released noised aggregate
# is POST-PROCESSING (certificate unchanged; the audit statistic mu is
# scale-invariant since inserted and null projections scale together). At
# gamma=1 the eps=16-calibrated per-round noise random-walks the LoRA
# adapters until frozen-backbone activations blow up (observed: NaN at cell
# 1 of Arm A); gamma=0.2 keeps the walk bounded. Recorded per row.
SERVER_LR = float(os.environ.get("RL_SERVER_LR",
                                 "0.2" if IS_DECODER else "1.0"))
# Arm C: weak-adversary canary realism for decoders. Set RL_CANARY_KIND to
# random|natural|indist to enable data-canary clients of that kind (audited
# with the loss-based weak statistic at the weak cells).
CANARY_KIND = os.environ.get("RL_CANARY_KIND", "")

OUT = os.environ.get("RL_OUT", f"results/step17_decomposition_{DATASET}.json")


DECODER_HF_NAMES = {
    "pythia-70m": "EleutherAI/pythia-70m",
    "pythia-160m": "EleutherAI/pythia-160m",
    "pythia-410m": "EleutherAI/pythia-410m",
    "pythia-1b": "EleutherAI/pythia-1b",
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B",
    "gpt2-124m": "gpt2",
}


def make_task():
    if IS_DECODER:
        from harness.decoder_task import (get_decoder_model,
                                          make_decoder_task, reset_adapters)
        # dataset names like 'pythia-160m-reddit' select the LEAF Reddit
        # corpus; bare model names default to StackOverflow
        base = DATASET.replace("-reddit", "")
        source = "reddit" if DATASET.endswith("-reddit") else "stackoverflow"
        hf_name = DECODER_HF_NAMES[base]
        t = make_decoder_task(model_name=hf_name, num_clients=N_CLIENTS,
                              max_elements_per_client=MAX_ELEMENTS,
                              seq_len=SEQ_LEN, adapter_r=ADAPTER_R,
                              seed=0, amp=False, source=source)
        # fp32 on MPS: fp16 autocast is marginally unstable for multi-day
        # unattended runs (and no faster); see step24 smoke logs v3-v5.
        t._decoder_reset = (
            lambda s, _n=hf_name, _r=ADAPTER_R:
            reset_adapters(get_decoder_model(_n, adapter_r=_r, amp=False), s))
        return t
    if DATASET == "stackoverflow":
        from harness.tasks import make_stackoverflow_task
        return make_stackoverflow_task(num_clients=N_CLIENTS, seed=0)
    from harness.tasks import make_image_benchmark_task
    return make_image_benchmark_task(dataset=DATASET, num_clients=N_CLIENTS,
                                     samples_per_client=SPC,
                                     non_iid_alpha=0.3, seed=0)


def is_holdout(k, E, C, lr):
    """Pre-registered held-out configs for the C2 prediction protocol:
    k x C corners, with the two 'diagonal' (E, eta) combos per corner.

    Single-E grids (decoder arms, PRE-REGISTERED 2026-08-01 before any
    decoder cell ran): the (E, eta) diagonal is vacuous and would hold out
    every k-corner cell, leaving the training set without any k contrast
    (the degeneracy the StackOverflow LSTM grid exposed). Hold only the
    high-eta corners instead: 4 of 12 configs out, k-slope identifiable
    in-sample."""
    corner_k = k in (min(K_LEVELS), max(K_LEVELS))
    corner_c = C in (min(C_GRID), max(C_GRID))
    if len(E_GRID) == 1 and len(ETA_GRID) == 1:
        # PRE-REGISTERED 2026-08-04 (before any single-E single-eta cell
        # ran): with both theta axes collapsed, hold the k-corners at the
        # HIGH C only, so training keeps a k contrast at low C.
        return corner_k and C == max(C_GRID)
    if len(E_GRID) == 1:
        diag = lr == ETA_GRID[-1]
    else:
        diag = ((E == E_GRID[0] and lr == ETA_GRID[-1])
                or (E == E_GRID[-1] and lr == ETA_GRID[0]))
    return corner_k and corner_c and diag


def is_weak_cell(k, E, C, lr):
    """A-term (weak adversary) measured at sparsest/densest k, viable theta."""
    return (k in (min(K_LEVELS), max(K_LEVELS))
            and E == max(E_GRID) and C == min(C_GRID) and lr == max(ETA_GRID))


def make_outliers(n, seed, label=0):
    """Crafted data canaries (weak adversary): noise images, fixed label."""
    import torch
    g = torch.Generator().manual_seed(seed + 777)
    return [(torch.rand(1, 1, 28, 28, generator=g),
             torch.tensor([float(label)])) for _ in range(n)]


def spread_schedules(n, k, rounds, seed):
    import torch
    g = torch.Generator().manual_seed(seed + 555)
    return [sorted(torch.randperm(rounds, generator=g)[:k].tolist())
            for _ in range(n)]


def neg_losses(task, vec, samples, device):
    import torch
    import torch.nn.functional as F
    from torch.nn.utils import vector_to_parameters
    model = task.model_fn().to(device)
    with torch.no_grad():
        vector_to_parameters(vec.to(device).clone(), model.parameters())
        out = []
        for X, y in samples:
            loss = F.cross_entropy(model(X.to(device)), y.long().to(device))
            out.append(-loss.item())
    return out


def load_existing():
    if RESUME and os.path.exists(OUT):
        with open(OUT) as f:
            saved = json.load(f)
        rows = saved.get("rows", [])
        # diverged rows stay in the record but do not count as done: the
        # cell is retried (post-fix) and appends a fresh row
        done = {(r["k"], r["E"], r["C"], r["lr"], r["seed"])
                for r in rows if not r.get("diverged")}
        print(f"[resume] {len(rows)} cells already in {OUT} "
              f"({len(done)} count as done)\n")
        return rows, done
    return [], set()


def save(meta, rows):
    os.makedirs("results", exist_ok=True)
    tmp = OUT + ".tmp"
    with open(tmp, "w") as f:
        json.dump(dict(meta=meta, rows=rows), f, indent=1)
    os.replace(tmp, OUT)


def make_decoder_canaries(n, seed, start=0):
    from harness.decoder_task import make_text_canaries
    return make_text_canaries(CANARY_KIND, n,
                              DECODER_HF_NAMES[DATASET],
                              seq_len=SEQ_LEN, seed=seed, start=start)


def main():
    dev = pick_device(DEVICE)
    task = make_task()
    # image outliers for image tasks; text canaries for decoders when a
    # canary kind is requested (Arm C)
    weak_ok = (DATASET in ("emnist", "fashion")
               or (IS_DECODER and CANARY_KIND != ""))

    configs = list(itertools.product(K_LEVELS, E_GRID, C_GRID, ETA_GRID))
    print(f"device={dev}  {DATASET} dim={task.dim}  N={N_CLIENTS} T={T}  "
          f"target_eps={TARGET_EPS} delta={DELTA:.2e}")
    print(f"grid: k={K_LEVELS} x E={E_GRID} x C={C_GRID} x eta={ETA_GRID} "
          f"= {len(configs)} configs x {len(SEEDS)} seeds")
    n_hold = sum(is_holdout(*c) for c in configs)
    print(f"held-out configs (C2 protocol): {n_hold}/{len(configs)}\n")

    # z depends only on (q, T) -> calibrate once per k level
    zs = {}
    for k in K_LEVELS:
        cohort = max(1, round(k * N_CLIENTS / T))
        q = cohort / N_CLIENTS
        z = calibrate_noise_multiplier(TARGET_EPS, q, T, DELTA, "rdp")
        eps_rdp = theoretical_epsilon(z, q, T, DELTA, "rdp")
        eps_pld = theoretical_epsilon(z, q, T, DELTA, "pld")
        zs[k] = (cohort, q, z, eps_rdp, eps_pld)
        print(f"  k={k:>4}: cohort={cohort:>4} q={q:.3f} z={z:.4f} "
              f"-> eps_rdp={eps_rdp:.3f} eps_pld={eps_pld:.3f}  "
              f"(crosstalk bound d*z^2/T={task.dim * z**2 / T:.0f} vs m={N_INS})")
    print()

    meta = dict(dataset=DATASET, N=N_CLIENTS, T=T, target_eps=TARGET_EPS,
                delta=DELTA, device=str(dev), seeds=len(SEEDS),
                server_lr=SERVER_LR,
                k_levels=K_LEVELS, E_grid=E_GRID, C_grid=C_GRID,
                eta_grid=ETA_GRID, n_ins=N_INS, n_null=N_NULL,
                calibration={str(k): dict(cohort=v[0], q=v[1], z=v[2],
                                          eps_rdp=v[3], eps_pld=v[4])
                             for k, v in zs.items()})
    rows, done = load_existing()

    header = (f"{'k':>4} {'E':>3} {'C':>5} {'eta':>4} {'seed':>4} | "
              f"{'eps_final':>9} {'eps_allit':>9} {'gap':>6} {'eps_weak':>8} "
              f"{'acc':>6} {'mins':>5}")
    print(header)
    t0 = time.time()
    for (k, E, C, lr) in configs:
        cohort, q, z, eps_rdp, eps_pld = zs[k]
        for s in SEEDS:
            if (k, E, C, lr, s) in done:
                continue
            tc = time.time()
            if hasattr(task, "_decoder_reset"):
                task._decoder_reset(s)
            can = make_canaries(dim=task.dim, num_inserted=N_INS,
                                num_held_out=N_NULL, clip_norm=C, rounds=T,
                                participate_rounds=k, spread=True, seed=s)
            weak = weak_ok and is_weak_cell(k, E, C, lr)
            dc = dc_sched = None
            if weak:
                dc = (make_decoder_canaries(N_INS, seed=s) if IS_DECODER
                      else make_outliers(N_INS, seed=s))
                dc_sched = spread_schedules(N_INS, k, T, seed=s)

            sim = DPFedAvg(task, clip_norm=C, noise_multiplier=z,
                           server_lr=SERVER_LR, local_lr=lr, local_epochs=E,
                           device=DEVICE, seed=s)
            res = sim.run(rounds=T, cohort_size=cohort, add_noise=True,
                          record_iterates=True, canaries=can,
                          data_canaries=dc, data_canary_rounds=dc_sched)

            if not torch.isfinite(res.final_vec).all():
                rows.append(dict(
                    k=k, E=E, C=C, lr=lr, seed=s, T=T,
                    cohort=cohort, q=q, z=z,
                    eps_theory_rdp=eps_rdp, eps_theory_pld=eps_pld,
                    eps_final=-1.0, mu_final=0.0,
                    eps_allit=-1.0, mu_allit=0.0,
                    eps_strong=-1.0, eps_weak=None, mu_weak=None,
                    null_cosine_rms=None, acc=None,
                    holdout=is_holdout(k, E, C, lr), diverged=True,
                    elapsed_s=time.time() - tc,
                ))
                save(meta, rows)
                print(f"{k:>4} {E:>3} {C:>5.1f} {lr:>4.1f} {s:>4} | "
                      f"DIVERGED (non-finite final vec) -- recorded, continuing",
                      flush=True)
                continue

            fin = audit_final_model(res, can, DELTA)
            allit = audit_all_iterates(res, can, DELTA)
            eps_weak = mu_weak = None
            if weak:
                nul = (make_decoder_canaries(N_NULL, seed=s + 10_000,
                                             start=N_INS)
                       if IS_DECODER else make_outliers(N_NULL, seed=s + 10_000))
                a_w = audit_scalar(neg_losses(task, res.final_vec, dc, dev),
                                   neg_losses(task, res.final_vec, nul, dev),
                                   DELTA)
                eps_weak, mu_weak = a_w.epsilon_hat, a_w.mu_hat
            acc = evaluate_accuracy(task, res.final_vec, device=dev)

            rows.append(dict(
                k=k, E=E, C=C, lr=lr, seed=s, T=T,
                cohort=cohort, q=q, z=z,
                eps_theory_rdp=eps_rdp, eps_theory_pld=eps_pld,
                eps_final=fin.epsilon_hat, mu_final=fin.mu_hat,
                eps_allit=allit.epsilon_hat, mu_allit=allit.mu_hat,
                eps_strong=fin.epsilon_hat,  # update-control adversary alias
                eps_weak=eps_weak, mu_weak=mu_weak,
                null_cosine_rms=fin.null_cosine_rms,
                acc=acc, holdout=is_holdout(k, E, C, lr),
                elapsed_s=time.time() - tc,
            ))
            save(meta, rows)
            if SAVE_PROJ and res.iterates is not None:
                # per-round projections for inserted AND held-out canaries
                # (checkpoint-granularity re-audits need the null set too)
                os.makedirs("results/projections", exist_ok=True)
                dirs = can.directions.to(res.final_vec.device)
                held = can.held_out.to(res.final_vec.device)
                prev, P, Q = res.init_vec, [], []
                for it in res.iterates:
                    upd = it - prev
                    P.append(((dirs @ upd) / C).cpu().numpy())
                    Q.append(((held @ upd) / C).cpu().numpy())
                    prev = it
                np.savez(f"results/projections/{DATASET}_T{T}_k{k}_E{E}"
                         f"_C{C}_lr{lr}_s{s}.npz",
                         ins=np.stack(P).astype(np.float32),
                         null=np.stack(Q).astype(np.float32))
            gap = allit.mu_hat / max(fin.mu_hat, 1e-9)
            ew = f"{eps_weak:8.3f}" if eps_weak is not None else "       -"
            print(f"{k:>4} {E:>3} {C:>5.1f} {lr:>4.1f} {s:>4} | "
                  f"{fin.epsilon_hat:>9.3f} {allit.epsilon_hat:>9.3f} "
                  f"{gap:>6.2f} {ew} {acc:>6.3f} "
                  f"{(time.time() - tc) / 60:>5.1f}", flush=True)

    print(f"\n[{(time.time() - t0) / 60:.1f} min this session] "
          f"{len(rows)} total cells saved -> {OUT}")

    # quick H-law readout (informal; step 18 does the real fit)
    by_k = {}
    for r in rows:
        if r["mu_final"] > 1e-9 and r["mu_allit"] > 0:
            by_k.setdefault(r["k"], []).append(r["mu_allit"] / r["mu_final"])
    if len(by_k) >= 2:
        print("\nH-law preview: mean mu_allit/mu_final vs sqrt(T/k)")
        for k in sorted(by_k):
            print(f"  k={k:>4}: gap={np.mean(by_k[k]):6.2f}  "
                  f"sqrt(T/k)={np.sqrt(T / k):6.2f}")


if __name__ == "__main__":
    main()
