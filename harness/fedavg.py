"""Simulated DP-FedAvg with user-level (client-level) differential privacy.

Single-process simulation. One round:
  1. sample a cohort of clients (fixed cohort size m; Poisson option later),
  2. each client runs `local_epochs` of SGD from the current global model and
     returns its model DELTA (update),
  3. clip each client delta to L2 norm `C`,
  4. sum the clipped deltas, add Gaussian noise N(0, (z*C)^2 I) -- sensitivity of
     the sum to one client is C, so `z` (noise_multiplier) sets the DP level,
  5. divide by cohort size and apply with server LR.

This is exactly the mechanism the theoretical accountant (sampled Gaussian,
composed over rounds) and the one-shot canary auditor both assume, so the
"theoretical vs empirical epsilon" comparison is apples-to-apples.

Canary-client injection (for the auditor) is added in Step 3; the `run` loop
already exposes the hooks it needs (per-round cohort, recorded iterates).
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from .tasks import Task


def pick_device(prefer: str = "auto") -> torch.device:
    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class CanarySet:
    """Canary clients for the one-shot auditor (Andrew & Kairouz, ICLR'24).

    Each canary is a FIXED update direction of L2-norm `C` (already a valid
    clipped client update). `participation[i]` lists the rounds canary i joins.
    `held_out` are same-distribution directions NEVER inserted -> they give the
    empirical null (noise-only) distribution for the epsilon estimate.
    """
    directions: torch.Tensor          # (m, d), each row L2-norm == C
    participation: List[List[int]]    # participation[i] = rounds canary i is in
    held_out: torch.Tensor            # (m_null, d), each row L2-norm == C


@dataclass
class AdaptiveClip:
    """DP clipped-count side release (Andrew et al., NeurIPS'21 adaptive clipping).

    Each round the server also releases the CENTRED count
        sum_i (b_i - 1/2) + N(0, sigma_b^2),   b_i = 1[||delta_i|| <= C_t],
    and moves the clip norm geometrically toward the `target_quantile`:
        C_{t+1} = C_t * exp(-lr * (b_tilde - target_quantile)),
        b_tilde = noised_count / cohort_size + 1/2.
    One client shifts the centred count by +-1/2, so the joint (update, count)
    release of a round is ONE Gaussian mechanism with effective multiplier z,
        1/z^2 = 1/z_Delta^2 + 1/(2 sigma_b)^2,
    where z_Delta is the simulator's `noise_multiplier` (noise on the update
    sum is z_Delta * C_t).

    lr = 0 keeps C_t fixed: the count is then a released training diagnostic
    only, and the model stream is bit-identical for every sigma_b (the count
    noise has its own RNG stream).
    """
    sigma_b: float
    target_quantile: float = 0.5
    lr: float = 0.2
    canary_bit: int = 1               # bit a canary client reports
    seed_offset: int = 7919           # count-noise RNG = seed + seed_offset


@dataclass
class RoundLog:
    round: int
    preclip_norms: List[float]        # per-client delta norm BEFORE clipping
    frac_clipped: float               # fraction of cohort whose delta was clipped
    update_norm: float                # norm of the applied (noised, averaged) update


@dataclass
class SimResult:
    final_vec: torch.Tensor
    init_vec: torch.Tensor            # theta_0 (so Delta = final - init)
    logs: List[RoundLog] = field(default_factory=list)
    iterates: Optional[List[torch.Tensor]] = None  # per-round global vec (for all-iterates auditor)
    # adaptive_clip runs only: per-round lists `C` (clip norm used), `count_noised`
    # (released centred count), `count_honest` (centred honest sum, auditor-side
    # white-box baseline), `n_canaries`; scalars `cohort_size`, `server_lr`
    clip_log: Optional[dict] = None


class DPFedAvg:
    def __init__(
        self,
        task: Task,
        clip_norm: float = 1.0,
        noise_multiplier: float = 1.0,
        server_lr: float = 1.0,
        local_lr: float = 0.5,
        local_epochs: int = 1,
        local_batch_size: int = 16,
        server_momentum: float = 0.0,
        device: str = "auto",
        seed: int = 0,
    ):
        self.task = task
        self.C = clip_norm
        self.z = noise_multiplier
        self.server_lr = server_lr
        self.local_lr = local_lr
        self.local_epochs = local_epochs
        self.local_batch_size = local_batch_size
        self.server_momentum = server_momentum
        self.device = pick_device(device)
        self.seed = seed
        self._loss = (torch.nn.CrossEntropyLoss() if task.multiclass
                      else torch.nn.BCEWithLogitsLoss())

    # --- helpers -----------------------------------------------------------
    def _init_vec(self) -> torch.Tensor:
        # Deterministic per-seed init. Fresh model construction draws from
        # the GLOBAL torch RNG, so without seeding it here the init depends
        # on process history (e.g. how many cells ran before a resume) and
        # the run is not reproducible from `seed` alone. fork_rng keeps the
        # global stream itself untouched. Decoder tasks return a cached
        # singleton re-initialized via reset_adapters(seed) and draw nothing
        # here; _local_update overwrites params from global_vec, so this is
        # the only init-relevant draw.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            model = self.task.model_fn().to(self.device)
            return parameters_to_vector(model.parameters()).detach().clone()

    def _local_update(self, global_vec: torch.Tensor, X: torch.Tensor,
                      y: torch.Tensor) -> torch.Tensor:
        """Return delta = (params after E local epochs) - global_vec."""
        if self.task.task_kind == "lm":
            return self._local_update_lm(global_vec, X, y)
        model = self.task.model_fn().to(self.device)
        vector_to_parameters(global_vec.clone(), model.parameters())
        X, y = X.to(self.device), y.to(self.device)
        opt = torch.optim.SGD(model.parameters(), lr=self.local_lr)
        for _ in range(self.local_epochs):
            opt.zero_grad()
            out = model(X)
            if self.task.multiclass:
                loss = self._loss(out, y.long())
            else:
                loss = self._loss(out.squeeze(-1), y)
            loss.backward()
            opt.step()
        new_vec = parameters_to_vector(model.parameters()).detach()
        return new_vec - global_vec

    def _local_update_lm(self, global_vec: torch.Tensor, X: torch.Tensor,
                         y: torch.Tensor) -> torch.Tensor:
        model = self.task.model_fn().to(self.device)
        vector_to_parameters(global_vec.clone(), model.parameters())
        X, y = X.to(self.device), y.long().to(self.device)
        n = X.shape[0]
        bs = max(1, self.local_batch_size)
        opt = torch.optim.SGD(model.parameters(), lr=self.local_lr)
        gen = torch.Generator(device="cpu").manual_seed(self.seed + n)
        for _ in range(self.local_epochs):
            perm = torch.randperm(n, generator=gen)
            for start in range(0, n, bs):
                idx = perm[start:start + bs]
                opt.zero_grad()
                loss = self._loss(model(X[idx]), y[idx])
                loss.backward()
                opt.step()
        new_vec = parameters_to_vector(model.parameters()).detach()
        return new_vec - global_vec

    @staticmethod
    def _clip(delta: torch.Tensor, C: float):
        norm_t = delta.norm()
        if not torch.isfinite(norm_t):
            # a locally diverged client contributes nothing (a zero update is
            # a valid clipped update); without this, inf * 0 = NaN cascades
            # through the server aggregate
            return torch.zeros_like(delta), float("inf")
        norm = norm_t.item()
        scale = min(1.0, C / (norm + 1e-12))
        return delta * scale, norm

    # --- main loop ---------------------------------------------------------
    def run(
        self,
        rounds: int,
        cohort_size: int,
        record_iterates: bool = False,
        add_noise: bool = True,
        canaries: Optional[CanarySet] = None,
        single_epoch: bool = False,
        data_canaries: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        data_canary_rounds: Optional[List[List[int]]] = None,
        noise_stream: Optional[torch.Tensor] = None,
        fixed_cohorts: Optional[List[List[int]]] = None,
        adaptive_clip: Optional[AdaptiveClip] = None,
    ) -> SimResult:
        """single_epoch=True: shuffle clients once and walk through them in
        cohort-sized slices, so each client participates EXACTLY once
        (the Andrew & Kairouz StackOverflow protocol). Requires
        rounds * cohort_size <= num_clients.

        noise_stream: pre-generated (rounds, d) noise, ALREADY scaled by z*C
        (correlated-noise mechanisms, step 20); overrides the iid draw.
        fixed_cohorts: deterministic per-round client lists (min-separation
        participation); overrides sampling.

        adaptive_clip: also release the DP clipped count each round and (if
        its lr > 0) adapt the clip norm; `clip_norm` is then C_0 and canary
        updates are rescaled to the current C_t. None -> unchanged behaviour."""
        if adaptive_clip is not None and (noise_stream is not None
                                          or data_canaries is not None):
            raise NotImplementedError(
                "adaptive_clip is implemented for iid noise and update "
                "canaries only")
        gen = torch.Generator().manual_seed(self.seed)
        gen_b = (torch.Generator().manual_seed(
            self.seed + adaptive_clip.seed_offset)
            if adaptive_clip is not None else None)
        C_t = self.C
        clip_log = (dict(C=[], count_noised=[], count_honest=[], n_canaries=[],
                         cohort_size=cohort_size, server_lr=self.server_lr)
                    if adaptive_clip is not None else None)
        init_vec = self._init_vec()
        global_vec = init_vec.clone()
        velocity = torch.zeros_like(global_vec)
        n = self.task.num_clients
        logs: List[RoundLog] = []
        iterates: List[torch.Tensor] = [] if record_iterates else None

        epoch_cohorts = None
        if single_epoch:
            if rounds * cohort_size > n:
                raise ValueError("single_epoch needs rounds*cohort_size <= num_clients")
            perm = torch.randperm(n, generator=gen).tolist()
            epoch_cohorts = [perm[r * cohort_size:(r + 1) * cohort_size]
                             for r in range(rounds)]

        # round -> list of canary indices participating that round
        canary_rounds = {}
        if canaries is not None:
            for i, rlist in enumerate(canaries.participation):
                for r in rlist:
                    canary_rounds.setdefault(r, []).append(i)

        # round -> list of DATA-canary indices (weak adversary: a real client
        # holding crafted data; its update is honestly computed each round)
        dc_rounds = {}
        if data_canaries is not None:
            for i, rlist in enumerate(data_canary_rounds):
                for r in rlist:
                    dc_rounds.setdefault(r, []).append(i)

        for r in range(rounds):
            if fixed_cohorts is not None:
                cohort = fixed_cohorts[r]
            elif epoch_cohorts is not None:
                cohort = epoch_cohorts[r]
            else:
                cohort = torch.randperm(n, generator=gen)[:cohort_size].tolist()
            summed = torch.zeros_like(global_vec)
            preclip, n_clipped = [], 0
            for ci in cohort:
                delta = self._local_update(global_vec, *self.task.client_data[ci])
                clipped, norm = self._clip(delta, C_t)
                summed += clipped
                preclip.append(norm)
                if norm > C_t:
                    n_clipped += 1

            # canary clients contribute their fixed (already norm-C) updates;
            # under adaptive clipping they are rescaled to the current C_t
            idx = canary_rounds.get(r)
            if idx:
                summed = summed + (canaries.directions[idx].to(self.device)
                                   .sum(dim=0) * (C_t / self.C))

            # data-canary clients train honestly on their crafted data
            for i in dc_rounds.get(r, []):
                Xc, yc = data_canaries[i]
                delta = self._local_update(global_vec, Xc, yc)
                clipped, _ = self._clip(delta, self.C)
                summed += clipped

            if add_noise:
                if noise_stream is not None:
                    noise = noise_stream[r].to(self.device)
                else:
                    noise = torch.randn(summed.shape, generator=gen).to(self.device) * (self.z * C_t)
                summed = summed + noise
            update = summed / cohort_size

            if adaptive_clip is not None:
                # centred clipped count: honest clients, then canaries, then noise
                honest_c = (len(cohort) - n_clipped) - 0.5 * len(cohort)
                n_can = len(idx) if idx else 0
                noise_b = (torch.randn(1, generator=gen_b).item()
                           * adaptive_clip.sigma_b) if add_noise else 0.0
                noised_c = (honest_c + n_can * (adaptive_clip.canary_bit - 0.5)
                            + noise_b)
                clip_log["C"].append(C_t)
                clip_log["count_noised"].append(noised_c)
                clip_log["count_honest"].append(honest_c)
                clip_log["n_canaries"].append(n_can)
                b_tilde = noised_c / cohort_size + 0.5
                C_next = C_t * float(torch.exp(torch.tensor(
                    -adaptive_clip.lr * (b_tilde - adaptive_clip.target_quantile))))
            else:
                C_next = C_t
            if self.server_momentum > 0:
                velocity = self.server_momentum * velocity + update
                global_vec = global_vec + self.server_lr * velocity
            else:
                global_vec = global_vec + self.server_lr * update

            logs.append(RoundLog(
                round=r,
                preclip_norms=preclip,
                frac_clipped=n_clipped / cohort_size,
                update_norm=update.norm().item(),
            ))
            if record_iterates:
                iterates.append(global_vec.clone())
            C_t = C_next
            if self.device.type == "mps":
                # Metal heap hygiene: without this, large transient
                # allocations fragment the heap over thousands of steps and
                # command buffers start faulting (garbage reads -> NaN).
                torch.mps.empty_cache()

        return SimResult(final_vec=global_vec, init_vec=init_vec, logs=logs,
                         iterates=iterates, clip_log=clip_log)


@torch.no_grad()
def evaluate_accuracy(task: Task, vec: torch.Tensor, device: Optional[torch.device] = None) -> float:
    """Train-set accuracy across all clients' data (sanity metric only)."""
    device = device or pick_device("cpu")
    model = task.model_fn().to(device)
    vector_to_parameters(vec.to(device).clone(), model.parameters())
    correct = total = 0
    for X, y in task.client_data:
        X, y = X.to(device), y.to(device)
        if task.task_kind == "lm":
            # Subsample for speed on large LM clients
            n = X.shape[0]
            take = min(n, 512)
            if n > take:
                X, y = X[:take], y[:take]
            pred = model(X).argmax(dim=1)
            correct += (pred == y.long()).sum().item()
            total += y.numel()
            continue
        out = model(X)
        if task.multiclass:
            pred = out.argmax(dim=1).float()
        else:
            pred = (torch.sigmoid(out.squeeze(-1)) > 0.5).float()
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / total
