"""Result figures for the refocused draft (paper/PoPETS/main_v2.tex).

Reads only stored results; trains nothing.

  v2_readings.pdf   audited epsilon of the final-model, standard and whitened
                    audits per mechanism, against the certificate
                    (EMNIST k=10, EMNIST k=100, StackOverflow/Pythia-160M k=5)
  v2_predicted.pdf  deflection predicted from the strategy matrix alone
                    against the measured deflection (EMNIST mechanism arm)
  v2_hidden.pdf     adaptive clipping: audit of the model updates against the
                    joint audit, over the hidden share lambda

Run:  .venv/bin/python -m experiments.plot_v2_figures
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from harness.accountant import calibrate_noise_multiplier
from harness.auditor import gdp_epsilon
from harness.mechanisms import make_mechanism, minsep_schedule

OUT = os.environ.get("RL_FIG_DIR", "paper/PoPETS/figures")

# categorical slots 1-3 of the validated palette (all-pairs safe for three)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3de"
MECHS = ["fedavg", "tree", "bandmf", "blt"]
NAMES = {"fedavg": "DP-FedAvg", "tree": "Tree", "bandmf": "BandMF", "blt": "BLT"}

plt.rcParams.update({
    "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID,
    "grid.linewidth": 0.6, "grid.linestyle": "-", "axes.axisbelow": True,
    "legend.frameon": False, "pdf.fonttype": 42, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


SUMMARY = "results/step48_v2_summary.json"
MAGENTA, YELLOW = "#e87ba4", "#eda100"
# five audits, fixed order and fixed colour in every figure (palette slots
# 1-5; validated on the adjacent pairlist that grouped bars need)
AUDIT5 = [("final", BLUE, "final-model audit"), ("max", ORANGE, "max audit"),
          ("sum", AQUA, "sum audit"), ("single", YELLOW, "single-round audit"),
          ("whitened", MAGENTA, "whitened audit")]
SHORT = {"fedavg": "DP-FedAvg", "tree": "Tree", "sqrtmf": "Square-root",
         "bandmf_opt": "BandMF", "blt_opt": "BLT", "bandmf": "BandMF (h)",
         "blt": "BLT (h)"}


def summary():
    with open(SUMMARY) as f:
        return json.load(f)


def merged(S, tags):
    """cells and runs of several result files (same task) in one list."""
    cells, runs = [], []
    for t in tags:
        if t in S:
            cells += S[t]["cells"]
            runs += S[t]["runs"]
    return cells, runs


def per_seed(runs, m, k, key):
    rs = [r for r in runs if r["mechanism"] == m and r["k"] == k]
    return [float(np.mean([r[key] for r in rs if r["seed"] == s]))
            for s in sorted({r["seed"] for r in rs})]


# ---------------------------------------------------------------------------
def fig_readings():
    """Audited epsilon of the five audits per mechanism, against the certificate."""
    S = summary()
    em_cells, em_runs = merged(S, ["emnist", "emnist_production"])
    have = {(c["mechanism"], c["k"]) for c in em_cells}
    left = [m for m in ("fedavg", "tree", "sqrtmf", "bandmf_opt", "blt_opt") if (m, 10) in have]
    if len(left) < 4:                       # production run not finished yet
        left = [m for m in ("fedavg", "tree", "bandmf", "blt") if (m, 10) in have]
    panels = [("EMNIST", em_cells, em_runs, left, 10, S["emnist"]["meta"]["eps_cert"])]
    if "stackoverflow" in S and len(S["stackoverflow"]["cells"]) >= 4:   # all four mechanisms
        so = S["stackoverflow"]
        k = so["cells"][0]["k"]
        panels.append(("StackOverflow, LSTM", so["cells"], so["runs"],
                       [c["mechanism"] for c in so["cells"] if c["k"] == k], k,
                       so["meta"]["eps_cert"]))
    else:
        right = [m for m in ("fedavg", "tree", "bandmf", "blt") if (m, 100) in have]
        panels.append(("EMNIST, every round", em_cells, em_runs, right, 100,
                       S["emnist"]["meta"]["eps_cert"]))
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.3), sharey=True,
                             gridspec_kw=dict(width_ratios=[len(p[3]) for p in panels]))
    w = 0.16
    top = 0
    for ax, (title, cells, runs, mechs, k, cert) in zip(axes, panels):
        for i, m in enumerate(mechs):
            c = [c for c in cells if c["mechanism"] == m and c["k"] == k][0]
            for j, (a, col, lab) in enumerate(AUDIT5):
                x = i + (j - 2) * w
                ax.bar(x, c[f"eps_{a}"], width=w - 0.025, color=col, edgecolor="white",
                       linewidth=0.5, label=lab if i == 0 else None, zorder=2)
                sm = per_seed(runs, m, k, f"eps_{a}")
                ax.scatter([x] * len(sm), sm, s=3.5, color=INK, zorder=3,
                           linewidths=0, alpha=0.7)
                top = max(top, max(sm))
        ax.axhline(cert, color=INK, linewidth=0.8, zorder=1)
        ax.set_xticks(range(len(mechs)))
        ax.set_xticklabels([SHORT[m] for m in mechs])
        ax.set_title(f"{title}, $k={k}$ of $T={S['emnist']['meta']['T'] if 'EMNIST' in title else so['meta']['T']}$",
                     loc="left", pad=4)
        ax.set_xlim(-0.55, len(mechs) - 0.45)
        ax.tick_params(axis="x", length=0)
    axes[-1].text(len(panels[-1][3]) - 0.42, panels[-1][5], "certificate", ha="left",
                  va="center", fontsize=6.5, color=INK2, clip_on=False)
    axes[0].set_ylabel(r"audited privacy level $\hat\varepsilon$")
    axes[0].set_ylim(0, max(21, 1.06 * top))
    h, l = axes[0].get_legend_handles_labels()
    h.append(plt.Line2D([], [], marker="o", color=INK, linestyle="", markersize=2.2))
    l.append("one seed")
    fig.legend(h, l, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.1),
               handlelength=1.0, columnspacing=1.2, handletextpad=0.4)
    fig.subplots_adjust(wspace=0.05)
    fig.savefig(os.path.join(OUT, "v2_readings.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------------------
def fig_predicted():
    """Measured against predicted deflection, one small panel per audit
    (small multiples: five audits cannot share one scatter legibly)."""
    S = summary()
    cells, runs = merged(S, ["emnist", "emnist_production"])
    fig, axes = plt.subplots(1, 5, figsize=(7.1, 1.75), sharex=True, sharey=True)
    table = []
    for ax, (a, col, lab) in zip(axes, AUDIT5):
        ax.plot([0, 3.1], [0, 3.1], color=GRID, linewidth=1.0, zorder=1)
        for c in cells:
            sm = per_seed(runs, c["mechanism"], c["k"], f"mu_{a}")
            x = c["pred"][a]
            ax.plot([x, x], [min(sm), max(sm)], color=col, linewidth=0.8, zorder=2)
            every = c["k"] == S["emnist"]["meta"]["T"]
            ax.scatter(x, c[f"mu_{a}"], s=16, marker="s" if every else "o",
                       facecolor="white" if every else col, edgecolor=col,
                       linewidth=0.9, zorder=3)
            table.append((c["mechanism"], c["k"], a, x, c[f"mu_{a}"]))
        ax.set_title(lab, loc="left", pad=3, fontsize=7)
        ax.set_xlim(-0.15, 3.1)
        ax.set_ylim(-0.15, 3.1)
        ax.set_aspect("equal")
        ax.set_xticks([0, 1, 2, 3])
        ax.set_yticks([0, 1, 2, 3])
        ax.grid(True, axis="both")
    axes[0].set_ylabel(r"measured $\hat\mu$")
    axes[2].set_xlabel("deflection predicted from the strategy matrix, before training")
    h = [plt.Line2D([], [], marker="o", color=INK2, linestyle="", markersize=3.5),
         plt.Line2D([], [], marker="s", color=INK2, markerfacecolor="white",
                    linestyle="", markersize=3.5)]
    axes[0].legend(h, ["$k=10$", "every round"], loc="upper left", handletextpad=0.1,
                   borderaxespad=0.1, fontsize=6.5)
    fig.subplots_adjust(wspace=0.08)
    fig.savefig(os.path.join(OUT, "v2_predicted.pdf"))
    plt.close(fig)
    return table


# ---------------------------------------------------------------------------
def fig_hidden():
    with open("results/step44_side_release_emnist_cpu.json") as f:
        d = json.load(f)
    z = d["meta"]["z"]
    cells = [c for c in d["pooled"]
             if (c["E"], c["C0"], c["clip_lr"]) == (1, 0.1, 0.0)]
    cells.sort(key=lambda c: c["lam"])
    lam = np.array([c["lam"] for c in cells])
    fig, ax = plt.subplots(figsize=(3.25, 2.3))
    grid = np.linspace(0, 0.8, 200)
    ax.plot(grid, np.sqrt(1 - grid) / z, color=BLUE, linewidth=1.2, zorder=2)
    ax.plot(grid, np.full_like(grid, 1 / z), color=ORANGE, linewidth=1.2, zorder=2)

    def pts(key, col, mk, lab):
        sel = [c for c in cells if key in c]
        x = np.array([c["lam"] for c in sel])
        y = np.array([c[key] for c in sel])
        lo = np.array([c[key + "_ci"][0] for c in sel])
        hi = np.array([c[key + "_ci"][1] for c in sel])
        ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt="none", ecolor=col,
                    elinewidth=0.9, capsize=2, zorder=3)
        ax.scatter(x, y, s=26, color=col, marker=mk, edgecolor="white",
                   linewidth=0.7, zorder=4, label=lab)

    pts("mu_model", BLUE, "o", "audit of the model updates")
    pts("mu_joint", ORANGE, "s", "joint audit: updates and count")
    ax.text(0.125, 1 / z + 0.07, "certificate", ha="center", va="bottom",
            fontsize=6.5, color=INK2)
    ax.text(0.5, np.sqrt(0.5) / z - 0.2, r"$\sqrt{1-\lambda}\,\times$ certificate",
            ha="right", va="top", fontsize=6.5, color=INK2)
    ax.set_xlim(-0.03, 0.82)
    ax.set_ylim(0, 3.4)
    ax.set_xticks(lam)
    ax.set_xlabel(r"share $\lambda$ of the certified leakage carried by the count")
    ax.set_ylabel(r"measured deflection $\hat\mu$")
    ax.legend(loc="lower left", handletextpad=0.3, borderaxespad=0.2)
    fig.savefig(os.path.join(OUT, "v2_hidden.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------------------
def fig_finalmodel(T=100, nm=None):
    """Predicted audited level of the FINAL MODEL at the same certificate,
    against the number of rounds a client takes part in."""
    YELLOW = "#eda100"
    if nm is None:
        nm = calibrate_noise_multiplier(16.0, 1.0, 1, 1e-5, "rdp")
    ks = [1, 2, 4, 5, 10, 20, 25, 50, 100]
    series = [("fedavg", "DP-FedAvg", BLUE, "o"), ("tree", "Tree", ORANGE, "s"),
              ("bandmf_opt", "BandMF", AQUA, "^"), ("blt_opt", "BLT", YELLOW, "D")]
    one = np.ones(T)
    fig, ax = plt.subplots(figsize=(3.25, 2.4))
    cert = gdp_epsilon(1.0 / nm, 1e-5)
    ax.axhline(cert, color=INK, linewidth=0.8, zorder=1)
    ax.text(1.0, cert + 0.3, "certificate (same for all)", ha="left", va="bottom",
            fontsize=6.5, color=INK2)
    out = {}
    for name, lab, col, mk in series:
        M = make_mechanism(name, T, band=10)
        eps = []
        for k in ks:
            z = nm * M.sensitivity_norm(minsep_schedule(k, T))
            eps.append(gdp_epsilon(k / (z * np.sqrt(one @ M.Sigma_unit @ one)), 1e-5))
        out[name] = eps
        ax.plot(ks, eps, color=col, linewidth=1.3, label=lab, zorder=3)
    # measured final-model readings (mean over runs, range over seeds)
    measured = {}
    if os.path.exists(SUMMARY):
        S = summary()
        cells, runs = merged(S, ["emnist", "emnist_production", "emnist_ksweep"])
        for name, lab, col, mk in series:
            for c in cells:
                if c["mechanism"] != name:
                    continue
                sm = per_seed(runs, name, c["k"], "eps_final")
                ax.plot([c["k"]] * 2, [min(sm), max(sm)], color=col, linewidth=0.8, zorder=4)
                ax.scatter(c["k"], c["eps_final"], s=18, marker=mk, facecolor="white",
                           edgecolor=col, linewidth=0.9, zorder=5)
                measured[(name, c["k"])] = c["eps_final"]
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10, 20, 50, 100])
    ax.set_xticklabels(["1", "2", "5", "10", "20", "50", "100"])
    ax.minorticks_off()
    ax.set_ylim(0, 17)
    ax.set_xlabel(f"rounds a client takes part in (of $T={T}$)")
    ax.set_ylabel(r"$\hat\varepsilon$ of the final-model audit")
    h, l = ax.get_legend_handles_labels()
    if measured:
        h.append(plt.Line2D([], [], marker="o", markerfacecolor="white", markeredgecolor=INK,
                            color=INK, linestyle="", markersize=3.5))
        l.append("measured")
    fig.legend(h, l, loc="lower center", ncol=5,
               bbox_to_anchor=(0.54, -0.2), handletextpad=0.3, columnspacing=0.9,
               handlelength=1.4)
    fig.savefig(os.path.join(OUT, "v2_finalmodel.pdf"))
    plt.close(fig)
    return ks, out, measured


def fig_crossover():
    """DP-FedAvg (independent noise): deflection of the final-model, max and
    whitened audits against the number of rounds a client takes part in.
    Lines are predictions from the strategy matrix; hollow markers are
    measured means (range over seeds) of the runs with one local epoch and
    clip norm 0.1. On DP-FedAvg the sum audit and the whitened audit are the
    same statistic."""
    S = summary()
    cells, runs = merged(S, ["emnist", "emnist_ksweep"])
    runs = [r for r in runs if r["mechanism"] == "fedavg" and r["E"] == 1
            and abs(r["C"] - 0.1) < 1e-9]
    cells = sorted([c for c in cells if c["mechanism"] == "fedavg"], key=lambda c: c["k"])
    if len(cells) < 4:
        return None
    ks = [c["k"] for c in cells]
    series = [("whitened", MAGENTA, "D", "whitened audit"), ("final", BLUE, "o", "final-model audit"),
              ("max", ORANGE, "s", "max audit")]
    fig, ax = plt.subplots(figsize=(3.25, 2.25))
    cert = S["emnist"]["meta"]["mu_cert"]
    ax.axhline(cert, color=INK, linewidth=0.8, zorder=1)
    ax.text(112, cert, "certificate", ha="left", va="center", fontsize=6.5, color=INK2,
            clip_on=False)
    out = {}
    for a, col, mk, lab in series:
        ax.plot(ks, [c["pred"][a] for c in cells], color=col, linewidth=1.3, label=lab, zorder=3)
        for c in cells:
            sm = per_seed(runs, "fedavg", c["k"], f"mu_{a}")
            ax.plot([c["k"]] * 2, [min(sm), max(sm)], color=col, linewidth=0.8, zorder=4)
            ax.scatter(c["k"], np.mean(sm), s=16, marker=mk, facecolor="white",
                       edgecolor=col, linewidth=0.9, zorder=5)
            out[(a, c["k"])] = (c["pred"][a], float(np.mean(sm)), len(sm))
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10, 20, 50, 100])
    ax.set_xticklabels(["1", "2", "5", "10", "20", "50", "100"])
    ax.minorticks_off()
    ax.set_ylim(-0.15, 3.5)
    ax.set_xlabel(f"rounds a client takes part in (of $T={S['emnist']['meta']['T']}$)")
    ax.set_ylabel(r"deflection $\hat\mu$")
    h, l = ax.get_legend_handles_labels()
    h.append(plt.Line2D([], [], marker="o", markerfacecolor="white", markeredgecolor=INK,
                        color=INK, linestyle="", markersize=3.5))
    l.append("measured")
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.54, -0.13),
               handletextpad=0.3, columnspacing=0.9, handlelength=1.4)
    fig.savefig(os.path.join(OUT, "v2_crossover.pdf"))
    plt.close(fig)
    return out


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_readings()
    tab = fig_predicted()
    fig_hidden()
    fig_finalmodel()
    cross = fig_crossover()
    if cross:
        print("crossover (DP-FedAvg): audit, k, predicted, measured, seeds")
        for (au, k), v in sorted(cross.items()):
            print(f"  {au:>9} {k:>4} {v[0]:5.2f} {v[1]:5.2f} {v[2]}")
    print("wrote v2_readings, v2_predicted, v2_hidden, v2_finalmodel (.pdf) to", OUT)
    for a, _, _ in AUDIT5:
        e = np.array([abs(me - p) for _, _, aa, p, me in tab if aa == a])
        print(f"{a:>9}: mean |measured - predicted| = {e.mean():.2f}, max = {e.max():.2f}")
