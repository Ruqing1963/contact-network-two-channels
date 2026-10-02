#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""plot_figures.py -- figures of the paper from data/*.csv.
Fig. 1  bulk channel: z of the divisor energy against the number K of Legendre modes removed
Fig. 2  per-globule z against the own-sequence null for four channel statistics (memory)
Fig. 3  closure flux: end-to-end effective resistance of the contact network"""
import os, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data"); FIG = os.path.join(HERE, "..", "paper", "figures")
COL = {"True": "#000000", "S1": "#d62728", "S6pp": "#1f77b4"}
LAB = {"True": "true primes", "S1": "S1 shuffle", "S6pp": "S6$''$ (PNT only)"}
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "legend.fontsize": 7, "pdf.fonttype": 42,
                     "ps.fonttype": 42, "mathtext.fontset": "stix", "font.family": "STIXGeneral"})


def read(name):
    with open(os.path.join(DATA, name), newline="") as f:
        return list(csv.DictReader(f))


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, name + ".pdf")); fig.savefig(os.path.join(FIG, name + ".png"), dpi=200)
    plt.close(fig); print("written", name)


def sem(v): v = np.asarray(v, float); return v.std(ddof=1) / np.sqrt(len(v))


def fig1():
    rows = read("legendre_full_results.csv"); Ks = [0, 4, 8, 16, 32]
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6))
    for a, proto in zip(ax, ("production", "relaxed")):
        for j, arm in enumerate(("True", "S1", "S6pp")):
            v = [x for x in rows if x["protocol"] == proto and x["arm"] == arm]
            if not v: continue
            m = [np.mean([float(x[f"z_{K}"]) for x in v]) for K in Ks]; s = [sem([float(x[f"z_{K}"]) for x in v]) for K in Ks]
            a.errorbar(np.arange(len(Ks)) + (j - 1) * 0.08, m, s, fmt="o-", color=COL[arm], ms=3.5, lw=1, label=f"{LAB[arm]} ($n={len(v)}$)")
        a.set_xticks(range(len(Ks))); a.set_xticklabels(["full", "$K{=}4$", "$8$", "$16$", "$32$"])
        a.set_xlabel("Legendre modes removed from the divisor"); a.set_ylabel(r"$z$ of $D^{\top}\Delta^{+}D$ vs own gap shuffles")
        a.set_title(f"({'a' if proto == 'production' else 'b'}) {proto} globules"); a.legend(frameon=False)
    fig.tight_layout(); save(fig, "fig1_bulk_channel")


def fig2():
    leg = {x["file"]: x for x in read("legendre_full_results.csv")}
    hod = {x["file"]: x for x in read("hodge_results.csv")}
    bnd = {x["file"]: x for x in read("boundary_results.csv")}
    stats_ = [("full divisor", lambda f: float(leg[f]["z_0"])), ("divisor, $K{=}16$ removed", lambda f: float(leg[f]["z_16"])),
              ("cycle share $H_c$", lambda f: float(hod[f]["z_H"])), ("Perron $\\rho_P$", lambda f: float(bnd[f]["z_P"]))]
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True)
    for a, proto in zip(ax, ("production", "relaxed")):
        for i, (name, fn) in enumerate(stats_):
            for j, arm in enumerate(("True", "S1", "S6pp")):
                files = [f for f, x in leg.items() if x["protocol"] == proto and x["arm"] == arm]
                if not files: continue
                v = np.array([fn(f) for f in files]); xs = i + (j - 1) * 0.25
                a.scatter(np.full(len(v), xs) + np.random.default_rng(i * 10 + j).uniform(-0.06, 0.06, len(v)), v, s=6, color=COL[arm], alpha=0.5, lw=0)
                a.errorbar(xs, v.mean(), sem(v), fmt="_", color=COL[arm], ms=12, mew=2, label=LAB[arm] if i == 0 else None)
        a.axhline(0, color="k", lw=0.5); a.axhspan(-2, 2, color="0.9", zorder=0)
        a.set_xticks(range(len(stats_))); a.set_xticklabels([s[0] for s in stats_], rotation=15)
        a.set_title(f"({'a' if proto == 'production' else 'b'}) {proto} globules")
    ax[0].set_ylabel("$z$ against the globule's own gap shuffles"); ax[0].legend(frameon=False, loc="upper right")
    fig.tight_layout(); save(fig, "fig2_memory")


def fig3():
    rows = read("albanese_results.csv")
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for j, (proto, arm) in enumerate([("production", "True"), ("production", "S1"), ("production", "S6pp"), ("relaxed", "True"), ("relaxed", "S1")]):
        v = np.array([float(x["R_end"]) for x in rows if x["protocol"] == proto and x["arm"] == arm])
        ax.scatter(np.full(len(v), j) + np.random.default_rng(j).uniform(-0.12, 0.12, len(v)), v, s=8, color=COL[arm], alpha=0.6, lw=0)
        ax.plot([j - 0.25, j + 0.25], [np.median(v)] * 2, color=COL[arm], lw=2)
    ax.set_yscale("log"); ax.set_xticks(range(5))
    ax.set_xticklabels(["true", "S1", "S6$''$", "true", "S1"]); ax.axvline(2.5, color="0.6", lw=0.6)
    ax.text(1, ax.get_ylim()[1] * 0.6, "production", ha="center"); ax.text(3.5, ax.get_ylim()[1] * 0.6, "relaxed", ha="center")
    ax.set_ylabel(r"end-to-end resistance $R$;  $H_{\rm flux}=1/(1+R)$")
    fig.tight_layout(); save(fig, "fig3_closure_flux")


if __name__ == "__main__":
    fig1(); fig2(); fig3()
