#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_legendre_full_ensemble.py -- Task 1: divisor energy of the contact network, with smooth
positional modes removed, over the full conformation library.

Conformations (PDTP v4.0 data/v4/): production protocol 24 True + 8 S1 + 40 S6'' (N = 4e4,
8000 steps); relaxed (1e6-step isothermal hold) 5 True + 5 S1.
Per globule:
    D = rank(k) - mean;  D_K = D - P_{<=K} D for K in {4, 8, 16, 32}  (K = 0: the full divisor)
    E_K = D_K^T Delta^+ D_K  on the contact graph (r_c = 1.5, backbone + contacts)
    null: NSHUF gap shuffles of the globule's own sequence on the same conformation, processed
    identically -> z_K = (E_K - mean_null)/sd_null.
Arm contrasts on the per-globule z_K: sequence-level (random-effects, m = 1) statistic and naive
Welch; Holm and Benjamini-Hochberg corrections over the family of envelope-removed tests
(K in {4, 8, 16, 32} x all contrasts).
Output: ../data/legendre_full_results.csv (per globule), ../data/legendre_full_tests.csv
"""
import os, sys, csv, time, zlib, argparse
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
from scipy import stats
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))
import contact_graph as cg          # noqa: E402
from legendre_filter import LegendreFilter  # noqa: E402
import stats_re                     # noqa: E402

DATA = os.path.join(HERE, "..", "data")
KS = (0, 4, 8, 16, 32)
NSHUF = 20
_FILTERS = {}


def filters(n):
    if n not in _FILTERS:
        _FILTERS[n] = {K: LegendreFilter(n, K) for K in KS if K > 0}
    return _FILTERS[n]


def energies(G, k, F):
    D = stats.rankdata(k); D = D - D.mean()
    out = {0: G.energy(D)}
    for K, f in F.items():
        out[K] = G.energy(f.residual(D))
    frac = {K: f.removed_fraction(D) for K, f in F.items()}
    return out, frac


def job(fn):
    t0 = time.time()
    r, k, proto, arm, run = cg.load(fn)
    G = cg.ContactGraph(r); F = filters(len(k))
    obs, frac = energies(G, k, F)
    rng = np.random.default_rng(zlib.crc32(f"P4|legendre|{os.path.basename(fn)}".encode()) & 0xFFFFFFFF)
    null = [energies(G, cg.kappa_shuffled(k, rng), F)[0] for _ in range(NSHUF)]
    row = dict(file=os.path.basename(fn), protocol=proto, arm=arm, run=run, n_contacts=len(G.contacts))
    for K in KS:
        v = np.array([x[K] for x in null])
        row[f"E_{K}"] = obs[K]; row[f"z_{K}"] = (obs[K] - v.mean()) / v.std(ddof=1)
        if K: row[f"removed_frac_{K}"] = frac[K]
    row["seconds"] = time.time() - t0
    return row


def analyse(rows):
    tests = []
    pairs = [("production", "True", "S1"), ("production", "True", "S6pp"), ("production", "S6pp", "S1"),
             ("relaxed", "True", "S1")]
    for proto, a, b in pairs:
        A = [x for x in rows if x["protocol"] == proto and x["arm"] == a]
        B = [x for x in rows if x["protocol"] == proto and x["arm"] == b]
        if len(A) < 2 or len(B) < 2: continue
        for K in KS:
            s = stats_re.seq_level([x[f"z_{K}"] for x in A], [x[f"z_{K}"] for x in B])
            tests.append(dict(protocol=proto, contrast=f"{a}-{b}", K=K, n_a=len(A), n_b=len(B),
                              mean_a=np.mean([x[f"z_{K}"] for x in A]), mean_b=np.mean([x[f"z_{K}"] for x in B]), **s))
    fam = [t for t in tests if t["K"] > 0]
    ph, pb = stats_re.holm([t["p_seq"] for t in fam]), stats_re.bh([t["p_seq"] for t in fam])
    for t, h, q in zip(fam, ph, pb): t["p_holm"], t["p_bh"] = h, q
    for t in tests:
        t.setdefault("p_holm", float("nan")); t.setdefault("p_bh", float("nan"))
    return tests


def write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("written", path)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--analyse-only", action="store_true"); a = ap.parse_args()
    out = os.path.join(DATA, "legendre_full_results.csv")
    if a.analyse_only:
        with open(out, newline="") as f:
            rows = [{k: (v if k in ("file", "protocol", "arm") else float(v)) for k, v in r.items()} for r in csv.DictReader(f)]
    else:
        files = [f for arm in ("True", "S1", "S6pp") for f in cg.production_files(arm)] + \
                [f for arm in ("True", "S1") for f in cg.relaxed_files(arm)]
        print(f"{len(files)} conformations, {NSHUF} gap shuffles each, workers={a.workers}", flush=True)
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            rows = list(ex.map(job, files))
        print(f"done in {(time.time()-t0)/60:.1f} min", flush=True)
        os.makedirs(DATA, exist_ok=True); write(out, rows)
    tests = analyse(rows)
    write(os.path.join(DATA, "legendre_full_tests.csv"), tests)
    print(f"\n{'protocol':>10} {'contrast':>10} {'K':>3} {'mean_a':>8} {'mean_b':>8} {'Z_seq':>7} {'p_seq':>8} {'p_welch':>8} {'p_holm':>7} {'p_bh':>7}  sigma_seq")
    for t in tests:
        print(f"{t['protocol']:>10} {t['contrast']:>10} {t['K']:>3} {t['mean_a']:>+8.2f} {t['mean_b']:>+8.2f} {t['Z_seq']:>+7.2f}"
              f" {t['p_seq']:>8.3g} {t['p_welch']:>8.3g} {t['p_holm']:>7.3g} {t['p_bh']:>7.3g}  {t['sigma_seq']:.2f}")


if __name__ == "__main__":
    main()
