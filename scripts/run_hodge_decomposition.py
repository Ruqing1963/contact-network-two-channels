#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_hodge_decomposition.py -- Task 2: cycle (harmonic) share of the stiffness field carried by
the contact edges.

The full coboundary d^T k is exact and has no harmonic part on any graph (tested in
tests/test_operators.py), so the sequence-sensitive 1-form is the stiffness-rank difference on
NON-BONDED contacts only:  f_e = rank(k)_j - rank(k)_i,  zero on backbone bonds.
H_c = ||h||^2/||f||^2 with h the cycle part of f (code/hodge.py).  Computed for the raw rank and
for the rank with Legendre modes <= 16 removed.  Null per globule: NSHUF gap shuffles of the
globule's own sequence on the same conformation -> z.
Output: ../data/hodge_results.csv, ../data/hodge_tests.csv
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
import hodge                        # noqa: E402
from legendre_filter import LegendreFilter  # noqa: E402
import stats_re                     # noqa: E402

DATA = os.path.join(HERE, "..", "data")
NSHUF = 20
_F = {}


def shares(G, k):
    if G.n not in _F: _F[G.n] = LegendreFilter(G.n, 16)
    rk = stats.rankdata(k); rk = rk - rk.mean()
    return (hodge.cycle_share(G, hodge.contact_rank_form(G, rk)),
            hodge.cycle_share(G, hodge.contact_rank_form(G, _F[G.n].residual(rk))))


def job(fn):
    t0 = time.time(); r, k, proto, arm, run = cg.load(fn); G = cg.ContactGraph(r)
    obs = shares(G, k)
    rng = np.random.default_rng(zlib.crc32(f"P4|hodge|{os.path.basename(fn)}".encode()) & 0xFFFFFFFF)
    null = np.array([shares(G, cg.kappa_shuffled(k, rng)) for _ in range(NSHUF)])
    z = (np.array(obs) - null.mean(0)) / null.std(0, ddof=1)
    return dict(file=os.path.basename(fn), protocol=proto, arm=arm, run=run, H_c=obs[0], z_H=z[0],
                H_c_fluct=obs[1], z_H_fluct=z[1], null_H_mean=null[:, 0].mean(), seconds=time.time() - t0)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=6); a = ap.parse_args()
    files = [f for arm in ("True", "S1", "S6pp") for f in cg.production_files(arm)] + \
            [f for arm in ("True", "S1") for f in cg.relaxed_files(arm)]
    print(f"{len(files)} conformations, workers={a.workers}", flush=True); t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        rows = list(ex.map(job, files))
    print(f"done in {(time.time()-t0)/60:.1f} min", flush=True)
    with open(os.path.join(DATA, "hodge_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    tests = []
    for proto, A_, B_ in [("production", "True", "S1"), ("production", "True", "S6pp"), ("production", "S6pp", "S1"), ("relaxed", "True", "S1")]:
        A = [x for x in rows if x["protocol"] == proto and x["arm"] == A_]; B = [x for x in rows if x["protocol"] == proto and x["arm"] == B_]
        for key in ("H_c", "z_H", "z_H_fluct"):
            s = stats_re.seq_level([x[key] for x in A], [x[key] for x in B])
            tests.append(dict(protocol=proto, contrast=f"{A_}-{B_}", stat=key, mean_a=np.mean([x[key] for x in A]),
                              mean_b=np.mean([x[key] for x in B]), **s))
    ph, pb = stats_re.holm([t["p_seq"] for t in tests]), stats_re.bh([t["p_seq"] for t in tests])
    for t, h, q in zip(tests, ph, pb): t["p_holm"], t["p_bh"] = h, q
    with open(os.path.join(DATA, "hodge_tests.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(tests[0])); w.writeheader(); w.writerows(tests)
    print(f"\nper arm (mean +- s.d.):")
    for proto in ("production", "relaxed"):
        for arm in ("True", "S1", "S6pp"):
            v = [x for x in rows if x["protocol"] == proto and x["arm"] == arm]
            if v: print(f"  {proto:>10} {arm:>5} n={len(v):2d}  H_c={np.mean([x['H_c'] for x in v]):.4f}+-{np.std([x['H_c'] for x in v], ddof=1):.4f}"
                        f"  z_H={np.mean([x['z_H'] for x in v]):+.2f}+-{np.std([x['z_H'] for x in v], ddof=1):.2f}"
                        f"  z_H_fluct={np.mean([x['z_H_fluct'] for x in v]):+.2f}+-{np.std([x['z_H_fluct'] for x in v], ddof=1):.2f}")
    print(f"\n{'protocol':>10} {'contrast':>10} {'stat':>10} {'mean_a':>9} {'mean_b':>9} {'Z_seq':>7} {'p_seq':>8} {'p_holm':>7}")
    for t in tests:
        print(f"{t['protocol']:>10} {t['contrast']:>10} {t['stat']:>10} {t['mean_a']:>9.4f} {t['mean_b']:>9.4f} {t['Z_seq']:>+7.2f} {t['p_seq']:>8.3g} {t['p_holm']:>7.3g}")


if __name__ == "__main__":
    main()
