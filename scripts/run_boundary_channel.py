#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_boundary_channel.py -- the boundary (Perron) channel on the full conformation library.

ANCCFT Ch. 2: KMS states of the boundary algebra see only the Perron branch of the spectrum.
The polymer counterpart used here: rho_P = Spearman(k, v_P), v_P the Perron eigenvector of the
contact-graph adjacency.  Null per globule: NSHUF gap shuffles on the same conformation -> z.
Run next to run_legendre_full_ensemble.py, it gives the two channels on the same 82 globules.
Output: ../data/boundary_results.csv, ../data/boundary_tests.csv
"""
import os, sys, csv, time, zlib, argparse
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
from scipy import stats, sparse
from scipy.sparse.linalg import eigsh
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))
import contact_graph as cg   # noqa: E402
import stats_re              # noqa: E402

DATA = os.path.join(HERE, "..", "data")
NSHUF = 20


def job(fn):
    t0 = time.time(); r, k, proto, arm, run = cg.load(fn); G = cg.ContactGraph(r)
    A = sparse.coo_matrix((np.ones(G.m), (G.edges[:, 0], G.edges[:, 1])), shape=(G.n, G.n)); A = (A + A.T).tocsr()
    v = eigsh(A.astype(float), k=1, which="LA")[1][:, 0]; v *= np.sign(v.sum())
    rho = lambda kk: stats.spearmanr(kk, v).correlation
    obs = rho(k)
    rng = np.random.default_rng(zlib.crc32(f"P4|perron|{os.path.basename(fn)}".encode()) & 0xFFFFFFFF)
    null = np.array([rho(cg.kappa_shuffled(k, rng)) for _ in range(NSHUF)])
    return dict(file=os.path.basename(fn), protocol=proto, arm=arm, run=run, rho_P=obs,
                z_P=(obs - null.mean()) / null.std(ddof=1), seconds=time.time() - t0)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=6); a = ap.parse_args()
    files = [f for arm in ("True", "S1", "S6pp") for f in cg.production_files(arm)] + \
            [f for arm in ("True", "S1") for f in cg.relaxed_files(arm)]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        rows = list(ex.map(job, files))
    print(f"{len(rows)} conformations in {(time.time()-t0)/60:.1f} min")
    with open(os.path.join(DATA, "boundary_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    tests = []
    for proto, A_, B_ in [("production", "True", "S1"), ("production", "True", "S6pp"), ("production", "S6pp", "S1"), ("relaxed", "True", "S1")]:
        A = [x["z_P"] for x in rows if x["protocol"] == proto and x["arm"] == A_]
        B = [x["z_P"] for x in rows if x["protocol"] == proto and x["arm"] == B_]
        tests.append(dict(protocol=proto, contrast=f"{A_}-{B_}", mean_a=np.mean(A), mean_b=np.mean(B), **stats_re.seq_level(A, B)))
    ph = stats_re.holm([t["p_seq"] for t in tests])
    for t, h in zip(tests, ph): t["p_holm"] = h
    with open(os.path.join(DATA, "boundary_tests.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(tests[0])); w.writeheader(); w.writerows(tests)
    for proto in ("production", "relaxed"):
        for arm in ("True", "S1", "S6pp"):
            v = [x["z_P"] for x in rows if x["protocol"] == proto and x["arm"] == arm]
            if v: print(f"  {proto:>10} {arm:>5} n={len(v):2d}  z_P = {np.mean(v):+.2f} +- {np.std(v, ddof=1):.2f}")
    for t in tests:
        print(f"  {t['protocol']:>10} {t['contrast']:>10}  Z_seq={t['Z_seq']:+.2f}  p_seq={t['p_seq']:.3g}  p_holm={t['p_holm']:.3g}")


if __name__ == "__main__":
    main()
