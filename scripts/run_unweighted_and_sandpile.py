#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_unweighted_and_sandpile.py -- invariants that do not use the sequence, and the exact sandpile
arithmetic of the coarse-grained contact graph.

Per globule (full graph): contact count, mean degree, ln kappa (spanning trees, sparse LU of the
reduced Laplacian), algebraic connectivity lambda_2, adjacency spectral radius.
Per globule (coarse graph, blobs of B = 100 beads, PARI/GP exact): log10 |Jac|, number of
invariant factors > 1, 2-rank and 3-rank, log10 of the group exponent, and for the sequence divisor
(dense rank of k summed over blobs, made integer and degree zero) log10 of its order in Jac and
log10 index = log10(exponent / order); the same index for NSHUF gap shuffles (null).
Output: ../data/unweighted_sandpile_results.csv, ../data/unweighted_sandpile_tests.csv
"""
import os, sys, csv, time, zlib, subprocess, tempfile, argparse
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
from scipy import stats
from scipy.sparse.linalg import eigsh
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))
import contact_graph as cg   # noqa: E402
import albanese              # noqa: E402  (GP path)
import stats_re              # noqa: E402

DATA = os.path.join(HERE, "..", "data")
B, NSHUF = 100, 5


def coarse_divisor(k, n, mb):
    blob = np.minimum(np.arange(n) // B, mb - 1)
    S = np.bincount(blob, weights=stats.rankdata(k, method="dense"), minlength=mb).round().astype(np.int64)
    D = S - S.sum() // mb; D[-1] -= int(D.sum()); return D


def gp_snf(L, Dcols):
    """(log10|Jac|, #inv, rank2, rank3, log10 exponent, [log10 ord per column]) via PARI/GP."""
    Lr, Dr = L[1:, 1:], Dcols[1:, :]
    lm = "[" + ";".join(",".join(str(int(v)) for v in row) for row in Lr) + "]"
    dm = "[" + ";".join(",".join(str(int(v)) for v in row) for row in Dr) + "]"
    with tempfile.TemporaryDirectory() as wd:
        src, out = os.path.join(wd, "a.gp"), os.path.join(wd, "a.out")
        prog = (f"default(parisizemax, 2000000000);\nL={lm};\nD={dm};\n"
                "s=matsnf(L); s=select(t->t>1, s); ex=if(#s,lcm(s),1);\n"
                "X=matsolve(L,D); o=vector(#D[1,], j, denominator(X[,j]));\n"
                f'write("{out.replace(chr(92), "/")}", [log(vecprod(s)+0.)/log(10), #s, #select(t->t%2==0,s), '
                "#select(t->t%3==0,s), log(ex+0.)/log(10), apply(t->log(t+0.)/log(10), o)]);\nquit;\n")
        with open(src, "w", encoding="ascii") as f: f.write(prog)
        subprocess.run([albanese.GP, "-q", "-f", src], check=True, capture_output=True, timeout=3600)
        v = [float(t) for t in open(out).read().replace("[", "").replace("]", "").split(",") if t.strip()]
    return v[0], int(v[1]), int(v[2]), int(v[3]), v[4], v[5:]


def job(fn):
    t0 = time.time(); r, k, proto, arm, run = cg.load(fn); G = cg.ContactGraph(r)
    d = G.lu.U.diagonal(); lnk = float(np.sum(np.log(np.abs(d)))) + float(np.sum(np.log(np.abs(G.lu.L.diagonal()))))
    lam2 = float(np.sort(eigsh(G.L.astype(float), k=2, sigma=-1e-3, which="LM", return_eigenvectors=False))[1])
    from scipy import sparse
    A = sparse.coo_matrix((np.ones(G.m), (G.edges[:, 0], G.edges[:, 1])), shape=(G.n, G.n)); A = (A + A.T).tocsr()
    lamA = float(eigsh(A.astype(float), k=1, which="LA", return_eigenvectors=False)[0])
    Lc = G.coarse_laplacian(B); mb = len(Lc)
    rng = np.random.default_rng(zlib.crc32(f"P4|snf|{os.path.basename(fn)}".encode()) & 0xFFFFFFFF)
    cols = [coarse_divisor(k, G.n, mb)] + [coarse_divisor(cg.kappa_shuffled(k, rng), G.n, mb) for _ in range(NSHUF)]
    lJ, ninv, r2, r3, lex, lords = gp_snf(Lc, np.stack(cols, 1))
    return dict(file=os.path.basename(fn), protocol=proto, arm=arm, run=run, n_contacts=len(G.contacts),
                mean_degree=2 * G.m / G.n, ln_kappa=lnk, lambda2=lam2, lambda_max=lamA,
                log10_Jac=lJ, n_invfac=ninv, rank2=r2, rank3=r3, log10_exponent=lex,
                log10_index_D=lex - lords[0], log10_index_null_mean=float(np.mean(lex - np.array(lords[1:]))),
                seconds=time.time() - t0)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=6); a = ap.parse_args()
    files = [f for arm in ("True", "S1", "S6pp") for f in cg.production_files(arm)] + \
            [f for arm in ("True", "S1") for f in cg.relaxed_files(arm)]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        rows = list(ex.map(job, files))
    print(f"{len(rows)} conformations in {(time.time()-t0)/60:.1f} min")
    with open(os.path.join(DATA, "unweighted_sandpile_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    keys = ["mean_degree", "ln_kappa", "lambda2", "lambda_max", "log10_Jac", "rank2", "rank3", "log10_index_D"]
    tests = []
    for proto, A_, B_ in [("production", "True", "S1"), ("production", "True", "S6pp"), ("production", "S6pp", "S1"), ("relaxed", "True", "S1")]:
        A = [x for x in rows if x["protocol"] == proto and x["arm"] == A_]; Bb = [x for x in rows if x["protocol"] == proto and x["arm"] == B_]
        for key in keys:
            s = stats_re.seq_level([x[key] for x in A], [x[key] for x in Bb])
            tests.append(dict(protocol=proto, contrast=f"{A_}-{B_}", stat=key, mean_a=np.mean([x[key] for x in A]),
                              mean_b=np.mean([x[key] for x in Bb]), **s))
    ph = stats_re.holm([t["p_seq"] for t in tests])
    for t, h in zip(tests, ph): t["p_holm"] = h
    with open(os.path.join(DATA, "unweighted_sandpile_tests.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(tests[0])); w.writeheader(); w.writerows(tests)
    for proto in ("production", "relaxed"):
        for arm in ("True", "S1", "S6pp"):
            v = [x for x in rows if x["protocol"] == proto and x["arm"] == arm]
            if v: print(f"  {proto:>10} {arm:>5} n={len(v):2d} " + "  ".join(f"{k}={np.mean([x[k] for x in v]):.4g}" for k in keys)
                        + f"  idx_null={np.mean([x['log10_index_null_mean'] for x in v]):.3f}")
    print(f"  smallest p_seq = {min(t['p_seq'] for t in tests):.3g}, smallest p_holm = {min(t['p_holm'] for t in tests):.3g}")
    for t in sorted(tests, key=lambda t: t["p_seq"])[:5]:
        print(f"    {t['protocol']} {t['contrast']} {t['stat']}: Z={t['Z_seq']:+.2f} p={t['p_seq']:.3g} holm={t['p_holm']:.3g}")


if __name__ == "__main__":
    main()
