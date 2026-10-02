#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_albanese_mass.py -- Task 3: Albanese energy of winding cochains on the closed contact graph
Y* = Y + e*, e* = (0, N-1)  (ANCCFT Ch. 22).

  backbone winding: H = N for every conformation (the ring is a cycle; Cor. 22.4.3). Computed as a
                    check only.
  closure flux:     H = 1/(1+R), R = effective resistance between the chain ends in the open contact
                    graph (Thm. 22.4.1); inverse effective mass M^{-1} = (2t/n) H, here t = 1, n = N.
R does not depend on the stiffness, only on the conformation, so there is no gap-shuffle null; arms
are compared directly (sequence-level statistic).  Exact check on the 400-vertex coarse graph
with PARI/GP: kappa_0 * H is an integer (Cor. 22.4.2).
Output: ../data/albanese_results.csv, ../data/albanese_tests.csv
"""
import os, sys, csv, time, argparse
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code"))
import contact_graph as cg          # noqa: E402
import albanese                     # noqa: E402
import stats_re                     # noqa: E402

DATA = os.path.join(HERE, "..", "data")


def job(fn):
    t0 = time.time(); r, k, proto, arm, run = cg.load(fn); G = cg.ContactGraph(r)
    R = albanese.end_resistance(G); H = 1.0 / (1.0 + R)
    Hw = albanese.backbone_winding_energy(G)
    row = dict(file=os.path.basename(fn), protocol=proto, arm=arm, run=run, R_end=R, H_flux=H,
               M_star_over_N=(1.0 + R) / 2.0, H_winding_minus_N=Hw - G.n,
               end_distance=float(np.linalg.norm(r[0] - r[-1])))
    try:
        logk, is_int, Hc, new_edge = albanese.coarse_exact_check(G.coarse_laplacian(100))
        row.update(coarse_log10_kappa=logk, coarse_kappaH_integer=int(is_int), coarse_H=Hc, coarse_new_edge=int(new_edge))
    except Exception as e:
        row.update(coarse_log10_kappa=float("nan"), coarse_kappaH_integer=-1, coarse_H=float("nan"), coarse_new_edge=-1)
    row["seconds"] = time.time() - t0
    return row


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=6); a = ap.parse_args()
    files = [f for arm in ("True", "S1", "S6pp") for f in cg.production_files(arm)] + \
            [f for arm in ("True", "S1") for f in cg.relaxed_files(arm)]
    print(f"{len(files)} conformations, workers={a.workers}", flush=True); t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        rows = list(ex.map(job, files))
    print(f"done in {(time.time()-t0)/60:.1f} min", flush=True)
    with open(os.path.join(DATA, "albanese_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"checks: max |H_winding - N| = {max(abs(x['H_winding_minus_N']) for x in rows):.2e};"
          f" kappa_0*H integer on coarse graphs: {sum(x['coarse_kappaH_integer'] == 1 for x in rows)}/{len(rows)}")
    tests = []
    for proto, A_, B_ in [("production", "True", "S1"), ("production", "True", "S6pp"), ("production", "S6pp", "S1"), ("relaxed", "True", "S1")]:
        A = [x for x in rows if x["protocol"] == proto and x["arm"] == A_]; B = [x for x in rows if x["protocol"] == proto and x["arm"] == B_]
        for key in ("R_end", "H_flux", "end_distance"):
            s = stats_re.seq_level([x[key] for x in A], [x[key] for x in B])
            tests.append(dict(protocol=proto, contrast=f"{A_}-{B_}", stat=key, mean_a=np.mean([x[key] for x in A]),
                              mean_b=np.mean([x[key] for x in B]), **s))
    ph = stats_re.holm([t["p_seq"] for t in tests])
    for t, h in zip(tests, ph): t["p_holm"] = h
    with open(os.path.join(DATA, "albanese_tests.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(tests[0])); w.writeheader(); w.writerows(tests)
    print(f"\nper arm (mean +- s.d.):")
    for proto in ("production", "relaxed"):
        for arm in ("True", "S1", "S6pp"):
            v = [x for x in rows if x["protocol"] == proto and x["arm"] == arm]
            if v: print(f"  {proto:>10} {arm:>5} n={len(v):2d}  R_end={np.mean([x['R_end'] for x in v]):.3f}+-{np.std([x['R_end'] for x in v], ddof=1):.3f}"
                        f"  H_flux={np.mean([x['H_flux'] for x in v]):.4f}  |r0-rN|={np.mean([x['end_distance'] for x in v]):.1f}")
    print(f"\n{'protocol':>10} {'contrast':>10} {'stat':>12} {'mean_a':>9} {'mean_b':>9} {'Z_seq':>7} {'p_seq':>8} {'p_holm':>7}")
    for t in tests:
        print(f"{t['protocol']:>10} {t['contrast']:>10} {t['stat']:>12} {t['mean_a']:>9.4f} {t['mean_b']:>9.4f} {t['Z_seq']:>+7.2f} {t['p_seq']:>8.3g} {t['p_holm']:>7.3g}")


if __name__ == "__main__":
    main()
