# -*- coding: utf-8 -*-
"""
albanese.py -- harmonic energy of a winding cochain on the closed contact graph (ANCCFT Ch. 22).

Close the chain by one extra bond e* = (0, N-1).  For the winding cochain alpha (a 1-form on the
edges of the closed graph Y*), the harmonic representative is h = Pi alpha with
Pi = I - d^T Delta^+ d, and the Albanese energy is H = <h, h> = alpha^T Pi alpha; the inverse
effective mass of the Z-periodic lattice built on the unit cell Y* is M^{-1} = (2t/n) H
(Thm. 22.3.1).

Two winding cochains:
  * backbone winding  alpha = indicator of the backbone ring (all backbone bonds + e*).  This is
    an oriented cycle of Y*, so alpha is already harmonic and H = N exactly, for EVERY
    conformation (Cor. 22.4.3).  It carries no information and is computed only as a check.
  * closure flux      alpha = indicator of e* alone.  Then H = 1 - R_eff^{Y*}(e*) (Thm. 22.4.1).
    Since Y* = Y + e*, R_eff^{Y*}(e*) = R/(1+R) with R the effective resistance between the two
    chain ends in the open contact graph Y, so
        H = 1/(1 + R),   M* = n (1 + R) / (2t).
    This depends on the conformation, and through the chain ends possibly on the sequence.
Exact check on coarse graphs (PARI/GP): kappa_0 * H is an integer (Cor. 22.4.2), with
kappa_0 = det of the reduced Laplacian of Y* (the spanning-tree count).
"""
from __future__ import annotations
import os, subprocess, tempfile
import numpy as np

GP = os.environ.get("GP_EXE", r"C:\Program Files (x86)\Pari64-2-17-3\gp.exe")


def end_resistance(G) -> float:
    """Effective resistance between beads 0 and N-1 in the open contact graph."""
    b = np.zeros(G.n); b[0] = 1.0; b[-1] = -1.0
    x = G.solve(b); return float(x[0] - x[-1])


def closure_flux_energy(G) -> float:
    R = end_resistance(G); return 1.0 / (1.0 + R)


def backbone_winding_energy(G) -> float:
    """alpha^T Pi alpha for the backbone ring on Y* = Y + e*, e* oriented 0 -> N-1.
    The ring i -> i+1 (i < N-1), then N-1 -> 0, is alpha = +1 on backbone bonds and -1 on e*.
    It is a cycle (d* alpha = 0), so Pi alpha = alpha and the energy is ||alpha||^2 = N."""
    from scipy import sparse
    n, m = G.n, G.m
    d_star = sparse.hstack([G.d, sparse.csr_matrix(([-1.0, 1.0], ([0, n - 1], [0, 0])), shape=(n, 1))]).tocsr()
    alpha = np.zeros(m + 1); alpha[:G.n_backbone] = 1.0; alpha[-1] = -1.0
    flux = np.asarray(d_star @ alpha).ravel()
    if np.abs(flux).max() > 1e-12:
        raise ValueError("backbone ring is not closed")
    return float(alpha @ alpha)


def coarse_exact_check(L_open: np.ndarray, gp: str = GP):
    """Exact kappa_0 and kappa_0 * H for the closure flux on the coarse graph Y* (blob 0 joined to
    the last blob).  Returns (log10 kappa_0, kappa_0*H is integer, H as float)."""
    L = L_open.copy(); k = len(L) - 1
    new_edge = L[0, k] == 0          # False: ends already adjacent, the closure is a parallel edge
    L[0, k] -= 1; L[k, 0] -= 1; L[0, 0] += 1; L[k, k] += 1   # multigraph Laplacian of Y + e*
    mat = "[" + ";".join(",".join(str(int(v)) for v in row) for row in L[1:, 1:]) + "]"
    with tempfile.TemporaryDirectory() as wd:
        src = os.path.join(wd, "a.gp"); out = os.path.join(wd, "a.out")
        # R*(e) via the open graph: R = (e_0 - e_k)^T L_open^+ (e_0 - e_k); grounded at vertex 0
        lo = "[" + ";".join(",".join(str(int(v)) for v in row) for row in L_open[1:, 1:]) + "]"
        prog = (f'default(parisizemax, 1000000000);\nLs={mat};\nLo={lo};\n'
                f'kap=matdet(Ls);\nb=vector({k},j,0)~; b[{k}]=-1;\n'
                f'R=-(matsolve(Lo,b))[{k}];\nH=1/(1+R);\n'
                f'write("{out.replace(chr(92), "/")}", [log(kap+0.)/log(10), type(kap*H)=="t_INT", H+0.]);\nquit;\n')
        with open(src, "w", encoding="ascii") as f: f.write(prog)
        subprocess.run([gp, "-q", "-f", src], check=True, capture_output=True, timeout=600)
        v = open(out).read().strip().strip("[]").split(",")
    return float(v[0]), int(v[1]) == 1, float(v[2]), new_edge
