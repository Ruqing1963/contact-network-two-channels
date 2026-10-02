# -*- coding: utf-8 -*-
"""
contact_graph.py -- contact networks of PDTP conformations and their Laplacians.

Graph Y on beads 0..N-1: backbone bonds (i, i+1) and non-bonded contacts |r_i - r_j| < r_c,
|i - j| > 1.  Edges are oriented i < j; the backbone edges come first (indices 0..N-2), then
the contacts.  d is the N x m incidence (coboundary transpose) with d[i, e] = -1, d[j, e] = +1,
so (d^T f)_e = f_j - f_i and Delta = d d^T is the graph Laplacian (ANCCFT Ch. 3, Lemma 3.3.2).

Conformations (from the PDTP v4.0 repository, data/v4/):
    conf/N40000_{True,S1,S6pp}_r*.npz        production protocol (8000 steps); fields r, k
    relaxed_endpoints/c_N40000_{True,S1}_r*.npz   1e6-step isothermal hold; field r only.
The stiffness of a relaxed endpoint is rebuilt from the seed rule of the run that produced it
(PDTP v4.0, code/v4/region_decomposition.py).
"""
from __future__ import annotations
import os, glob
import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree
from scipy.sparse.linalg import splu
import core

RC = 1.5
N_DEFAULT = 40_000
RELAXED_BASE_SEED = 20261101
DATA_DIR = os.environ.get("PDTP_V4_DATA", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "Paper1", "GitHub",
    "prime-driven-topological-polymer", "data", "v4"))


# ------------------------------------------------------------------ conformations
def production_files(arm: str):
    return sorted(glob.glob(os.path.join(DATA_DIR, "conf", f"N40000_{arm}_r*.npz")),
                  key=lambda f: int(f.split("_r")[-1].split(".")[0]))


def relaxed_files(arm: str):
    return sorted(glob.glob(os.path.join(DATA_DIR, "relaxed_endpoints", f"c_N40000_{arm}_r*.npz")),
                  key=lambda f: int(f.split("_r")[-1].split(".")[0]))


def relaxed_kappa(arm: str, run: int, N: int = N_DEFAULT) -> np.ndarray:
    pr = np.where(core.sieve(N - 1))[0]
    rng = np.random.default_rng(RELAXED_BASE_SEED + 500_000 + {"True": 0, "S1": 1}[arm] * 10_000 + run)
    g = np.diff(pr).copy()
    if arm == "S1":
        rng.shuffle(g)
    return core.pdtp_kappa_from_gaps(pr[0], g, N)


def load(fn: str):
    """(r, k, protocol, arm, run) for a production or relaxed conformation file."""
    z = np.load(fn); base = os.path.basename(fn)
    run = int(base.split("_r")[-1].split(".")[0])
    if base.startswith("c_"):
        arm = base.split("_")[2]
        return z["r"].astype(float), relaxed_kappa(arm, run), "relaxed", arm, run
    arm = base.split("_")[1]
    # stored k is float32; rebuild the identical sequence in float64 from its anchors so that the
    # observed divisor and its gap-shuffle nulls are built at the same precision
    return z["r"].astype(float), kappa_rebuilt(z["k"].astype(float)), "production", arm, run


# ------------------------------------------------------------------ sequences
def anchors_from_kappa(k: np.ndarray) -> np.ndarray:
    return np.where(np.isclose(k, core.K_HINGE))[0]


def kappa_shuffled(k: np.ndarray, rng) -> np.ndarray:
    """Gap-shuffle null: the globule's own gap multiset in random order, same first anchor."""
    a = anchors_from_kappa(k); g = np.diff(a)
    return core.pdtp_kappa_from_gaps(int(a[0]), rng.permutation(g), len(k))


def kappa_rebuilt(k: np.ndarray) -> np.ndarray:
    """The stiffness rebuilt from the anchors of k (identity check of the null construction)."""
    a = anchors_from_kappa(k)
    return core.pdtp_kappa_from_gaps(int(a[0]), np.diff(a), len(k))


# ------------------------------------------------------------------ graph
class ContactGraph:
    def __init__(self, r: np.ndarray, rc: float = RC):
        n = len(r)
        p = cKDTree(r).query_pairs(rc, output_type="ndarray")
        self.contacts = p[np.abs(p[:, 0] - p[:, 1]) > 1]
        bb = np.stack([np.arange(n - 1), np.arange(1, n)], 1)
        self.edges = np.vstack([bb, self.contacts]); self.n = n; self.m = len(self.edges)
        self.n_backbone = n - 1
        e = np.arange(self.m)
        self.d = sparse.coo_matrix((np.r_[-np.ones(self.m), np.ones(self.m)],
                                    (np.r_[self.edges[:, 0], self.edges[:, 1]], np.r_[e, e])),
                                   shape=(n, self.m)).tocsr()
        self.L = (self.d @ self.d.T).tocsc()
        self._lu = None

    @property
    def lu(self):
        """LU of the grounded Laplacian (vertex 0 removed); the graph is connected (backbone)."""
        if self._lu is None:
            self._lu = splu(self.L[1:, 1:].tocsc())
        return self._lu

    def solve(self, b: np.ndarray) -> np.ndarray:
        """x with Delta x = b - mean(b), x[0] = 0  (a representative of Delta^+ b)."""
        b = b - b.mean(); x = np.zeros(self.n); x[1:] = self.lu.solve(b[1:]); return x

    def energy(self, D: np.ndarray) -> float:
        """D^T Delta^+ D for a degree-zero (mean-zero) vertex function D."""
        D = D - D.mean(); return float(D @ self.solve(D))

    def coarse_laplacian(self, B: int = 100):
        """Integer Laplacian of the blob graph (B consecutive beads per blob)."""
        mb = self.n // B; blob = np.minimum(np.arange(self.n) // B, mb - 1)
        bi, bj = blob[self.contacts[:, 0]], blob[self.contacts[:, 1]]; keep = bi != bj
        A = np.zeros((mb, mb), np.int64)
        A[bi[keep], bj[keep]] = 1; A[bj[keep], bi[keep]] = 1
        idx = np.arange(mb - 1); A[idx, idx + 1] = 1; A[idx + 1, idx] = 1
        return np.diag(A.sum(1)) - A
