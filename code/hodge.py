# -*- coding: utf-8 -*-
"""
hodge.py -- discrete Hodge decomposition of edge 1-forms on a contact graph.

For a 1-form f on the m oriented edges, C^1 = im d^T  (+)  ker d  orthogonally:
    f = d^T phi + h,   phi = Delta^+ d f,   h = f - d^T phi,   d h = 0,
with d the N x m incidence of ContactGraph (d^T is the coboundary, ANCCFT Ch. 3,
Def. 3.3.1: the Hecke-Dirac operator is d + d*).  h is the cycle (harmonic) part: zero net
flux at every vertex.

Note.  The full coboundary of a vertex function, f = d^T k, is exact by definition, so its
harmonic part vanishes identically for any graph and any k.  A sequence-sensitive cycle
energy needs a 1-form that is not a coboundary; the one used here is the stiffness-rank
difference carried by the NON-BONDED contact edges only (zero on the backbone):
    f_e = rank(k)_j - rank(k)_i  for contacts e = (i, j),   f_e = 0 on backbone bonds,
and H_c = ||h||^2 / ||f||^2 is the share of that field that circulates around contact cycles.
"""
from __future__ import annotations
import numpy as np


def decompose(G, f: np.ndarray):
    """Return (gradient part d^T phi, harmonic part h) of the 1-form f on graph G."""
    phi = G.solve(np.asarray(G.d @ f).ravel())
    grad = np.asarray(G.d.T @ phi).ravel()
    return grad, f - grad


def contact_rank_form(G, rank: np.ndarray) -> np.ndarray:
    f = np.zeros(G.m)
    c = G.contacts
    f[G.n_backbone:] = rank[c[:, 1]] - rank[c[:, 0]]
    return f


def cycle_share(G, f: np.ndarray) -> float:
    """H_c = ||h||^2 / ||f||^2 computed as 1 - <f, d^T Delta^+ d f>/||f||^2 (no explicit h)."""
    df = np.asarray(G.d @ f).ravel()
    return float(1.0 - (df @ G.solve(df)) / (f @ f))
