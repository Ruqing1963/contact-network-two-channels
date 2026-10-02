# -*- coding: utf-8 -*-
"""Tests of the contact-graph operators: Laplacian solve, Hodge projection, Legendre filter,
Albanese energies, multiple-testing corrections."""
import os, sys, shutil
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))
import core                    # noqa: E402
import contact_graph as cg     # noqa: E402
import hodge                   # noqa: E402
import albanese                # noqa: E402
from legendre_filter import LegendreFilter  # noqa: E402
import stats_re                # noqa: E402


def compact_walk(n=400, seed=0):
    """A random walk folded into a small box, so that it has many contacts."""
    rng = np.random.default_rng(seed); r = np.zeros((n, 3))
    for i in range(1, n):
        for _ in range(100):
            v = rng.normal(size=3); v /= np.linalg.norm(v); x = r[i - 1] + v
            if np.all(np.abs(x) < 4): break
        r[i] = x
    return r


G = cg.ContactGraph(compact_walk())
Lp = np.linalg.pinv(G.L.toarray())


def test_graph_basics():
    assert G.n == 400 and G.m == G.n_backbone + len(G.contacts) and len(G.contacts) > 400
    assert np.allclose(G.L.toarray(), (G.d @ G.d.T).toarray())
    assert np.allclose(np.asarray(G.L.sum(1)).ravel(), 0)


def test_solve_is_pseudoinverse():
    rng = np.random.default_rng(1); b = rng.normal(size=G.n); b -= b.mean()
    x = G.solve(b); x -= x.mean()
    assert np.allclose(x, Lp @ b, atol=1e-8)
    assert abs(G.energy(b) - b @ Lp @ b) < 1e-8 * abs(b @ Lp @ b)


def test_hodge_orthogonal_and_closed():
    rng = np.random.default_rng(2); f = rng.normal(size=G.m)
    grad, h = hodge.decompose(G, f)
    assert abs(grad @ h) < 1e-8 * (f @ f)
    assert np.allclose(np.asarray(G.d @ h).ravel(), 0, atol=1e-8)
    assert abs(hodge.cycle_share(G, f) - (h @ h) / (f @ f)) < 1e-10


def test_full_coboundary_has_no_harmonic_part():
    """f = d^T k is exact for every k: its cycle share is identically zero."""
    k = np.random.default_rng(3).normal(size=G.n)
    f = np.asarray(G.d.T @ k).ravel()
    assert hodge.cycle_share(G, f) < 1e-10


def test_contact_form_is_not_exact():
    k = np.random.default_rng(4).normal(size=G.n)
    assert hodge.cycle_share(G, hodge.contact_rank_form(G, k)) > 0.05


def test_legendre_filter():
    n = 1000; F = LegendreFilter(n, 8)
    assert np.allclose(F.Q.T @ F.Q, np.eye(9), atol=1e-10)
    x = 2 * np.arange(n) / (n - 1) - 1; p = 3 * x ** 5 - x + 0.5
    assert np.allclose(F.residual(p), 0, atol=1e-9) and abs(F.removed_fraction(p) - 1) < 1e-12
    noise = np.random.default_rng(5).normal(size=n); res = F.residual(noise)
    assert np.allclose(F.Q.T @ res, 0, atol=1e-10)


def test_backbone_winding_is_N():
    """Cor. 22.4.3 of ANCCFT: the backbone ring is a cycle, so its harmonic energy is N,
    checked against the explicit projector Pi = I - d*^T (Delta*)^+ d* of the closed graph."""
    from scipy import sparse
    n, m = G.n, G.m
    dstar = sparse.hstack([G.d, sparse.csr_matrix(([-1.0, 1.0], ([0, n - 1], [0, 0])), shape=(n, 1))]).toarray()
    Pi = np.eye(m + 1) - dstar.T @ np.linalg.pinv(dstar @ dstar.T) @ dstar
    alpha = np.zeros(m + 1); alpha[:G.n_backbone] = 1.0; alpha[-1] = -1.0
    assert abs(alpha @ Pi @ alpha - n) < 1e-6 * n
    assert abs(albanese.backbone_winding_energy(G) - n) < 1e-9


def test_closure_flux_matches_projector():
    """H = 1 - R_eff(e*) on Y + e* equals 1/(1+R) with R the open end-to-end resistance."""
    from scipy import sparse
    n, m = G.n, G.m
    dstar = sparse.hstack([G.d, sparse.csr_matrix(([1.0, -1.0], ([n - 1, 0], [0, 0])), shape=(n, 1))]).toarray()
    Ls = dstar @ dstar.T; Pi = np.eye(m + 1) - dstar.T @ np.linalg.pinv(Ls) @ dstar
    alpha = np.zeros(m + 1); alpha[-1] = 1.0
    assert abs(alpha @ Pi @ alpha - albanese.closure_flux_energy(G)) < 1e-8
    e = np.zeros(n); e[0], e[-1] = 1, -1
    assert abs(albanese.end_resistance(G) - e @ Lp @ e) < 1e-8


def test_coarse_exact_integrality():
    if not os.path.exists(albanese.GP):
        print("    (PARI/GP not found; skipped)"); return
    Lc = cg.ContactGraph(compact_walk(800, seed=6)).coarse_laplacian(B=20)
    log_k, is_int, H, _ = albanese.coarse_exact_check(Lc)
    assert is_int and 0 < H <= 1 and log_k > 0


def test_kappa_rebuilt_from_anchors():
    """The gap-shuffle null starts from the exact stiffness: rebuilding from anchors is the identity."""
    pr = np.where(core.sieve(39_999))[0]
    k = core.pdtp_kappa_from_gaps(pr[0], np.diff(pr), 40_000)
    assert np.array_equal(cg.kappa_rebuilt(k), k)
    s = cg.kappa_shuffled(k, np.random.default_rng(7))
    assert np.array_equal(np.sort(s), np.sort(k))       # same multiset of bead stiffnesses


def test_stored_conformations_rebuild():
    files = cg.production_files("True")[:1] + cg.production_files("S6pp")[:1]
    if not files:
        print("    (PDTP v4.0 conformations not found; skipped)"); return
    for f in files:
        stored = np.load(f)["k"]                                  # float32 on disk
        _, k, _, _, _ = cg.load(f)                                # float64, rebuilt from anchors
        assert np.allclose(k, stored, rtol=1e-6, atol=0)          # same sequence up to float32 rounding
        assert np.array_equal(cg.kappa_rebuilt(k), k)


def test_multiple_testing():
    p = [0.01, 0.04, 0.03, 0.005]
    assert np.allclose(stats_re.holm(p), [0.03, 0.06, 0.06, 0.02])
    assert np.allclose(stats_re.bh(p), [0.02, 0.04, 0.04, 0.02])
