# -*- coding: utf-8 -*-
"""
stats_re.py -- sequence-level statistics and multiple-testing corrections.

The true arm is ONE fixed sequence observed in n_T trajectories; every surrogate trajectory uses
its own fresh sequence (m = 1).  Under the null "the true sequence is a typical member of the
surrogate class",
    x_T = mu + s_T + eps,   s_T ~ (0, sigma_seq^2),   eps ~ (0, sigma_th^2),
the surrogate values have variance sigma_seq^2 + sigma_th^2 and the true mean has variance
sigma_seq^2 + sigma_th^2/n_T about mu.  sigma_th^2 is estimated from the true arm's own spread
(one sequence, many trajectories), sigma_seq^2 = max(0, s_S^2 - s_T^2), and
    Z_seq = (xbar_T - xbar_S) / sqrt(sigma_seq^2 + s_T^2/n_T + s_S^2/n_S),
referred to a t distribution with n_S - 1 degrees of freedom.  This is the m = 1 form of the
random-effects test of PDTP v4.0, Sec. 7.3.
"""
from __future__ import annotations
import numpy as np
from scipy import stats


def seq_level(T, S):
    T, S = np.asarray(T, float), np.asarray(S, float)
    sT2, sS2 = T.var(ddof=1), S.var(ddof=1)
    s2_seq = max(0.0, sS2 - sT2)
    den = np.sqrt(s2_seq + sT2 / len(T) + sS2 / len(S))
    Z = (T.mean() - S.mean()) / den
    return dict(delta=T.mean() - S.mean(), sigma_seq=np.sqrt(s2_seq), sigma_th=np.sqrt(sT2), Z_seq=Z,
                p_seq=2 * stats.t.sf(abs(Z), len(S) - 1),
                p_welch=stats.ttest_ind(T, S, equal_var=False).pvalue,
                rank=int(np.sum(S > T.mean())), n_S=len(S))


def holm(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); adj = np.empty(n); run = 0.0
    for i, j in enumerate(o):
        run = max(run, (n - i) * p[j]); adj[j] = min(1.0, run)
    return adj


def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)[::-1]; adj = np.empty(n); run = 1.0
    for i, j in enumerate(o):
        run = min(run, p[j] * n / (n - i)); adj[j] = run
    return adj
