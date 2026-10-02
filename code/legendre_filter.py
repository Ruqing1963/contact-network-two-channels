# -*- coding: utf-8 -*-
"""
legendre_filter.py -- orthogonal removal of smooth positional modes along the backbone.

P_{<=K} is the orthogonal projector onto span{P_0(x_i), ..., P_K(x_i)}, x_i = 2 i/(N-1) - 1,
built by QR of the Legendre Vandermonde matrix (so the basis is exactly orthonormal on the
discrete grid).  residual(D) = D - P_{<=K} D is the divisor with every chain-scale profile of
degree <= K removed.
"""
from __future__ import annotations
import numpy as np
from numpy.polynomial import legendre as Lg


class LegendreFilter:
    def __init__(self, n: int, deg: int):
        x = 2 * np.arange(n) / (n - 1) - 1
        self.Q, _ = np.linalg.qr(Lg.legvander(x, deg))
        self.deg = deg

    def project(self, D: np.ndarray) -> np.ndarray:
        return self.Q @ (self.Q.T @ D)

    def residual(self, D: np.ndarray) -> np.ndarray:
        return D - self.project(D)

    def removed_fraction(self, D: np.ndarray) -> float:
        """Share of ||D||^2 carried by the removed modes."""
        P = self.project(D); return float(P @ P / (D @ D))
