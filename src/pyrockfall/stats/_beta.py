from __future__ import annotations
from scipy.stats import beta as base
import numpy as np

from typing import Callable, Optional, Tuple, Union

from ._distribution import (
    Distribution,
    DistributionID,
    DistributionParameters,
    registerDistribution,
)


# ===============================================================
# BETA
# ===============================================================

@registerDistribution(DistributionID.BETA)
class Beta(Distribution):
    """Beta(α, β) on [0, 1]."""
    def __init__(self, alpha: float, beta: float):
        if alpha <= 0 or beta <= 0:
            raise ValueError("alpha and beta must be > 0.")
        self.alpha = float(alpha)
        self.beta  = float(beta)
        self._dist = base(a=self.alpha, b=self.beta)

    def rvs(self, size: int = 1, random_state=None) -> np.ndarray: return self._dist.rvs(size=size, random_state=random_state)
    def pdf(self, x) -> np.ndarray: return self._dist.pdf(x)  # type: ignore
    def cdf(self, x) -> np.ndarray: return self._dist.cdf(x)
    def ppf(self, q) -> np.ndarray: return self._dist.ppf(q)
    def expect(self, func: Callable, lb: float = 0, ub: float = 1) -> float: return float(self._dist.expect(func, lb=lb, ub=ub))
    def median(self) -> float: return float(self._dist.median())
    def mean(self) -> float:   return float(self._dist.mean())
    def var(self) -> float:    return float(self._dist.var())
    def interval(self, confidence: float = 1.0) -> Tuple[float, float]:
        lo, hi = self._dist.interval(confidence); return float(lo), float(hi)
    def native_params(self) -> dict[str, float]: return {"alpha": self.alpha, "beta": self.beta}
    def generic_params(self) -> DistributionParameters:
        den1 = self.alpha + self.beta
        den2 = den1 + 1
        loc = self.alpha / den1
        scale = np.sqrt(loc * (1 - loc) / den2)
        return DistributionParameters.relative(
            id=int(self.DistID),
            loc=loc,
            scale=scale,
            rel_min=loc,
            rel_max=1-loc,
        )
