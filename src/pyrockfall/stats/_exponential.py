from __future__ import annotations
from scipy.stats import expon as base
import numpy as np

from typing import Callable, Optional, Tuple, Union

from ._distribution import (
    Distribution,
    DistributionID,
    DistributionParameters,
    registerDistribution,
)


# ===============================================================
# EXPONENTIAL
# ===============================================================

@registerDistribution(DistributionID.EXPONENTIAL)
class Exponential(Distribution):
    """Exponential with rate λ (lam). Support x ≥ 0.  scale = 1/λ."""
    def __init__(self, lam: float):
        if lam <= 0:
            raise ValueError("lam (rate λ) must be > 0.")
        self.lam = float(lam)
        self._dist = base(loc=0.0, scale=1.0 / self.lam)

    def rvs(self, size: int = 1, random_state=None) -> np.ndarray: return self._dist.rvs(size=size, random_state=random_state)
    def pdf(self, x) -> np.ndarray: return self._dist.pdf(x)  # type: ignore
    def cdf(self, x) -> np.ndarray: return self._dist.cdf(x)
    def ppf(self, q) -> np.ndarray: return self._dist.ppf(q)
    def expect(self, func: Callable, lb: float = -np.inf, ub: float = np.inf) -> float: return float(self._dist.expect(func, lb=lb, ub=ub))
    def median(self) -> float: return float(self._dist.median())
    def mean(self) -> float:   return float(self._dist.mean())
    def var(self) -> float:    return float(self._dist.var())
    def interval(self, confidence: float = 1.0) -> Tuple[float, float]:
        lo, hi = self._dist.interval(confidence); return float(lo), float(hi)
    def native_params(self) -> dict[str, float]: return {"lam": self.lam}
    def generic_params(self) -> DistributionParameters:
        return DistributionParameters.relative(
            id=int(self.DistID),
            loc=1/self.lam,
            scale=0.0,
            rel_min=1/self.lam,
            rel_max=np.inf,
        )
