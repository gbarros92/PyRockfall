from __future__ import annotations
from scipy.stats import gamma as base
import numpy as np

from typing import Callable, Optional, Tuple, Union

from ._distribution import (
    Distribution,
    DistributionID,
    DistributionParameters,
    registerDistribution,
)


# ===============================================================
# GAMMA
# ===============================================================

@registerDistribution(DistributionID.GAMMA)
class Gamma(Distribution):
    """Gamma with shape α and rate λ (i.e., θ = 1/λ is the scale)."""
    def __init__(self, alpha: float, lam: float):
        if alpha <= 0 or lam <= 0:
            raise ValueError("alpha and lam (rate λ) must be > 0.")
        self.alpha = float(alpha)
        self.lam   = float(lam)
        # SciPy's gamma uses 'a' for shape and 'scale' (θ). Here θ = 1/λ.
        self._dist = base(a=self.alpha, scale=1.0 / self.lam)

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
    def native_params(self) -> dict[str, float]: return {"alpha": self.alpha, "lam": self.lam}
    def generic_params(self) -> DistributionParameters:
        return DistributionParameters.relative(
            id=int(self.DistID),
            loc=self.alpha / self.lam,
            scale=np.sqrt(self.alpha) / self.lam,
            rel_min=self.alpha / self.lam,
            rel_max=np.inf,
        )
