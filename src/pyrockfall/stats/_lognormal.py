from __future__ import annotations
from scipy.stats import lognorm as base
import numpy as np

from typing import Callable, Optional, Tuple, Union

from ._distribution import (
    Distribution,
    DistributionID,
    DistributionParameters,
    registerDistribution,
)


# ===============================================================
# LOGNORMAL (μ, σ)
# ===============================================================

@registerDistribution(DistributionID.LOGNORMAL)
class Lognormal(Distribution):
    """Lognormal with log-parameters μ and σ (i.e., ln X ~ N(μ, σ²))."""
    def __init__(self, mu: float, sigma: float):
        if sigma <= 0:
            raise ValueError("sigma must be > 0.")
        self.mu = float(mu)
        self.sigma = float(sigma)
        # SciPy: s = σ (shape), scale = exp(μ)
        self._dist = base(s=self.sigma, scale=np.exp(self.mu))

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
    def native_params(self) -> dict[str, float]: return {"mu": self.mu, "sigma": self.sigma}
    def generic_params(self) -> DistributionParameters:
        loc = np.exp(self.mu + 0.5 * self.sigma ** 2)
        scale = np.sqrt((np.exp(self.sigma ** 2) - 1) * np.exp(2 * self.mu + self.sigma ** 2))
        return DistributionParameters.relative(
            id=int(self.DistID),
            loc=loc,
            scale=scale,
            rel_min=loc,
            rel_max=np.inf,
        )


