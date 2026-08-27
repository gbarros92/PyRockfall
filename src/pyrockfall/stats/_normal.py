from __future__ import annotations
from scipy.stats import norm as base
import numpy as np

from typing import Callable, Optional, Tuple, Union

from ._distribution import (
    Distribution,
    DistributionID,
    DistributionParameters,
    registerDistribution,
)


# ===============================================================
# NORMAL
# ===============================================================

@registerDistribution(DistributionID.NORMAL)
class Normal(Distribution):
    """Gaussian distribution N(mu, sigma^2)."""
    def __init__(self, mu: float, sigma: float):
        if sigma <= 0:
            raise ValueError("sigma must be > 0.")
        self.mu = float(mu)
        self.sigma = float(sigma)
        # scipy.stats frozen RV used for all operations
        self._dist = base(loc=self.mu, scale=self.sigma)

    # --- core API (all delegate to scipy.stats) ---
    def rvs(self, size: int = 1, random_state: Optional[Union[int, np.random.Generator]] = None) -> np.ndarray:
        return self._dist.rvs(size=size, random_state=random_state)
    def pdf(self, x) -> np.ndarray: return self._dist.pdf(x)  # type: ignore
    def cdf(self, x) -> np.ndarray: return self._dist.cdf(x)
    def ppf(self, q) -> np.ndarray: return self._dist.ppf(q)
    def expect(self, func: Callable, lb: float = -np.inf, ub: float = np.inf) -> float:
        return float(self._dist.expect(func, lb=lb, ub=ub))
    def median(self) -> float: return float(self._dist.median())
    def mean(self) -> float: return float(self._dist.mean())
    def var(self) -> float: return float(self._dist.var())
    def interval(self, confidence: float = 1.0) -> Tuple[float, float]:
        lo, hi = self._dist.interval(confidence)
        return float(lo), float(hi)
    def native_params(self) -> dict[str, float]: return {"mu": self.mu, "sigma": self.sigma}
    def generic_params(self) -> DistributionParameters:
        return DistributionParameters.relative(
            id=int(self.DistID),
            loc=self.mu,
            scale=self.sigma,
            rel_min=np.inf,
            rel_max=np.inf,
        )
