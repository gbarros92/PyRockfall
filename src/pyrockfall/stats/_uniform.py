from __future__ import annotations
from scipy.stats import uniform as base
import numpy as np

from typing import Callable, Optional, Tuple, Union

from ._distribution import (
    Distribution,
    DistributionID,
    DistributionParameters,
    registerDistribution,
)


# ===============================================================
# UNIFORM
# ===============================================================

@registerDistribution(DistributionID.UNIFORM)
class Uniform(Distribution):
    """Continuous Uniform distribution on [lower, upper]."""
    def __init__(self, lower: float, upper: float):
        if not (lower < upper):
            raise ValueError("Uniform requires lower < upper.")
        self.lower = float(lower)
        self.upper = float(upper)
        self._dist = base(loc=self.lower, scale=self.upper - self.lower)

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
    def native_params(self) -> dict[str, float]: return {"lower": self.lower, "upper": self.upper}
    def generic_params(self) -> DistributionParameters:
        return DistributionParameters.relative(
            id=int(self.DistID),
            loc=(self.upper + self.lower) * 0.5,
            scale=0.0,
            rel_min=self.upper*0.5 - self.lower*0.5,
            rel_max=self.upper*0.5 - self.lower*0.5,
        )
