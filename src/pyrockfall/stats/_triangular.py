from __future__ import annotations
from scipy.stats import triang as base
import numpy as np

from typing import Callable, Optional, Tuple, Union

from ._distribution import (
    Distribution,
    DistributionID,
    DistributionParameters,
    registerDistribution,
)


# ===============================================================
# TRIANGULAR
# ===============================================================

@registerDistribution(DistributionID.TRIANGULAR)
class Triangular(Distribution):
    """Triangular distribution on [lower, upper] with mode at `mode`."""
    def __init__(self, lower: float, mode: float, upper: float):
        lower = float(lower); mode = float(mode); upper = float(upper)
        if not (lower < mode < upper):
            raise ValueError("Triangular requires lower < mode < upper.")
        self.lower, self.mode, self.upper = lower, mode, upper
        # SciPy parameterization: c = (mode - loc) / scale, loc=lower, scale=upper-lower
        c = (mode - lower) / (upper - lower)
        self._dist = base(c=c, loc=lower, scale=upper - lower)

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
    def native_params(self) -> dict[str, float]: return {"lower": self.lower, "mode": self.mode, "upper": self.upper}
    def generic_params(self) -> DistributionParameters:
        return DistributionParameters.relative(
            id=int(self.DistID),
            loc=self.mode,
            scale=0.0,
            rel_min=self.mode - self.lower,
            rel_max=self.upper - self.mode,
        )
