from __future__ import annotations
import numpy as np

from typing import Callable, Optional, Tuple, Union

from ._distribution import (
    Distribution,
    DistributionID,
    DistributionParameters,
    registerDistribution,
)


# ===============================================================
# DETERMINISTIC
# ===============================================================

@registerDistribution(DistributionID.NONE)
class Deterministic(Distribution):
    """Constant-valued (degenerate) distribution at `value`."""
    def __init__(self, value: float):
        self.value = float(value)

    def rvs(self, size: int = 1, random_state: Optional[Union[int, np.random.Generator]] = None) -> np.ndarray:
        return np.full(size, self.value, dtype=float)
    def pdf(self, x) -> np.ndarray:
        # Dirac delta isn't representable as a regular pdf → zeros
        x = np.asarray(x, dtype=float)
        return np.zeros_like(x)
    def cdf(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return (x >= self.value).astype(float)
    def ppf(self, q) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        return np.full_like(q, self.value)
    def expect(self, func: Callable, lb: float = -np.inf, ub: float = np.inf) -> float:
        v = self.value
        return float(func(v)) if (lb <= v <= ub) else 0.0
    def median(self) -> float: return self.value
    def mean(self) -> float: return self.value
    def var(self) -> float: return 0.0
    def interval(self, confidence: float = 1.0) -> Tuple[float, float]:
        if not (0.0 < confidence <= 1.0):
            raise ValueError("confidence must be in (0, 1].")
        return (self.value, self.value)    
    def native_params(self) -> dict[str, float]: return {"value": self.value}
    def generic_params(self) -> DistributionParameters:
        return DistributionParameters.relative(
            id=int(self.DistID),
            loc=self.value,
            scale=0.0,
            rel_min=0.0,
            rel_max=0.0,
        )
