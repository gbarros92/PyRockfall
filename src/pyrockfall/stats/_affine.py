from __future__ import annotations
import numpy as np

from typing import Callable, Tuple

from ._distribution import (
    Distribution,
    DistributionID,
    DistributionParameters,
)
from ._utils import formatTextDist


# ===============================================================
# AFFINE
# ===============================================================

class Affine(Distribution):
    """
    Runtime wrapper applying Y = scale * X + translate to a base Distribution.
    - Preserves the base DistID (wrappers are not registered).
    """
    DistID = DistributionID.NOTDEFINED  # class-level default; instance will mirror base

    def __init__(self, base: Distribution, scale: float = 1.0, translate: float = 0.0):
        if scale == 0:
            raise ValueError("Affine scale must be non-zero.")
        s_acc = float(scale)
        t_acc = float(translate)

        # Compose with any nested Affine(base, s_b, t_b): y = s_acc*(s_b*x + t_b) + t_acc
        b = base
        while isinstance(b, Affine):
            s_old = s_acc
            s_acc = s_acc * b.scale
            t_acc = s_old * b.translate + t_acc
            b = b.base  # step down to the next base

        self.base = b
        self.scale = s_acc
        self.translate = t_acc

        # Preserve family ID from the ultimate non-Affine base
        self.DistID = b.DistID

    # Helpers
    def _z(self, y):
        return (np.asarray(y, float) - self.translate) / self.scale
    
    def rvs(self, size: int = 1, random_state=None) -> np.ndarray:
        # Using base.rvs is fine even for negative scale (it just flips samples)
        return self.scale * self.base.rvs(size=size, random_state=random_state) + self.translate

    def pdf(self, y) -> np.ndarray:
        z = self._z(y)
        return self.base.pdf(z) / abs(self.scale)

    def cdf(self, y) -> np.ndarray:
        z = self._z(y)
        if self.scale > 0:
            return self.base.cdf(z)
        # For negative scale, CDF flips: F_Y(y) = 1 - F_X(z)
        return 1.0 - self.base.cdf(z)

    def ppf(self, q) -> np.ndarray:
        q = np.asarray(q, float)
        if self.scale > 0:
            x = self.base.ppf(q)
        else:
            x = self.base.ppf(1.0 - q)
        return self.scale * x + self.translate
    
    def expect(self, func: Callable, lb: float = -np.inf, ub: float = np.inf) -> float:
        # Map Y-interval to X-interval
        x1 = (lb - self.translate) / self.scale
        x2 = (ub - self.translate) / self.scale
        xlb, xub = (min(x1, x2), max(x1, x2))
        # E[f(Y)] = E[f(scale*X + translate)]
        return float(self.base.expect(lambda x: func(self.scale * x + self.translate),
                                      lb=xlb, ub=xub))

    def median(self) -> float:
        return float(self.ppf(0.5))

    def mean(self) -> float:
        return float(self.scale * self.base.mean() + self.translate)

    def var(self) -> float:
        return float((self.scale ** 2) * self.base.var())

    def interval(self, confidence: float = 1.0) -> Tuple[float, float]:
        lo_x, hi_x = self.base.interval(confidence)
        lo_y = self.scale * lo_x + self.translate
        hi_y = self.scale * hi_x + self.translate
        if self.scale >= 0:
            return (float(lo_y), float(hi_y))
        # Negative scale reverses endpoints
        return (float(min(lo_y, hi_y)), float(max(lo_y, hi_y)))
    
    def __repr__(self) -> str:
        return f"({self.base!r}) * {formatTextDist(self.scale)} + {formatTextDist(self.translate)}"
    __str__ = __repr__

    def native_params(self) -> dict[str, float]:
        return self.base.native_params()

    def generic_params(self) -> DistributionParameters:
        g = self.base.generic_params()
        # transform endpoints and center by y = s*x + t
        a_y = self.scale * g.a + self.translate
        b_y = self.scale * g.b + self.translate
        loc_y = self.scale * g.loc + self.translate
        scale_y = abs(self.scale) * g.scale
        return DistributionParameters.absolute(
            id=int(self.DistID),
            loc=float(loc_y),
            scale=float(scale_y),
            abs_min=float(a_y),
            abs_max=float(b_y),
        )
