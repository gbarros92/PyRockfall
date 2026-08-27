from __future__ import annotations
import numpy as np

from typing import Callable, Tuple

from ._distribution import (
    Distribution,
    DistributionID,
    DistributionParameters,
)
from ._uniform import Uniform
from ._utils import formatTextDist


# ===============================================================
# TRUNCATE
# ===============================================================

class Truncate(Distribution):
    """
    Runtime wrapper that truncates any base Distribution to [lower, upper].
    - Preserves the base DistID (wrappers are not registered).
    - Uses a single generic logic via CDF/PPF mapping (no scipy.truncnorm).
    """
    DistID = DistributionID.NOTDEFINED  # class-level default; instance will mirror base

    def __init__(self, base: Distribution, lower: float | None = None, upper: float | None = None):
        if lower is None and upper is None:
            raise ValueError("Truncate needs at least one bound.")
        if (lower is not None) and (upper is not None) and not (lower < upper):
            raise ValueError("Truncate requires lower < upper when both bounds are provided.")
        self.base: Distribution = base
        self._lower: float = lower if lower is not None else -np.inf
        self._upper: float = upper if upper is not None else np.inf

        # Preserve the base family ID at the instance level
        self.DistID = base.DistID

        # Precompute tail probabilities of the base
        F = self.base.cdf
        self._Fa = 0.0 if self._lower is None else float(F(self._lower))
        self._Fb = 1.0 if self._upper is None else float(F(self._upper))
        self._den = self._Fb - self._Fa
        if self._den == 0.0:
            print("Warning: Truncate has zero probability mass in the given range. Changing to uniform distribution on [lower, upper].")
            self.base = Uniform(self._lower, self._upper)
            self._Fa = 0.0
            self._Fb = 1.0
            self._den = 1.0
        if self._den < 0.0:
            raise ValueError("Invalid truncation: F(upper) must be > F(lower).")

    # ---- core transforms ----
    def rvs(self, size: int = 1, random_state=None) -> np.ndarray:
        rng = np.random.default_rng(random_state)
        u = rng.uniform(0.0, 1.0, size=size)
        return self.ppf(u)

    def pdf(self, x) -> np.ndarray:
        x = np.asarray(x, float)
        fx = self.base.pdf(x) / self._den
        if self._lower is not None:
            fx = np.where(x < self._lower, 0.0, fx)
        if self._upper is not None:
            fx = np.where(x > self._upper, 0.0, fx)
        return fx

    def cdf(self, x) -> np.ndarray:
        x = np.asarray(x, float)
        Fx = self.base.cdf(x)
        out = (Fx - self._Fa) / self._den
        if self._lower is not None:
            out = np.where(x < self._lower, 0.0, out)
        if self._upper is not None:
            out = np.where(x >= self._upper, 1.0, out)
        return np.clip(out, 0.0, 1.0)

    def ppf(self, q) -> np.ndarray:
        q = np.asarray(q, float)
        return self.base.ppf(self._Fa + q * self._den)
    
    def expect(self, func: Callable, lb: float = -np.inf, ub: float = np.inf) -> float:
        # Truncated RV T has pdf f_T(t) = f_X(t) / den on [a,b], else 0
        lo = max(lb, self._lower)
        hi = min(ub, self._upper)
        if not (lo < hi):
            return 0.0
        num = self.base.expect(func, lb=lo, ub=hi)
        return float(num / self._den)
    
    def median(self) -> float:
        return float(self.ppf(0.5))

    def mean(self) -> float:
        ex = self.base.expect(lambda t: t, lb=self._lower, ub=self._upper)
        return float(ex / self._den)

    def var(self) -> float:
        ex  = self.base.expect(lambda t: t,   lb=self._lower, ub=self._upper) / self._den
        ex2 = self.base.expect(lambda t: t*t, lb=self._lower, ub=self._upper) / self._den
        return float(ex2 - ex * ex)

    def interval(self, confidence: float = 1.0) -> Tuple[float, float]:
        if not (0.0 < confidence <= 1.0):
            raise ValueError("confidence must be in (0, 1].")
        alpha = (1.0 - confidence) / 2.0
        lo = float(self.ppf(alpha))
        hi = float(self.ppf(1.0 - alpha))
        return (lo, hi)
    
    def __repr__(self) -> str:
        return f"({self.base!r}) in [{formatTextDist(self._lower)}, {formatTextDist(self._upper)}]"
    __str__ = __repr__
    
    def native_params(self) -> dict[str, float]:
        return self.base.native_params()

    def generic_params(self) -> DistributionParameters:
        g = self.base.generic_params()
        # base endpoints
        a_base, b_base = g.a, g.b
        # apply truncation (cap only the endpoints)
        a = max(a_base, self._lower)
        b = min(b_base, self._upper)
        # loc/scale unchanged for generic view
        return DistributionParameters.relative(
            id=int(self.DistID),
            loc=g.loc,
            scale=g.scale,
            rel_min=float(g.loc - a),
            rel_max=float(b - g.loc),
        )
