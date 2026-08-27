from __future__ import annotations
import numpy as np

from numbers import Real
from typing import Protocol, runtime_checkable, TypeAlias, overload

from ._distribution import (
    Distribution,
    DistributionID,
    DistributionParameters,
)
from ._affine import Affine
from ._truncate import Truncate
from ._deterministic import Deterministic
from ._normal import Normal
from ._uniform import Uniform
from ._triangular import Triangular
from ._beta import Beta
from ._exponential import Exponential
from ._lognormal import Lognormal
from ._gamma import Gamma


# ===============================================================
# COERCE
# ===============================================================

@runtime_checkable
class SupportsToDistribution(Protocol):
    def to_distribution(self) -> "Distribution": ...

DistributionLike: TypeAlias = (
    "Distribution | SupportsToDistribution | Real | np.floating | np.integer | float | int"
)

@overload
def asDistribution(value: "Distribution") -> "Distribution": ...
@overload
def asDistribution(value: SupportsToDistribution) -> "Distribution": ...
@overload
def asDistribution(value: Real) -> "Distribution": ...
@overload
def asDistribution(value: np.floating) -> "Distribution": ...
@overload
def asDistribution(value: np.integer) -> "Distribution": ...
@overload
def asDistribution(value: float) -> "Distribution": ...
@overload
def asDistribution(value: int) -> "Distribution": ...

def asDistribution(value):  # runtime
    """Coerce a value into a Distribution instance.

    If `value` is already a Distribution, it is returned as-is.
    If `value` is a scalar (float or int), it is wrapped as a Deterministic.

    Args:
        value: A Distribution instance or a scalar.
    Returns:
        Distribution: The corresponding Distribution instance.
    """
    # Already a Distribution
    if isinstance(value, Distribution):
        return value

    # Custom objects that declare a converter
    if isinstance(value, SupportsToDistribution):
        return value.to_distribution()

    # Numpy scalars (float64/int64/etc.)
    if isinstance(value, (np.floating, np.integer)):
        return Deterministic(float(value))

    # Python numeric scalars (exclude bool)
    if isinstance(value, Real) and not isinstance(value, bool):
        return Deterministic(float(value))

    if isinstance(value, (float, int)):
        return Deterministic(float(value))

    raise TypeError(f"Cannot convert type {type(value)} to Distribution.")


# ===============================================================
# FACTORY
# ===============================================================

def makeDistribution(p: DistributionParameters) -> Distribution:
    """
    Reconstruct a Distribution from generic parameters:
      p.id, p.loc, p.scale, p.rel_min, p.rel_max
    Then apply Affine (when needed) and Truncate to [a,b] if finite.
    """
    did = int(p.id)
    a, b = p.a, p.b
    fin_a, fin_b = np.isfinite(a), np.isfinite(b)

    def _truncate_if_needed(rv: Distribution) -> Distribution:
        lo = a if fin_a else None
        hi = b if fin_b else None
        if lo is None and hi is None:
            return rv
        return Truncate(rv, lower=lo, upper=hi)

    # ---------------- Deterministic ----------------
    if did == int(DistributionID.NONE):
        return Deterministic(p.loc)

    # ---------------- Normal ----------------
    if did == int(DistributionID.NORMAL):
        if p.scale <= 0:
            return Deterministic(p.loc)
        base = Normal(mu=p.loc, sigma=p.scale)
        return _truncate_if_needed(base)

    # ---------------- Uniform ----------------
    if did == int(DistributionID.UNIFORM):
        if not (fin_a and fin_b):
            raise ValueError("Uniform requires finite [a, b].")
        if not (a < b):
            raise ValueError("Uniform needs a < b.")
        # generic mapping: a,b are exactly (lower, upper)
        return Uniform(lower=a, upper=b)

    # ---------------- Triangular ----------------
    if did == int(DistributionID.TRIANGULAR):
        if not (fin_a and fin_b):
            raise ValueError("Triangular requires finite [a, b].")
        if not (a < p.loc < b):
            raise ValueError("Triangular needs a < mode(loc) < b.")
        return Triangular(lower=a, mode=p.loc, upper=b)

    # ---------------- Beta ----------------
    if did == int(DistributionID.BETA):
        # Natural support is [0,1]. If [a,b] != [0,1], we treat it as an affine map
        # y = s*x + t, with s=b-a, t=a; invert mean/std to base, then re-apply affine.
        if not (fin_a and fin_b) or not (a < b):
            raise ValueError("Beta requires finite interval [a, b] with a < b.")

        s = b - a
        t = a

        # Map generic loc/scale to the base on [0,1]
        loc0 = (p.loc - t) / s
        scale0 = p.scale / abs(s) if s != 0 else 0.0

        if not (0.0 < loc0 < 1.0):
            raise ValueError("Beta: inferred base mean(loc0) must be in (0,1).")
        max_sd0 = np.sqrt(loc0 * (1.0 - loc0))
        if not (0.0 < scale0 < max_sd0):
            raise ValueError("Beta: inferred base std(scale0) must be in (0, sqrt(loc0*(1-loc0))).")

        S = loc0 * (1.0 - loc0) / (scale0 * scale0) - 1.0
        alpha = loc0 * S
        beta_ = (1.0 - loc0) * S

        base = Beta(alpha=alpha, beta=beta_)
        # The affine exactly restores [a,b], so truncation is not needed beyond that
        return Affine(base, scale=s, translate=t)

    # ---------------- Exponential ----------------
    if did == int(DistributionID.EXPONENTIAL):
        # Your generic mapping: loc = 1/λ, scale = 0, rel_min = 1/λ, rel_max = +inf.
        if p.loc <= 0:
            raise ValueError("Exponential requires loc > 0 (loc = 1/λ).")
        lam = 1.0 / p.loc
        base = Exponential(lam=lam)
        # If bounds differ from natural [0, +inf), truncate accordingly.
        return _truncate_if_needed(base)

    # ---------------- Lognormal ----------------
    if did == int(DistributionID.LOGNORMAL):
        # Your generic mapping uses mean (loc) and std (scale) of X.
        if p.loc <= 0 or p.scale < 0:
            raise ValueError("Lognormal requires loc > 0 and scale >= 0.")
        if p.scale == 0.0:
            return Deterministic(p.loc)
        sigma2 = np.log(1.0 + (p.scale * p.scale) / (p.loc * p.loc))
        sigma = np.sqrt(sigma2)
        mu = np.log(p.loc) - 0.5 * sigma2
        base = Lognormal(mu=mu, sigma=sigma)
        return _truncate_if_needed(base)

    # ---------------- Gamma ----------------
    if did == int(DistributionID.GAMMA):
        # Your generic mapping uses mean (loc) and std (scale).
        if p.loc <= 0 or p.scale <= 0:
            raise ValueError("Gamma requires loc > 0 and scale > 0.")
        alpha = (p.loc / p.scale) ** 2
        lam = p.loc / (p.scale * p.scale)
        base = Gamma(alpha=alpha, lam=lam)
        return _truncate_if_needed(base)

    raise ValueError(f"Unknown distribution id {p.id}")
