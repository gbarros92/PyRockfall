from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum
import numpy as np
import inspect

from numbers import Real
from typing import Dict, Type, Union, Optional, Tuple, Callable

from ._utils import formatTextDist

def _make_affine(base, s, t):
    from ._affine import Affine
    return Affine(base, s, t)

def _make_deterministic(value):
    from ._deterministic import Deterministic
    return Deterministic(value)

# ===============================================================
# ENUM & PARAMS
# ===============================================================

class DistributionID(IntEnum):
    """
    Enum for supported probability distribution types.
    """
    NOTDEFINED = 0
    NONE = 1
    NORMAL = 2
    UNIFORM = 3
    TRIANGULAR = 4
    BETA = 5
    EXPONENTIAL = 6
    LOGNORMAL = 7
    GAMMA = 8


class DistributionParameters:
    def __init__(
        self,
        id: int | DistributionID,
        loc: float,
        scale: float,
        min_raw: float,
        max_raw: float,
        relative: bool,
    ) -> None:
        if id == int(DistributionID.NOTDEFINED):
            raise ValueError("Distribution id 0 (NOTDEFINED) is reserved and cannot be used.")
        self.__id = int(id)
        self.__loc = float(loc)
        self.__scale = float(scale)
        self.__min_raw = float(min_raw)
        self.__max_raw = float(max_raw)
        self.__relative = bool(relative)

    @staticmethod
    def relative(
        id: int | DistributionID,
        loc: float,
        scale: float,
        rel_min: float,
        rel_max: float,
    ) -> "DistributionParameters":
        """
        min/max are interpreted as OFFSETS relative to loc.
        """
        return DistributionParameters(
            id=id,
            loc=loc,
            scale=scale,
            min_raw=rel_min,
            max_raw=rel_max,
            relative=True,
        )

    @staticmethod
    def absolute(
        id: int | DistributionID,
        loc: float,
        scale: float,
        abs_min: float,
        abs_max: float,
    ) -> "DistributionParameters":
        """min/max are interpreted as absolute values."""
        return DistributionParameters(
            id=id,
            loc=loc,
            scale=scale,
            min_raw=abs_min,
            max_raw=abs_max,
            relative=False,
        )

    @property
    def id(self) -> int: return self.__id

    @property
    def loc(self) -> float: return self.__loc

    @property
    def mean(self) -> float: return self.__loc

    @property
    def scale(self) -> float: return self.__scale

    @property
    def std(self) -> float: return self.__scale

    @property
    def is_relative(self) -> bool: return self.__relative

    @property
    def abs_min(self) -> float:
        return (self.__loc - self.__min_raw) if self.__relative else self.__min_raw

    @property
    def abs_max(self) -> float:
        return (self.__loc + self.__max_raw) if self.__relative else self.__max_raw

    @property
    def rel_min(self) -> float:
        return self.__min_raw if self.__relative else (self.__loc - self.__min_raw)

    @property
    def rel_max(self) -> float:
        return self.__max_raw if self.__relative else (self.__max_raw - self.__loc)

    @property
    def a(self) -> float: return self.abs_min

    @property
    def b(self) -> float: return self.abs_max

    def toTxt(self, relative=True):
        txt=''
        txt += f'{self.id}, '
        txt += f'{self.loc}, '
        txt += f'{self.scale}, '
        if relative:
            txt += f'{self.rel_min}, '
            txt += f'{self.rel_max}'
        else:
            txt += f'{self.abs_min}, '
            txt += f'{self.abs_max}'
        return txt


# ===============================================================
# REGISTRY & DECORATOR
# ===============================================================

_DISTRIBUTION_REGISTRY: Dict[int, Type["Distribution"]] = {}


def registerDistribution(id_: DistributionID):
    """Decorator to register concrete distributions."""
    dist_id = int(id_)

    def deco(cls):
        if inspect.isabstract(cls):
            raise TypeError(f"{cls.__name__} is abstract and cannot be registered.")
        if dist_id == int(DistributionID.NOTDEFINED):
            raise ValueError("ID 0 (NOTDEFINED) cannot be registered.")
        if dist_id in _DISTRIBUTION_REGISTRY:
            prev = _DISTRIBUTION_REGISTRY[dist_id].__name__
            raise ValueError(f"Distribution ID {dist_id} already registered by {prev}.")
        _DISTRIBUTION_REGISTRY[dist_id] = cls
        cls.DistID = dist_id
        return cls

    return deco

# ===============================================================
# BASE CLASS
# ===============================================================

class Distribution(ABC):
    """
    Abstract base class for random variables.
    
    Defines the API for all random variable classes in pyrockfall.
    """
    DistID: DistributionID = DistributionID.NOTDEFINED  # set by @registerDistribution

    # ---- required interface ----
    @abstractmethod
    def rvs(self, size: int = 1, random_state: Optional[Union[int, np.random.Generator]] = None) -> np.ndarray:
        """Random variates"""
        raise NotImplementedError

    @abstractmethod
    def pdf(self, x) -> np.ndarray:
        """Probability density function"""
        raise NotImplementedError

    @abstractmethod
    def cdf(self, x) -> np.ndarray:
        """Cumulative distribution function"""
        raise NotImplementedError
    
    def sf(self, x) -> np.ndarray:
        """Survival function (1 - CDF)"""
        return 1.0 - self.cdf(x)

    @abstractmethod
    def ppf(self, q) -> np.ndarray:
        """Percent point function (inverse of cdf)"""
        raise NotImplementedError

    def isf(self, q) -> np.ndarray:
        """Inverse survival function (inverse of sf)"""
        return self.ppf(1.0 - q)
    
    @abstractmethod
    def expect(self, func:Callable, lb: float = -np.inf, ub: float = np.inf) -> float:
        """Expected value of a function of the random variable over [lb, ub]"""
        raise NotImplementedError

    @abstractmethod
    def median(self) -> float:
        """Median of the distribution"""
        raise NotImplementedError

    @abstractmethod
    def mean(self) -> float:
        """Mean of the distribution"""
        raise NotImplementedError
    
    @abstractmethod
    def var(self) -> float:
        """Variance of the distribution"""
        raise NotImplementedError
    
    def std(self) -> float:
        """Standard deviation of the distribution"""
        return np.sqrt(self.var())
    
    @abstractmethod
    def interval(self, confidence: float = 1.0) -> Tuple[float, float]:
        """Confidence interval having equal areas around the median"""
        raise NotImplementedError
    
    # ---- Arithmetic operators (returns Affine wrappers) ----------
    def __add__(self, other):
        if isinstance(other, Real):
            return _make_affine(self, 1.0, float(other))
        return NotImplemented

    def __radd__(self, other):
        if isinstance(other, Real):
            return _make_affine(self, 1.0, float(other))
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, Real):
            return _make_affine(self, 1.0, -float(other))
        return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, Real):
            # other - X == (-1)*X + other
            return _make_affine(self, -1.0, float(other))
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, Real):
            s = float(other)
            if s == 0.0:
                return _make_deterministic(0.0)
            return _make_affine(self, s, 0.0)
        return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, Real):
            s = float(other)
            if s == 0.0:
                raise ValueError("Division by zero.")
            return _make_affine(self, 1.0 / s, 0.0)
        return NotImplemented

    def __neg__(self):
        return _make_affine(self, -1.0, 0.0)

    def __pos__(self):
        return self
    
    # ---- Representation ----
    def __repr__(self) -> str:
        params = self.native_params()
        # Stable key order for readability
        body = ", ".join(f"{k}={formatTextDist(v)}" for k, v in params.items())
        return f"{self.__class__.__name__}({body})"

    # make __str__ the same
    __str__ = __repr__
    
    # ---- Parameter access ----
    @abstractmethod
    def native_params(self) -> dict[str, float]:
        """Constructor-ready, family-specific parameters (e.g., {'mu':..., 'sigma':...})."""
        raise NotImplementedError

    @abstractmethod
    def generic_params(self) -> DistributionParameters:
        """Family-agnostic summary: loc, scale, rel_min, rel_max"""
        raise NotImplementedError
