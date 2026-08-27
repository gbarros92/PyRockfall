"""
pyrockfall's stats package
==========================

Defines distributions and statistics-related utilities for pyrockfall.
"""
from ._distribution import (
    Distribution,
    DistributionID,
    DistributionParameters,
    registerDistribution,
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
from ._vector import DistributionVector
from ._func import *
from ._func_vec import *
