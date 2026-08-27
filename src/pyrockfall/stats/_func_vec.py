from __future__ import annotations
from collections.abc import Sequence
import numpy as np

from typing import TypeAlias, overload, Optional
import numpy.typing as npt

from ._vector import DistributionVector
from ._func import DistributionLike


# ===============================================================
# COERCE
# ===============================================================

DistributionVectorLike: TypeAlias = (
    "DistributionVector | Sequence[DistributionLike] | npt.NDArray[np.generic] | DistributionLike"
)

@overload
def asDistributionVector(value: "DistributionVector") -> "DistributionVector": ...
@overload
def asDistributionVector(value: "DistributionVectorLike", *, length: Optional[int] = ...) -> "DistributionVector": ...

def asDistributionVector(value, *, length: Optional[int] = None) -> "DistributionVector":
    # Fast path
    if isinstance(value, DistributionVector):
        if length is None or len(value) == length or (len(value) == 1 and length is not None):
            # Rebuild to enforce broadcast/length if needed
            return DistributionVector(value.tolist(), length=length)
        raise ValueError(f"Length mismatch: got {len(value)} items, expected {length}.")
    # Generic path — let DistributionVector handle coercion/broadcast
    return DistributionVector(value, length=length)
