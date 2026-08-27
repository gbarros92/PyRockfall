from __future__ import annotations
from collections.abc import MutableSequence, Sequence
import numpy as np

from typing import Union, Optional

from ._distribution import Distribution
from ._deterministic import Deterministic
from ._func import asDistribution, DistributionLike


# ===============================================================
# VECTORS
# ===============================================================

class DistributionVector(MutableSequence[Distribution]):
    """
    1D container of Distributions with robust coercion.

    Examples
    --------
    DistributionVector([0.0, 2.0])                     # -> [Det(0), Det(2)]
    DistributionVector([Normal(2, 0.1), 0.0])          # mixed
    DistributionVector(0.0, length=3)                  # broadcast -> [Det(0), Det(0), Det(0)]
    v = DistributionVector([0.0, 1.0]); v[0] = 5       # per-entry coercion
    """

    def __init__(self, values: "DistributionLike | Sequence[DistributionLike] | np.ndarray",
                 length: Optional[int] = None):
        data = self._coerce_any(values)

        if length is not None:
            if len(data) == 1:
                # broadcast scalar/len-1 to requested length
                data = [data[0]] * length
            elif len(data) != length:
                raise ValueError(f"Length mismatch: got {len(data)} items, expected {length}.")
        self._data: list[Distribution] = data

    # ---------- core coercion ----------
    @staticmethod
    def _coerce_any(values) -> list[Distribution]:
        # Single Distribution-like: wrap and return [it]
        try_single = False
        if isinstance(values, Distribution):
            try_single = True
        else:
            # Numpy scalars or Python scalars or duck-typed .to_distribution
            is_scalar_like = (
                hasattr(values, "to_distribution") or
                isinstance(values, (float, int, np.floating, np.integer))
            )
            if is_scalar_like:
                try_single = True

        if try_single:
            return [asDistribution(values)]

        # Sequence/array path
        if isinstance(values, np.ndarray):
            flat = values.ravel().tolist()
        elif isinstance(values, Sequence):
            flat = list(values)
        else:
            raise TypeError(f"Unsupported type for DistributionVector: {type(values)}")

        return [asDistribution(v) for v in flat]

    # ---------- MutableSequence interface ----------
    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return DistributionVector(self._data[idx])
        return self._data[idx]

    def __setitem__(self, idx, value) -> None:
        if isinstance(idx, slice):
            # slice assignment: broadcast scalar or match lengths
            items = self._coerce_any(value)
            span = len(range(*idx.indices(len(self))))
            if len(items) == 1:
                items = items * span
            if len(items) != span:
                raise ValueError(f"Slice length {span} does not match {len(items)} new items.")
            self._data[idx] = items
        else:
            self._data[idx] = asDistribution(value)

    def __delitem__(self, idx) -> None:
        del self._data[idx]

    def insert(self, idx: int, value) -> None:
        self._data.insert(idx, asDistribution(value))

    # ---------- convenience ----------
    def tolist(self) -> list[Distribution]:
        return list(self._data)

    def __iter__(self):
        return iter(self._data)

    def __repr__(self) -> str:
        inner = ", ".join(repr(d) for d in self._data)
        return f"DistributionVector([{inner}])"

    # ---------- vectorised operations ----------
    def rvs(self, size: int = 1, random_state: Optional[Union[int, np.random.Generator]] = None) -> np.ndarray:
        """
        Draw independent samples from each component.
        Returns array shape (size, len(self)).
        """
        cols = [d.rvs(size=size, random_state=random_state) for d in self._data]
        return np.column_stack(cols) if cols else np.empty((size, 0))

    def mean(self) -> np.ndarray:
        return np.array([d.mean() for d in self._data], dtype=float)

    def std(self) -> np.ndarray:
        return np.array([d.std() for d in self._data], dtype=float)

    def var(self) -> np.ndarray:
        return np.array([d.var() for d in self._data], dtype=float)

    # ---------- scalar arithmetic (elementwise) ----------
    def _apply_scalar(self, fn) -> "DistributionVector":
        return DistributionVector([fn(d) for d in self._data])

    def __add__(self, other):
        if isinstance(other, (float, int, np.floating, np.integer)):
            return self._apply_scalar(lambda d: d + float(other))
        return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, (float, int, np.floating, np.integer)):
            return self._apply_scalar(lambda d: d - float(other))
        return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, (float, int, np.floating, np.integer)):
            return self._apply_scalar(lambda d: float(other) - d)
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, (float, int, np.floating, np.integer)):
            s = float(other)
            return self._apply_scalar(lambda d: (Deterministic(0.0) if s == 0 else d * s))
        return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, (float, int, np.floating, np.integer)):
            s = float(other)
            if s == 0.0:
                raise ValueError("Division by zero.")
            return self._apply_scalar(lambda d: d / s)
        return NotImplemented
