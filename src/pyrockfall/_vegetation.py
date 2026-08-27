"""
Vegetation class for pyrockfall
==========================

This module defines the :class:`Vegetation` class, which represents the canopy geometry
above a slope. A vegetation is defined by:

- Nodes: coordinates of the geometry (mandatory).
- Elements: connectivity between nodes (optional; if omitted, sequential connectivity is assumed).
- Drag: the drag coefficient that dissipates energy when block moves through vegetation.

The class supports stochastic perturbation of nodes, material assignment,
combination of contiguous slopes, and efficient queries.
"""

import numpy as np
from scipy.stats import norm
from typing import List, Optional, Tuple, Union, Sequence
from numpy.typing import NDArray

from . import stats
from ._geometry import Geometry
from ._utils import uniqueMaterialList


class Drag:
    _instance_count = 0

    def __init__(
            self,
            name: str = "",
            coefficient: stats.DistributionLike = 0.0
        ) -> None:
        Drag._instance_count += 1
        self._name: str = name or f"Drag {Drag._instance_count}"
        # Default to deterministic zeros
        self._coefficient: stats.Distribution = stats.asDistribution(coefficient)

    # ----------------------------
    # Properties
    # ----------------------------
    @property
    def name(self) -> str:
        """Name of the material."""
        return self._name

    @property
    def coefficient(self) -> stats.Distribution:
        return self._coefficient

    @coefficient.setter
    def coefficient(self, value: stats.DistributionLike) -> None:
        self._coefficient = stats.asDistribution(value)

    @property
    def numRandomVariables(self) -> int:
        """Number of stochastic variables (always ``1``)."""
        return 1

    # ----------------------------
    # Sampling / quantiles
    # ----------------------------
    def ppf(self, q: Union[Sequence[np.ndarray], np.ndarray]) -> np.ndarray:
        if not isinstance(q, (list, tuple, np.ndarray)) or len(q) != 1:
            raise ValueError("`q` must be a sequence of one quantile array.")
        out = [
            self.coefficient.ppf(np.asarray(q[0]))
        ]
        return np.asarray(out)

    def rvs(self, num_samples: int) -> np.ndarray:
        if num_samples < 0:
            raise ValueError("`num_samples` must be non-negative.")
        samples = [
            self.coefficient.rvs(num_samples)
        ]
        return np.asarray(samples)

    # ----------------------------
    # Representation
    # ----------------------------
    def __repr__(self) -> str:  # pragma: no cover - convenience only
        return (
            f"Drag(coefficient={type(self._coefficient).__name__})"
        )

class Vegetation:
    def __init__(
        self,
        geometry: Geometry,
        drag: Union[Drag, Sequence[Drag]],
        identities: Optional[Union[Sequence[int], NDArray[np.integer]]] = None,
    ):
        self._geometry = geometry
        if identities is not None:
            # user provided table + IDs
            if not isinstance(drag, (list, np.ndarray)):
                raise TypeError("If `identities` is provided, `materials` must be a list/array of Material (the table).")
            self._drag_table = list(drag)
            ids = np.asarray(identities, dtype=int)
        else:
            # materials is either a single material or per-element list
            if isinstance(drag, Drag):
                self._drag_table = [drag]
                ids = np.zeros(len(self._geometry.elements), dtype=int)
            elif isinstance(drag, (list, np.ndarray)):
                self._drag_table, ids = uniqueMaterialList(list(drag))
            else:
                raise TypeError("`materials` must be a Material, or a list/array of Material.")
        
        self._geometry.attributes = ids

    # Delegate to geometry
    def __getattr__(self, name):
        return getattr(self._geometry, name)
    
    @property
    def drag(self) -> List[Drag]:
        return [self._drag_table[mid] for mid in self.dragIdentities]

    @property
    def dragTable(self) -> List[Drag]:
        """list of Drag: Unique drag used."""
        return self._drag_table

    @property
    def dragIdentities(self) -> np.ndarray:
        """np.ndarray of shape (E,): Drag IDs per element (indices into `dragTable`)."""
        attrs = self._geometry.attributes
        if attrs is None:
            raise ValueError("Drag IDs not stored.")
        return attrs
    

