"""
Slope class for pyrockfall
==========================

This module defines the :class:`Slope` class, which represents a slope geometry
for rockfall simulation. A slope is defined by:

- Nodes: coordinates of the geometry (mandatory).
- Elements: connectivity between nodes (optional; if omitted, sequential connectivity is assumed).
- Materials: material assignment to elements, with optional support for material tables and IDs.

The class supports stochastic perturbation of nodes, material assignment,
combination of contiguous slopes, and efficient queries.
"""

import numpy as np
from scipy.stats import norm
from typing import List, Optional, Tuple, Union, Sequence
from numpy.typing import NDArray

from ._geometry import Geometry
from ._material import Material
from ._utils import uniqueMaterialList

class Slope:
    """
    Represents a slope geometry for rockfall simulations.

    Parameters
    ----------
    nodes : np.ndarray of shape (N, D)
        Node coordinates, where `N` is the number of nodes and `D` is the number of dimensions (2 or 3).
        Internally stored transposed as shape `(D, N)` for efficient vectorised operations.
    materials : Union[Material, List[Material], np.ndarray]
        Material assignment. Can be:
        - A single :class:`Material` (applied to all elements).
        - A list/array of length `E` (one :class:`Material` per element).
        - A list of unique materials (`materialTable`) if `materialIDs` is also provided.
    elements : Optional[np.ndarray of shape (E, M)], default=None
        Connectivity array. If not provided, defaults to sequential pairs
        `[[0, 1], [1, 2], ..., [E, E+1]]` with `E = len(materials)`.
    materialIDs : Optional[np.ndarray of shape (E,)], default=None
        Integer material IDs mapping each element to an entry in `materials`
        (interpreted as the material table). If provided, `materials` must be the
        material table. If omitted, `materials` is interpreted directly as
        per-element materials.

    Attributes
    ----------
    nodes : np.ndarray of shape (N, D)
        Node coordinates (always returned as `(N, D)`).
    elements : np.ndarray of shape (E, M)
        Element connectivity array.
    materialTable : List[Material]
        Unique materials used in the slope.
    materialIDs : np.ndarray of shape (E,)
        Integer material IDs mapping each element to an entry in `materialTable`.
    nodes_std : np.ndarray of shape (D, N)
        Standard deviations for stochastic node perturbations.
    """

    def __init__(
        self,
        geometry: Geometry,
        materials: Union[Material, Sequence[Material]],
        materialIDs: Optional[Union[Sequence[int], NDArray[np.integer]]] = None,
    ):
        self._geometry = geometry

        # --- materials handling ---
        if materialIDs is not None:
            # user provided table + IDs
            if not isinstance(materials, (list, np.ndarray)):
                raise TypeError("If `materialIDs` is provided, `materials` must be a list/array of Material (the table).")
            self._material_table = list(materials)
            material_ids = np.asarray(materialIDs, dtype=int)
        else:
            # materials is either a single material or per-element list
            if isinstance(materials, Material):
                self._material_table = [materials]
                material_ids = np.zeros(len(self._geometry.elements), dtype=int)
            elif isinstance(materials, (list, np.ndarray)):
                self._material_table, material_ids = uniqueMaterialList(list(materials))
            else:
                raise TypeError("`materials` must be a Material, or a list/array of Material.")
        
        self._geometry.attributes = material_ids


    # Delegate to geometry
    def __getattr__(self, name):
        return getattr(self._geometry, name)
    

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def materials(self) -> List[Material]:
        """
        Material object associated with each element.

        Returns
        -------
        list of Material
            A list of length equal to the number of elements,
            where entry ``i`` is the :class:`Material` object of element ``i``.
        """
        return [self._material_table[mid] for mid in self.materialIDs]

    @property
    def materialTable(self) -> List[Material]:
        """list of Material: Unique materials used in this slope."""
        return self._material_table

    @property
    def materialIDs(self) -> np.ndarray:
        """np.ndarray of shape (E,): Material IDs per element (indices into `materialTable`)."""
        attrs = self._geometry.attributes
        if attrs is None:
            raise ValueError("Material IDs not stored.")
        return attrs

    # ------------------------------------------------------------------
    # Methods
    # ------------------------------------------------------------------

    def __add__(self, other: "Slope") -> "Slope":
        """
        Combine two contiguous slopes into a new one.

        Raises
        ------
        TypeError
            If `other` is not a Slope.
        ValueError
            If slopes are not contiguous.
        """
        if not isinstance(other, Slope):
            raise TypeError(f"Unsupported operand type for +: 'Slope' and '{type(other).__name__}'")
        new_geometry = self._geometry + other._geometry
        new_material_table, material_ids = uniqueMaterialList(self.materials + other.materials)
        return Slope(
            geometry=new_geometry,
            materials=new_material_table,
            materialIDs=material_ids
        )

    def __iadd__(self, other: "Slope") -> "Slope":
        """In-place version of :meth:`__add__`."""
        tmp_obj = self + other
        self._geometry = tmp_obj._geometry
        self._material_table = tmp_obj._material_table
        return self

    def __repr__(self) -> str:
        return (
            f"Slope(num_nodes={self._nodes.shape[1]}, "
            f"num_elements={len(self._elements)}, "
            f"num_materials={len(self._material_table)})"
        )
