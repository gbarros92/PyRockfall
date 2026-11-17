# _mesh.py
# ========

"""
Mesh geometry (:class:`Mesh`) that implements the :class:`~.geometry.Geometry` API.

- Inherits :class:`~.geometry.Geometry` only (no simulation behaviour).
- Stores vertices and triangle connectivity in NumPy arrays.
- Uses :mod:`open3d` *optionally* for file I/O of non-``.npz`` formats.
- Save/load is driven by :class:`~.geometry.Geometry` via
  :meth:`_save_core_dict` / :meth:`_load_core`.
"""

from __future__ import annotations

from typing import Tuple, List, Mapping, Optional

import numpy as np
from numpy.typing import NDArray

try:  # optional dependency, used only for non-.npz formats
    import open3d as o3d  # type: ignore
    _HAS_O3D = True
except Exception:  # pragma: no cover
    _HAS_O3D = False

from ._model import Model
from ._geometry import Geometry
from ._geometry3d import Geometry3D
from ._frame import CurvilinearFrame
from ._utils import (
    getSubMesh,
    triangleCentroids,
    build_neighbours_mesh,
    tri_plane_intersections_yz_per_triangle
)

ArrayF = NDArray[np.floating]
ArrayI = NDArray[np.integer]

class Mesh(Model):
    """Triangle mesh that implements the :class:`~.geometry.Geometry` interface.

    This class stores vertices and triangle connectivity as NumPy arrays and
    provides small geometry utilities (segment/split and 2D slicing into
    :class:`~._geometry.Geometry` polylines). It has **no** simulation behaviour.

    Args:
        points: Optional initial vertex coordinates of shape ``(N, 3)`` or ``(N, 2)``.
            If 2D is provided, a zero ``z`` column is appended internally.
        triangles: Optional initial triangle connectivity of shape ``(E, 3)`` with
            0-based indices.
        colors: Optional per-vertex colors, attached as attribute ``"colors"``.
            Must have first dimension ``N``.
        normals: Optional per-vertex normals, attached as attribute ``"normals"``.
            Must have first dimension ``N``.
        attrs: Optional mapping of additional per-vertex attributes. Each array must
            have first dimension ``N`` and shape ``(N,)`` or ``(N, k)``.

    Notes:
        - Vertices are stored as ``float64`` by default.
        - Triangles are stored as ``int32``.
        - Persistence is handled by :class:`~.geometry.Geometry.save` / ``load``,
          which call :meth:`_save_core_dict` and :meth:`_load_core` to serialize
          only the geometry. Per-vertex attributes are included/excluded via the
          ``attributes`` argument on those base methods.
    """

    # --------------------------
    # Construction
    # --------------------------
    def __init__(
        self,
        points: Optional[np.ndarray] = None,
        triangles: Optional[np.ndarray] = None,
        *,
        colors: Optional[np.ndarray] = None,
        normals: Optional[np.ndarray] = None,
        attrs: Optional[Mapping[str, np.ndarray]] = None,
    ) -> None:
        super().__init__()
        self._points: ArrayF = np.zeros((0, 3), dtype=float)
        self._triangles: ArrayI = np.zeros((0, 3), dtype=np.int32)

        if points is not None:
            self.points = points  # triggers attribute sync
        if triangles is not None:
            self.triangles = triangles  # validates/set connectivity

        if colors is not None:
            self.set_attr("colors", np.asarray(colors))
        if normals is not None:
            self.set_attr("normals", np.asarray(normals))
        if attrs:
            for k, v in attrs.items():
                self.set_attr(k, np.asarray(v))

    # ----------------------------
    # Geometry API — points
    # ----------------------------
    @property
    def points(self) -> ArrayF:
        """Vertices as an array of shape ``(N, 3)``."""
        return self._points

    @points.setter
    def points(self, value: np.ndarray) -> None:
        arr = np.asarray(value, dtype=float)
        if arr.ndim != 2 or arr.shape[1] not in (2, 3):
            raise ValueError("Mesh.points must have shape (N, 3) or (N, 2).")
        if arr.shape[1] == 2:
            arr = np.column_stack([arr, np.zeros((arr.shape[0],), dtype=arr.dtype)])
        self._points = arr
        self._on_points_replaced()
        # If triangles reference out-of-range indices after point changes,
        # it is user's responsibility to update connectivity accordingly.

    # ----------------------------
    # Mesh connectivity
    # ----------------------------
    @property
    def triangles(self) -> ArrayI:
        """Triangle connectivity as an array of shape ``(E, 3)`` (int32)."""
        return self._triangles

    @triangles.setter
    def triangles(self, value: np.ndarray) -> None:
        arr = np.asarray(value, dtype=np.int32)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError("Mesh.triangles must have shape (E, 3).")
        if arr.size > 0:
            if arr.min() < 0:
                raise ValueError("Mesh.triangles contains negative indices.")
            if self.points.size and arr.max() >= self.points.shape[0]:
                raise ValueError("Mesh.triangles index out of range for current points.")
        self._triangles = arr

    # ----------------------------
    # Attribute dict API
    # ----------------------------

    def _check_attr_size(self, name: str, arr: np.ndarray) -> None:
        """
        Check if tentative attribute has appropriate size.
        `arr` must have first dimension N or T.
        Accepts (N,), (N, k), (T,), (T, k) with any dtype.
        """
        pts = self.points
        if pts.size == 0:
            raise ValueError("Set points before attaching attributes.")
        tri = self.triangles
        if tri.size == 0:
            raise ValueError("Set triangles before attaching attributes.")
        if (arr.shape[0] != pts.shape[0]) and (arr.shape[0] != tri.shape[0]):
            raise ValueError(f"Attribute '{name}' length {arr.shape[0]} "
                                f"!= number of points {pts.shape[0]}."
                                f"!= number of triangles {tri.shape[0]}.")
        
    def _on_points_replaced(self) -> None:
        """
        Default policy when points array is replaced.
        If any attribute length != number of points, clear them all.
        """
        num_points = self.points.shape[0]
        num_triangles = self.triangles.shape[0]
        for k, arr in list(self._attrs.items()):
            if arr.shape[0] == num_triangles:
                continue
            if arr.shape[0] != num_points:
                self.del_attr(k)
        self._invalidate_caches()

    def apply_point_mask_(self, mask: np.ndarray) -> None:
        """
        Keep only points[mask]. Triangles that reference a dropped point are
        removed and the remaining connectivity is reindexed. Per-point and
        per-triangle attributes are kept consistent.
        """
        mask = np.asarray(mask, dtype=bool)
        V = self.points
        T = self.triangles
        N, E = V.shape[0], T.shape[0]
        if mask.shape != (N,):
            raise ValueError("Mask must be boolean with length N points.")

        keep_tri = mask[T].all(axis=1) if E else np.zeros(0, dtype=bool)
        new_index = np.cumsum(mask) - 1
        new_T = new_index[T[keep_tri]].astype(np.int32).reshape(-1, 3)

        attrs: dict[str, np.ndarray] = {}
        for k, arr in self._attrs.items():
            if arr.shape[0] == E:
                attrs[k] = arr[keep_tri]
            elif arr.shape[0] == N:
                attrs[k] = arr[mask]

        # A subset lives in the same space: keep the frame and removed positions
        frame, removed = self._frame, self._removed
        self._points = V[mask]
        self._triangles = new_T
        self._attrs = attrs
        self._invalidate_caches()
        self._frame, self._removed = frame, removed

    def _steep_mask(self, max_nz: float = 0.7) -> Optional[NDArray[np.bool_]]:
        """
        Points on the face rather than on the floor or berms. Uses the
        ``"normals"`` attribute if present, otherwise the triangles around each
        point (area-weighted, independent of the triangle winding).
        """
        mask = super()._steep_mask(max_nz)
        if mask is not None:
            return mask
        V = self.points
        T = self.triangles
        if V.size == 0 or T.size == 0:
            return None
        fn = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])  # 2 * area * normal
        area = np.linalg.norm(fn, axis=1)
        ids = T.ravel()
        tot = np.bincount(ids, weights=np.repeat(area, 3), minlength=V.shape[0])
        vert = np.bincount(ids, weights=np.repeat(np.abs(fn[:, 2]), 3), minlength=V.shape[0])
        return (tot > 0) & (vert < max_nz * tot)

    def _triangle_strike(self) -> NDArray[np.float64]:
        """``s`` coordinate of each triangle centroid (NaN if any vertex is NaN)."""
        s = self.frame.xyz_to_sde(self.points)[:, 0]
        return s[self.triangles].mean(axis=1)

    def _submesh(self, ids: np.ndarray) -> "Mesh":
        """New mesh with the given triangles, their points and all attributes."""
        V = self.points
        T = self.triangles
        N, E = V.shape[0], T.shape[0]
        v_sub, e_sub, vertex_ids = getSubMesh(V, T, ids)
        sub = Mesh(points=v_sub, triangles=e_sub)
        for k, arr in self._attrs.items():
            if arr.shape[0] == E:
                sub._attrs[k] = arr[ids]
            elif arr.shape[0] == N:
                sub._attrs[k] = arr[vertex_ids]
        sub._frame = self._frame  # a subset lives in the same space
        return sub

    # ----------------------------
    # Persistence hooks (used by Geometry.save/load)
    # ----------------------------
    def _save_core_dict(self, **kwargs) -> dict[str, np.ndarray]:
        """Return the minimal geometry payload for serialization.

        Returns:
            dict: Contains ``{"points": (N, 3), "triangles": (E, 3)}``.
        """
        return {
            "points": np.asarray(self.points),
            "triangles": np.asarray(self.triangles, dtype=np.int32),
        }

    @classmethod
    def _load_core(cls, filename: str, **kwargs) -> "Mesh":
        """Build a :class:`Mesh` instance from a file.

        Only geometry (points/connectivity) is loaded here. Per-vertex attributes
        are loaded by :meth:`Geometry.load` if requested via its ``attributes`` arg.

        Supported formats:
            - ``.npz``: expects arrays ``points`` and ``triangles``.
            - ``.ply``, ``.obj``, ``.stl``, ``.off``: via :mod:`open3d`.

        Args:
            filename: Path to the file.

        Returns:
            Mesh: A new mesh with geometry loaded.
        """
        path = str(filename)
        low = path.lower()

        if low.endswith(".npz"):
            data = np.load(path)
            if "points" not in data or "triangles" not in data:
                raise ValueError("'.npz' must contain 'points' and 'triangles'.")
            pts = np.asarray(data["points"], dtype=float)
            tri = np.asarray(data["triangles"], dtype=np.int32)
            return cls(points=pts, triangles=tri)

        # Non-.npz formats via Open3D
        if not _HAS_O3D:
            raise ImportError("open3d is required to load non-.npz mesh formats.")
        m = o3d.io.read_triangle_mesh(path)
        if not m.has_triangles():
            raise ValueError("Loaded mesh has no triangles.")
        pts = np.asarray(m.vertices, dtype=float)
        tri = np.asarray(m.triangles, dtype=np.int32)
        return cls(points=pts, triangles=tri)

    # ----------------------------
    # Higher-level operations
    # ----------------------------
    def segment(self, dx: float, du: float = 0.0) -> List["Mesh"]:
        """Split the mesh along the strike into windows of length ``dx``.

        Triangles are assigned to a window using the ``s`` coordinate of their
        centroid in the curvilinear frame (see :meth:`Model.sde`), so the windows
        follow the wall around bends. Each window becomes an independent submesh
        that keeps the attributes and shares the frame of this mesh.

        Args:
            dx: Window length along the strike (must be ``> 0``).
            du: Overlap between consecutive windows (``0 <= du < dx``).

        Returns:
            list[Mesh]: Independent submeshes (0 or more), ordered by increasing ``s``.
        """
        if dx <= 0:
            raise ValueError("dx must be > 0.")
        if du < 0 or du >= dx:
            raise ValueError("du must satisfy 0 <= du < dx.")

        V = self.points
        T = self.triangles
        if V.size == 0 or T.size == 0:
            return []

        s = self.frame.xyz_to_sde(V)[:, 0]
        cs = s[T].mean(axis=1)  # (E,)
        ok = np.isfinite(cs)
        if not ok.any():
            return []
        s_min, s_max = float(np.nanmin(s)), float(np.nanmax(s))
        if not np.isfinite([s_min, s_max]).all() or s_max <= s_min:
            return []

        out: List[Mesh] = []
        for s0 in np.arange(s_min, s_max, dx - du):
            s1 = s0 + dx
            mask = ok & (cs >= s0) & ((cs < s1) if s1 < s_max else (cs <= s_max))
            ids = np.nonzero(mask)[0]
            if ids.size == 0:
                continue
            out.append(self._submesh(ids))
        return out

    def split(self, dx: float = 0.5, minPerc: float = 0.05) -> List["Mesh"]:
        """Split the mesh into **two** parts at the sharpest bend of the wall.

        The cut is made at the vertex of the frame's polyline with the largest
        turning angle that leaves at least ``minPerc`` of the triangles on each
        side. A wall without a bend is cut at the median ``s`` of the triangle
        centroids, which balances the two parts.

        Args:
            dx: Unused; kept for backward compatibility.
            minPerc: Minimum fraction of triangles per side (default ``0.05``).

        Returns:
            list[Mesh]: ``[before, after]`` along increasing ``s`` (or an empty
            list if no valid split exists).
        """
        V = self.points
        T = self.triangles
        if V.size == 0 or T.size == 0:
            return []

        cs = self._triangle_strike()
        s_cut = self._split_position(cs, minPerc)
        if not np.isfinite(s_cut):
            return []

        left_ids = np.nonzero(cs <= s_cut)[0]
        right_ids = np.nonzero(cs > s_cut)[0]
        if left_ids.size == 0 or right_ids.size == 0:
            return []
        return [self._submesh(left_ids), self._submesh(right_ids)]

    def slopeGeometry(
        self,
        *,
        label: str,
        nodes_std: Optional[np.ndarray] = None,
    ) -> Geometry:
        """Construct a :class:`Geometry` using labels from a mesh attribute.

        The attribute ``label`` can be either:
        - Per-triangle (length == E): used directly as ``materialIDs``; or
        - Per-vertex (length == N): converted to per-triangle IDs by:
            (i)   if all three vertex labels are equal → use that value;
            (ii)  if exactly two are equal → use the repeated label;
            (iii) if all different → use the label of the vertex closest
                  to the triangle's barycentre.

        Args:
            materials: Material table (indexable by the derived integer IDs).
            label: Name of the mesh attribute containing material IDs.
                (shape (N,), (N,1), (E,) or (E,1)).
            nodes_std: Optional node standard deviations, shape ``(N, 3)``.

        Returns:
            Geometry: A slope geometry with ``elements`` = triangles, and
            per-element ``materialIDs`` consistent with ``materials``.

        Raises:
            ValueError: If the mesh is empty, label shape is invalid, or
                        derived IDs fall outside ``materials`` range.
        """
        V = self.points                       # (N, 3)
        T = self.triangles                    # (E, 3)
        if V.size == 0 or T.size == 0:
            raise ValueError("Mesh must have points and triangles to build Geometry3D.")

        if not self.has_attr(label):
            raise ValueError(f"Attribute '{label}' not found.")
        raw = np.asarray(self.get_attr(label))
        if raw.ndim == 2 and raw.shape[1] == 1:
            raw = raw[:, 0]
        raw = raw.astype(int, copy=False)

        N = V.shape[0]
        E = T.shape[0]

        # Case 1: per-triangle labels → use directly
        if raw.shape[0] == E:
            mat_ids = raw.astype(int, copy=False)

        # Case 2: per-node labels → convert via mode
        elif raw.shape[0] == N:
            mat_ids = self.to_tri_attr(label, method="mode").astype(int, copy=False)

        else:
            raise ValueError(
                f"Attribute '{label}' must have length N={N} (per-node) or E={E} (per-triangle); got {raw.shape[0]}."
            )

        # nodes_std (optional)
        ns = None
        if nodes_std is not None:
            ns = np.asarray(nodes_std, dtype=float)
            if ns.shape != V.shape:
                raise ValueError(f"`nodes_std` must have shape {V.shape} (N, 3).")

        # Build Geometry3D
        return Geometry3D(
            nodes=V,
            elements=T,
            nodes_std=ns,
            attributes=mat_ids,
        )
    
    def to_tri_attr(
        self,
        attr: str,
        *,
        method: str = "mean",
        mode_tol: float = 1e-7,
        out: str | None = None,
    ) -> np.ndarray:
        """Convert a per-point attribute to a per-triangle attribute.

        Args:
            attr: Name of a per-point attribute. Supports shapes (N,) or (N,k).
            method: 'mean' for arithmetic mean; 'mode' for most-common label
                with barycentre tie-break (requires integer, 1-D).
            mode_tol: Tolerance for mode equality (only used if method='mode').
            out: If provided, stores the result as a mesh attribute with this name.

        Returns:
            np.ndarray: Per-triangle attribute of shape (E,) or (E,k).

        Raises:
            ValueError: If attribute is missing/shape-mismatched, or 'mode' used on
                non-integer / non-1D data.
        """
        if not self.has_attr(attr):
            raise ValueError(f"Attribute '{attr}' not found.")
        A = np.asarray(self.get_attr(attr))
        if A.ndim == 2 and A.shape[1] == 1:
            A = A[:, 0]

        V = self.points              # (N,3)
        T = self.triangles           # (E,3)
        N = V.shape[0]
        E = T.shape[0]

        if A.shape[0] != N:
            raise ValueError(f"Attribute '{attr}' must be per-point with length N={N}.")

        m = method.lower()
        if m in ("mean", "avg", "average"):
            tri_vals = A[T].mean(axis=1)   # (E,) if A was (N,), else (E,k)
        elif m in ("mode", "majority"):
            L = A[T]                                              # (E,3)
            # Equality helpers (scalar vs vector; int vs float with tol)
            if A.ndim == 1:
                if np.issubdtype(A.dtype, np.floating) and mode_tol is not None:
                    eq01 = np.isclose(L[:, 0], L[:, 1], atol=mode_tol, rtol=0)
                    eq02 = np.isclose(L[:, 0], L[:, 2], atol=mode_tol, rtol=0)
                    eq12 = np.isclose(L[:, 1], L[:, 2], atol=mode_tol, rtol=0)
                else:
                    eq01 = (L[:, 0] == L[:, 1])
                    eq02 = (L[:, 0] == L[:, 2])
                    eq12 = (L[:, 1] == L[:, 2])
            else:
                # vector-valued attribute (N, k) -> (E, 3, k)
                if np.issubdtype(A.dtype, np.floating) and mode_tol is not None:
                    eq01 = np.all(np.isclose(L[:, 0, :], L[:, 1, :], atol=mode_tol, rtol=0), axis=1)
                    eq02 = np.all(np.isclose(L[:, 0, :], L[:, 2, :], atol=mode_tol, rtol=0), axis=1)
                    eq12 = np.all(np.isclose(L[:, 1, :], L[:, 2, :], atol=mode_tol, rtol=0), axis=1)
                else:
                    eq01 = np.all(L[:, 0, :] == L[:, 1, :], axis=1)
                    eq02 = np.all(L[:, 0, :] == L[:, 2, :], axis=1)
                    eq12 = np.all(L[:, 1, :] == L[:, 2, :], axis=1)

            all_equal = eq01 & eq02
            has_pair  = (eq01 | eq02 | eq12) & (~all_equal)
            all_diff  = ~(all_equal | has_pair)

            # Initialise output
            if A.ndim == 1:
                tri_vals = np.empty(E, dtype=A.dtype)
            else:
                tri_vals = np.empty((E, A.shape[1]), dtype=A.dtype)

            # All equal → take any vertex value
            tri_vals[all_equal] = L[all_equal, 0] if A.ndim == 1 else L[all_equal, 0, :]

            # Exactly two equal → take the repeated one (majority)
            if np.any(has_pair):
                # pick based on which pair matches
                if A.ndim == 1:
                    maj = np.where(eq01, L[:, 0],
                        np.where(eq02, L[:, 0], L[:, 1]))
                    tri_vals[has_pair] = maj[has_pair]
                else:
                    maj = np.empty_like(tri_vals)
                    sel01 = has_pair & eq01
                    sel02 = has_pair & (~eq01) & eq02
                    sel12 = has_pair & (~eq01) & (~eq02) & eq12
                    if np.any(sel01): tri_vals[sel01] = L[sel01, 0, :]
                    if np.any(sel02): tri_vals[sel02] = L[sel02, 0, :]
                    if np.any(sel12): tri_vals[sel12] = L[sel12, 1, :]

            # All different → take the vertex whose point is closest to the triangle barycentre
            if np.any(all_diff):
                V = self.points
                T = self.triangles
                tri_pts = V[T]                          # (E, 3, 3)
                C = tri_pts.mean(axis=1)                # (E, 3)
                d2 = np.sum((tri_pts - C[:, None, :])**2, axis=2)  # (E, 3)
                idx_min = np.argmin(d2, axis=1)         # (E,)
                if A.ndim == 1:
                    pick = L[np.arange(E), idx_min]
                    tri_vals[all_diff] = pick[all_diff]
                else:
                    rows = np.arange(E)
                    tri_vals[all_diff] = L[rows[all_diff], idx_min[all_diff], :]
        else:
            raise ValueError("method must be one of {'mean','mode'} (or 'avg','majority').")

        if out:
            self.set_attr(out, tri_vals)
        return tri_vals

    def to_point_attr(
        self,
        attr: str,
        *,
        method: str = "mean",
        out: str | None = None,
        fill: float | int | None = None,
    ) -> np.ndarray:
        """Convert a per-triangle attribute to a per-point attribute.

        Args:
            attr: Name of a per-triangle attribute. Supports shapes (E,) or (E,k).
            method: 'mean' for arithmetic mean of incident triangles;
                'mode' for most-common integer label among incident triangles.
                Ties are broken by picking the label of the adjacent triangle
                whose centroid is closest to the point.
            out: If provided, stores the result as a mesh attribute with this name.
            fill: Value for isolated points (not referenced by any triangle).
                Defaults to np.nan for float arrays, and -1 for integer arrays.

        Returns:
            np.ndarray: Per-point attribute of shape (N,) or (N,k).

        Raises:
            ValueError: If attribute is missing/shape-mismatched, or 'mode'
                used on non-integer / non-1D data.
        """
        if not self.has_attr(attr):
            raise ValueError(f"Attribute '{attr}' not found.")
        A = np.asarray(self.get_attr(attr))
        if A.ndim == 2 and A.shape[1] == 1:
            A = A[:, 0]

        V = self.points                    # (N,3)
        T = self.triangles                 # (E,3)
        N = V.shape[0]
        E = T.shape[0]

        if A.shape[0] != E:
            raise ValueError(f"Attribute '{attr}' must be per-triangle with length E={E}.")

        m = method.lower()
        if m in ("mean", "avg", "average"):
            # scatter-add from triangles to points, then divide by counts
            point_count = np.zeros(N, dtype=np.int64)
            if A.ndim == 1:
                point_sum = np.zeros(N, dtype=float)
                tri_ids = np.repeat(np.arange(E), 3)           # (3E,)
                point_ids = T.ravel()                           # (3E,)
                np.add.at(point_sum, point_ids, A[tri_ids])
                np.add.at(point_count, point_ids, 1)
                with np.errstate(invalid="ignore", divide="ignore"):
                    point_vals = point_sum / point_count
                # fill isolated
                if fill is None:
                    fill_val = np.nan
                else:
                    fill_val = fill
                iso = point_count == 0
                if np.any(iso):
                    point_vals[iso] = fill_val
            else:  # (E,k)
                k = A.shape[1]
                point_sum = np.zeros((N, k), dtype=float)
                tri_ids = np.repeat(np.arange(E), 3)
                point_ids = T.ravel()
                np.add.at(point_sum, point_ids, A[tri_ids])
                np.add.at(point_count, point_ids, 1)
                with np.errstate(invalid="ignore", divide="ignore"):
                    point_vals = point_sum / point_count[:, None]
                if fill is None:
                    fill_val = np.nan
                else:
                    fill_val = fill
                iso = point_count == 0
                if np.any(iso):
                    point_vals[iso, :] = fill_val
        elif m in ("mode", "majority"):
            if A.ndim != 1 or not np.issubdtype(A.dtype, np.integer):
                raise ValueError("'mode' requires integer, 1-D per-triangle labels.")

            # Flatten triangle memberships
            tri_ids = np.repeat(np.arange(E), 3)            # (3E,)
            point_ids = T.ravel()                            # (3E,)

            # For tie-breaks, we need centroids and distances point↔centroid
            cent = triangleCentroids(V, T)                  # (E,3)
            d = np.linalg.norm(cent[tri_ids] - V[point_ids], axis=1)  # (3E,)

            labs = A[tri_ids]                               # (3E,)

            # Group by (point, label): counts and min distance
            order = np.lexsort((labs, point_ids))
            n_sorted = point_ids[order]
            l_sorted = labs[order]
            d_sorted = d[order]

            # group boundaries for consecutive equal (point,label)
            change = (np.diff(n_sorted) != 0) | (np.diff(l_sorted) != 0)
            grp_starts = np.r_[0, 1 + np.nonzero(change)[0]]
            grp_ends   = np.r_[grp_starts[1:], n_sorted.size]

            grp_points  = n_sorted[grp_starts]
            grp_labels = l_sorted[grp_starts]
            grp_counts = grp_ends - grp_starts

            # min distance per (point,label) group
            grp_min_d = np.minimum.reduceat(d_sorted, grp_starts)

            # Now pick, per point, the label with max count; tie → min distance
            # Sort groups by (point ASC, count DESC, min_d ASC)
            order2 = np.lexsort((grp_min_d, -grp_counts, grp_points))
            points_sorted = grp_points[order2]
            labels_sorted = grp_labels[order2]

            # first occurrence per point in this order is the winner
            uniq_points, first_idx = np.unique(points_sorted, return_index=True)
            chosen_labels = labels_sorted[first_idx]

            # Build output array and fill isolated points
            point_vals = np.empty(N, dtype=A.dtype)
            point_vals.fill(-1 if fill is None else fill)

            point_vals[uniq_points] = chosen_labels
        else:
            raise ValueError("method must be one of {'mean','mode'} (or 'avg','majority').")

        if out:
            self.set_attr(out, point_vals)
        return point_vals

    def section(
        self,
        increment: float,
        label: str,
        *,
        frame: Optional[CurvilinearFrame] = None,
        **kwargs,
    ):
        """Extract 2D profiles along the strike of the mesh, and their points in 3D.

        Each section is the exact cut of the triangles by a surface of constant
        ``s`` of the curvilinear frame (see :meth:`slice`), so every node lies on
        a triangle edge and every segment carries the material of the triangle it
        crosses. The mesh itself is not modified.

        Args:
            increment: Spacing between consecutive sections along the strike. A
                section must not pass exactly through mesh vertices; change the
                spacing slightly if :meth:`slice` reports that.
            label: Name of the attribute containing material IDs (per point or
                per triangle).
            frame: Frame to slice in. Defaults to the frame of the mesh.

        Returns:
            profiles: One 2D :class:`Geometry` per section, crest to toe, with
            nodes as (horizontal distance from the crest node, height above the
            lowest node) and material IDs per segment.

            sections_pc: A :class:`PointCloud` with all section nodes in the
            mesh's coordinates, the material attribute (``label``) and a
            ``"Profile"`` attribute numbering the sections.

        Raises:
            ValueError: If no valid sections are found.
        """
        return super().section(increment, label, frame=frame, **kwargs)

    def slice(
        self,
        direction: Optional[NDArray[np.floating]] = None,
        increment: Optional[float] = None,
        label: Optional[str] = None,
        *,
        frame: Optional[CurvilinearFrame] = None,
        **kwargs,
    ) -> Tuple[List[Geometry], NDArray]:
        """
        Per-triangle slicing with a simple local-walk ordering.

        By default the slices follow the strike of the wall: they are surfaces of
        constant ``s`` of the curvilinear frame (see :meth:`Model.sde`) and the
        returned 2D nodes are ``(d, e)``. With a ``direction`` the slices are
        parallel planes normal to it and the nodes are the two remaining axes of
        the points rotated so that ``direction`` is +X.

        Segment labels come from triangle materials. Triangles are intersected in
        the slicing coordinates, where a triangle spanning a bend of the wall is
        only approximately planar (the error is second order in its size).

        Args:
            direction: ``None`` to slice along the strike, or a fixed direction.
            increment: Spacing between consecutive slices.
            label: Name of the attribute containing material IDs (per point or
                per triangle).
            frame: Frame to slice in. Defaults to the frame of the mesh.

        Returns:
            Profiles as :class:`Geometry` objects and the slice positions (``s``,
            or the distance along ``direction``).
        """
        V = self.points              # (N,3)
        T = self.triangles           # (E,3)
        E = T.shape[0]
        if V.size == 0 or E == 0:
            return [], np.array([], dtype=float)

        # --- materials / labels
        if not self.has_attr(label):
            raise ValueError(f"Attribute '{label}' not found.")
        raw = np.asarray(self.get_attr(label))
        if raw.ndim == 2 and raw.shape[1] == 1:
            raw = raw[:, 0]
        raw = raw.astype(int, copy=False)

        N = V.shape[0]
        if raw.shape[0] == E:
            tri_labels = raw
        elif raw.shape[0] == N:
            tri_labels = self.to_tri_attr(label, method="mode").astype(int, copy=False)
        else:
            raise ValueError(f"Attribute '{label}' must have length N={N} or E={E}; got {raw.shape[0]}.")

        bad = (tri_labels < 0)
        if np.any(bad):
            raise ValueError(f"Triangle material IDs out of range: {np.unique(tri_labels[bad]).tolist()}.")

        if increment is None or increment <= 0:
            raise ValueError("increment must be > 0.")

        # --- slicing coordinates: column 0 is constant on a slice
        V, _ = self._slicing_coords(direction, frame)
        if not np.isfinite(V[:, 0]).any():
            return [], np.array([], dtype=float)

        # --- slice positions along x (slicing coordinates)
        x_min, x_max = float(np.nanmin(V[:, 0])), float(np.nanmax(V[:, 0]))
        if not np.isfinite(x_max - x_min) or x_max <= x_min:
            return [], np.array([], dtype=float)
        x_positions = np.arange(x_min + increment/2.0, x_max, increment, dtype=float)

        # --- neighbors (edge-based: edge i is v[i] -> v[(i+1)%3])
        neighbours = build_neighbours_mesh(T)

        tri_coords = V[T]                 # (E,3,3)
        # Never walk into a triangle with a point outside the frame (NaN)
        tri_ok = np.isfinite(tri_coords).all(axis=(1, 2))
        if not tri_ok.all():
            neighbours = np.where((neighbours >= 0) & ~tri_ok[neighbours], -1, neighbours)
        tri_centroids = tri_coords.mean(axis=1)  # (E,3)
        tri_min = tri_coords.min(axis=1)  # (E,3)
        tri_max = tri_coords.max(axis=1)  # (E,3)

        x_min = tri_min[:, 0]          # (E,)
        x_max = tri_max[:, 0]          # (E,)
        z_c = tri_centroids[:, 2]          # (E,)

        # Start triangle of each slice: the highest one it crosses. Triangles are
        # sorted by their lower bound once, so each slice only looks at the few
        # that can reach it (memory stays O(E) instead of a slices x triangles table).
        by_min = np.argsort(x_min, kind="stable")
        x_min_sorted = x_min[by_min]
        finite_ext = (x_max - x_min)[np.isfinite(x_max - x_min)]
        max_extent = float(finite_ext.max()) if finite_ext.size else 0.0
        lo = np.searchsorted(x_min_sorted, x_positions - max_extent, side="left")
        hi = np.searchsorted(x_min_sorted, x_positions, side="right")
        init_idx = np.full(x_positions.size, -1, dtype=np.int64)
        for i in range(x_positions.size):
            cand = by_min[lo[i]:hi[i]]
            cand = cand[x_max[cand] >= x_positions[i]]
            if cand.size:
                init_idx[i] = cand[np.argmax(z_c[cand])]
        nodes = []
        elements = []
        current_idx = np.array(init_idx, copy=True)

        def _check_walk(steps: int) -> None:
            # A walk can visit each triangle at most once; more means it is going
            # round in circles (inconsistent connectivity).
            if steps > E:
                raise ValueError(
                    "Slice walk did not terminate; check the mesh connectivity or "
                    "change `increment` slightly."
                )

        n_slices = x_positions.size

        def _walk(side: int):
            """Follow each cut from its start triangle to the edge of the mesh.

            The cut leaves the start triangle through its lower (`side` 0) or
            higher (`side` 1) intersection in the second coordinate. In every
            following triangle it leaves through the intersection it did not come
            in by, so the cut may turn back on itself (rough or overhanging
            surface). Returns, per step, the exit node of every slice, the
            triangle it was in and the triangle it moved to.
            """
            cur = np.array(init_idx, copy=True)
            prev = np.full(n_slices, -2, dtype=np.int64)
            out_nodes, out_cur, out_next = [], [], []
            steps = 0
            while np.any(cur >= 0):
                steps += 1
                _check_walk(steps)
                good = np.flatnonzero(cur >= 0)
                tri_ids = cur[good]
                Y, Zp, Eids = tri_plane_intersections_yz_per_triangle(
                    V[T[tri_ids, :]], x_positions[good], eps=1e-9
                )
                nb = np.where(Eids >= 0, neighbours[tri_ids[:, None], np.maximum(Eids, 0)], -1)  # (g,2)
                if steps == 1:
                    pick = np.full(good.size, side, dtype=np.int64)
                else:
                    pick = np.where(nb[:, 0] == prev[good], 1, 0)
                rows = np.arange(good.size)
                valid = Eids[rows, pick] >= 0
                nxt = np.where(valid, nb[rows, pick], -1)
                nxt[nxt == init_idx[good]] = -1          # a closed cut: stop where it started

                node = np.full((n_slices, 2), np.nan)
                node[good[valid], 0] = Y[rows, pick][valid]
                node[good[valid], 1] = Zp[rows, pick][valid]
                here = np.full(n_slices, -1, dtype=np.int64)
                here[good[valid]] = tri_ids[valid]
                there = np.full(n_slices, -1, dtype=np.int64)
                there[good] = nxt
                out_nodes.append(node)
                out_cur.append(here)
                out_next.append(there)

                prev[good] = tri_ids
                cur[good] = nxt
            return out_nodes, out_cur, out_next

        # One way from the start triangle, then the other. The first walk owns the
        # start triangle; the second contributes the triangles it moves into.
        left_nodes, left_cur, _ = _walk(0)
        right_nodes, _, right_next = _walk(1)
        nodes = left_nodes[::-1] + right_nodes
        elements = left_cur[::-1] + right_next

        nodes = np.stack(nodes, axis=1)
        elements = np.stack(elements, axis=1)

        profiles = []
        for i in range(len(x_positions)):
            nodes_i = nodes[i]
            mask = np.isfinite(nodes_i).all(axis=1)   # (N,) True if both y and z finite
            nodes_i = nodes_i[mask]
            elements_i = elements[i]
            elements_i = elements_i[elements_i >= 0]
            if nodes_i.shape[0] < 2 or elements_i.shape[0] < 1:
                raise ValueError("Invalid profile")
            if nodes_i.shape[0] != elements_i.shape[0] + 1:
                raise ValueError("Invalid profile")
            profiles.append(Geometry(nodes=nodes_i, attributes=tri_labels[elements_i]))

        return profiles, x_positions
