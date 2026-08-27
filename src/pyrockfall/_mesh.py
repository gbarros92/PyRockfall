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
from ._utils import (
    getSubMesh,
    triangleCentroids,
    rotationAlign2x,
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
        """Split the mesh along the **x** axis into slabs based on triangle centroids.

        Triangles are assigned to a bin using the x-coordinate of their centroid,
        and each bin becomes an independent submesh.

        Args:
            dx: Target slab width along x (must be ``> 0``).
            du: Overlap parameter (reserved/not used).

        Returns:
            list[Mesh]: Independent submeshes (0 or more).
        """
        if dx <= 0:
            raise ValueError("dx must be > 0.")

        V = self.points
        T = self.triangles
        if V.size == 0 or T.size == 0:
            return []

        cent = triangleCentroids(V, T)  # (E, 3)
        cx = cent[:, 0]
        x_min, x_max = float(np.min(cx)), float(np.max(cx))
        if not np.isfinite([x_min, x_max]).all() or x_max <= x_min:
            return []

        n_bins = int(np.ceil((x_max - x_min) / dx))
        out: List[Mesh] = []
        for i in range(n_bins):
            x0 = x_min + i * dx
            x1 = min(x0 + dx, x_max + 1e-12)
            mask = (cx >= x0) & ((cx < x1) if i < n_bins - 1 else (cx <= x1))
            ids = np.nonzero(mask)[0]
            if ids.size == 0:
                continue

            v_sub, e_sub, _ = getSubMesh(V, T, ids)  # reindexed vertices/connectivity
            out.append(Mesh(points=v_sub, triangles=e_sub))
        return out

    def split(self, dx: float = 0.5, minPerc: float = 0.05) -> List["Mesh"]:
        """Split the mesh into **two** parts by cutting at a percentile of centroid x.

        The method scans candidate cuts on ``x`` with step ``dx`` and chooses the
        most balanced split that leaves at least ``minPerc`` fraction of triangles
        on each side.

        Args:
            dx: Granularity for scanning the cut (default ``0.5``).
            minPerc: Minimum fraction of triangles per side (default ``0.05``).

        Returns:
            list[Mesh]: ``[left_mesh, right_mesh]`` (or empty list if no valid split).
        """
        V = self.points
        T = self.triangles
        if V.size == 0 or T.size == 0:
            return []

        cent = triangleCentroids(V, T)
        cx = cent[:, 0]
        x_min, x_max = float(np.min(cx)), float(np.max(cx))
        if not np.isfinite([x_min, x_max]).all() or x_max <= x_min:
            return []

        E = T.shape[0]
        min_e = int(np.ceil(minPerc * E))

        best_score = np.inf
        best_left: Optional[NDArray[np.int64]] = None
        n_steps = max(1, int(np.ceil((x_max - x_min) / max(dx, 1e-12))))

        for i in range(1, n_steps):  # skip extremes
            x_cut = x_min + i * dx
            left = np.nonzero(cx <= x_cut)[0]
            right = np.nonzero(cx > x_cut)[0]
            if left.size < min_e or right.size < min_e:
                continue
            score = abs(left.size - right.size)
            if score < best_score:
                best_score = score
                best_left = left

        if best_left is None:
            return []

        left_ids = best_left
        right_ids = np.setdiff1d(np.arange(E, dtype=np.int64), left_ids, assume_unique=True)

        vL, eL, _ = getSubMesh(V, T, left_ids)
        vR, eR, _ = getSubMesh(V, T, right_ids)
        return [Mesh(points=vL, triangles=eL), Mesh(points=vR, triangles=eR)]

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

    def slice(
        self,
        direction: NDArray[np.floating],   # ignored here; assume coords already aligned
        increment: float,
        label: str,
        **kwargs,
    ) -> Tuple[List[Geometry], NDArray]:
        """
        Per-triangle slicing with a simple local-walk ordering.
        Assumes points are already in an aligned frame where slice planes are x = xs.
        Returns 2D nodes in the slice plane (y, z). Segment labels come from triangle materials.
        """
        V = self.points              # (N,3) aligned coords: columns [x, y, z]
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

        if increment <= 0:
            raise ValueError("increment must be > 0.")
        
        R = rotationAlign2x(direction)
        centre = V.mean(axis=0, keepdims=True)  # (1,3)
        V = (V - centre) @ R.T
        centre = centre @ R.T
        centre_x = centre[0, 0]
        centre = centre[0, 1:]

        # --- slice positions along x (aligned frame)
        x_min, x_max = float(np.min(V[:, 0])), float(np.max(V[:, 0]))
        if not np.isfinite(x_max - x_min) or x_max <= x_min:
            return [], np.array([], dtype=float)
        x_positions = np.arange(x_min + increment/2.0, x_max, increment, dtype=float)

        # --- neighbors (edge-based: edge i is v[i] -> v[(i+1)%3])
        neighbours = build_neighbours_mesh(T)

        tri_coords = V[T]                 # (E,3,3)
        tri_centroids = tri_coords.mean(axis=1)  # (E,3)
        tri_min = tri_coords.min(axis=1)  # (E,3)
        tri_max = tri_coords.max(axis=1)  # (E,3)

        x_min = tri_min[:, 0]          # (E,)
        x_max = tri_max[:, 0]          # (E,)
        z_c = tri_centroids[:, 2]          # (E,)

        # Broadcasted slab mask: (S,E)
        M = (x_min[None, :] <= x_positions[:, None]) & (x_max[None, :] >= x_positions[:, None])

        # Use -inf outside the slab so argmax ignores them
        Z = np.where(M, z_c[None, :], -np.inf)   # (S,E)

        # Argmax per slice (S,)
        init_idx = Z.argmax(axis=1)
        nodes = []
        elements = []        
        nohit = ~M.any(axis=1)
        init_idx[nohit] = -1
        current_idx = np.array(init_idx, copy=True)

        # LEFT walk
        while np.any(current_idx >= 0):
            good = current_idx >= 0
            if not np.any(good):
                break
            current_tri = T[current_idx[good], :]
            Y, Zp, Eids = tri_plane_intersections_yz_per_triangle(
                V[current_tri], x_positions[good], eps=1e-9
            )
            y = np.full(len(x_positions), -np.inf, float)
            z = np.full(len(x_positions), -np.inf, float)
            y[good] = Y[:, 0]
            z[good] = Zp[:, 0]
            nodes = [np.stack((y, z), axis=1)] + nodes
            elements = [np.array(current_idx, copy=True)] + elements
            current_idx[good] = neighbours[current_idx[good], Eids[:, 0]]

        # RIGHT walk
        current_idx = np.array(init_idx, copy=True)
        while np.any(current_idx >= 0):
            good = current_idx >= 0
            if not np.any(good):
                break
            current_tri = T[current_idx[good], :]
            Y, Zp, Eids = tri_plane_intersections_yz_per_triangle(
                V[current_tri], x_positions[good], eps=1e-9
            )
            y = np.full(len(x_positions), np.inf, float)
            z = np.full(len(x_positions), np.inf, float)
            y[good] = Y[:, 1]
            z[good] = Zp[:, 1]
            current_idx[good] = neighbours[current_idx[good], Eids[:, 1]]
            nodes += [np.stack((y, z), axis=1)]
            elements += [np.array(current_idx, copy=True)]
        
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
            profiles.append(Geometry(nodes=nodes_i+centre, elements=elements_i, attributes=tri_labels[elements_i]))

        return profiles, x_positions + centre_x
