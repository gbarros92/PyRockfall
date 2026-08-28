"""
model.py
===========

Base class :class:`Model` for storing and manipulating rockface geometries.
Provides translation, rotation, clipping, centroid computation, and slicing
utilities. Subclasses must implement storage and I/O logic.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Iterable, List, Tuple, Union, Optional, Mapping, Sequence, Dict

import numpy as np
from scipy.optimize import minimize
from numpy.typing import NDArray

from ._geometry import Geometry
from ._filesystem import read_properties, write_ply_with_attrs, write_npz_full_fidelity, write_xyz_ascii, write_pcd_minimal
from ._utils import rotationAlign2x

class Model(ABC):
    """
    Base class for geometric data used in rockfall workflows.

    This class stores and manipulates point coordinates (2D or 3D) and provides
    general-purpose geometric utilities (translate, rotate, centroid, bounding box).
    Simulation-agnostic. Subclasses are responsible for concrete storage,
    loading/saving, and any topology (e.g., faces/triangles).

    ### Subclassing requirements
    - You **must** implement:
        - `points` property (getter & setter): returns/sets an array of shape (N, D), D in {2, 3}.
        - `save(output_file)`: persist model to disk in your preferred format.
        - `segment(dx, du)`: produce windowed segments as model objects.
        - `split(dx, minPerc)`: split into two model objects using your rule.
        - `slice(direction, increment, **kwargs)`: return extracted `Geometry` objects.

    - You **may** override:
        - `clip(min_val, max_val)`: default raises NotImplementedError because
          clipping model may require updating topology; base class cannot guess that.
    """

    # ----------------------------
    # Construction & core storage
    # ----------------------------
    def __init__(self) -> None:
        # Arbitrary per-point attribute map: name -> array of shape (N,) or (N, k)
        self._attrs: dict[str, np.ndarray] = {}
        self._normal: Optional[NDArray[np.float64]] = None  # cache for normalVector()

    # ----------------------------
    # Abstract point storage API
    # ----------------------------
    @property
    @abstractmethod
    def points(self) -> np.ndarray:
        """(N, D) array view of point coordinates."""
        raise NotImplementedError

    @points.setter
    @abstractmethod
    def points(self, value: np.ndarray) -> None:
        """Replace all points; subclasses must call _on_points_replaced(oldN, newN)."""
        raise NotImplementedError

    # ----------------------------
    # Attribute dict API
    # ----------------------------
    @property
    def attrs(self) -> Mapping[str, np.ndarray]:
        """
        Read-only mapping of per-point attributes: name -> ndarray.
        Each value has shape (N,) or (N, k) where N == number of points.
        """
        return self._attrs

    def has_attr(self, name: str) -> bool:
        return name in self._attrs

    def get_attr(self, name: str) -> np.ndarray:
        """Return the attribute array (by reference). Raises KeyError if missing."""
        return self._attrs[name]

    def set_attr(self, name: str, value: Optional[np.ndarray]) -> None:
        """
        Set/replace a per-point attribute. `value` must have first dimension N.
        Accepts (N,), (N, k) with any dtype. Use None or empty to remove.
        Stores by reference (no forced copy).
        """
        if value is None or (isinstance(value, np.ndarray) and value.size == 0):
            self._attrs.pop(name, None)
            return

        arr = np.asarray(value)
        self._check_attr_size(name, arr)
        self._attrs[name] = arr

    def del_attr(self, name: str) -> None:
        """Remove an attribute if present."""
        self._attrs.pop(name, None)

    def list_attrs(self) -> list[str]:
        """List available attribute names."""
        return list(self._attrs.keys())

    def attr_shape(self, name: str) -> tuple[int, ...]:
        return tuple(self._attrs[name].shape)

    def attr_dtype(self, name: str):
        return self._attrs[name].dtype

    def _check_attr_size(self, name: str, arr: np.ndarray) -> None:
        """
        Check if tentative attribute has appropriate size.
        `arr` must have first dimension N.
        Accepts (N,), (N, k) with any dtype.
        """
        pts = self.points
        if pts.size == 0:
            raise ValueError("Set points before attaching attributes.")
        if arr.shape[0] != pts.shape[0]:
            raise ValueError(f"Attribute '{name}' length {arr.shape[0]} "
                                f"!= number of points {pts.shape[0]}.")

    # ----------------------------
    # Keeping attributes in sync
    # ----------------------------
    def _on_points_replaced(self) -> None:
        """
        Default policy when points array is replaced.
        If any attribute length != number of points, clear them all.
        """
        n = self.points.shape[0]
        for k, arr in list(self._attrs.items()):
            if arr.shape[0] != n:
                self.del_attr(k)
        self._normal = None  # invalidate cached normal

    def apply_point_mask_(self, mask: np.ndarray) -> None:
        """
        Keep only points[mask]. Subclasses must override to update *topology*;
        this base method only handles attributes.
        """
        mask = np.asarray(mask, dtype=bool)
        pts = self.points
        if mask.shape != (pts.shape[0],):
            raise ValueError("Mask must be boolean with length N points.")
        # Reindex attributes consistently
        for k, arr in list(self._attrs.items()):
            self._attrs[k] = arr[mask]
        self.points = pts[mask]  # triggers _on_points_replaced

    def apply_point_index_map_(self, new_index_of_old: np.ndarray, new_N: Optional[int] = None) -> None:
        """
        Reorder/duplicate/drop points according to an index map.
        new_index_of_old[i] = j means old point i goes to new row j.
        If new_N is omitted, it is inferred as max(j)+1.

        This supports decimation, duplication, or arbitrary reindexing.
        """
        idx = np.asarray(new_index_of_old, dtype=np.int64)
        old_N = self.points.shape[0]
        if idx.shape != (old_N,):
            raise ValueError("new_index_of_old must have shape (N_old,)")

        if new_N is None:
            new_N = int(idx.max()) + 1
        if idx.min() < 0 or idx.max() >= new_N:
            raise ValueError("Index map contains out-of-range target indices.")

        # Build new arrays
        pts_old = self.points
        D = pts_old.shape[1]
        pts_new = np.empty((new_N, D), dtype=pts_old.dtype)
        pts_new[idx] = pts_old
        self.points = pts_new  # triggers _on_points_replaced

        # Reindex attributes
        for k, arr in list(self._attrs.items()):
            # shape (N, ...)  →  shape (new_N, ...)
            new_shape = (new_N,) + arr.shape[1:]
            new_arr = np.empty(new_shape, dtype=arr.dtype)
            new_arr[idx] = arr
            self._attrs[k] = new_arr

    # ----------------------------
    # I/O
    # ----------------------------
    def save(
        self,
        filename: str,
        attributes: Sequence[str] | str | None = '*',
        *,
        overwrite: bool = False,
        allow_lossy: bool = False,      # if True, permit dropping attrs unsupported by target format
        comments: dict[str, str] | None = None,  # extra PLY comments (units, CRS, etc.)
    ) -> None:
        """
        Save model to various formats.

        Parameters
        ----------
        filename : str
            Path with extension: .ply, .npz, .xyz, .pcd (limited), ...
        attributes : {'*', None, sequence of names}
            Which attributes from self._attrs to include (when supported).
            - '*': include all attributes
            - None: include none
            - sequence: include only those present
        overwrite : bool
            Overwrite target file if exists.
        allow_lossy : bool
            If False (default), refuse formats that would drop attributes.
        tri_reduce : {'mean','mode',None}
            For Mesh only: reduce per-triangle attributes to per-vertex before saving PLY/XYZ.
        comments : dict[str,str] | None
            Stored as PLY comments (e.g., {'units':'m','crs':'EPSG:28356'}).
        """
        ext = os.path.splitext(filename)[1].lower()
        if not overwrite and os.path.exists(filename):
            raise FileExistsError(f"{filename} exists. Use overwrite=True.")

        core = self._save_core_dict()

        # Collect selected attrs
        if attributes == '*':
            attrs = dict(self._attrs)
        elif attributes is None:
            attrs = {}
        else:
            attrs = {k: self._attrs[k] for k in attributes if k in self._attrs}

        # Route by extension
        if ext == ".ply":
            write_ply_with_attrs(filename, core, attrs, comments=comments)
        elif ext == ".npz":
            write_npz_full_fidelity(filename, core, self._attrs)
        elif ext == ".xyz":
            write_xyz_ascii(filename, core, attrs, allow_lossy=allow_lossy)
        elif ext == ".pcd":
            # Minimal writer (xyz + optional rgb) — warn if losing attrs
            write_pcd_minimal(filename, core, attrs, allow_lossy=allow_lossy)
        else:
            raise ValueError(f"Unsupported extension: {ext}. Prefer .ply or .npz")

    @abstractmethod
    def _save_core_dict(self, **kwargs) -> dict[str, np.ndarray]:
        """
        Return the minimal set of arrays that represent this model.
        The base class will merge these with any selected attributes and write them.
        Examples:
          - PointCloud -> {"points": (N,3)}
          - Mesh -> {"points": (N,3), "faces": (M,3)}  or {"points","tris"/"quads"}
        """
        raise NotImplementedError

    @classmethod
    def load(
        cls,
        filename: str,
        attributes: Sequence[str] | None = None,
        **kwargs
    ) -> "Model":
        """
        1) Build a minimal instance via subclass hook.
        2) If `attributes` requested, read & attach them post-construction.
        """
        if isinstance(attributes, str):
            attributes = [attributes]
        obj = cls._load_core(filename, **kwargs)  # subclass returns a ready instance
        if attributes:
            attrs = read_properties(filename, list(attributes))
            obj._attrs.update(attrs)
        return obj

    @classmethod
    @abstractmethod
    def _load_core(cls, filename: str, **kwargs) -> "Model":
        """Subclasses construct and return an instance (points/topology etc.)."""
        raise NotImplementedError

    # ----------------------------
    # Spatial utilities
    # ----------------------------
    def clip(self, min_val: np.ndarray, max_val: np.ndarray) -> None:
        """
        Clip the model to an axis-aligned bounding box (AABB).

        NOTE: The base class cannot safely implement this because clipping
        often requires updating topology (faces/edges). Subclasses should
        implement their own logic (e.g., remove points outside AABB and
        rebuild connectivity, or perform polygon/mesh clipping).

        Parameters
        ----------
        min_val : array-like (D,)
            Minimum coordinates of the AABB.
        max_val : array-like (D,)
            Maximum coordinates of the AABB.
        """
        raise NotImplementedError("Model.clip(): implement in subclass to handle topology safely.")

    def boundingBox(self) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Compute the axis-aligned bounding box of the current points.

        Returns
        -------
        (min_corner, max_corner) : tuple of np.ndarray
            Each is shape (D,). Raises ValueError if there are no points.
        """
        v = self.points
        if v.size == 0:
            raise ValueError("Cannot compute bounding box: no points.")
        return v.min(axis=0), v.max(axis=0)

    def centroid(self) -> NDArray[np.float64]:
        """
        Compute the centroid (arithmetic mean) of the points.

        Returns
        -------
        np.ndarray
            Shape (D,). Raises ValueError if there are no points.
        """
        v = self.points
        if v.size == 0:
            raise ValueError("Cannot compute centroid: no points.")
        return np.mean(v, axis=0)

    # ----------------------------
    # Rigid transforms
    # ----------------------------
    def translate(self, value: Union[float, Iterable, np.ndarray], direction: str = "xyz") -> None:
        """
        Translate the model along specified axes.

        Parameters
        ----------
        value : float | array-like
            If scalar, applied to all axes listed in `direction`.
            If array-like, it is zipped with `direction` (1:1).
        direction : str
            Axes along which to translate, subset of {'x','y','z'}.
            For 2D data, only 'x' and 'y' are meaningful.

        Notes
        -----
        - If `value` is scalar and `direction='xyz'` on a 2D model, the 'z'
          component is ignored.
        - Values are applied in the order given by `direction`.
        """
        v = self.points
        if v.size == 0:
            return
        D = v.shape[1]
        allowed = "xy" if D == 2 else "xyz"
        for d in direction:
            if d not in allowed:
                raise ValueError(f"Direction '{d}' not valid for {D}D model.")

        vals = np.asarray(value, float).ravel()
        if vals.size == 1:
            vals = np.repeat(vals, len(direction))
        if vals.size != len(direction):
            raise ValueError("Length of `value` must match number of axes in `direction`.")

        t = np.zeros(D, dtype=v.dtype)
        idx = {"x":0, "y":1, "z":2}
        for val, d in zip(vals, direction):
            t[idx[d]] = val

        # in-place, single pass
        v += t


    def rotate(self, theta: float, axis: str = "z") -> None:
        """
        Rotate the model by an angle around a principal axis (in-place).

        Parameters
        ----------
        theta : float
            Rotation angle in degrees.
        axis : {'x','y','z'}
            Axis of rotation. For 2D data, only 'z' is meaningful.

        Notes
        -----
        - Rotation is performed about the origin. If you want to rotate about
          the centroid, call `c = centroid(); translate(-c); rotate(...); translate(c)`.
        """
        v = self.points
        if v.size == 0:
            return
        
        c, s = np.cos(np.radians(theta)), np.sin(np.radians(theta))
        
        D = v.shape[1]
        if D == 2:
            if axis != "z":
                raise ValueError("2D model can only rotate around 'z'.")
            x = v[:, 0].copy()                  # temp to avoid aliasing
            y = v[:, 1]
            v[:, 0] = c * x - s * y
            v[:, 1] = s * x + c * y
            return

        # 3D case
        if axis not in {"x","y","z"}:
            raise ValueError("Axis must be one of {'x','y','z'}.")

        x, y, z = v[:, 0], v[:, 1], v[:, 2]

        if axis == "x":
            y0 = y.copy()
            v[:, 1] =  c * y0 - s * z
            v[:, 2] =  s * y0 + c * z
        elif axis == "y":
            x0 = x.copy()
            v[:, 0] =  c * x0 + s * z
            v[:, 2] = -s * x0 + c * z
        else:  # 'z'
            x0 = x.copy()
            v[:, 0] =  c * x0 - s * y
            v[:, 1] =  s * x0 + c * y
        self._normal = None  # invalidate cached normal

    # ----------------------------
    # Orientation utilities
    # ----------------------------
    def normalVector(self) -> NDArray[np.float64]:
        """
        Fit a plane ``z = a x + b y + d``.

        Returns
        -------
        np.ndarray
            ``plane_normal``: unit normal vector (-a, -b, 1) normalized.
            Orientation is adjusted to have a positive Z component
            (for stable convention).

        Notes
        -----
        - For 2D data, this method is not defined and raises a ``ValueError``.
        - Plane is fit in the least squares sense.
        """
        if self._normal is None:
            v = self.points
            if v.shape[1] != 3 or v.shape[0] < 3:
                raise ValueError("normalVector() needs >=3 points in 3D.")

            # Work in float64 and centre to improve conditioning.
            P = np.asarray(v, dtype=np.float64)
            Pc = P - P.mean(axis=0, keepdims=True)
            x, y, z = Pc[:, 0], Pc[:, 1], Pc[:, 2]

            # Design matrix and least-squares solve: [a, b, d_c]
            # (intercept is relative to the centred coords)
            A = np.column_stack([x, y, np.ones_like(x)])
            coef, *_ = np.linalg.lstsq(A, z, rcond=None)
            a, b, d_c = coef

            # Unit normal is proportional to (a, b, -1)
            n = np.array([-a, -b, 1.0], dtype=np.float64)
            n /= np.linalg.norm(n)
            self._normal = n
        return self._normal
    
    def dip(self, return_normal: bool = False) -> float:
        """
        Dip angle of the fitted plane ``z = a x + b y + d``.

        Returns
        -------
        float
            Dip angle in degrees, defined as ``atan(b)`` where ``z ≈ a x + b y + d``.
            This follows the convention of tilt along Y used in the original implementation.

        Notes
        -----
        - For 2D data, this method is not defined and raises a ``ValueError``.
        - Plane is fit in the least squares sense.
        """
        n = self.normalVector()  # unit normal
        n_xy = np.hypot(n[0], n[1])
        alpha = np.arctan2(n[2], n_xy)  # angle between n and horizontal plane
        return float(90.0 - np.degrees(alpha))

    def dipDirection(self) -> float:
        """
        Dip direction of the fitted plane ``z = a x + b y + d``.

        Returns
        -------
        float
            Dip direction in degrees, defined as ``atan2(a, b)`` where ``z ≈ a x + b y + d``.
        """
        n = self.normalVector()  # unit normal
        angle = 180 - np.degrees(np.arctan2(n[0], -n[1]))
        if angle < 0:
            angle += 360.0
        return float(angle % 360)

    def alignWithX(self, sense: int = 1) -> None:
        """
        Rotate model in the XY-plane so that the fitted plane’s horizontal
        strike vector aligns with X.
        Parameters
        ----------
        sense : {1, -1}
            Sense of direction: 1 = X+, -1 = X-.

        Notes
        -----
        - The plane is fit as ``z = a x + b y + d`` (vertical least squares)
        via :meth:`normalVector`.
        - The plane’s unit normal is projected into XY. Its perpendicular
        (strike) is rotated onto +X using a single atan2.
        - If the fitted normal points downward (n_z < 0), it is flipped
        upward for consistency.
        - Translation to and from the centroid is applied so that rotation
        is about the global origin.
        """
        if self.points.shape[1] != 3:
            raise ValueError("alignWithX() requires 3D model.")

        # Translate to origin for a stable rotation about Z
        c = self.centroid()
        self.translate(-c)

        # Fit plane and get its unit normal
        n = self.normalVector()
        direction = np.array([-n[1], n[0], 0.0])
        norm = np.linalg.norm(direction)
        if norm > 1e-12:
            direction /= norm
            direction *= float(sense)

            R = rotationAlign2x(direction)

            self.points = self.points @ R.T

        # Restore original position
        self.translate(c)
    
    def coneApproximation(self) -> Tuple[float, float, float, float]:
        if self.points.size == 0:
            return 0.0, 0.0, 0.0, 0.0
        
        c = self.centroid()
        self.translate(-c)

        # Fit plane and get its unit normal
        n = self.normalVector()
        direction = np.array([-n[1], n[0], 0.0])
        direction /= np.linalg.norm(direction)
        direction *= float(-1)

        R = rotationAlign2x(direction)

        # Restore original position
        v = self.points.copy()
        self.translate(c)

        v = v @ R.T
        z_min = v.min(axis=0)[2]
        v[:, 2] -= z_min

        x = v[:, 0]
        y = v[:, 1]
        z = v[:, 2]


        def lin_fit(p, return_func=True, return_grad=False, return_coeff=False, eps=1e-12):
            x0, y0 = p
            dx = x - x0
            dy = y - y0
            rho = np.sqrt(dx*dx + dy*dy)
            rho_safe = np.maximum(rho, eps)  # avoid division by zero
            A = np.column_stack((np.ones_like(rho), rho))
            ab, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
            a, b = ab
            r = z - (a + b * rho)
            out = []
            if return_func:
                f = float(r @ r)
                out.append(f)
            if return_grad:
                drho_dx0 = (x0 - x) / rho_safe
                drho_dy0 = (y0 - y) / rho_safe
                gx0 = -2.0 * b * np.sum(r * drho_dx0)
                gy0 = -2.0 * b * np.sum(r * drho_dy0)
                g = np.array([gx0, gy0], dtype=float)
                out.append(g)
            if return_coeff:
                out.append(a)
                out.append(b)
            if len(out) == 1:
                return out[0]
            return out
        
        x0_init = 0.0
        y0_init = (x.min() - x.max()) * 0.25
        p0 = np.array([x0_init, y0_init], dtype=float)
        res = minimize(
            fun=lambda p: lin_fit(p, return_func=True, return_grad=True),
            x0=p0,
            jac=True,
            method="BFGS",
            options=dict(gtol=1e-8, maxiter=500)
        )
        x0, y0 = res.x

        # Recover cone parameters
        a, b = lin_fit((x0, y0), return_func=False, return_coeff=True)
        height = a
        radius = np.nan
        if abs(b) > 1e-14:
            radius = -a / b

        # Get cone center
        cone_center = np.array((x0, y0, 0.0))
        cone_center = cone_center @ R
        cone_center += c
        # cone_center[2] += z_min
        return cone_center[0], cone_center[1], height, radius


    # ----------------------------
    # Higher-level operations (abstract) — return project-specific types
    # ----------------------------
    @abstractmethod
    def segment(self, dx: float, du: float = 0.0) -> List["Model"]:
        """
        Segment the model into smaller sections/windows.

        Parameters
        ----------
        dx : float
            Window width (project-specific meaning).
        du : float, default 0
            Overlap between segments (same units as dx).

        Returns
        -------
        list[Model]
            New model objects representing each segment/window.
        """
        raise NotImplementedError

    @abstractmethod
    def split(self, dx: float = 0.5, minPerc: float = 0.05) -> List["Model"]:
        """
        Split the model into two parts according to your rule.

        Parameters
        ----------
        dx : float, default 0.5
            Cutting distance/parameter controlling the split.
        minPerc : float, default 0.05
            Minimum fraction of points allowed in either output to avoid degenerate splits.

        Returns
        -------
        list[Model]
            Two model objects (or potentially more, per your design).
        """
        raise NotImplementedError

    @abstractmethod
    def slice(self, direction: NDArray[np.float64], increment: float, label:str, **kwargs) -> Tuple[List["Geometry"], NDArray]:
        """
        Slice the model into profiles (e.g., 2D lines extracted along a direction).

        Parameters
        ----------
        direction : np.ndarray
            Direction vector along which to slice (length 2 or 3). Will be normalized.
        increment : float
            Spacing between consecutive slices.
        materials : Sequence[Material]
            List of materials to consider during slicing.
        label : str
            Name of the mesh attribute containing material IDs.
        **kwargs :
            Implementation-specific options.

        Returns
        -------
        list[Geometry]
            Extracted profiles as `Geometry` type.
        ndarray
            The slice positions along the direction vector.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        txt = f"{cls}("
        try:
            v = self.points
            shape = tuple(v.shape)
            dtype = v.dtype
            txt += f"points_shape={shape}, dtype={dtype}, "
        except Exception:
            pass
        n_attrs = len(self._attrs)
        return txt + f"n_attrs={n_attrs})"

