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
from ._frame import CurvilinearFrame
from ._filesystem import read_properties, write_ply_with_attrs, write_npz_full_fidelity, write_xyz_ascii, write_pcd_minimal
from ._utils import rotationAlign2x


def _slice_positions(positions) -> NDArray[np.float64]:
    """
    Validate slice positions given explicitly to ``slice()``.

    Returns them as a 1D float array, in the order given. They must be finite;
    repeated positions are allowed (each gives its own profile).
    """
    pos = np.asarray(positions, dtype=float)
    if pos.ndim == 0:
        pos = pos[None]
    if pos.ndim != 1:
        raise ValueError("positions must be a 1D sequence of slice positions.")
    if not np.all(np.isfinite(pos)):
        raise ValueError("positions must be finite.")
    return pos

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
        self._frame: Optional[CurvilinearFrame] = None      # cache for the curvilinear frame
        self._removed: Optional[NDArray[np.float64]] = None  # xyz of points dropped by removePoints()

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
        self._invalidate_caches()

    def _invalidate_caches(self, normal: bool = True) -> None:
        """
        Forget everything derived from the point coordinates: the fitted plane
        normal, the curvilinear frame and the positions remembered by
        :meth:`removePoints`. Called whenever the points are replaced or moved.
        """
        if normal:
            self._normal = None
        self._frame = None
        self._removed = None

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
        # A subset lives in the same space: keep the frame and removed positions
        frame, removed = self._frame, self._removed
        self.points = pts[mask]  # triggers _on_points_replaced
        self._frame, self._removed = frame, removed

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
        self._invalidate_caches(normal=False)  # a translation keeps the plane normal


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
        self._invalidate_caches()

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
    # Curvilinear coordinates: (s, d, e) and (u, v, w)
    # ----------------------------
    def _steep_mask(self, max_nz: float = 0.7) -> Optional[NDArray[np.bool_]]:
        """
        Points that belong to the face rather than to the floor or berms, i.e.
        whose normal is far from vertical (``|nz| < max_nz``). Used to fit the
        polyline of the curvilinear frame.

        The base implementation reads the ``"normals"`` attribute and returns
        ``None`` when it is missing. Subclasses with topology may compute normals.
        """
        if not self.has_attr("normals"):
            return None
        nrm = np.asarray(self.get_attr("normals"), dtype=float)
        if nrm.ndim != 2 or nrm.shape != (self.points.shape[0], 3):
            return None
        length = np.linalg.norm(nrm, axis=1)
        return (length > 0) & (np.abs(nrm[:, 2]) < max_nz * length)

    def fitFrame(
        self,
        *,
        tol: float = 1.0,
        res: float = 0.5,
        corner_length: Union[str, float, None] = "auto",
        **kwargs,
    ) -> CurvilinearFrame:
        """
        Fit the curvilinear frame of the model and keep it as :attr:`frame`.

        A plan-view polyline is fitted along the wall and the points are unrolled
        along it, which gives the coordinates returned by :meth:`sde` and :meth:`uvw`.

        Parameters
        ----------
        tol : float
            How far the centreline of the wall may stray from each straight segment
            of the polyline (model units).
        res : float
            Bin width along the wall used by the fit (model units).
        corner_length : {"auto", float, None}
            Length either side of each bend over which the slicing direction turns.
            ``None`` spreads the turn along each whole segment.
        **kwargs :
            Passed to :meth:`CurvilinearFrame.fit`.

        Returns
        -------
        CurvilinearFrame
        """
        v = self.points
        if v.ndim != 2 or v.shape[1] != 3 or v.shape[0] < 3:
            raise ValueError("fitFrame() needs >=3 points in 3D.")
        self._frame = CurvilinearFrame.fit(
            v, steep=self._steep_mask(), tol=tol, res=res, corner_length=corner_length, **kwargs
        )
        return self._frame

    @property
    def frame(self) -> CurvilinearFrame:
        """
        Curvilinear frame of the model. Fitted with default settings on first use
        (see :meth:`fitFrame`). It can also be assigned, e.g. a frame loaded with
        :meth:`CurvilinearFrame.load` or shared with another model.
        """
        if self._frame is None:
            self.fitFrame()
        return self._frame

    @frame.setter
    def frame(self, value: Optional[CurvilinearFrame]) -> None:
        if value is not None and not isinstance(value, CurvilinearFrame):
            raise TypeError("frame must be a CurvilinearFrame or None.")
        self._frame = value

    def sde(self, store: bool = False) -> NDArray[np.float64]:
        """
        Curvilinear coordinates of the points.

        Returns
        -------
        np.ndarray
            Shape (N, 3): ``s`` distance along the strike, ``d`` distance along the
            dip direction (increasing down-dip) and ``e`` elevation (equal to z).
            Points outside the invertible region of the frame are NaN.

        Parameters
        ----------
        store : bool
            Also store the columns as attributes ``"strike"``, ``"dip_direction"``
            and ``"elevation"``.
        """
        sde = self.frame.xyz_to_sde(self.points)
        if store:
            for name, col in zip(("strike", "dip_direction", "elevation"), sde.T):
                self.set_attr(name, np.ascontiguousarray(col))
        return sde

    def uvw(self, store: bool = False) -> NDArray[np.float64]:
        """
        Principal coordinates of the unrolled points.

        Returns
        -------
        np.ndarray
            Shape (N, 3): ``u`` and ``v`` span the best-fit plane of ``(s, d, e)``
            (``u`` along the strike, ``v`` up-dip) and ``w`` is the height above it.

        Parameters
        ----------
        store : bool
            Also store the columns as attributes ``"U"``, ``"V"`` and ``"W"``.
        """
        frame = self.frame
        sde = frame.xyz_to_sde(self.points)
        if frame.pca_axes is None:
            frame.fit_pca(sde)
        uvw = frame.sde_to_uvw(sde)
        if store:
            for name, col in zip(("U", "V", "W"), uvw.T):
                self.set_attr(name, np.ascontiguousarray(col))
        return uvw

    def _slicing_coords(
        self,
        direction: Optional[NDArray[np.float64]] = None,
        frame: Optional[CurvilinearFrame] = None,
    ) -> Tuple[NDArray[np.float64], Optional[CurvilinearFrame]]:
        """
        Coordinates in which slices are planes of constant first column.

        - ``frame`` given, or neither given: ``(s, d, e)`` of that frame (default:
          the model's own frame), i.e. slices follow the strike of the wall.
        - ``direction`` given: the points rotated so that ``direction`` is the first
          axis, i.e. parallel slices normal to a fixed direction.

        Returns the coordinates (N, 3) and the frame used (``None`` for a fixed
        direction).
        """
        if frame is None and direction is not None:
            R = rotationAlign2x(direction)
            return self.points @ R.T, None
        frame = self.frame if frame is None else frame
        return frame.xyz_to_sde(self.points), frame

    # ----------------------------
    # Removing points and filling the holes left behind
    # ----------------------------
    def removePoints(self, mask_or_attr: Union[str, np.ndarray], value=None) -> int:
        """
        Remove points, either flagged by a boolean mask or selected by the value of
        an attribute (in-place).

        The positions of the removed points are remembered, so that
        :meth:`fillHoles` can rebuild the parts left empty, including those that
        reach the edge of the model.

        Parameters
        ----------
        mask_or_attr : str | array-like of bool
            Name of a per-point attribute, or a boolean mask of length N that is
            True for the points to remove.
        value : scalar | sequence, optional
            With an attribute name: the value, or values, to remove.

        Returns
        -------
        int
            Number of points removed.

        Examples
        --------
        >>> pc.removePoints("Material", 0)               # one class
        >>> pc.removePoints("Material", [0, 3])          # several classes
        >>> pc.removePoints(pc.get_attr("height") > 2)   # any condition
        """
        n = self.points.shape[0]
        if isinstance(mask_or_attr, str):
            if value is None:
                raise ValueError("Give the attribute value(s) to remove.")
            if not self.has_attr(mask_or_attr):
                raise ValueError(f"Attribute '{mask_or_attr}' not found.")
            arr = np.asarray(self.get_attr(mask_or_attr))
            if arr.ndim == 2 and arr.shape[1] == 1:
                arr = arr[:, 0]
            if arr.ndim != 1 or arr.shape[0] != n:
                raise ValueError(f"Attribute '{mask_or_attr}' must hold one value per point.")
            mask = np.zeros(n, dtype=bool)
            for v in np.atleast_1d(value):
                if np.issubdtype(arr.dtype, np.floating):
                    mask |= np.isclose(arr, v)
                else:
                    mask |= arr == v
        else:
            mask = np.asarray(mask_or_attr)
            if mask.dtype != bool or mask.shape != (n,):
                raise ValueError("Mask must be boolean with length N points.")

        n_removed = int(np.count_nonzero(mask))
        if n_removed == 0:
            return 0
        removed = self.points[mask].copy()
        previous = self._removed
        self.apply_point_mask_(~mask)
        self._removed = removed if previous is None else np.vstack([previous, removed])
        return n_removed

    def fillHoles(self, attributes: Sequence[str] | str = (), **kwargs) -> "Model":
        """
        Create points in the missing parts of the model and krige their coordinates
        and the selected attributes. See :meth:`PointCloud.fillHoles`.

        Not available for this type of model: convert it to a point cloud
        (:func:`mesh2PointCloud`), fill the holes and triangulate again
        (:func:`pointCloud2Mesh`).
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.fillHoles() is not implemented; "
            "use mesh2PointCloud() -> PointCloud.fillHoles() -> pointCloud2Mesh()."
        )

    def _split_position(self, s: NDArray[np.float64], minPerc: float = 0.05, min_turn: float = 5.0) -> float:
        """
        Position along the strike at which to split the model in two.

        Returns ``s`` of the polyline vertex with the largest turning angle (at
        least `min_turn` degrees) that leaves a fraction `minPerc` of the given
        ``s`` values on each side; the median of ``s`` if there is no such bend.
        """
        frame = self.frame
        fin = s[np.isfinite(s)]
        if fin.size == 0:
            return np.nan
        n_min = int(max(1, minPerc * fin.size))
        turn = frame.turning_angles()
        for i in np.argsort(-turn):
            if turn[i] < np.radians(min_turn):
                break
            cut = float(frame.S0[i + 1])
            n_left = int(np.count_nonzero(fin <= cut))
            if n_min <= n_left <= fin.size - n_min:
                return cut
        return float(np.median(fin))

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
    def slice(
        self,
        direction: Optional[NDArray[np.float64]] = None,
        increment: Optional[float] = None,
        label: Optional[str] = None,
        **kwargs,
    ) -> Tuple[List["Geometry"], NDArray]:
        """
        Slice the model into 2D profiles.

        Parameters
        ----------
        direction : np.ndarray, optional
            ``None`` (default) slices along the strike of the wall using the
            curvilinear frame: slices are surfaces of constant ``s`` and the
            profile coordinates are ``(d, e)``. A vector instead cuts parallel
            slices normal to that fixed direction.
        increment : float
            Spacing between consecutive slices.
        label : str
            Name of the attribute containing material IDs.
        **kwargs :
            Implementation-specific options (e.g. ``frame`` to use a frame other
            than the model's own).

        Returns
        -------
        list[Geometry]
            Extracted profiles as `Geometry` type.
        ndarray
            The slice positions (``s``, or the distance along ``direction``).
        """
        raise NotImplementedError

    def section(
        self,
        increment: float,
        label: str,
        *,
        frame: Optional[CurvilinearFrame] = None,
        **kwargs,
    ) -> Tuple[List["Geometry"], "Model"]:
        """
        Extract 2D profiles along the strike of the model, and their points in 3D.

        This is :meth:`slice` along the strike (slices of constant ``s`` in the
        curvilinear frame), with each profile also mapped back to world
        coordinates. The model itself is not modified.

        Parameters
        ----------
        increment : float
            Spacing between consecutive sections along the strike.
        label : str
            Name of the attribute holding material IDs.
        frame : CurvilinearFrame, optional
            Frame to slice in. Defaults to the frame of the model.
        **kwargs :
            Passed to :meth:`slice` of the subclass.

        Returns
        -------
        profiles : list[Geometry]
            One 2D profile per section, ordered from crest to toe, with node
            coordinates given as (horizontal distance from the crest node, height
            above the lowest node) and material IDs per segment as attributes.
        sections_pc : PointCloud
            All section nodes as one point cloud in the model's coordinates, with
            the material attribute (`label`, one value per node) and a
            ``"Profile"`` attribute numbering the sections along the strike.

        Raises
        ------
        ValueError
            If no valid sections are found.
        """
        from ._pointcloud import PointCloud  # here to avoid a circular import

        frame = self.frame if frame is None else frame
        slices, positions = self.slice(None, increment, label, frame=frame, **kwargs)
        if len(slices) == 0:
            raise ValueError("No valid sections found. Try adjusting parameters.")

        profiles = []
        section_pts = []
        section_mat = []
        section_id = []
        for k, (prof, s_k) in enumerate(zip(slices, positions)):
            d, e = prof.nodes[:, 0], prof.nodes[:, 1]
            mats = np.asarray(prof.attributes)
            if e[0] < e[-1]:
                # crest first
                d, e, mats = d[::-1], e[::-1], mats[::-1]

            # World coordinates of the section nodes
            sde = np.column_stack([np.full(d.shape, s_k), d, e])
            section_pts.append(frame.sde_to_xyz(sde))
            section_mat.append(np.append(mats, mats[-1]))  # one material per node
            section_id.append(np.full(d.shape[0], k, dtype=int))

            # Profile relative to the crest node and the base of the section
            dist = np.abs(d - d[np.argmax(e)])
            profiles.append(Geometry(nodes=np.column_stack([dist, e - e.min()]), attributes=mats))

        sections_pc = PointCloud(points=np.concatenate(section_pts, axis=0))
        sections_pc.set_attr(label, np.concatenate(section_mat, axis=0))
        sections_pc.set_attr("Profile", np.concatenate(section_id, axis=0))
        return profiles, sections_pc

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

