from __future__ import annotations

from pathlib import Path
from typing import Tuple, List, Mapping, Optional, Sequence, Dict
from numpy.typing import NDArray, ArrayLike

import numpy as np
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors, RadiusNeighborsClassifier, KNeighborsClassifier

try:
    import open3d as o3d  # optional, used if available for IO
    _HAS_O3D = True
except Exception:  # pragma: no cover
    _HAS_O3D = False

ArrayF = NDArray[np.float64]

from ._model import Model, _slice_positions
from ._geometry import Geometry
from ._frame import CurvilinearFrame
from ._kriging import fill_holes_uv, _class_label
from ._utils import pcWalk, profile_from_slab

class PointCloud(Model):
    """
    Geometry represented as a point set (no inherent connectivity).

    - `points` are stored as a NumPy array of shape (N, 3).
    - Per-point attributes live in `self.attrs` (e.g. 'colors', 'normals', 'material_id').
      Each must have first dimension N.

    Notes
    -----
    - This class avoids keeping an internal Open3D object; we construct one
      transiently only for IO when needed.
    """

    # ----------------------------
    # Construction
    # ----------------------------
    def __init__(
        self,
        points: Optional[np.ndarray] = None,
        *,
        colors: Optional[np.ndarray] = None,
        normals: Optional[np.ndarray] = None,
        attrs: Optional[Mapping[str, np.ndarray]] = None,
    ) -> None:
        super().__init__()
        self._points: ArrayF = np.zeros((0, 3), dtype=float)
        if points is not None:
            self.points = points  # will call _on_points_replaced()
        if colors is not None:
            self.set_attr("colors", np.asarray(colors))
        if normals is not None:
            self.set_attr("normals", np.asarray(normals))
        if attrs:
            for k, v in attrs.items():
                self.set_attr(k, np.asarray(v))
        self._resolution: Optional[float] = None  # cached mean NN distance

    # ----------------------------
    # Abstract point storage API
    # ----------------------------
    @property
    def points(self) -> ArrayF:
        return self._points

    @points.setter
    def points(self, value: np.ndarray) -> None:
        arr = np.asarray(value, dtype=float)
        if arr.ndim != 2 or arr.shape[1] not in (2, 3):
            raise ValueError("PointCloud.points must be an array of shape (N, 3) (or (N, 2) for 2D).")
        # Promote to 3D if given as 2D
        if arr.shape[1] == 2:
            arr = np.column_stack([arr, np.zeros((arr.shape[0],), dtype=arr.dtype)])
        self._points = arr
        self._on_points_replaced()

    # ----------------------------
    # Persistence
    # ----------------------------
    def _save_core_dict(self, **kwargs) -> dict[str, np.ndarray]:
        return {"points": np.asarray(self.points)}
    
    @classmethod
    def _load_core(cls, filename: str, **kwargs) -> "PointCloud":
        """
        Load from file by extension:

        - .npz: restores points + attrs
        - .pcd/.ply/.xyz: via Open3D for PCD/PLY; .xyz is plain ASCII (x y z)
        """

        path = Path(filename)
        ext = path.suffix.lower()

        has_colors = False
        has_normals = False

        if ext == '.npz':
            data = np.load(path, allow_pickle=False)
            pts = np.asarray(data["points"])
            has_colors = 'colors' in data.files
            has_normals = 'normals' in data.files
            if has_colors:
                cols = np.asarray(data['colors'])
            if has_normals:
                nmls = np.asarray(data['normals'])

        if ext == '.xyz':
            pts = np.loadtxt(path)
            if pts.ndim == 1:
                pts = pts[None, :]
            pts = pts[:, :3]

        if ext in {'.pcd', '.ply', '.pts', '.las'}:
            if not _HAS_O3D:
                raise ValueError(f"Loading '{ext}' requires Open3D installed.")
            pc = o3d.io.read_point_cloud(str(path))
            pts = np.asarray(pc.points, dtype=np.float64)
            if pc.has_colors():
                has_colors = True
                cols = np.asarray(pc.colors, dtype=np.float64)
            if pc.has_normals():
                has_normals = True
                nmls = np.asarray(pc.normals, dtype=np.float64)
        try:
            obj = cls(points=pts)
            if has_colors:
                obj.set_attr("colors", cols)
            if has_normals:
                obj.set_attr("normals", nmls)
            return obj
        except Exception as e:
            raise ValueError(f"Unsupported input extension: '{ext}'")

    # ----------------------------
    # Spatial utilities
    # ----------------------------
    def clip(self, min_val: np.ndarray, max_val: np.ndarray) -> None:
        """Keep only points within an AABB; attributes remain consistent."""
        if self.points.size == 0:
            return
        mn = np.asarray(min_val, dtype=float).ravel()
        mx = np.asarray(max_val, dtype=float).ravel()
        if mn.size != 3 or mx.size != 3:
            raise ValueError("clip(): min_val and max_val must have shape (3,).")
        v = self.points
        mask = np.all((v >= mn) & (v <= mx), axis=1)
        self.apply_point_mask_(mask)

    # ----------------------------
    # Convenience
    # ----------------------------
    def has_colors(self) -> bool:
        return self.has_attr("colors")

    def has_normals(self) -> bool:
        return self.has_attr("normals")

    # ----------------------------
    # Higher-level operations
    # ----------------------------
    def segment(self, ds: float, du: float = 0.0) -> List["PointCloud"]:
        """
        Split along the strike into contiguous windows of length `ds`.

        The windows are taken on the ``s`` coordinate of the curvilinear frame
        (see :meth:`Model.sde`), so they follow the wall around bends.

        Parameters
        ----------
        ds : float
            Window length along the strike.
        du : float, default 0
            Overlap between consecutive windows (``0 <= du < ds``).

        Returns
        -------
        list[PointCloud]
            One point cloud per non-empty window, ordered by increasing ``s``.
            They share the frame of this point cloud.
        """
        if ds <= 0:
            raise ValueError("ds must be > 0.")
        if du < 0 or du >= ds:
            raise ValueError("du must satisfy 0 <= du < ds.")
        if self.points.shape[0] == 0:
            return []

        s = self.frame.xyz_to_sde(self.points)[:, 0]
        idx = np.flatnonzero(np.isfinite(s))
        if idx.size == 0:
            return []
        order = idx[np.argsort(s[idx], kind="stable")]
        ss = s[order]

        t0 = np.arange(ss[0], ss[-1], ds - du)
        if t0.size == 0:
            t0 = np.array([ss[0]])
        t1 = t0 + ds
        start = np.searchsorted(ss, t0, side="left")
        end = np.searchsorted(ss, t1, side="left")
        end[t1 >= ss[-1]] = ss.size  # the last window includes the end of the wall

        out: List[PointCloud] = []
        for lo, hi in zip(start, end):
            if hi > lo:
                out.append(self._from_indices(np.sort(order[lo:hi])))
        return out

    def split(self, dx: float = 0.5, minPerc: float = 0.05) -> List["PointCloud"]:
        """
        Split into two parts at the sharpest bend of the wall.

        The cut is made at the vertex of the frame's polyline with the largest
        turning angle that leaves at least `minPerc` of the points on each side.
        A wall without a bend is cut at its median ``s``.

        Parameters
        ----------
        dx : float
            Unused; kept for backward compatibility.
        minPerc : float, default 0.05
            Minimum fraction of points on each side of the cut.

        Returns
        -------
        list[PointCloud]
            ``[before, after]`` along increasing ``s``.
        """
        v = self.points
        if v.size == 0:
            return []

        s = self.frame.xyz_to_sde(v)[:, 0]
        s_cut = self._split_position(s, minPerc)
        left = s <= s_cut
        return [self._from_indices(np.flatnonzero(left)), self._from_indices(np.flatnonzero(~left))]

    def resolution(self) -> float:
        """
        Estimate mean nearest-neighbour spacing using 2-NN distances.

        Returns
        -------
        float
        """
        if self._resolution is None:
            v = self.points
            if v.shape[0] < 2:
                return 0.0
            nn = NearestNeighbors(n_neighbors=2, algorithm="auto")
            nn.fit(v[:, :3])
            d, _ = nn.kneighbors(return_distance=True)
            self._resolution = float(np.mean(d[:, 0]))
        return self._resolution

    def slice(
        self,
        direction: Optional[NDArray[np.float64]] = None,
        increment: Optional[float] = None,
        label: Optional[str] = None,
        *,
        frame: Optional[CurvilinearFrame] = None,
        slice_height_increment: float = -1.0,
        slice_resolution_factor: float = 5.0,
        min_points: int = 1,
        transverse_radius: Optional[float] = None,
        outlier_fraction: float = 0.0,
        outlier_radius: Optional[float] = None,
        method: str = "interp",
        refine: bool = True,
        positions: Optional[ArrayLike] = None,
        **kwargs,
    ) -> Tuple[List["Geometry"], NDArray]:
        """
        Slice the point cloud into 2D profiles.

        By default the slices follow the strike of the wall: they are surfaces of
        constant ``s`` of the curvilinear frame, `increment` apart, and each profile
        is given in ``(d, e)`` (distance along the dip direction, elevation). No
        prior alignment or segmentation of the model is needed, and slabs tile the
        wall around bends without gaps or overlaps.

        Processing outline
        ------------------
        1. Map the points to slicing coordinates (:meth:`Model._slicing_coords`).
        2. Bin all points to slabs of ``+-increment/2`` around the slice centres.
        3. Build an ordered profile, crest to toe, through each slab (see `method`).
        4. Predict materials at segment midpoints via KNN classification.

        Parameters
        ----------
        direction : np.ndarray, optional
            ``None`` (default) slices along the strike. A vector cuts parallel slices
            normal to that fixed direction instead; the profile coordinates are then
            the two remaining axes of the points rotated so that `direction` is +X.
        increment : float
            Spacing between consecutive slices, and thickness of each slab
            (``+-increment/2`` around its centre).
        label : str
            Name of the point attribute holding material IDs.
        frame : CurvilinearFrame, optional
            Frame to slice in. Defaults to the frame of the point cloud.
        positions : array-like, optional
            Slice centres (``s``, or the distance along `direction`) to use instead
            of a grid starting at the first point. Give the same positions, and the
            same `frame`, to slice several models of one wall (e.g. the original
            and a filled copy) at exactly the same places. `increment` is then only
            the slab thickness. Positions with too few points are skipped, so match
            the returned positions against these.
        method : {"interp", "walk"}
            How the profile is built from the points of a slab.

            - ``"interp"`` (default): the slab is a height field over the direction
              up the face, binned every `slice_height_increment`, with one node at
              the median of each bin (:func:`profile_from_slab`). Not iterative,
              and insensitive to stray points. Along the strike the direction up
              the face is the V axis of the frame; with a fixed `direction` it is
              the main direction of each slab.
            - ``"walk"``: step through the points from crest to toe
              (:func:`pcWalk`). Can follow a surface that folds back over the
              face plane, but may stop early.
        refine : bool
            With ``"interp"``: second pass that evens out the node spacing along
            the profile.
        slice_height_increment : float
            Node spacing of the profile. If not positive,
            ``resolution() * slice_resolution_factor`` is used.
        slice_resolution_factor : float
            See `slice_height_increment`.
        min_points : int
            Slabs with fewer points are skipped.
        transverse_radius : float, optional
            Keep only the points of each slab within this distance (along the
            profile's horizontal axis) of the slab's median.
        outlier_fraction : float
            Stray-point filter, applied to each slab in the profile plane. A point
            is dropped if it has fewer neighbours within `outlier_radius` than this
            fraction of the slab's median. Stray points off the surface (a sparse
            second layer, isolated points above the crest) otherwise pull the walk
            away from it. ``0`` (default) disables the filter; ``0.3`` is a
            reasonable value to try. It changes which walks fail rather than
            removing failures altogether, so check the result.
        outlier_radius : float, optional
            Radius used to count neighbours. Defaults to
            ``1.25 * resolution()``.

        Returns
        -------
        profiles : list[Geometry]
            One profile per valid slice, ordered from crest to toe, with material
            IDs per segment as attributes.
        positions : ndarray
            Position of each slice (``s``, or the distance along `direction`). With
            the strike-based slicing, world coordinates of a profile are
            ``frame.sde_to_xyz(np.column_stack([np.full(n, s_k), d, e]))``.
        """
        if self.points.size == 0:
            return [], np.array([], dtype=float)

        if label is None or not self.has_attr(label):
            raise ValueError(f"Attribute '{label}' not found.")
        mat_ids = np.asarray(self.get_attr(label))
        if mat_ids.ndim == 2 and mat_ids.shape[1] == 1:
            mat_ids = mat_ids[:, 0]
        mat_ids = mat_ids.astype(int, copy=False)

        if increment is None or increment <= 0:
            raise ValueError("increment must be > 0.")
        if method not in ("interp", "walk"):
            raise ValueError("method must be 'interp' or 'walk'.")

        # --- Decide vertical resolution (shared walk step)
        dlt_z = slice_height_increment if slice_height_increment > 0 else self.resolution() * slice_resolution_factor
        if not np.isfinite(dlt_z) or dlt_z <= 0:
            raise ValueError("slice_height_increment and resolution are invalid.")

        # --- Slicing coordinates: column 0 is constant on a slice
        points, used_frame = self._slicing_coords(direction, frame)
        # Direction up the face in the profile plane: the frame's V axis without
        # its component along the strike (None: use each slab's main direction)
        face_axis = None
        if method == "interp" and used_frame is not None and used_frame.pca_axes is not None:
            v_axis = used_frame.pca_axes[1, 1:]
            if np.linalg.norm(v_axis) > 0.1:
                face_axis = v_axis / np.linalg.norm(v_axis)
        ok = np.isfinite(points).all(axis=1)
        if not ok.all():
            points = points[ok]
            mat_ids = mat_ids[ok]
        if points.shape[0] == 0:
            return [], np.array([], dtype=float)

        # --- Slice centres
        x = points[:, 0]
        x_min = float(np.min(x))
        x_max = float(np.max(x))
        if not np.isfinite(x_max - x_min) or x_max <= x_min:
            return [], np.array([], float)

        half = 0.5 * increment  # typical "+-inc/2" slab

        if positions is None:
            x_positions = np.arange(x_min + half, x_max + 1e-12, increment, dtype=float)
        else:
            x_positions = _slice_positions(positions)
        num_slices = x_positions.size
        if num_slices == 0:
            return [], np.array([], float)

        # ---- Sort once by x and keep views for the profile plane ----
        order = np.argsort(x, kind="stable")
        xs = x[order]
        yz = points[order, 1:]   # (N,2): profile-plane coordinates

        # ---- For each slab m, find contiguous [start:end) in xs ----
        starts = np.searchsorted(xs, x_positions - half, side="left")
        ends   = np.searchsorted(xs, x_positions + half, side="right")

        valid_slices = starts < ends
        if not valid_slices.any():
            return [], np.array([], dtype=float)
        min_points = max(int(min_points), 1)
        if outlier_fraction > 0 and outlier_radius is None:
            outlier_radius = 1.25 * self.resolution()
        polylines = []
        max_pts = 0
        for i in range(num_slices):
            lo, hi = starts[i], ends[i]
            slab_points = yz[lo:hi]
            dlt_x = np.abs(xs[lo:hi] - x_positions[i])

            if transverse_radius is not None and slab_points.shape[0] > 0:
                keep = np.abs(slab_points[:, 0] - np.median(slab_points[:, 0])) <= transverse_radius
                slab_points = slab_points[keep]
                dlt_x = dlt_x[keep]

            if outlier_fraction > 0 and slab_points.shape[0] > 2:
                # Stray points have few neighbours compared with the surface
                count = cKDTree(slab_points).query_ball_point(slab_points, outlier_radius, return_length=True)
                keep = count >= outlier_fraction * np.median(count)
                slab_points = slab_points[keep]
                dlt_x = dlt_x[keep]

            if slab_points.shape[0] < min_points:
                # empty or too sparse slab
                valid_slices[i] = False
                continue

            # PCA of the slab: its main direction
            X = slab_points - slab_points.mean(0)
            _, _, vh = np.linalg.svd(X, full_matrices=False)

            if method == "interp":
                # Height field over the direction up the face: no iteration
                axis_i = vh[0] if face_axis is None else face_axis
                nodes = profile_from_slab(slab_points, dlt_z, axis_i, refine=refine)
            else:
                # Order points along the main direction, crest first
                pc1 = X @ vh[0]
                order_i = np.argsort(pc1)
                slab_points = slab_points[order_i]
                dlt_x = dlt_x[order_i]
                if slab_points[0, 1] < slab_points[-1, 1]:
                    slab_points = slab_points[::-1]
                    dlt_x = dlt_x[::-1]

                # Walk along profile
                nodes = pcWalk(slab_points[0], slab_points[-1], slab_points, dlt_x, dlt_z)

            # Check whether profile is valid
            if nodes.shape[0] < 2:
                # Invalid profile: not enough points
                valid_slices[i] = False
                continue

            polylines.append(nodes)
            max_pts = max(max_pts, nodes.shape[0])

        nodes = np.full((len(polylines), max_pts, 2), np.nan, dtype=float)
        for i, pl in enumerate(polylines):
            nodes[i, :pl.shape[0], :] = pl
        if nodes.shape[0] == 0:
            return [], np.array([], dtype=float)
        x_positions = x_positions[valid_slices]
        num_slices = x_positions.size

        # Reshape to (num_slices, num_heights) arrays
        x_grid = np.repeat(x_positions[:, None], max_pts, axis=1)
        y_grid = nodes[:, :, 0]
        z_grid = nodes[:, :, 1]

        # --- Midpoints for materials (predict in one batch too)
        x_mid = x_grid[:, :-1]          # constant per slice
        y_mid = 0.5 * (y_grid[:, :-1] + y_grid[:, 1:])
        z_mid = 0.5 * (z_grid[:, :-1] + z_grid[:, 1:])

        mats_all = np.full_like(y_mid, -1, dtype=int)
        mid_nan = np.isnan(y_mid) | np.isnan(z_mid)

        mid = np.column_stack((x_mid[~mid_nan], y_mid[~mid_nan], z_mid[~mid_nan]))

        # material classifier (3D, in slicing coordinates)
        cls = KNeighborsClassifier(n_neighbors=min(10, points.shape[0]), weights="distance")
        cls.fit(points, mat_ids)
        mats_all[~mid_nan] = cls.predict(mid)

        profiles = []
        for i in range(len(x_positions)):
            nodes_i = nodes[i]
            mask = np.isfinite(nodes_i).all(axis=1)   # (N,) True if both coordinates finite
            nodes_i = nodes_i[mask]
            mats_all_i = mats_all[i]
            mats_all_i = mats_all_i[mats_all_i >= 0]
            if nodes_i.shape[0] < 2 or mats_all_i.shape[0] < 1:
                raise ValueError("Invalid profile")
            if nodes_i.shape[0] != mats_all_i.shape[0] + 1:
                raise ValueError("Invalid profile")
            profiles.append(Geometry(nodes=nodes_i, attributes=mats_all_i))

        return profiles, x_positions


    def section(
        self,
        increment: float,
        label: str,
        *,
        transverse_radius: float | None = None,
        initial_direction: tuple[float, float] = (1.0, 0.0),
        min_points: int = 20,
        max_nodes: int | None = None,
        max_iter: int = 20,
        tol: float = 1e-3,
        max_turn_angle: float | None = np.deg2rad(45.0),
        weight_power: float = 1.0,
        min_new_fraction: float = 0.10,
        slice_height_increment: float = 0.5,
        frame: Optional[CurvilinearFrame] = None,
        method: str = "interp",
        positions: Optional[ArrayLike] = None,
    ):
        """
        Extract 2D profiles along the strike of the model.

        This is :meth:`slice` along the strike (``direction=None``) with the
        profiles also returned as a 3D point cloud in world coordinates (see
        :meth:`Model.section`). The point cloud itself is not modified.

        Parameters
        ----------
        increment : float
            Spacing, in the point cloud's units, between consecutive sections
            along the strike.
        label : str
            Name of the point attribute holding material/class IDs.
        transverse_radius : float, optional
            Half-width, along the dip direction, of the strip of points used
            for each section (centred on the section's median). ``None`` uses
            all points of the slab.
        min_points : int, optional
            Minimum number of points required for a section to be considered
            valid; sections with fewer points are discarded.
        slice_height_increment : float, optional
            Node spacing of the profiles (see :meth:`slice`).
        method : {"interp", "walk"}, optional
            How each profile is built from its points (see :meth:`slice`).
        frame : CurvilinearFrame, optional
            Frame to slice in. Defaults to the frame of the point cloud.
        positions : array-like, optional
            Positions ``s`` of the sections, instead of a grid every `increment`
            (see :meth:`slice`).
        initial_direction, max_nodes, max_iter, tol, max_turn_angle, weight_power, min_new_fraction :
            Ignored. They controlled the polyline tracing that the curvilinear
            frame replaces and are kept for backward compatibility.

        Returns
        -------
        profiles : list[Geometry]
            One 2D :class:`Geometry` per section, with node coordinates
            given as (distance along section, height above section base)
            and material IDs as attributes.
        sections_pc : PointCloud
            All section points combined into a single point cloud, in the
            original global coordinate system, with the material attribute
            (`label`) and a `"Profile"` attribute identifying which
            section each point belongs to.

        Raises
        ------
        ValueError
            If no valid sections are found.
        """
        return super().section(
            increment,
            label,
            frame=frame,
            slice_height_increment=slice_height_increment,
            min_points=min_points,
            transverse_radius=transverse_radius,
            method=method,
            positions=positions,
        )

    # ----------------------------
    # Filling holes
    # ----------------------------
    def fillHoles(
        self,
        attributes: Sequence[str] | str = (),
        *,
        kinds: Optional[Dict[str, str]] = None,
        spacing: Optional[float] = None,
        cell: Optional[float] = None,
        min_cells: int = 4,
        use_removed: bool = True,
        ring_scale: float = 0.5,
        attr_ring_scale: float = 1.0,
        max_train: int = 2000,
        n_closest: Optional[int] = None,
        variogram: str = "spherical",
        attr_variogram: Optional[str] = None,
        trend_order: int = 1,
        nlags: int = 20,
        aniso: float = 10.0,
        bedding_angle: float = 0.0,
        geometry_aniso: float = 3.0,
        shared_variogram: bool = True,
        dense_variogram: bool = True,
        dense_lag: float = 3.0,
        surface_tol: Optional[float] = 0.15,
        surface_radius: Optional[float] = None,
        others: str = "nearest",
        cover=None,
        cover_side="auto",
        cover_margin: float = 0.0,
        max_cover: int = 500,
        cover_samples: int = 200,
        cover_sweeps: int = 50,
        seed: int = 0,
        return_info: bool = False,
    ):
        """
        Create points in the missing parts of the point cloud and krige their
        coordinates and the selected attributes.

        The work is done in the ``(u, v, w)`` coordinates of the curvilinear frame
        (see :meth:`Model.uvw`), where the wall is unrolled onto its best-fit plane:
        ``w`` is a small residual even for a wall with corners, ``u`` runs along
        the strike and ``v`` up-dip, so layering is anisotropic along ``u``.

        Processing outline
        ------------------
        1. Find the missing parts on an occupancy grid in ``(u, v)``: empty cells
           enclosed by points, plus empty cells that held points taken out with
           :meth:`removePoints`. Each connected region is one hole.
        2. Create new points in every hole on a regular ``(u, v)`` grid.
        3. Height: take the points in a ring around the hole that lie on the main
           surface (second layers, loose blocks and steps are left out), remove a
           polynomial trend from their ``w``, fit a variogram to the residual
           (nugget + short- and long-range structure, from all pairs at short
           lags, so it describes the roughness of the rock), krige it at the new
           points and add the trend back. Map ``(u, v, w)`` back to ``(x, y, z)``.
        4. Attributes: integer-valued ones by indicator kriging (a probability per
           class, corrected to lie in [0, 1] and sum to 1), the others by ordinary
           kriging.

        Parameters
        ----------
        attributes : sequence of str
            Per-point attributes to krige at the new points.
        kinds : dict, optional
            ``name -> "categorical" | "continuous"`` to override the automatic
            choice (integer-valued attributes are categorical).
        spacing : float, optional
            Spacing of the new points. Defaults, for each hole, to the spacing
            that matches the density of the points around it.
        cell : float, optional
            Cell size of the occupancy grid. Defaults to ``5 * resolution()``.
        min_cells : int
            Smallest hole, in cells; smaller gaps are sampling noise.
        use_removed : bool
            Also fill the cells emptied by :meth:`removePoints`.
        ring_scale, attr_ring_scale : float
            Width of the ring of points used for the height and for the
            attributes, as a fraction of the hole's equivalent diameter.
        max_train : int
            Largest number of ring points used per kriging system; the ring is
            thinned evenly to this number.
        n_closest : int, optional
            ``None`` (default) solves one system per hole with all ring points.
            A number uses only that many nearest ring points per new point.
        variogram, attr_variogram : str
            Variogram model for the height and for the attributes: "spherical",
            "exponential", "gaussian" or "linear".
        trend_order : {0, 1, 2}
            Polynomial trend removed from ``w`` before kriging.
        nlags : int
            Lag bins of the experimental variograms.
        aniso : float
            Attribute ranges along the layers divided by the range across them
            (1 = isotropic).
        bedding_angle : float
            Angle (degrees, counter-clockwise) of the layers from horizontal,
            within the plane of the face.
        geometry_aniso : float
            Same ratio as `aniso` for the height ``w``: values above 1 assume the
            shape of the face persists further along the strike (e.g. benches)
            than up-dip.
        shared_variogram : bool
            Use one variogram for all classes of a categorical attribute.
        dense_variogram : bool
            Fit the height variogram as nugget + short- + long-range structure
            from all ring points (default), or as one structure on the thinned
            ring (False, smoother).
        dense_lag : float
            Boundary, in metres, between the short- and long-range structures.
        surface_tol, surface_radius : float
            Ring points more than `surface_tol` (m) from the median ``w`` of their
            24 nearest neighbours (within `surface_radius` m, if given) are left
            out of the height kriging: second layers, loose blocks, steps.
            ``surface_tol=None`` keeps every point.
        others : {"nearest", "nan"}
            Value given to the attributes that are not kriged: that of the nearest
            existing point, or NaN.
        cover : "removed" or (r, 3) array, optional
            Points that hid the surface and lie in front of it, e.g. vegetation:
            ``"removed"`` uses the points taken out with :meth:`removePoints`, an
            array gives their ``(x, y, z)``. The filled surface is kept behind them
            (on the side away from the air, see
            :meth:`CurvilinearFrame.outward_sign`): its height at the deepest
            cover point of each small cell follows a truncated Gaussian, and the
            mean and ``kriging_std`` of the new points account for it. See
            ``cover_uvw`` in :func:`~._kriging.fill_holes_uv`.
        cover_side : {"auto", 1, -1}
            Which way along ``w`` is "in front" (towards the air). ``"auto"``:
            from the frame when the face clearly dips (the outward direction has
            an upward component, see :meth:`CurvilinearFrame.outward_sign`),
            otherwise from the point normals (assumed to point to the air), and
            an error if there are none.
        cover_margin : float
            Least distance (m) of the surface behind the cover.
        max_cover, cover_samples, cover_sweeps : int
            Bounds per hole, draws and Gibbs sweeps used for the cover.
        seed : int
            Random seed for thinning the rings.
        return_info : bool
            Also return a dict with the holes found and the fitted variograms.

        Returns
        -------
        PointCloud
            A new point cloud with the existing points followed by the new ones.
            Extra attributes: ``"filled"`` (1 for new points), ``"hole_id"``,
            ``"kriging_std"`` (standard deviation of the kriged height); with
            `cover`, ``"cover_bound"`` (the bound on ``w`` of each new point, NaN
            where there is none) and ``"cover_free_w"`` (``w`` before it); per
            categorical attribute ``"<name>_P<class>"`` and ``"<name>_Pmax"``; per
            continuous attribute ``"<name>_std"``.
        dict
            Only with `return_info`: ``holes`` (per hole: size, ring, the height
            variogram ``w_variogram``, the share of ring points off the surface
            ``off_surface``, and ``train_idx``, the indices in the output of the
            points the height was kriged from), ``n_new``, ``spacing``, ``cell``.
            Pass it to :func:`sampleRealisations` to draw realisations consistent
            with the kriging.
        """
        if isinstance(attributes, str):
            attributes = [attributes]
        attributes = list(attributes)
        kinds = {} if kinds is None else dict(kinds)
        if others not in ("nearest", "nan"):
            raise ValueError("others must be 'nearest' or 'nan'.")
        n = self.points.shape[0]
        if n < 3:
            raise ValueError("fillHoles() needs >=3 points.")

        # ---- (u, v, w) of the existing points
        frame = self.frame
        uvw = self.uvw()
        ok = np.isfinite(uvw).all(axis=1)
        idx_ok = np.flatnonzero(ok)

        spec = {}
        for name in attributes:
            if not self.has_attr(name):
                raise ValueError(f"Attribute '{name}' not found.")
            arr = np.asarray(self.get_attr(name))
            if arr.ndim == 2 and arr.shape[1] == 1:
                arr = arr[:, 0]
            kind = kinds.get(name)
            if kind is None:
                integer_like = arr.ndim == 1 and (
                    np.issubdtype(arr.dtype, np.integer) or np.issubdtype(arr.dtype, np.bool_)
                    or (np.issubdtype(arr.dtype, np.floating) and np.all(np.mod(arr[ok], 1) == 0))
                )
                kind = "categorical" if integer_like else "continuous"
            spec[name] = (arr[ok], kind)

        if cell is None:
            cell = 5.0 * self.resolution()
        cover_uvw = None
        if cover is not None:
            if isinstance(cover, str):
                if cover != "removed":
                    raise ValueError("cover must be 'removed' or an (r, 3) array of points.")
                cover_xyz = self._removed if self._removed is not None else np.empty((0, 3))
            else:
                cover_xyz = np.asarray(cover, dtype=float).reshape(-1, 3)
            cover_uvw = frame.xyz_to_uvw(cover_xyz) if cover_xyz.shape[0] else None
            side = self._cover_side(cover_side, uvw, ok)
        removed_uv = None
        if use_removed and self._removed is not None and self._removed.shape[0] > 0:
            removed_uv = frame.xyz_to_uvw(self._removed)[:, :2]

        res = fill_holes_uv(
            uvw[ok], spec,
            cell=cell, removed_uv=removed_uv, spacing=spacing, min_cells=min_cells,
            ring_scale=ring_scale, attr_ring_scale=attr_ring_scale, max_train=max_train,
            n_closest=n_closest, variogram=variogram, attr_variogram=attr_variogram,
            trend_order=trend_order, nlags=nlags, aniso=aniso,
            angle=frame.horizontal_angle_uv() + bedding_angle, w_aniso=geometry_aniso,
            shared_variogram=shared_variogram, dense_variogram=dense_variogram, dense_lag=dense_lag,
            surface_tol=surface_tol, surface_radius=surface_radius,
            cover_uvw=cover_uvw, cover_side=side if cover_uvw is not None else 1.0, cover_margin=cover_margin,
            max_cover=max_cover, cover_samples=cover_samples, cover_sweeps=cover_sweeps, seed=seed,
        )
        for h in res["holes"]:
            if "train_idx" in h:                    # indices into the output (existing points come first)
                h["train_idx"] = idx_ok[h["train_idx"]]
        m = res["uvw"].shape[0]

        # ---- assemble: existing points followed by the new ones
        xyz_new = frame.uvw_to_xyz(res["uvw"]) if m else np.empty((0, 3))
        out = PointCloud(points=np.vstack([self.points, xyz_new]))
        out.set_attr("filled", np.r_[np.zeros(n, dtype=np.int8), np.ones(m, dtype=np.int8)])
        out.set_attr("hole_id", np.r_[np.zeros(n, dtype=np.int32), res["hole"].astype(np.int32)])
        out.set_attr("kriging_std", np.r_[np.zeros(n), res["std"]].astype(np.float32))
        if cover_uvw is not None:
            out.set_attr("cover_bound", np.r_[np.full(n, np.nan), res["cover_bound"]])
            out.set_attr("cover_free_w", np.r_[np.full(n, np.nan), res["free_w"]])

        nearest = idx_ok[res["nearest"]] if m else np.empty(0, dtype=np.int64)
        for name, arr in self.attrs.items():
            if name in spec or name in ("filled", "hole_id", "kriging_std", "cover_bound", "cover_free_w"):
                continue
            if others == "nearest":
                out.set_attr(name, np.concatenate([arr, arr[nearest]], axis=0))
            else:
                pad = np.full((m,) + arr.shape[1:], np.nan)
                out.set_attr(name, np.concatenate([arr.astype(float), pad], axis=0))

        for name in attributes:
            r = res["attrs"][name]
            arr = np.asarray(self.get_attr(name))
            if arr.ndim == 2 and arr.shape[1] == 1:
                arr = arr[:, 0]
            if r["kind"] == "categorical":
                out.set_attr(name, np.concatenate([arr, r["value"].astype(arr.dtype)]))
                P_all = np.vstack([(arr[:, None] == r["classes"][None, :]).astype(float), r["probs"]])
                for j, c in enumerate(r["classes"]):
                    out.set_attr(f"{name}_P{_class_label(c)}", P_all[:, j].astype(np.float32))
                out.set_attr(f"{name}_Pmax", P_all.max(axis=1).astype(np.float32))
            else:
                out.set_attr(name, np.concatenate([arr.astype(float), r["value"]], axis=0))
                out.set_attr(f"{name}_std", np.r_[np.zeros(n), r["std"]].astype(np.float32))

        out._frame = frame
        if return_info:
            info = dict(holes=res["holes"], n_new=int(m), spacing=res["spacing"], cell=res["cell"],
                        cover_side=res["cover_side"])
            return out, info
        return out

    # ----------------------------
    # Protected
    # ----------------------------
    def _cover_side(self, cover_side, uvw, ok) -> float:
        """+1 or -1: the direction of the air along ``w`` (see :meth:`fillHoles`)."""
        if cover_side in (1, -1, 1.0, -1.0):
            return float(cover_side)
        if cover_side != "auto":
            raise ValueError("cover_side must be 'auto', 1 or -1.")
        frame = self.frame
        if abs(frame.pca_axes[2, 2]) >= 0.2:                  # the face clearly dips
            return frame.outward_sign()
        if self.has_attr("normals"):
            nrm = np.asarray(self.get_attr("normals"), dtype=float)
            sub = np.flatnonzero(ok)[:: max(1, int(ok.sum()) // 20000)]
            ahead = frame.xyz_to_uvw(self.points[sub] + 0.01 * nrm[sub])[:, 2]
            dw = ahead - uvw[sub, 2]
            dw = dw[np.isfinite(dw)]
            if dw.size and np.median(dw) != 0:
                return float(np.sign(np.median(dw)))
        raise ValueError("The face is nearly vertical and there are no normals: give cover_side=1 or -1 "
                         "(the direction of the air along w).")

    def _from_indices(self, ids: np.ndarray) -> "PointCloud":
        """Create a new PointCloud with a subset of rows, preserving attributes."""
        ids = np.asarray(ids, dtype=int)
        sub_pts = self.points[ids]
        sub_attrs: dict[str, np.ndarray] = {}
        for k, arr in self.attrs.items():
            sub_attrs[k] = arr[ids] if arr.ndim == 1 else arr[ids, ...]
        sub = PointCloud(points=sub_pts, attrs=sub_attrs)
        sub._frame = self._frame  # a subset lives in the same space
        return sub
