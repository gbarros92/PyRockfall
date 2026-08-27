from __future__ import annotations

from pathlib import Path
from typing import Tuple, List, Mapping, Optional
from numpy.typing import NDArray

import numpy as np
from sklearn.neighbors import NearestNeighbors, RadiusNeighborsClassifier, KNeighborsClassifier

try:
    import open3d as o3d  # optional, used if available for IO
    _HAS_O3D = True
except Exception:  # pragma: no cover
    _HAS_O3D = False

ArrayF = NDArray[np.float64]

from ._model import Model
from ._geometry import Geometry
from ._utils import rotationAlign2x, pcWalk, trace_polyline_nodes, assign_sections_from_nodes

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
        Split along arclength into contiguous windows of width `ds`.

        Returns
        -------
        list[PointCloud]
        """
        x0, y0, height, radius = self.coneApproximation()
        if radius == 0:
            raise ValueError('Zero radius found in cone approximation.')

        v = self.points.copy()
        x = v[:, 0] - x0
        y = v[:, 1] - y0
        z = v[:, 2] - v.min(axis=0)[2]

        if height < 0:
            radius *= 1 - z.max() / height

        # Find new polar coords
        r = np.hypot(x, y)
        theta = np.arctan2(y, x)

        # reorder
        order = np.argsort(theta)
        theta = theta[order]
        r = r[order]
        v = v[order]

        # Divide in segments
        tot_len = radius * (theta[-1] - theta[0])
        dt = ds / radius
        du = du / radius
        step = dt - du
        t0 = np.arange(theta[0], theta[-1] + 1e-12, step)
        t1 = t0 + dt
        mask = t0 < tot_len - 1e-12
        t0, t1 = t0[mask], t1[mask]

        start_idx = np.searchsorted(theta, t0, side="left")
        end_idx   = np.searchsorted(theta, np.minimum(t1, theta[-1]), side="left")

        good = end_idx > start_idx
        start_idx = start_idx[good]
        end_idx = end_idx[good]

        out: List[PointCloud] = []
        for s, e in zip(start_idx, end_idx):
            idx = np.array([order[i] for i in range(s, e)], dtype=int)
            seg = self._from_indices(idx)
            for k, arr in self.attrs.items():
                seg.set_attr(k, arr[idx] if arr.ndim == 1 else arr[idx, ...])
            out.append(seg)
        return out

    def split(self, dx: float = 0.5, minPerc: float = 0.05) -> List["PointCloud"]:
        """
        Split into two parts by scanning vertical cuts along X and maximising
        the difference of `alignWithX()` angles (original heuristic).
        """
        v = self.points
        if v.size == 0:
            return []

        pmin, pmax = self.boundingBox()
        vv = v - pmin  # local frame
        lx = float(pmax[0] - pmin[0])
        nx = max(1, int(lx / dx) - 1)

        deltaA = np.zeros(nx, dtype=float)
        n_total = vv.shape[0]
        n_min = int(max(1, minPerc * n_total))
        n_max = n_total - n_min

        def _angle_of(points_xyz: ArrayF) -> float:
            tmp = PointCloud(points=points_xyz + pmin)
            return tmp.dipDirection()

        for i in range(nx):
            x_cut = dx + i * dx
            left_ids = np.where(vv[:, 0] <= x_cut)[0]
            n_left = left_ids.size
            if n_left < n_min or n_left > n_max:
                continue
            right_ids = np.setdiff1d(np.arange(n_total), left_ids, assume_unique=False)
            a1 = _angle_of(vv[left_ids, :] + pmin)
            a2 = _angle_of(vv[right_ids, :] + pmin)
            deltaA[i] = a1 - a2

        i_best = int(np.argmax(np.abs(deltaA)))
        x_cut = dx + i_best * dx
        left_ids = np.where(vv[:, 0] <= x_cut)[0]
        right_ids = np.setdiff1d(np.arange(n_total), left_ids, assume_unique=False)

        return [self._from_indices(left_ids), self._from_indices(right_ids)]

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
        direction: NDArray[np.float64],
        increment: float,
        label: str,
        *,
        slice_height_increment: float = -1.0,
        slice_resolution_factor: float = 5.0,
        **kwargs,
    ) -> Tuple[List["Geometry"], NDArray]:
        """
        Vectorised point-cloud slicing:
        - Rotate/centre so `direction` aligns with +X'
        - Bin all points to slice centers at once (±increment/2 segments)
        - Predict materials at segment midpoints via KNN classification
        - Assemble Geometry objects (light per-slice loop only)

        Returns
        -------
        list[Geometry], ndarray
        """
        points = self.points.copy()  # copy because we modify
        if points.size == 0:
            return [], np.array([], dtype=float)

        if not self.has_attr(label):
            raise ValueError(f"Attribute '{label}' not found.")
        mat_ids = np.asarray(self.get_attr(label))
        if mat_ids.ndim == 2 and mat_ids.shape[1] == 1:
            mat_ids = mat_ids[:, 0]
        mat_ids = mat_ids.astype(int, copy=False)

        if increment <= 0:
            raise ValueError("increment must be > 0.")
        
        # --- Decide vertical resolution (shared Z' grid)
        dlt_z = slice_height_increment if slice_height_increment > 0 else self.resolution() * slice_resolution_factor
        if not np.isfinite(dlt_z) or dlt_z <= 0:
            raise ValueError("slice_height_increment and resolution are invalid.")
        
        # --- Align: rotate so direction -> +x'
        R = rotationAlign2x(direction)               # Get rotation matrix
        centre = points.mean(axis=0, keepdims=True)  # (1,3)
        points = (points - centre) @ R.T             # row-vector convention
        centre = (centre @ R.T)
        centre_x = centre[0, 0]
        centre = centre[0, 1:]

        # --- Slice centers along x'
        x = points[:, 0]
        x_min = float(np.min(x))
        x_max = float(np.max(x))
        if not np.isfinite(x_max - x_min) or x_max <= x_min:
            return [], np.array([], float)

        half = 0.5 * increment  # typical "±inc/2" slab

        # Slice centres
        x_positions = np.arange(x_min + half, x_max + 1e-12, increment, dtype=float)
        num_slices = x_positions.size
        if num_slices == 0:
            return [], np.array([], float)

        # ---- Sort once by x and keep views for YZ ----
        order = np.argsort(x, kind="stable")
        xs = x[order]
        yz = points[order, 1:]   # (N,2) view: [y,z]

        # ---- For each slab m, find contiguous [start:end) in xs ----
        left  = x_positions - half  # min(dlt_z/2, half)
        right = x_positions + half  # min(dlt_z/2, half)
        starts = np.searchsorted(xs, left,  side="left")
        ends   = np.searchsorted(xs, right, side="right")

        # Start
        valid_slices = starts < ends
        if not valid_slices.any():
            return [], np.array([], dtype=float)
        polylines = []
        max_pts = 0
        for i in range(num_slices):
            # --- Bin all points to their nearest slice centre (slab ±inc/2)
            lo, hi = starts[i], ends[i]
            slab_points = yz[lo:hi]
            dlt_x = np.abs(xs[lo:hi] - x_positions[i])

            if slab_points.size == 0:
                # empty slab
                valid_slices[i] = False
                continue

            # Compute PCA to order points along main direction
            X = slab_points - slab_points.mean(0)
            _, _, vh = np.linalg.svd(X, full_matrices=False)
            pc1 = X @ vh[0]
            order = np.argsort(pc1)
            slab_points = slab_points[order]
            dlt_x = dlt_x[order]
            if slab_points[0, 1] < slab_points[-1, 1]:
                # ensure ascending order in Y
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
        # Midpoints along vertical segments (drop last row because it has no pair)
        x_mid = x_grid[:, :-1]          # x is constant per slice
        y_mid = 0.5 * (y_grid[:, :-1] + y_grid[:, 1:])
        z_mid = 0.5 * (z_grid[:, :-1] + z_grid[:, 1:])

        mats_all = np.full_like(y_mid, -1, dtype=int)
        mid_nan = np.isnan(y_mid) | np.isnan(z_mid)

        mid = np.column_stack((x_mid[~mid_nan], y_mid[~mid_nan], z_mid[~mid_nan]))
        
        # material classifier (3D)
        # cls = RadiusNeighborsClassifier(radius=max(increment/2, dlt_z/2), weights="distance")
        cls = KNeighborsClassifier(n_neighbors=10, weights="distance")
        cls.fit(points, mat_ids)
        mats_all[~mid_nan] = cls.predict(mid)

        profiles = []
        for i in range(len(x_positions)):
            nodes_i = nodes[i]
            mask = np.isfinite(nodes_i).all(axis=1)   # (N,) True if both y and z finite
            nodes_i = nodes_i[mask]
            mats_all_i = mats_all[i]
            mats_all_i = mats_all_i[mats_all_i >= 0]
            if nodes_i.shape[0] < 2 or mats_all_i.shape[0] < 1:
                raise ValueError("Invalid profile")
            if nodes_i.shape[0] != mats_all_i.shape[0] + 1:
                raise ValueError("Invalid profile")
            profiles.append(Geometry(nodes=nodes_i+centre, attributes=mats_all_i))

        return profiles, x_positions + centre_x
 

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
    ):
        """
        Extract 2D profiles along the strike of the model.

        Unlike :meth:`slice`, which cuts parallel profiles along a single
        fixed direction, ``section`` traces a polyline that follows the
        curvature of the point cloud along strike, then extracts a profile
        perpendicular to that polyline at every ``increment`` step. This
        makes it suitable for curved or non-planar features (e.g. a bench
        or wall that changes orientation along its length), where a
        fixed-direction slice would cut obliquely through the surface.

        Processing outline
        -------------------
        1. Align the point cloud so its mean strike lies along +X
           (via :meth:`dipDirection` / :meth:`alignWithX`).
        2. Trace a polyline of nodes along strike in the XY-plane, spaced
           roughly ``increment`` apart, following the local point density
           (:func:`trace_polyline_nodes`).
        3. Assign the original points to the section (slab) centred on
           each polyline segment midpoint, using a strip of half-width
           ``transverse_radius`` normal to the segment tangent
           (:func:`assign_sections_from_nodes`).
        4. For each section, walk the assigned points in local
           (distance-along-section, height) coordinates to build an
           ordered profile (:func:`pcWalk`), then rotate the profile back
           into global XYZ coordinates.
        5. Classify materials at the profile points via KNN
           (using the `label` attribute) and package the results as
           :class:`Geometry` profiles plus a combined :class:`PointCloud`
           of all section points.

        Parameters
        ----------
        increment : float
            Target spacing, in the point cloud's units, between
            consecutive polyline nodes along strike (and hence between
            sections).
        label : str
            Name of the point attribute holding material/class IDs, used
            to classify the extracted profile points.
        transverse_radius : float, optional
            Half-width of the strip (normal to the local strike direction)
            used to assign points to each section. Defaults to half the
            point cloud's vertical (Z) extent.
        initial_direction : tuple[float, float], optional
            Initial (x, y) marching direction used to seed the polyline
            trace before it locks onto the local point distribution.
        min_points : int, optional
            Minimum number of points required for a section to be
            considered valid; sections with fewer points are discarded.
        max_nodes : int, optional
            Maximum number of nodes to trace along the polyline. ``None``
            for no limit.
        max_iter : int, optional
            Maximum number of refinement iterations per traced node.
        tol : float, optional
            Convergence tolerance used while tracing each node.
        max_turn_angle : float, optional
            Maximum allowed change in marching direction between
            consecutive nodes, in radians. ``None`` disables the limit.
        weight_power : float, optional
            Power applied to distance-based weighting when estimating the
            local marching direction/centre at each step.
        min_new_fraction : float, optional
            Minimum fraction of newly covered points required per step for
            the trace to continue advancing.

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
            If no valid sections are found, or if a constructed profile is
            inconsistent (mismatched node/material counts).
        """
        self_dip_dir = self.dipDirection()
        self_centre = self.centroid()
        self.translate(-self_centre)
        self.alignWithX(sense=-1)

        if transverse_radius is None:
            transverse_radius = float((self.points[:,-1].max() - self.points[:,-1].min()) / 2)
        nodes = trace_polyline_nodes(
            points=self.points,
            increment=increment,
            transverse_radius=transverse_radius,
            initial_direction=initial_direction,
            min_points=min_points,
            max_nodes=max_nodes,
            max_iter=max_iter,
            tol=tol,
            max_turn_angle=max_turn_angle,
            weight_power=weight_power,
            min_new_fraction=min_new_fraction,
        )

        # ------------------------------------------------------------
        # Assign original point-cloud points to each section
        # ------------------------------------------------------------
        section_ids, centres, tangents, section_dirs = assign_sections_from_nodes(
            points=self.points,
            nodes=nodes,
            transverse_radius=transverse_radius,
            min_points=min_points,
        )

        # ------------------------------------------------------------
        # Walk each section in local 2D coordinates, then rotate back
        # to the same global XYZ system as self.points.
        # ------------------------------------------------------------
        sections = []
        valid = np.ones(len(section_ids), dtype=bool)
        section_id = []

        for k, idx in enumerate(section_ids):
            if idx.size < min_points:
                valid[k] = False
                continue

            # Original points belonging to this slab, in global XYZ coordinates
            slab_points = self.points[idx]

            # Relative XY coordinates with respect to the section centre
            rel_xy = slab_points[:, :2] - centres[k]

            # Local coordinates
            #
            # dlt_s: coordinate along the wall/segment tangent
            # dlt_r: coordinate along the 2D section direction
            dlt_s = rel_xy @ tangents[k]
            dlt_r = rel_xy @ section_dirs[k]
            z = slab_points[:, 2]

            # Local 2D profile coordinates used by pcWalk:
            # column 0 = distance along section direction
            # column 1 = height
            profile_points = np.column_stack([dlt_r, z])

            # --------------------------------------------------------
            # Order points before pcWalk
            # --------------------------------------------------------
            X = profile_points - profile_points.mean(axis=0)
            _, _, vh = np.linalg.svd(X, full_matrices=False)

            pc1 = X @ vh[0]
            order = np.argsort(pc1)

            profile_points = profile_points[order]
            dlt_s = dlt_s[order]
            dlt_r = dlt_r[order]

            # Keep your existing orientation convention
            if profile_points[0, 1] < profile_points[-1, 1]:
                profile_points = profile_points[::-1]
                dlt_s = dlt_s[::-1]
                dlt_r = dlt_r[::-1]

            # --------------------------------------------------------
            # Walk along profile in local 2D coordinates
            # section[:, 0] = local dlt_r
            # section[:, 1] = z
            # --------------------------------------------------------
            section = pcWalk(
                profile_points[0],
                profile_points[-1],
                profile_points,
                dlt_s,
                0.5,
            )

            # --------------------------------------------------------
            # Rotate section back to global XYZ coordinates.
            #
            # The extracted profile lies exactly on the section plane,
            # so its coordinate along the tangent is zero.
            #
            # global_xy = centre + r * section_direction
            # --------------------------------------------------------
            r_sec = section[:, 0]
            z_sec = section[:, 1]

            xy_sec = (
                centres[k][None, :]
                + r_sec[:, None] * section_dirs[k][None, :]
            )

            section_global = np.column_stack([xy_sec, z_sec])
            sections.append(section_global)
            section_id.append([k] * section_global.shape[0])

        # ------------------------------------------------------------
        # Pack variable-length global 3D sections into padded array
        # ------------------------------------------------------------
        if len(sections) == 0:
            raise ValueError("No valid sections found. Try adjusting parameters.")
        
        section_pts = np.concatenate(sections, axis=0)
        section_id = np.concatenate(section_id, axis=0).astype(int)

        # ------------------------------------------------------------
        # Material classifier in global XYZ coordinates
        # ------------------------------------------------------------
        mat_ids = self.get_attr(label)
        cls = KNeighborsClassifier(
            n_neighbors=10,
            weights="distance",
        )
        cls.fit(self.points, mat_ids)
        material_id = cls.predict(section_pts)

        # ------------------------------------------------------------
        # Construct PointCloud sections with material attributes
        # ------------------------------------------------------------
        sections_pc = PointCloud(points=section_pts)
        sections_pc.set_attr(label, material_id)
        sections_pc.set_attr("Profile", section_id)
        sections_pc.rotate(-self_dip_dir)  # rotate back to original orientation
        sections_pc.translate(self_centre)  # translate back to original position

        self.rotate(-self_dip_dir)  # rotate back to original orientation
        self.translate(self_centre)  # translate back to original position

        # ------------------------------------------------------------
        # Create Geometry 2D profiles
        # ------------------------------------------------------------
        profiles = []

        for id in np.unique(section_id):
            nodes_i = section_pts[section_id == id]
            mats_i = material_id[section_id == id]

            if nodes_i.shape[0] < 2 or mats_i.shape[0] < 2:
                raise ValueError("Invalid profile")

            if nodes_i.shape[0] != mats_i.shape[0] :
                raise ValueError("Invalid profile")
            
            z_min = nodes_i[:, 2].min()
            id_z_max = np.argmax(nodes_i[:, 2])
            nodes_i[:, :2] = nodes_i[:, :2] - nodes_i[id_z_max, :2]  # relative to base of section
            nodes_i[:, 2] = nodes_i[:, 2] - z_min  # relative to base of section
            dist_xy = np.linalg.norm(nodes_i[:, :2], axis=1)
            prof_coords = np.column_stack([dist_xy, nodes_i[:, 2]])

            profiles.append(
                Geometry(
                    nodes=prof_coords,
                    attributes=mats_i[:-1],
                )
            )
        return profiles, sections_pc



    # ----------------------------
    # Protected
    # ----------------------------
    def _from_indices(self, ids: np.ndarray) -> "PointCloud":
        """Create a new PointCloud with a subset of rows, preserving attributes."""
        ids = np.asarray(ids, dtype=int)
        sub_pts = self.points[ids]
        sub_attrs: dict[str, np.ndarray] = {}
        for k, arr in self.attrs.items():
            sub_attrs[k] = arr[ids] if arr.ndim == 1 else arr[ids, ...]
        return PointCloud(points=sub_pts, attrs=sub_attrs)
