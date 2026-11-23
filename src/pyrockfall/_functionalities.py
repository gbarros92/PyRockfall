"""
User functionalities in pyrockfall
==================================

This module implements functions to facilitate rockfall simulation.
"""
import numpy as np
from scipy.spatial import Delaunay, cKDTree
from numpy.typing import NDArray
from typing import Union, List, Tuple, Optional, Callable, Literal, Dict, Any

from ._geometry import Geometry
from ._material import Material
from ._slope import Slope
from ._rock import Rock
from ._seeder import Seeder
from ._model import Model
from ._pointcloud import PointCloud
from ._mesh import Mesh


def singleMaterialSlope(slope: Slope, material: Union[Material, int]) -> List[Slope]:
    """Create a list of slopes, each containing only a single material from the original slope.

    Parameters
    ----------
    slope : Slope
        The original slope.
    material : Material or int
        The material to isolate. Can be the material instance or its index (int).
        
    Returns
    -------
    List[Slope]
        A list of slopes, each containing only one material.
    """
    if isinstance(material, Material):
        slope_materials = slope.materials
    elif isinstance(material, int):
        slope_materials = slope.materialIDs
    else:
        raise TypeError("material must be an instance of Material or an integer (index).")

    slopes = []
    if material not in slope_materials:
        return slopes  # Material not found, return empty list
    
    # Get material instance
    if isinstance(material, int):
        mat_obj = slope.materialTable[material]
    else:
        mat_obj = material

    new_nodes = []
    new_nodes_std = []
    collecting = False
    for mat_id, mat in enumerate(slope_materials):
        if mat == material:
            collecting = True
            if len(new_nodes) == 0:
                new_nodes.append(slope.nodes[mat_id])
                if slope.hasUncertainty:
                    new_nodes_std.append(slope.nodes_std[mat_id])
            new_nodes.append(slope.nodes[mat_id+1])
            if slope.hasUncertainty:
                new_nodes_std.append(slope.nodes_std[mat_id+1])
        else:
            if collecting:
                collecting = False
                # Create new slope
                slopes.append(Slope(
                    Geometry(
                        nodes=np.array(new_nodes),
                        nodes_std=np.array(new_nodes_std) if slope.hasUncertainty else None,
                    ),
                    materials=[mat_obj],
                    materialIDs=[0]*(len(new_nodes)-1),
                ))
                new_nodes = []
                new_nodes_std = []
    if collecting:
        # Create last new slope
        slopes.append(Slope(
            Geometry(
                nodes=np.array(new_nodes),
                nodes_std=np.array(new_nodes_std) if slope.hasUncertainty else None,
            ),
            materials=[mat_obj],
            materialIDs=[0]*(len(new_nodes)-1),
        ))
    return slopes

def removeMaterial(slope: Slope, material: Union[Material, int]) -> Slope:
    """Remove a material from the slope.

    Parameters
    ----------
    slope : Slope
        The original slope.
    material : Material or int
        The material to remove. Can be the material instance or its index (int).
        
    Returns
    -------
    Slope
        The modified slope.
    """
    if isinstance(material, Material):
        slope_materials = slope.materials
    elif isinstance(material, int):
        slope_materials = slope.materialIDs
    else:
        raise TypeError("material must be an instance of Material or an integer (index).")

    if material not in slope_materials:
        return slope  # Material not found, return original slope

    # Get new nodes and materials identities
    new_nodes = []
    new_nodes_std = []
    new_materials = []
    do_add_first = slope_materials[0] != material
    have_kept_run = False  # True once at least one kept segment has been emitted
    for mat_id, mat in enumerate(slope_materials):
        if mat != material:
            # Add nodes
            if do_add_first:
                do_add_first = False
                new_nodes.append(slope.nodes[mat_id])
                if slope.hasUncertainty:
                    new_nodes_std.append(slope.nodes_std[mat_id])
                if have_kept_run:
                    # Bridging across a removed run in the middle of the
                    # profile: the new connecting edge (from the end of the
                    # previous kept run to this segment's start) needs its
                    # own material entry, taking on this segment's material.
                    new_materials.append(slope.materialIDs[mat_id])

            # Add material for the (unchanged) kept segment itself
            new_materials.append(slope.materialIDs[mat_id])
            new_nodes.append(slope.nodes[mat_id+1])
            if slope.hasUncertainty:
                new_nodes_std.append(slope.nodes_std[mat_id+1])
            have_kept_run = True
        else:
            do_add_first = True

    # Get new material table
    mat_tab = slope.materialTable
    if isinstance(material, int):
        material_id = material
    else:
        material_id = mat_tab.index(material)
    new_mat_tab = mat_tab[:material_id] + mat_tab[material_id+1:]
    new_materials = np.array(new_materials, dtype=int)
    new_materials[new_materials > material_id] -= 1  # Shift material IDs

    if len(new_nodes) < 2:
        raise ValueError("Cannot remove the material as it would leave an empty slope.")
    
    # Create new slope
    return Slope(
        Geometry(
            nodes=np.array(new_nodes),
            nodes_std=np.array(new_nodes_std) if slope.hasUncertainty else None,
        ),
        materials=new_mat_tab,
        materialIDs=new_materials,
    )

def findClosest(slope: Slope, height: float) -> Tuple[float, int]:
    """
    Find the closest point on the slope to a given height.

    Parameters
    ----------
    height : float
        The height (y-coordinate) to find the closest point to.

    Returns
    -------
    Tuple[float, int]
        A tuple (x_closest, element_id) with the x coordinate in the slope
        for the given height and the ID of the element where the closest
        point is located.
    """
    if slope.nodes.shape[1] != 2:
        raise ValueError("findClosest is only implemented for 2D slopes.")
    
    # Given y find x in the profile
    for e, element in enumerate(slope.elements):
        i, j = element
        if slope.nodes[i+1][1] <= height <= slope.nodes[i][1]:
            seg_len = slope.nodes[j][1] - slope.nodes[i][1]
            if seg_len == 0:
                continue
            t = (height - slope.nodes[i][1]) / seg_len
            x = (slope.nodes[j][0] - slope.nodes[i][0]) * t + slope.nodes[i][0]
            break
    return x, e

def extrudePolyline(nodes_2d: np.ndarray, dy: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Extrudes a 2D polyline into a 3D surface mesh.

    Parameters
    ----------
    nodes_2d : np.ndarray
        The input 2D nodes (N, 2) where columns are (x, z).
    dy : float
        The extrusion distance along the y-axis.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        - nodes_3d: (2N, 3) with (x, 0, z) and (x, 1, z)
        - tris: (2*(N-1), 3) with triangles per segment:
                [i, i+1, N+i], [N+i, i+1, N+i+1]
    """
    nodes_2d = np.asarray(nodes_2d, dtype=float)  # (N,2) where columns are (x,z)
    assert nodes_2d.ndim == 2 and nodes_2d.shape[1] == 2
    N = nodes_2d.shape[0]

    # Build 3D nodes
    x = nodes_2d[:, 0]
    z = nodes_2d[:, 1]
    lower = np.column_stack([x, np.zeros(N),   z])       # y=0
    upper = np.column_stack([x, dy*np.ones(N), z])       # y=dy
    nodes_3d = np.vstack([lower, upper])               # (2N,3)

    # Triangulate each quad (i -> i+1)
    tris = []
    for i in range(N - 1):
        tris.append([i, i + 1, N + i])         # [i, i+1, N+i]
        tris.append([N + i, i + 1, N + i + 1]) # [N+i, i+1, N+i+1]
    tris = np.asarray(tris, dtype=int)         # (2*(N-1), 3)

    return nodes_3d, tris

# Convert grid to triangular mesh for saving as PLY
def grid2mesh(x: np.ndarray, y: np.ndarray, z0: float = 0.0):
    """
    Build a 2-triangle-per-cell mesh over a rectilinear grid defined by
    x -> (nx,) and y -> (ny,). Vertices lie on the z=z0 plane.

    Returns:
        points   : (ny*nx, 3) in row-major (y major, then x)
        triangles: (2*(ny-1)*(nx-1), 3) int32, CCW winding
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ny, nx = y.size, x.size

    # Vertices in row-major: for each row i (y[i]), iterate j over x
    Xg, Yg = np.meshgrid(x, y, indexing='ij')
    points = np.column_stack([Xg.ravel(), Yg.ravel(),
                              np.full(nx*ny, z0, dtype=float)])

    # Vectorised cell indices
    i = np.arange(nx - 1)[:, None]   # (nx-1, 1)
    j = np.arange(ny - 1)[None, :]   # (1, ny-1)

    # Vertex ids with stride = ny
    v00 = (i    )*ny + (j    )
    v01 = (i    )*ny + (j + 1)
    v10 = (i + 1)*ny + (j    )
    v11 = (i + 1)*ny + (j + 1)

    # Two triangles per quad (CCW if x and y are ascending)
    tri1 = np.stack([v00, v10, v11], axis=-1).reshape(-1, 3)  # CCW
    tri2 = np.stack([v00, v11, v01], axis=-1).reshape(-1, 3)  # CCW
    triangles = np.vstack([tri1, tri2]).astype(np.int32)

    # Keep CCW if exactly one axis is descending
    if (x[-1] < x[0]) ^ (y[-1] < y[0]):
        triangles[:, [1, 2]] = triangles[:, [2, 1]]

    return points, triangles

def mesh2PointCloud(mesh: Mesh) -> PointCloud:
    """Build a :class:`PointCloud` from a mesh's nodes.

    Only per-node attributes are carried over (any attribute whose first
    dimension matches the triangle count instead of the point count is
    treated as per-triangle and dropped). Triangle connectivity is
    discarded entirely.

    Parameters
    ----------
    mesh : Mesh
        The source mesh.

    Returns
    -------
    PointCloud
        A new point cloud with the mesh's nodes and per-node attributes.
    """
    n = mesh.points.shape[0]
    node_attrs = {k: v for k, v in mesh.attrs.items() if v.shape[0] == n}
    return PointCloud(mesh.points.copy(), attrs=node_attrs)

def pointCloud2Mesh(
    pc: PointCloud,
    *,
    method: Literal["uv", "plane"] = "uv",
    max_edge: Optional[float] = None,
) -> Mesh:
    """Build a :class:`Mesh` from a point cloud via automatic Delaunay
    triangulation.

    By default the points are triangulated in the ``(u, v)`` plane of the
    point cloud's curvilinear frame (see :meth:`Model.uvw`), where the wall is
    unrolled flat along its strike. This also works for walls that change
    direction (corners, curved walls), which no single plane can represent.
    The triangle connectivity is attached to the original (untouched) points,
    and all per-point attributes are carried over as per-node attributes. The
    mesh shares the frame of the point cloud.

    Parameters
    ----------
    pc : PointCloud
        The source point cloud.
    method : {"uv", "plane"}
        ``"uv"`` (default) triangulates in the unrolled ``(u, v)`` plane.
        ``"plane"`` levels a copy of the cloud with one rotation (strike onto
        X, then the dip about X) and triangulates its (X, Y); only suitable
        for a wall that is close to a single plane.
    max_edge : float, optional
        Drop triangles with an edge longer than this (measured in 3D). A
        Delaunay triangulation fills the convex outline of the points, so long
        triangles bridge gaps and concave parts of the outline. ``None`` keeps
        every triangle.

    Returns
    -------
    Mesh
        A new mesh with the point cloud's nodes, automatically triangulated
        connectivity, and per-node attributes. Points outside the frame (NaN
        coordinates) are kept as nodes but belong to no triangle.
    """
    points = pc.points
    if method == "uv":
        frame = pc.frame
        uvw = pc.uvw()
        ok = np.isfinite(uvw).all(axis=1)
        if ok.all():
            simplices = Delaunay(uvw[:, :2]).simplices
        else:
            simplices = np.flatnonzero(ok)[Delaunay(uvw[ok, :2]).simplices]
    elif method == "plane":
        frame = None
        levelled = PointCloud(points.copy())
        levelled.alignWithX()
        levelled.rotate(-levelled.dip(), axis="x")
        simplices = Delaunay(levelled.points[:, :2]).simplices  # (X, Y) plane
    else:
        raise ValueError("method must be 'uv' or 'plane'.")

    simplices = simplices.astype(np.int32)
    if max_edge is not None:
        tri_pts = points[simplices]                                   # (E, 3, 3)
        edges = np.linalg.norm(tri_pts - np.roll(tri_pts, -1, axis=1), axis=2)
        simplices = simplices[edges.max(axis=1) <= max_edge]

    mesh = Mesh(points.copy(), simplices, attrs=dict(pc.attrs))
    if frame is not None:
        mesh.frame = frame  # same points, same space
    return mesh

def _probability_attributes(pc: PointCloud, label: str) -> Tuple[List[str], NDArray[np.float64]]:
    """Names of the ``<label>_P<class>`` attributes of a point cloud, and their classes."""
    prefix = f"{label}_P"
    names, classes = [], []
    for name in pc.list_attrs():
        if not name.startswith(prefix) or name == f"{label}_Pmax":
            continue
        code = name[len(prefix):]
        try:
            classes.append(float(code.replace("m", "-").replace("_", ".")))
        except ValueError:
            continue
        names.append(name)
    order = np.argsort(classes)
    return [names[i] for i in order], np.asarray(classes, dtype=float)[order]


def sampleRealisations(
    pc: PointCloud,
    n_realisations: int,
    label: str,
    *,
    std: str = "kriging_std",
    info: Optional[Dict[str, Any]] = None,
    frame=None,
    variogram: str = "spherical",
    geometry_aniso: float = 3.0,
    aniso: float = 10.0,
    bedding_angle: float = 0.0,
    ring_scale: float = 1.0,
    max_train: int = 2000,
    dense_lag: float = 3.0,
    surface_tol: Optional[float] = 0.15,
    surface_radius: Optional[float] = None,
    resolution: Optional[float] = None,
    uncertain_only: bool = False,
    cover_sweeps: int = 50,
    seed: int = 0,
) -> List[PointCloud]:
    """
    Draw realisations of a point cloud whose position and material are uncertain,
    e.g. the output of :meth:`PointCloud.fillHoles`.

    Each uncertain point carries the standard deviation of its height above the
    face plane (attribute `std`) and one probability per material (attributes
    ``<label>_P<class>``, summing to 1). In every realisation:

    - **Coordinates** (conditional simulation): the height above the face plane
      is ``w_r = w + (s - s_ok)``, where ``s`` is a random surface with the
      variogram of the height (roughness included) and ``s_ok`` is its kriging
      from the existing points around the hole. ``s - s_ok`` is a draw of the
      kriging error, so over many realisations each point has mean ``w`` and
      standard deviation `std`, while each realisation is as rough as the
      variogram everywhere and joins the existing rock at the edge of the hole.
      The point moves by ``w_r - w`` along the ``w`` axis of the curvilinear
      frame, keeping its ``(u, v)``.
    - **Material:** ``q = Phi(g)`` with ``g`` a spatially correlated Gaussian
      field (coherent layers), and the point gets the first class whose
      cumulative probability exceeds ``q`` (inverse CDF), so class ``k`` is drawn
      with probability ``P_k``.

    Points with ``std == 0`` and a probability of 1 for one class are certain and
    are copied unchanged.

    If the holes were filled with a cover (``fillHoles(..., cover=...)``, and
    `info` is given), each realisation first draws the height at the cover
    bounds from its truncated Gaussian (:func:`~._kriging.truncated_gaussian_gibbs`),
    so it lies behind the cover there, and the kriging error is simulated
    conditionally on those heights as well as on the existing points. Points
    with a bound of their own (attribute ``"cover_bound"``) are then clipped at
    it, so the realisations average to the stored height and spread as
    ``kriging_std``.

    The variograms and conditioning points are set per hole (attribute
    ``"hole_id"``; one group if it is missing):

    - from `info`, the dict returned by ``fillHoles(..., return_info=True)``:
      the variograms and the points used to fill each hole, so the spread of
      ``w`` matches `std` exactly; or
    - otherwise from the certain points around the hole that lie on the main
      surface (see :func:`~._kriging.surface_mask`): the nested variogram of
      the detrended ``w`` (see :func:`~._kriging.fit_nested_variogram`), and the
      indicator variogram of the class whose proportion is closest to 0.5,
      with ranges `aniso` times longer along the layers. The spread of ``w``
      then matches `std` only approximately.

    Parameters
    ----------
    pc : PointCloud
        Point cloud with the attributes `std`, ``<label>_P<class>`` and `label`.
    n_realisations : int
        Number of realisations.
    label : str
        Material attribute. Its probabilities are ``<label>_P<class>``.
    std : str
        Attribute with the standard deviation of the height above the face plane.
    info : dict, optional
        Output of ``fillHoles(..., return_info=True)``, to reuse its variograms
        and conditioning points. Its indices refer to the cloud returned by that
        call.
    frame : CurvilinearFrame, optional
        Frame in which ``w`` is measured. Defaults to the frame of `pc` (the one
        used to fill the holes, if `pc` came straight from :meth:`fillHoles`).
    variogram : str
        Variogram model fitted when `info` is not given.
    geometry_aniso, aniso, bedding_angle : float
        Anisotropy of the fitted ``w`` and material variograms (see
        :meth:`PointCloud.fillHoles`).
    ring_scale : float
        Width of the ring of certain points used when `info` is not given, as a
        fraction of the hole's size.
    max_train : int
        Largest number of ring points used as conditioning data.
    dense_lag, surface_tol, surface_radius : float
        As in :meth:`PointCloud.fillHoles`, when `info` is not given.
    resolution : float, optional
        Grid spacing of the simulated surfaces. Defaults, per hole, to half the
        spacing of its points, so the roughness between neighbouring points is
        reproduced (a coarser grid is faster but slightly too rough at the
        shortest distances).
    uncertain_only : bool
        Return only the uncertain points of each realisation.
    cover_sweeps : int
        Gibbs sweeps for the heights at the cover bounds.
    seed : int
        Random seed.

    Returns
    -------
    list[PointCloud]
        `n_realisations` point clouds, each with all attributes of `pc`, the
        sampled material in `label` and the sampled displacement along ``w`` in
        ``"dw"``. Attributes other than `label` and ``"dw"`` are shared with `pc`,
        not copied.

    Raises
    ------
    ValueError
        If attributes are missing or the probabilities are not valid.
    """
    from ._kriging import (Variogram, GaussianField, RandomField, conditional_simulation, fit_variogram,
                           fit_nested_variogram, poly_design, surface_mask, variogram_from_dict, _thin,
                           kriging_conditional, truncated_gaussian_gibbs)
    from scipy.spatial import cKDTree
    from scipy.special import ndtr

    if n_realisations < 1:
        raise ValueError("n_realisations must be >= 1.")
    for name in (std, label):
        if not pc.has_attr(name):
            raise ValueError(f"Attribute '{name}' not found.")
    p_names, classes = _probability_attributes(pc, label)
    if not p_names:
        raise ValueError(f"No probability attributes '{label}_P<class>' found.")

    sigma = np.asarray(pc.get_attr(std), dtype=float).ravel()
    P = np.column_stack([np.asarray(pc.get_attr(n), dtype=float) for n in p_names])
    if P.min() < -1e-9 or np.abs(P.sum(axis=1) - 1.0).max() > 1e-4:
        raise ValueError(f"'{label}_P<class>' must be >= 0 and sum to 1 at every point.")
    if np.any(sigma < 0) or not np.all(np.isfinite(sigma)):
        raise ValueError(f"'{std}' must be finite and >= 0.")
    labels = np.asarray(pc.get_attr(label))

    N = pc.points.shape[0]
    uncertain = (sigma > 0) | (P.max(axis=1) < 1.0 - 1e-9)
    idx = np.flatnonzero(uncertain)
    n_unc = idx.size
    frame = pc.frame if frame is None else frame
    uvw = frame.xyz_to_uvw(pc.points)
    if np.isnan(uvw[idx]).any():
        raise ValueError("Some uncertain points are outside the frame.")
    cdf = np.cumsum(P[idx], axis=1)
    cdf[:, -1] = 1.0                                      # guard against round-off

    rng = np.random.default_rng(seed)
    DW = np.zeros((n_realisations, n_unc))
    Q = np.empty((n_realisations, n_unc))
    groups = (np.asarray(pc.get_attr("hole_id"))[idx].astype(np.int64)
              if pc.has_attr("hole_id") else np.zeros(n_unc, dtype=np.int64))
    by_hole = {} if info is None else {h["hole"]: h for h in info.get("holes", []) if "skipped" not in h}
    angle = frame.horizontal_angle_uv() + bedding_angle
    certain = np.flatnonzero(~uncertain & np.isfinite(uvw).all(axis=1))
    tree = cKDTree(uvw[certain, :2]) if certain.size else None

    for g in (np.unique(groups) if n_unc else []):
        sel = np.flatnonzero(groups == g)
        uv_g = uvw[idx[sel], :2]
        sp_g = _point_spacing(uv_g)
        res_g = resolution if resolution is not None else (None if sp_g is None else 0.5 * sp_g)
        vw = vc = train = None

        known = by_hole.get(int(g))
        if known is not None:
            vw = variogram_from_dict(known["w_variogram"])
            cv = known.get("attr_variograms", {}).get(label)
            vc = variogram_from_dict(cv) if cv and "model" in cv else None
            t = known.get("train_idx")
            if t is not None:
                t = np.asarray(t, dtype=np.int64)
                if t.size and t.min() >= 0 and t.max() < N and not uncertain[t].any():
                    train = t

        size = float(np.sqrt(np.prod(np.maximum(np.ptp(uv_g, axis=0), 1e-6))))
        if known is None or train is None:
            # Conditioning points: certain points on the main surface around the group
            ring = np.empty(0, dtype=np.int64)
            if tree is not None:
                lo, hi = uv_g.min(axis=0) - ring_scale * size, uv_g.max(axis=0) + ring_scale * size
                box = certain[np.all((uvw[certain, :2] >= lo) & (uvw[certain, :2] <= hi), axis=1)]
                if box.size:
                    dist, _ = cKDTree(uv_g).query(uvw[box, :2], k=1, distance_upper_bound=ring_scale * size)
                    ring = box[np.isfinite(dist)]
            if surface_tol is not None and ring.size >= 3:
                on = surface_mask(uvw[ring, :2], uvw[ring, 2], radius=surface_radius, tol=surface_tol)
                if on.sum() >= 3:
                    ring = ring[on]
            if ring.size >= 10:
                train = _thin(uvw[:, :2], ring, max_train, rng)
                if known is None:
                    ur = uvw[train, :2]
                    c0 = ur.mean(axis=0)
                    sc = float(np.abs(ur - c0).max()) or 1.0
                    coef = np.linalg.lstsq(poly_design(ur, 1, c0, sc), uvw[train, 2], rcond=None)[0]
                    resid = uvw[ring, 2] - poly_design(uvw[ring, :2], 1, c0, sc) @ coef
                    if variogram == "linear":
                        vw = fit_variogram(uvw[ring, :2], resid, variogram, aniso=geometry_aniso, angle=angle, seed=seed)
                    else:
                        vw = fit_nested_variogram(uvw[ring, :2], resid, variogram, dense_lag=dense_lag,
                                                  aniso=geometry_aniso, angle=angle, max_points=max_train, seed=seed)
                    ind = labels[train][:, None] == classes[None, :]
                    props = ind.mean(axis=0)
                    present = ind.any(axis=0)
                    if present.sum() > 1:
                        ref = int(np.argmin(np.where(present, np.abs(props - 0.5), np.inf)))
                        vc = fit_variogram(ur, ind[:, ref].astype(float), variogram,
                                           aniso=aniso, angle=angle, seed=seed)
            elif known is None:
                # Too few certain points around: a range of a quarter of the hole
                vw = Variogram(variogram, psill=1.0, range=max(0.25 * size, 1e-3), aniso=geometry_aniso, angle=angle)
                vc = Variogram(variogram, psill=1.0, range=max(0.25 * size, 1e-3), aniso=aniso, angle=angle)

        # ---- coordinates: kriging error drawn by conditional simulation
        cover = known.get("cover") if (known is not None and train is not None and "trend" in known) else None
        if cover is not None and len(cover.get("uv", [])) > 0:
            # Heights at the cover bounds: truncated Gaussian given the existing points
            tr = known["trend"]
            t_args = (int(tr["order"]), np.asarray(tr["center"], dtype=float), float(tr["scale"]))
            t_coef = np.asarray(tr["coef"], dtype=float)
            cu = np.asarray(cover["uv"], dtype=float).reshape(-1, 2)
            side = float(cover["side"])
            r_d = uvw[train, 2] - poly_design(uvw[train, :2], *t_args) @ t_coef
            b = side * (np.asarray(cover["bound_w"], dtype=float) - poly_design(cu, *t_args) @ t_coef)
            mu_c, S_c = kriging_conditional(uvw[train, :2], r_d, cu, vw)
            Y = side * truncated_gaussian_gibbs(side * mu_c, S_c, b, n_realisations, rng=rng, sweeps=cover_sweeps)
            # w_r - w = OK of (0 at the existing points, Y - mean at the bounds) + kriging error
            vals = np.vstack([np.zeros((train.size, n_realisations)), (Y - np.asarray(cover["mean"], dtype=float)).T])
            DW[:, sel], _, _ = conditional_simulation(np.vstack([uvw[train, :2], cu]), vals, uv_g, vw,
                                                      n_realisations, rng=rng, resolution=res_g)
        elif train is not None:
            DW[:, sel], _, _ = conditional_simulation(uvw[train, :2], np.zeros(train.size), uv_g, vw,
                                                      n_realisations, rng=rng, resolution=res_g)
        else:
            # no conditioning points: an unconditional surface scaled to `std`
            z = RandomField(uv_g, vw, resolution=res_g).sample(rng, n_realisations)
            DW[:, sel] = sigma[idx[sel]] * z / np.sqrt(max(vw.psill + vw.nugget, 1e-300))

        # ---- materials: probability field
        Q[:, sel] = ndtr(GaussianField(uv_g, vc).sample(rng, n_realisations)) if vc is not None \
            else rng.random((n_realisations, sel.size))

    # ---- Cover: clip at the bound of each point, as fillHoles did for its mean
    side = None if info is None else info.get("cover_side")
    if side is not None and pc.has_attr("cover_bound") and pc.has_attr("cover_free_w") and n_unc:
        B = np.asarray(pc.get_attr("cover_bound"), dtype=float)[idx]
        F = np.asarray(pc.get_attr("cover_free_w"), dtype=float)[idx]
        has = np.isfinite(B) & np.isfinite(F)
        if has.any():
            side = float(side)
            X = F[has] + DW[:, has]                       # realisation before the bound
            Y = side * np.minimum(side * X, side * B[has])
            DW[:, has] = Y - uvw[idx[has], 2]             # relative to the stored (censored) mean

    # ---- Realisations
    keep = idx if uncertain_only else np.arange(N)
    base_attrs = {k: (v[keep] if uncertain_only else v) for k, v in pc.attrs.items() if k not in (label, "dw")}
    uvw_u = uvw[idx]
    out: List[PointCloud] = []
    for r in range(n_realisations):
        dw = DW[r]
        k = (Q[r][:, None] < cdf).argmax(axis=1)          # first class with CDF > q

        moved = uvw_u.copy()
        moved[:, 2] += dw
        xyz = pc.points.copy()
        xyz[idx] = frame.uvw_to_xyz(moved)
        lab = labels.copy()
        lab[idx] = classes[k].astype(labels.dtype, copy=False)
        dw_all = np.zeros(N)
        dw_all[idx] = dw

        real = PointCloud(points=xyz[keep])
        for name, arr in base_attrs.items():
            real._attrs[name] = arr                        # shared, not copied
        real.set_attr(label, lab[keep])
        real.set_attr("dw", dw_all[keep])
        real._frame = frame
        out.append(real)
    return out


def _point_spacing(uv, n: int = 5000) -> Optional[float]:
    """Median nearest-neighbour distance of 2D points (on a subset of `n`)."""
    uv = np.asarray(uv, dtype=float)
    if uv.shape[0] < 2:
        return None
    sub = uv if uv.shape[0] <= n else uv[np.random.default_rng(0).choice(uv.shape[0], n, replace=False)]
    d, _ = cKDTree(uv).query(sub, k=2)
    s = float(np.median(d[:, 1]))
    return s if s > 0 else None


def slopeBelow(slope: Slope, seeder: Seeder) -> Slope:
    """
    Create a slope that extends below the seeder position down to the base of the slope.

    Parameters
    ----------
    slope : Slope
        The original slope.
    seeder : Seeder
        The seeder with the position.

    Returns
    -------
    Slope
        The modified slope extending below the seeder.
    """
    if not isinstance(seeder, Seeder):
        raise TypeError("seeder must be an instance of Seeder.")
    if seeder.points.shape[1] > 1:
        raise ValueError("Must be a point seeder.")
    if slope.nodes.shape[1] > 2:
        raise ValueError("Slope must be 2D.")
    if np.all(seeder.points[-1] <= slope.nodes[:, -1]):
        return slope  # Seeder is already below the slope

    # Given seeder's y find x in the profile
    y = seeder.points[1]
    
    for e, elem in enumerate(slope.elements):
        i, j = elem
        pi = slope.nodes[i]
        pj = slope.nodes[j]
        if pj[1] <= y <= pi[1]:
            t = (y - pi[1]) / (pj[1] - pi[1])
            x = (pj[0] - pi[0]) * t + pi[0]
            break
    else:
        return slope  # No intersection found, return original slope
    e_initial = e
    new_nodes = [np.array((x, y)).reshape(-1)]
    if slope.hasUncertainty:
        new_nodes_std = [slope.nodes_std[slope.elements[e_initial][0]]]
    for e in range(e_initial, len(slope.elements)):
        new_nodes.append(slope.nodes[slope.elements[e][1]])
        if slope.hasUncertainty:
            new_nodes_std.append(slope.nodes_std[slope.elements[e][1]])

    # Create new materials
    new_materials = slope.materialIDs[e_initial:]

    return Slope(
        Geometry(
            nodes=np.array(new_nodes),
            nodes_std=np.array(new_nodes_std) if slope.hasUncertainty else None,
        ),
        materials=slope.materialTable,
        materialIDs=np.array(new_materials, dtype=int),
    )

def materialLayers(slope: Slope) -> Tuple[List[Material], np.ndarray]:
    """
    Get the list of material layers and their corresponding heights in the slope.
    Parameters
    ----------
    slope : Slope
        The slope to analyze.
    Returns
    -------
    Tuple[List[Material], np.ndarray]
        - List of materials in the order they appear from bottom to top.
        - Corresponding heights (y-coordinates) where each material layer starts.
    """
    material = []
    heights = []
    cur_mat = -1
    for e in range(len(slope.elements)):
        if slope.materialIDs[e] != cur_mat:
            heights.append(slope.nodes[slope.elements[e][0], 1])
            material.append(slope.materialTable[slope.materialIDs[e]])
            cur_mat = slope.materialIDs[e]
    return material, np.array(heights)

def interpPercentiles(
    data: NDArray[np.float64],
    percentiles: NDArray[np.float64],
    ranks: Optional[NDArray[np.float64]] = None,
) -> NDArray[np.float64]:
    """
    Calculate in each percentile each value of data is in.

    Parameters
    ----------
    data : (K,) array of float
        The data find the percentiles they belong.
    percentiles : (N,) array of float
        The desired percentile levels (in percent, within [0, 100]).
    ranks : (N,) array of float, optional
        The percentile levels (in percent, within [0, 100]) corresponding to the
        values in `percentiles`. If not given, defaults to `np.linspace(0, 100, N)`.
        (Used for validation only; no interpolation is performed here.)
    """
    N = percentiles.shape[0]
    if ranks is None:
        ranks = np.linspace(0, 100, N)
    else:
        ranks = np.asarray(ranks, dtype=float)
        if ranks.shape != (N,):
            raise ValueError(f"`ranks` must have shape {(N,)}, got {ranks.shape}.")
    # Interpolate each component CDF onto the common grid
    return np.interp(data, percentiles, ranks, left=ranks.min(), right=ranks.max())


def combinePercentiles(
    percentiles: NDArray[np.float64],
    ranks: Optional[NDArray[np.float64]] = None,
    likelihood: Optional[NDArray[np.float64]] = None,
) -> NDArray[np.float64]:
    """
    Combine percentiles across N variables at matched ranks using per-variable weights.

    Parameters
    ----------
    percentiles : (N, M) array of float
        Percentile *values* for N variables evaluated at M percentile levels (ranks).
        Row i contains the M percentile values for variable i; column j aligns the
        j-th rank across variables.
    ranks : (M,) array of float, optional
        The percentile levels (in percent, within [0, 100]) corresponding to the
        columns of `percentiles`. If not given, defaults to `np.linspace(0, 100, M)`.
        (Used for validation only; no interpolation is performed here.)
    likelihood : (N,) array of float, optional
        Per-variable nonnegative weights. If not given, uses uniform weights
        `p = np.ones(N) / N`. If an array is provided, it will be normalized to sum to 1.

    Returns
    -------
    combined : (M,) array of float
        The weighted combination of percentiles at each rank:
        `combined[j] = sum_i w[i] * percentiles[i, j]`.

    Notes
    -----
    - This function assumes the j-th column of `percentiles` represents the *same*
      percentile level for all variables (i.e., columns are aligned by rank).
    - If you work with quantile levels in [0, 1], convert upstream (e.g., with
      `np.quantile`) and still pass `ranks` only for shape/monotonicity checks.
    - `ranks` are not numerically used unless you extend this function to interpolate.
    """
    N, M = percentiles.shape
    if ranks is None:
        ranks = np.linspace(0, 100, M)
    else:
        ranks = np.asarray(ranks, dtype=float)
        if ranks.shape != (M,):
            raise ValueError(f"`ranks` must have shape {(M,)}, got {ranks.shape}.")
        
    # Weights (normalise so CDF remains in [0,1])
    if likelihood is None:
        w = np.full(N, 1.0 / N, dtype=float)
    else:
        w = np.asarray(likelihood, dtype=float)
        if w.shape != (N,):
            raise ValueError(f"`likelihood` must have shape {(N,)}, got {w.shape}.")
        if np.any(w < 0):
            raise ValueError("`likelihood` must be nonnegative.")
        s = w.sum()
        if s > 0:
            w = w / s
        else:
            return np.full_like(ranks, np.nan, dtype=float)

    # Common support grid over all component percentiles        
    percentiles_grid = np.unique(percentiles.ravel())

    # Interpolate each component CDF onto the common grid
    # Fi[i, :] = F_i(percentiles_grid)
    Fi = np.array([
        np.interp(percentiles_grid, percentiles[i], ranks, left=0.0, right=100.0)
        for i in range(N)
    ])

    # Weighted sum (equivalent to weighted mean because w sums to 1)
    F_mix = (w[:, None] * Fi).sum(axis=0)

    # Numerical safety: ensure non-decreasing and within [0, 1]
    F_mix = np.maximum.accumulate(np.clip(F_mix, 0.0, 100.0))

    # Deduplicate before inverting: whenever combined rows have different
    # value ranges, extrapolation (left=0/right=100 above) creates flat
    # plateaus in F_mix, i.e. repeated values. np.interp resolves ties on a
    # non-strictly-increasing xp by the last match, which is only correct
    # for the F=100 plateau (there, the true end of support is the
    # smallest x reaching 100 -- so we want the *first* occurrence). At the
    # F=0 plateau it's the opposite: the true start of support is the
    # *largest* x still at F=0 (the smallest x is just the extrapolated
    # tail below every component's own support).
    F_unique, first_idx = np.unique(F_mix, return_index=True)
    if F_unique[0] == 0.0:
        first_idx[0] = np.flatnonzero(F_mix == 0.0)[-1]
    x_unique = percentiles_grid[first_idx]

    # Invert to get mixture quantiles at requested ranks
    q_mix = np.interp(ranks, F_unique, x_unique)
    return q_mix

def _is_categorical(arr: np.ndarray) -> bool:
    # Heuristic: strings, bytes, or small int enums treated as categorical
    if arr.dtype.kind in ("U", "S", "O"):
        return True
    if arr.dtype.kind in ("i", "u"):
        # If few unique values relative to length, treat as categorical
        # (You can replace with explicit schema if you have one)
        uniq = np.unique(arr)
        return len(uniq) <= max(32, int(0.01 * arr.size))
    return False

def interpAttributes(
    source: "Model",
    destination: "Model",
    attributes: List[str],
    *,
    method: Literal["nearest", "knn", "radius"] = "knn",
    **kwargs: Any,
) -> None:
    """
    Interpolate attributes from `source` to `destination` points in-place.

    Parameters
    ----------
    source, destination : Geometry
        Must expose `.points: (N,3)` and `.get_attr(name)`, `.set_attr(name, values)`.
    attributes : list of str
        Attribute names to transfer. Each may be shape (N,), (N,D) with D>1.
    method : {"nearest","knn","radius"}, default "knn"
        - "nearest": k=1
        - "knn": use k nearest neighbors (parameters['n_neighbors'] required; default 8)
        - "radius": use neighbors within radius (parameters['radius'] required; optional 'max_k')
    parameters : dict
        Method-specific parameters, e.g. {'n_neighbors': 8}, {'radius': 2.0, 'max_k': 64},
        IDW power via {'power': 2.0}, Gaussian sigma via {'sigma': 1.0}.
    weighting : {"none","idw","gaussian"}, default "idw"
        Aggregation weights for continuous attributes. Categorical uses (weighted) mode.
    """
    src_pts = np.asarray(source.points, float)
    dst_pts = np.asarray(destination.points, float)

    if src_pts.ndim != 2 or dst_pts.ndim != 2 or src_pts.shape[1] != dst_pts.shape[1]:
        raise ValueError("source.points and destination.points must be (N,D) with same D.")
    
    categorical_vals = []
    continuous_vals = []

    for name in attributes:
        vals = np.asarray(source.get_attr(name))
        if vals.shape[0] != src_pts.shape[0]:
            raise ValueError(f"Attribute '{name}' length mismatch with source points.")
        categorical = _is_categorical(vals if vals.ndim == 1 else vals[:, 0])
        if categorical:
            categorical_vals.append(vals)
        else:
            continuous_vals.append(vals)
    continuous_vals = np.array(continuous_vals).transpose() if continuous_vals else None
    categorical_vals = np.array(categorical_vals).transpose() if categorical_vals else None
    if continuous_vals is None and categorical_vals is None:
        return  # Nothing to do
    if continuous_vals is not None:
        if method == "nearest":
            from sklearn.neighbors import KNeighborsRegressor as knn
            knn_model = knn(n_neighbors=1)
        elif method == "knn":
            from sklearn.neighbors import KNeighborsRegressor as knn
            knn_model = knn(**kwargs)
        elif method == "radius":
            from sklearn.neighbors import RadiusNeighborsRegressor as knn
            knn_model = knn(**kwargs)
        else:
            raise ValueError(f"Unknown method '{method}'")
        knn_model.fit(src_pts, continuous_vals)
        continuous_interp = knn_model.predict(dst_pts)
        if continuous_interp.ndim == 1:
            # sklearn squeezes predict() to 1D for a single-column target.
            continuous_interp = continuous_interp.reshape(-1, 1)
    if categorical_vals is not None:
        if method == "nearest":
            from sklearn.neighbors import KNeighborsClassifier as knn
            knn_model = knn(n_neighbors=1)
        elif method == "knn":
            from sklearn.neighbors import KNeighborsClassifier as knn
            knn_model = knn(**kwargs)
        elif method == "radius":
            from sklearn.neighbors import RadiusNeighborsClassifier as knn
            knn_model = knn(**kwargs)
        else:
            raise ValueError(f"Unknown method '{method}'")
        knn_model.fit(src_pts, categorical_vals)
        categorical_interp = knn_model.predict(dst_pts)
        if categorical_interp.ndim == 1:
            # sklearn squeezes predict() to 1D for a single-column target.
            categorical_interp = categorical_interp.reshape(-1, 1)

    count_continuous = 0
    count_categorical = 0
    for name in attributes:
        vals = np.asarray(source.get_attr(name))
        if _is_categorical(vals if vals.ndim == 1 else vals[:, 0]):
            destination.set_attr(name, categorical_interp[:, count_categorical])
            count_categorical += 1
        else:
            destination.set_attr(name, continuous_interp[:, count_continuous])
            count_continuous += 1

def aggregateOnFloor(
    pointsOnWall: Model,
    profile_name: str,
    attributes: List[str],
    percentiles: NDArray[np.float64],
    likelihood: Optional[Callable[[NDArray[np.float64]], NDArray[np.float64]]] = None,
) -> Dict[str, NDArray[np.float64]]:
    """
    Aggregate profile attributes from wall points onto corresponding floor points.

    For each point along the floor (e.g. toe of the wall), this function identifies
    the set of wall points located directly above it (the profile). The specified
    attributes are then aggregated across the profile according to the requested
    percentiles. The result is a new set of attributes defined at the floor points,
    with names following the convention ``<attribute>_<percentile>``. For example,
    providing ``attributes=['E1', 'df']`` and ``percentiles=[5, 50, 95]`` will create
    attributes ``'E1_5'``, ``'E1_50'``, ``'E1_95'``, ``'df_5'``, ``'df_50'`` and
    ``'df_95'``.

    The aggregation accounts for the statistical likelihood of each wall point
    contributing to the profile. If ``likelihood`` is provided, it should be a
    callable that returns a vector of likelihood weights for the set of wall points
    above a given floor point. If ``likelihood`` is ``None``, equal likelihood is
    assumed, corresponding to a uniform weighting
    ``np.full(n_points, 1.0 / n_points)`` for each profile.

    Parameters
    ----------
    pointsOnWall : Geometry
        Geometry object containing wall points where the original attributes are
        defined. Must support ``.get_attr(name)`` to access attributes.
    attributes : List[str]
        Base names of the attributes to aggregate (e.g. ``['E1', 'df']``). For each
        base name and each percentile, a new attribute will be created on the floor.
    percentiles : np.ndarray of float
        Percentile ranks to evaluate (e.g. ``[5, 10, ..., 95]``). Must be within the
        range ``[0, 100]``.
    likelihood : callable or None, optional
        Function returning a likelihood vector for the wall points above a given
        floor point. If ``None``, equal likelihood are assumed. This allows
        different profiles with varying numbers of points to be handled consistently.

    Notes
    -----
    - The function returns a dictionary of the aggregated attributes.
    - The likelihood weighting enables non-uniform treatment of wall points, which
      may be useful if probabilities of occurrence vary along the profile.
    - Percentile aggregation is performed independently for each attribute.
    """
    # Store original orientation for rotating back later
    dip_dir = pointsOnWall.dipDirection()
    centroid = pointsOnWall.centroid()

    # Rotate wall and floor to align wall with x axis
    pointsOnWall.translate(-centroid)
    pointsOnWall.rotate(dip_dir, axis='z')
        
    # Extract attributes from wall
    wall_attrs = dict()
    floor_attrs = dict()
    for name in attributes:
        wall_attrs[name] = []
        for percentile in percentiles:
            name_pct = f"{name}_{int(percentile)}"
            floor_attrs[name_pct] = []
            wall_attrs[name].append(np.asarray(pointsOnWall.get_attr(name_pct)))
        wall_attrs[name] = np.array(wall_attrs[name]).transpose()  # shape (num_points, len(percentiles))

    for prof_id in np.unique(pointsOnWall.get_attr(profile_name)):
        # Find wall points in profile
        mask = pointsOnWall.get_attr(profile_name) == prof_id
        wall_pts = pointsOnWall.points[mask]
        if wall_pts.size == 0:
            for name in attributes:
                for percentile in percentiles:
                    name_pct = f"{name}_{int(percentile)}"
                    floor_attrs[name_pct].append(np.nan)
            continue  # No wall points above this floor point
        n_wall = wall_pts.shape[0]

        # Get likelihood for these wall points
        if likelihood is None:
            w = np.full(n_wall, 1.0 / n_wall, dtype=float)
        else:
            wall_pc = PointCloud(wall_pts)
            wall_pc.rotate(-dip_dir, axis='z')
            wall_pc.translate(centroid)
            w = likelihood(wall_pc.points)
            if w.shape != (n_wall,):
                raise ValueError(f"likelihood function must return shape {(n_wall,)}, got {w.shape}.")
            if np.any(w < 0):
                raise ValueError("likelihood function must return nonnegative values.")
            s = w.sum()
            if s > 0:
                w /= s  # normalize to sum to 1

        # Extract wall attributes for these points
        profile_attrs = {name: wall_attrs[name][mask] for name in attributes}

        # Aggregate each attribute at requested percentiles
        for name, vals in profile_attrs.items():
            floor_attr = combinePercentiles(vals, percentiles, likelihood=w)
            # Store on floor with new name
            for p, v in zip(percentiles, floor_attr):
                name_pct = f"{name}_{int(p)}"
                floor_attrs[name_pct].append(v)

    outputs = dict()
    for name_pct, vals in floor_attrs.items():
        outputs[name_pct] = np.asarray(vals)

    # Restore original orientation
    pointsOnWall.rotate(-dip_dir, axis='z')
    pointsOnWall.translate(centroid)

    return outputs

def wallFloorElevation(
    model: Model,
    profile_spacing: float,
    *,
    label: Optional[str] = None,
    ignore_ids: List[int] = [],
    frame=None,
) -> float:
    """
    Common floor elevation for all profiles of a wall.

    The wall is cut into slabs of width `profile_spacing` along the strike (the
    ``s`` coordinate of the curvilinear frame). Slabs that do not span the full
    height of the wall are ignored; the floor is the highest of the lowest
    elevations of the remaining slabs, so every complete profile reaches it.

    Parameters
    ----------
    model : Model
        Wall geometry.
    profile_spacing : float
        Width of the slabs along the strike.
    label : str, optional
        Attribute holding material IDs. Needed only with `ignore_ids`.
    ignore_ids : list[int], optional
        Material IDs left out when looking for the floor (e.g. talus). Their
        elevation range sets the tolerance used to decide whether a slab is
        complete; without them the tolerance is `profile_spacing`.
    frame : CurvilinearFrame, optional
        Frame to use. Defaults to the frame of the model.

    Returns
    -------
    float
        Floor elevation.
    """
    frame = model.frame if frame is None else frame
    sde = frame.xyz_to_sde(model.points)
    ok = np.isfinite(sde).all(axis=1)
    s, e = sde[ok, 0], sde[ok, 2]

    delta = profile_spacing
    if label is not None and len(ignore_ids) > 0:
        mat = np.asarray(model.get_attr(label))
        if mat.ndim == 2 and mat.shape[1] == 1:
            mat = mat[:, 0]
        if mat.shape[0] != model.points.shape[0]:
            # per-triangle labels: give each point the label of a triangle around it
            tri = model.triangles
            point_mat = np.full(model.points.shape[0], -1, dtype=mat.dtype)
            point_mat[tri.ravel()] = np.repeat(mat, 3)
            mat = point_mat
        mat = mat[ok]
        for m_id in ignore_ids:
            ignored = mat == m_id
            if ignored.any():
                delta = float(e[ignored].max() - e[ignored].min())
            s, e, mat = s[~ignored], e[~ignored], mat[~ignored]
    if e.size == 0:
        raise ValueError("No points left to find the floor elevation.")

    e_max, e_min = float(e.max()), float(e.min())
    k = np.floor((s - s.min()) / profile_spacing).astype(np.int64)
    order = np.argsort(k, kind="stable")
    k_sorted, e_sorted = k[order], e[order]
    starts = np.r_[0, 1 + np.flatnonzero(np.diff(k_sorted))]
    top = np.maximum.reduceat(e_sorted, starts)
    bottom = np.minimum.reduceat(e_sorted, starts)
    complete = (top >= e_max - delta) & (bottom <= e_min + delta)  # ignore incomplete slabs
    if not complete.any():
        return e_min
    return float(max(e_min, bottom[complete].max()))


def runWallProfiles(
    model: Model,
    segment_length: float,
    profile_spacing: float,
    profile_resolution: float,
    material_label: str,
    materials: Dict[int, Material],
    runner: Callable[[Slope], Tuple[NDArray[np.float64], NDArray[np.float64]]],
    *,
    do_remove_talus: bool = True,
    results_labels: List[str] = [],
) -> Tuple[PointCloud, PointCloud]:
    """
    Generate wall and floor point clouds by slicing a geological model into profiles
    and running block-release analyses.

    This function extracts vertical profiles along a given slope (wall), removes talus if
    requested, and runs a user-specified analysis (`runner`) for each profile. The function
    consolidates the results into two point clouds:

    - **Wall point cloud**: points at block release locations with associated result attributes.
    - **Floor point cloud**: points at the floor with direction vectors, representing the
      projection of block trajectories to the base of the slope.

    Profiles are slices of constant ``s`` in the curvilinear frame of the model (see
    :meth:`Model.sde`), so they follow the strike of the wall around bends. The model
    itself is not modified.

    Args:
        model (Geometry):
            Geological wall geometry to be sliced into segments and profiles.
        segment_length (float):
            Length of wall segments (along strike) used for profile slicing.
        profile_spacing (float):
            Horizontal spacing between consecutive vertical profiles.
        profile_resolution (float):
            Vertical discretization step used when slicing profiles.
        material_label (str):
            Attribute name in the model that indicates material IDs.
        materials (Dict[int, Material]):
            Dictionary mapping material IDs to material definitions, including talus if present.
        runner (Callable[[Slope], Tuple[NDArray[np.float64], NDArray[np.float64]]]):
            Function that runs the block-release analysis for a single slope profile.
            Must return a tuple of:
            
            - positions (ndarray, shape (N, 2)): x,y positions of block release points.
            - results (ndarray, shape (N, M)): analysis results for each release point,
              with M result metrics.
        do_remove_talus (bool, optional):
            Whether to identify and remove talus layers from profiles. Defaults to True.
        results_labels (List[str], optional):
            Names of result metrics to be stored in the wall point cloud. If empty,
            generic labels (`result_0`, `result_1`, …) will be created automatically.

    Returns:
        Tuple[PointCloud, PointCloud]:
            - **pc_wall** (PointCloud): Point cloud of release points on the wall. Each
              point stores results from the analysis (`results_labels`).
            - **pc_floor** (PointCloud): Point cloud of points projected onto the floor
              with an attribute `"direction"` storing trajectory direction vectors.

    Raises:
        ValueError:
            If the output of `runner` does not match expected shapes:
            - positions not 2D (x,y),
            - number of results inconsistent across profiles,
            - mismatch in number of positions and results.

    Notes:
        - Profile coordinates are (d, e): distance along the dip direction and elevation.
        - Profiles containing incomplete or ambiguous talus sections are skipped.
        - The returned point clouds are in the model's own coordinates.

    Example:
        >>> pc_wall, pc_floor = runWallProfiles(
        ...     model=geom,
        ...     segment_length=5.0,
        ...     profile_spacing=2.0,
        ...     profile_resolution=0.5,
        ...     material_label="material",
        ...     materials=mat_list,
        ...     runner=my_runner,
        ...     do_remove_talus=True,
        ...     results_labels=["first_impact", "runout"]
        ... )
        >>> pc_wall.points.shape
        (1200, 3)
        >>> pc_wall.get_attr("first_impact").shape
        (1200,)
        >>> pc_floor.get_attr("direction").shape
        (100, 3)
    """
    # Get talus ID for removing talus from profiles
    talus_id = -1
    if do_remove_talus:
        for key, entry in materials.items():
            if 'talus' in entry.name.lower():
                talus_id = key
                break

    # Curvilinear frame: s along the strike, d down-dip, e elevation
    frame = model.frame

    # Create a common floor for all profiles by finding the minimum z value in the wall excluding talus
    min_z = wallFloorElevation(
        model, profile_spacing, label=material_label, ignore_ids=[talus_id], frame=frame
    )

    # Work on a copy without the points below the common floor
    keep = model.points[:, 2] >= min_z
    if isinstance(model, Mesh):
        wall = Mesh(points=model.points.copy(), triangles=model.triangles.copy(), attrs=dict(model.attrs))
    else:
        wall = PointCloud(points=model.points.copy(), attrs=dict(model.attrs))
    wall.frame = frame
    wall.apply_point_mask_(keep)

    # Store positions from where blocks are released, as (s, d, e)
    X_wall = []
    Y_wall = []
    Z_wall = []

    # Store positions on the floor, as (s, d)
    X_floor = []
    Y_floor = []

    # Store results on the wall
    results_wall = []
    num_results = -1 if len(results_labels) == 0 else len(results_labels)

    for seg in wall.segment(segment_length):
        # Extracts profiles from the wall segment: slices of constant s, nodes in (d, e)
        profiles, x_profiles = seg.slice(
            None,
            profile_spacing,
            slice_height_increment=profile_resolution,
            label=material_label,
            frame=frame,
        )

        # Store positions from where blocks are released
        X_seg = []
        Y_seg = []
        Z_seg = []

        # Store positions on the floor
        X_floor_seg = []
        Y_floor_seg = []

        # Loop over profiles
        for prof, x_prof in zip(profiles, x_profiles):
            # Find beginning of the floor
            prof.nodes[-1][1] = min_z  # Ensure profile ends at the floor

            # Find mean segment length in the profile for tolerance
            len_seg_mean = 0.0
            for i in range(len(prof.nodes)-1):
                xi, yi = prof.nodes[i]
                xj, yj = prof.nodes[i+1]
                len_seg_mean += np.sqrt((xj-xi)**2 + (yj-yi)**2)
            len_seg_mean /= len(prof.nodes)-1

            materialIDs = prof.attributes
            if materialIDs is not None:
                mat_list = [materials[m] for m in np.unique(materialIDs)]
            else:
                raise ValueError('No materials in profile!')
            slope = Slope(
                prof,
                materialIDs=materialIDs,
                materials=mat_list
            )

            if do_remove_talus:
                # Get talus profiles
                talus_list = singleMaterialSlope(slope, talus_id)
                
                # Check if talus is only on top or bottom of the profile
                do_skip_profile = False
                for talus in talus_list:
                    talus_prof = talus.nodes
                    max_y_talus = talus_prof[:,1].max()
                    min_y_talus = talus_prof[:,1].min()
                    max_y_profile = prof.nodes[:,1].max()
                    min_y_profile = prof.nodes[:,1].min()
                    do_remove_talus_prof = 0  # 0: do not remove talus, 1: remove talus from top, 2: remove talus from bottom
                    if abs(max_y_talus - max_y_profile) < len_seg_mean/2:
                        do_remove_talus_prof = 1
                    if abs(min_y_talus - min_y_profile) < len_seg_mean/2:
                        do_remove_talus_prof = 2
                    if do_remove_talus_prof == 0:
                        do_skip_profile = True
                        break
                if do_skip_profile:
                    continue
                try:
                    # Remove talus from profile
                    slope = removeMaterial(slope, talus_id)
                except ValueError:
                    continue  # Skip profile if talus removal fails (potentially talus-only section)

            # Centre profile's toe at origin
            x_centre = slope.nodes[-1, 0]
            y_centre = slope.nodes[-1, 1]
            slope.nodes[:, 0] -= x_centre
            slope.nodes[:, 1] -= y_centre

            # Run analysis
            positions_prof, results_prof = runner(slope)
            if positions_prof.shape[1] != 2:
                raise ValueError('Positions from runner should be 2D (x,y)')
            if positions_prof.shape[0] != results_prof.shape[0]:
                raise ValueError('Results from runner should be per seeder')
            if num_results > 0 and results_prof.shape[1] != num_results:
                raise ValueError(f'Number of results from runner ({results_prof.shape[1]}) does not match the previous number of results ({num_results})')
            num_results = results_prof.shape[1]
            for i in range(len(results_labels), num_results):
                results_labels.append(f"result_{i}")

            # Filter out points with NaN values in any of the results
            mask = ~np.isnan(results_prof).any(axis=1)
            if not np.any(mask):
                continue  # Skip profile if all points are invalid

            # Store profile results on wall
            results_wall.append(results_prof[mask])

            # Store positions from where blocks are released to segment
            X_seg.append(np.full(len(positions_prof[mask]), x_prof))
            Y_seg.append(positions_prof[mask,0] + x_centre)
            Z_seg.append(positions_prof[mask,1] + y_centre)

            # Store position on the floor from this profile to the segment
            X_floor_seg.append(x_prof)
            Y_floor_seg.append(x_centre)

        if len(X_seg) == 0:
            continue  # No results obtained in this segment

        X_floor.append(np.asarray(X_floor_seg, float))
        Y_floor.append(np.asarray(Y_floor_seg, float))

        X_wall.append(np.concatenate(X_seg))
        Y_wall.append(np.concatenate(Y_seg))
        Z_wall.append(np.concatenate(Z_seg))
    
    if len(X_wall) == 0:
        raise ValueError("No results obtained from any profile.")
        
    # Consolidate arrays: release points in (s, d, e)
    sde_wall = np.column_stack([np.concatenate(X_wall), np.concatenate(Y_wall), np.concatenate(Z_wall)])
    results_wall = np.concatenate(results_wall)

    # Points on the floor, at the toe of each profile
    s_floor = np.concatenate(X_floor)
    d_floor = np.concatenate(Y_floor)
    sde_floor = np.column_stack([s_floor, d_floor, np.full_like(s_floor, min_z)])

    # Back to the model's coordinates through the frame
    pc_wall = PointCloud(frame.sde_to_xyz(sde_wall))
    for i in range(num_results):
        pc_wall.set_attr(results_labels[i], results_wall[:,i])

    points = frame.sde_to_xyz(sde_floor)
    pc_floor = PointCloud(points)

    # Direction on the floor: horizontal, away from the wall (increasing d)
    ahead = frame.sde_to_xyz(sde_floor + np.array([0.0, 1.0, 0.0]))
    direction = ahead - points
    direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-12)
    pc_floor.set_attr('direction', direction)

    return pc_wall, pc_floor


def rocksSeeders(seeders: List[Seeder]) -> List[Rock]:
    rock_types: List[Rock] = []
    for seeder in seeders:
        for rock_type in seeder.rocks:
            if rock_type not in rock_types:
                rock_types.append(rock_type)
    return rock_types


def slopeFeatures(slope: Slope) -> Tuple[float, float, float]:
    nodes = np.array(slope.nodes, dtype=float)  # Ensure copy
    height = nodes[:, 1].max() - nodes[:, 1].min()
    d = np.diff(nodes, axis=0)
    nodes -= nodes.mean(axis=0)
    _, _, Vt = np.linalg.svd(nodes, full_matrices=False)
    v = Vt[0]  # unit direction along best-fit line (vx, vy)
    vy = nodes[-1, 1] - nodes[0, 1]
    if v[1] * vy < 0:
        v *= -1

    slope_angle = -np.degrees(np.arctan2(v[1], v[0]))

    local_slope = -np.degrees(np.arctan2(d[:, 1], d[:, 0]))
    local_length = np.hypot(d[:, 0], d[:, 1])

    roughness = np.sqrt(sum(local_length * (local_slope - slope_angle) ** 2) / sum(local_length))

    return height, slope_angle, roughness

