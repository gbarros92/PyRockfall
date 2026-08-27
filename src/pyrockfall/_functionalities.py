"""
User functionalities in pyrockfall
==================================

This module implements functions to facilitate rockfall simulation.
"""
import numpy as np
from numpy.typing import NDArray
from typing import Union, List, Tuple, Optional, Callable, Literal, Dict, Any

from ._geometry import Geometry
from ._material import Material
from ._slope import Slope
from ._rock import Rock
from ._seeder import Seeder
from ._model import Model
from ._pointcloud import PointCloud

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

    The geometry is temporarily rotated so the wall aligns with the X axis for processing,
    and then rotated back to its original orientation before returning.

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
        - The function aligns the slope with the X axis for slicing and re-aligns it back
          after processing.
        - Profiles containing incomplete or ambiguous talus sections are skipped.
        - The returned point clouds are rotated back to the model's original orientation.

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
    # Store original orientation for rotating back later
    dip_dir = model.dipDirection()
    centroid = model.centroid()

    # Get talus ID for removing talus from profiles
    talus_id = -1
    if do_remove_talus:
        for key, entry in materials.items():
            if 'talus' in entry.name.lower():
                talus_id = key
                break

    # Align wall with x axis
    model.alignWithX(sense=-1)

    # Create a common floor for all profiles by finding the minimum z value in the wall excluding talus
    pmin, pmax = model.boundingBox()
    points = model.points.copy()
    points_talus = points[model.get_attr(material_label) == talus_id]
    points = points[model.get_attr(material_label) != talus_id]
    if points_talus.size == 0:
        delta_remove = profile_spacing
    else:
        delta_remove = points_talus[:,2].max() - points_talus[:,2].min()
    max_z_pts = points[:,2].max()
    min_z_pts = points[:,2].min()
    min_z = min_z_pts
    for x in np.arange(pmin[0]+profile_spacing/2, pmax[0], profile_spacing):
        mask = (points[:,0] >= x-profile_spacing/2) & (points[:,0] < x+profile_spacing/2)
        points_in_slice = points[mask]
        if points_in_slice.size == 0:
            continue
        max_z_slice = points_in_slice[:,2].max()
        if max_z_slice < max_z_pts - delta_remove:
            continue  # Ignore incomplete slices
        min_z_slice = points_in_slice[:,2].min()
        if min_z_slice > min_z_pts + delta_remove:
            continue  # Ignore incomplete slices
        min_z = max(min_z, min_z_slice)
    pmin[2] = min_z  # Set minimum z value to the lowest point in
    model.clip(pmin, pmax)  # Clip wall to remove points below the minimum z value

    # Re-align wall with x axis after clipping
    dip_dir += model.dipDirection()
    dip_dir = dip_dir % 360
    model.alignWithX(sense=-1)

    # Store positions from where blocks are released
    X_wall = []
    Y_wall = []
    Z_wall = []

    # Store positions on the floor
    X_floor = []
    Y_floor = []
    dX_floor = []
    dY_floor = []

    # Store results on the wall
    results_wall = []
    num_results = -1 if len(results_labels) == 0 else len(results_labels)

    for seg in model.segment(segment_length):
        # Store original orientation for rotating back later
        dip_dir_seg = seg.dipDirection()
        seg.alignWithX(sense=-1)

        # Extracts profiles from the wall segment
        profiles, x_profiles = seg.slice(
            np.array([1.0, 0.0, 0.0]),
            profile_spacing,
            slice_height_increment=profile_resolution,
            label=material_label
        )

        # Store positions from where blocks are released
        X_seg = []
        Y_seg = []
        Z_seg = []

        # Store positions on the floor
        X_floor_seg = []
        Y_floor_seg = []
        dX_floor_seg = []
        dY_floor_seg = []

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

            # Store position and direction on the floor from this profile to the segment
            X_floor_seg.append(x_prof)
            Y_floor_seg.append(x_centre)
            dX_floor_seg.append(0)
            dY_floor_seg.append(1)

        # Rotate segment back to original orientation
        def rotateSegBack(x, y, centre=(0,0)):
            x = np.asarray(x, float) - centre[0]
            y = np.asarray(y, float) - centre[1]
            c, s = np.cos(np.radians(-dip_dir_seg)), np.sin(np.radians(-dip_dir_seg))
            x_rot =  c * x - s * y + centre[0]
            y_rot =  s * x + c * y + centre[1]
            return x_rot, y_rot
        
        if len(X_seg) == 0:
            continue  # No results obtained in this segment

        X_seg = np.concatenate(X_seg)
        Y_seg = np.concatenate(Y_seg)
        Z_seg = np.concatenate(Z_seg)
        X_seg, Y_seg = rotateSegBack(X_seg, Y_seg, seg.centroid())
        dX_floor_seg, dY_floor_seg = rotateSegBack(dX_floor_seg, dY_floor_seg)

        X_floor.append(np.asarray(X_floor_seg))
        Y_floor.append(np.asarray(Y_floor_seg))
        dX_floor.append(dX_floor_seg)
        dY_floor.append(dY_floor_seg)

        X_wall.append(X_seg)
        Y_wall.append(Y_seg)
        Z_wall.append(Z_seg)
    
    if len(X_wall) == 0:
        raise ValueError("No results obtained from any profile.")
        
    # Consolidate arrays
    X_wall = np.concatenate(X_wall)
    Y_wall = np.concatenate(Y_wall)
    Z_wall = np.concatenate(Z_wall)
    results_wall = np.concatenate(results_wall)

    # Create grid on the floor
    X_floor = np.concatenate(X_floor)
    Y_floor = np.concatenate(Y_floor)
    dX_floor = np.concatenate(dX_floor)
    dY_floor = np.concatenate(dY_floor)

    pc_wall = PointCloud(np.column_stack([X_wall, Y_wall, Z_wall]))
    for i in range(num_results):
        pc_wall.set_attr(results_labels[i], results_wall[:,i])

    points = np.column_stack([X_floor, Y_floor, np.full_like(X_floor, min_z)])
    pc_floor = PointCloud(points)
    pc_floor.set_attr('direction', np.column_stack([dX_floor, dY_floor, np.zeros_like(dX_floor)]))

    # Rotate wall back to original orientation
    model.translate(-centroid)
    model.rotate(-dip_dir, 'z')
    model.translate(centroid)

    # Rotate points on wall to original orientation
    pc_wall.translate(-centroid)
    pc_wall.rotate(-dip_dir, 'z')
    pc_wall.translate(centroid)

    # Rotate points on floor back to original orientation
    pc_floor.translate(-centroid)
    pc_floor.rotate(-dip_dir, 'z')
    pc_floor.translate(centroid)
    
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

