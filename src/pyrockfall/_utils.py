"""
Utility functions for geometric and mesh operations in model3d.

Includes helpers for angles, mesh processing, and PLY scalar field reading.
"""

import struct
import math as m
import numpy as np
from scipy.linalg import lstsq
from scipy.spatial import cKDTree

from typing import Tuple, List, Callable
from numpy.typing import NDArray

def rot2AlignWithX(direction: np.ndarray) -> np.ndarray:
    """Return a 3x3 rotation matrix R such that v_rot = v @ R aligns `direction` to +X."""
    d = np.asarray(direction, dtype=float)
    r = np.linalg.norm(d)
    if r == 0:
        raise ValueError("direction must be non-zero")
    d = d / r
    # angles as in your code
    phi = np.arccos(np.clip(d[2], -1.0, 1.0))
    theta = np.arctan2(d[1], d[0])

    c, s = np.cos(-theta), np.sin(-theta)
    Rz = np.array([[c, -s, 0.0],
                [s,  c, 0.0],
                [0.0, 0.0, 1.0]])

    b = np.pi * 0.5 - phi
    c, s = np.cos(b), np.sin(b)
    Ry = np.array([[ c, 0.0, -s],
                [0.0, 1.0,  0.0],
                [ s, 0.0,  c]])
    # Apply z-then-y as in your original sequence: v' = v @ (Rz @ Ry)
    return Rz @ Ry


def angleBetweenVectors(cc: np.ndarray, cn: np.ndarray) -> float:
    """
    Calculate the angle between two vectors.

    Args:
        cc (numpy.ndarray): The first vector.
        cn (numpy.ndarray): The second vector.

    Returns:
        float: The angle between the two vectors in degrees.
    """
    nvp = (cn.dot(cc))/(np.linalg.norm(cn)*np.linalg.norm(cc))
    if nvp > 1:
        nvp = 1
    elif nvp < -1:
        nvp = -1
    angle_s = m.degrees(m.acos(nvp))      
    return angle_s


def triangleCentroids(coords, connect):
    """
    Compute centroids for triangles defined by connectivity.

    Args:
        coords (np.ndarray): Vertex coordinates.
        connect (np.ndarray): Connectivity indices.

    Returns:
        np.ndarray: Centroids of each triangle.
    """
    return np.mean(coords[connect], axis=1)


def getSubMesh(V: np.ndarray, T: np.ndarray, ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract a sub-mesh by keeping only the given triangles, dropping any
    vertex not referenced by them and remapping connectivity accordingly.

    Args:
        V: Vertex coordinates, shape (N, D).
        T: Triangle connectivity, shape (E, 3) (0-based indices into V).
        ids: Indices of triangles (rows of T) to keep.

    Returns:
        V_sub: Reindexed vertex coordinates, shape (N_sub, D).
        T_sub: Reindexed triangle connectivity, shape (len(ids), 3), indices
            into V_sub.
        vertex_ids: Original vertex indices kept, shape (N_sub,), such that
            ``V_sub == V[vertex_ids]``.
    """
    V = np.asarray(V)
    T = np.asarray(T)
    ids = np.asarray(ids, dtype=np.int64)

    T_kept = T[ids]
    vertex_ids, inverse = np.unique(T_kept, return_inverse=True)
    V_sub = V[vertex_ids]
    T_sub = inverse.reshape(T_kept.shape).astype(T.dtype)

    return V_sub, T_sub, vertex_ids


def build_neighbours_mesh(triangles: np.ndarray) -> np.ndarray:
    """
    Build triangle adjacency for a triangle mesh.

    Parameters
    ----------
    triangles : (E, 3) int array (0-based)
        Each row is a face with three vertex indices.

    Returns
    -------
    neighbours : (E, 3) int32
        For each face, neighbour across the edge opposite local vertex 0, 1, 2.
        -1 indicates a boundary (no neighbouring face).
    """
    tri = np.asarray(triangles, dtype=np.int64)
    if tri.ndim != 2 or tri.shape[1] != 3:
        raise ValueError("triangles must be (M,3)")
    M = tri.shape[0]
    N = np.int64(int(tri.max()) + 1)

    # Directed edges in local order: (v0->v1), (v1->v2), (v2->v0)
    u = tri[:, [0, 1, 2]].reshape(-1)          # (E,)
    v = tri[:, [1, 2, 0]].reshape(-1)          # (E,)
    E = u.size
    owner = np.repeat(np.arange(M, dtype=np.int64), 3)  # triangle per edge

    # Directed keys and reversed keys
    key_fwd = u * N + v
    key_rev = v * N + u

    # Sort forward keys once, then lookup reversed keys via searchsorted
    order = np.argsort(key_fwd)
    key_sorted = key_fwd[order]
    pos = np.searchsorted(key_sorted, key_rev, side="left")

    # Only index key_sorted where pos is in-bounds
    inbounds = pos < E
    match = np.zeros_like(pos, dtype=bool)
    match[inbounds] = (key_sorted[pos[inbounds]] == key_rev[inbounds])

    # Triangle that owns the matching reversed edge
    partner_owner = np.full(E, -1, dtype=np.int64)
    # Only read from 'order' and 'pos' where we actually have a match
    hit = inbounds & match
    partner_owner[hit] = owner[order[pos[hit]]]

    # Drop self-matches (degenerate tri like [0,0,0] yields self edges)
    valid = hit & (partner_owner != owner)

    # Fill neighbours (flattened edge order mirrors u/v construction)
    neigh_flat = np.full(E, -1, dtype=np.int64)
    neigh_flat[valid] = partner_owner[valid]
    neighbours = neigh_flat.reshape(M, 3).astype(np.int32, copy=False)
    return neighbours


def build_neighbours_polygon(elements: np.ndarray) -> np.ndarray:
    """
    Build segment adjacency for a polygonal chain / polygon.

    Parameters
    ----------
    elements : (E, 2) int array
        Each row is a segment with two vertex indices.

    Returns
    -------
    neighbours : (E, 2) int32
        neighbours[i, 0] = segment adjacent through elements[i, 0]
        neighbours[i, 1] = segment adjacent through elements[i, 1]
        -1 indicates a boundary node.

    Notes
    -----
    This assumes a manifold 1D topology:
    each node is connected to at most two segments.
    """
    elems = np.asarray(elements, dtype=np.int64)
    if elems.ndim != 2 or elems.shape[1] != 2:
        raise ValueError("elements must be of shape (E, 2)")
    if elems.size == 0:
        return np.empty((0, 2), dtype=np.int32)

    E = elems.shape[0]
    n_nodes = int(elems.max()) + 1

    # Flatten node incidence:
    # seg_id[k] is incident to node_id[k] at local endpoint loc[k] (0 or 1)
    node_id = elems.reshape(-1)                          # (2E,)
    seg_id  = np.repeat(np.arange(E, dtype=np.int64), 2)
    loc     = np.tile(np.array([0, 1], dtype=np.int64), E)

    # Sort by node so equal nodes are grouped
    order = np.argsort(node_id)
    node_sorted = node_id[order]
    seg_sorted  = seg_id[order]
    loc_sorted  = loc[order]

    # Find groups of equal nodes
    start = np.flatnonzero(np.r_[True, node_sorted[1:] != node_sorted[:-1]])
    end   = np.r_[start[1:], len(node_sorted)]
    count = end - start

    # Result
    neighbours = np.full((E, 2), -1, dtype=np.int64)

    # Boundary nodes: only one incident segment
    mask1 = count == 1
    if np.any(mask1):
        idx = start[mask1]
        neighbours[seg_sorted[idx], loc_sorted[idx]] = -1

    # Regular nodes: exactly two incident segments
    mask2 = count == 2
    if np.any(mask2):
        idx0 = start[mask2]
        idx1 = idx0 + 1

        s0, l0 = seg_sorted[idx0], loc_sorted[idx0]
        s1, l1 = seg_sorted[idx1], loc_sorted[idx1]

        neighbours[s0, l0] = s1
        neighbours[s1, l1] = s0

    # Non-manifold nodes: more than two incident segments
    if np.any(count > 2):
        bad_nodes = node_sorted[start[count > 2]]
        raise ValueError(
            f"Non-manifold polygon: nodes {bad_nodes.tolist()} belong to more than two segments."
        )

    return neighbours.astype(np.int32, copy=False)


def angle(vert):
    """
    Calculate the rotation angle needed to align a set of 3D points with a reference axis.

    Args:
        vert (np.ndarray): Array of vertex coordinates.

    Returns:
        float: The rotation angle in degrees.
    """
    # averaged normalised plane normal + coefficients of plane equation Z = coeff[0]*X + coeff[1]*Y + coeff[2]
    v = np.asarray(vert)
    A = np.c_[v[:,0], v[:,1], np.ones(v.shape[0])]
    coeff,_,_,_ = lstsq(A, v[:,2])  # type: ignore # coefficients
    # plane equation  ax+by+cz+d = 0
    a = coeff[0]
    b = coeff[1]
    c = -1
    # d = coeff[2]
    n = [a,b,c]
    n = np.array(n)/np.linalg.norm(n)

    ## normal has to point in positive y-direction 
    if (b<0):
        n = -n
        coeff = -coeff

    # project normal vector in x-y plane 
    zDirection = n[2]
    n[2]=0
    n = n/np.linalg.norm(n)
    n_ref = np.array([0,1,0]) # reference normal 
    #compute angle
    a_rotate = np.sign(n[0])*angleBetweenVectors(n, n_ref)
    # special case: 
    if zDirection<0:
        print('Add 180 deg rotation to wall')
        a_rotate=a_rotate-180   
    return a_rotate


def read_scalar_field_from_ply(filename):
    """
    Read a scalar field from a PLY file.

    Args:
        filename (str): Path to the PLY file.

    Returns:
        np.ndarray: Scalar field as a numpy array.
    """
    with open(filename, 'rb') as f:
        # Read the header
        reading_vertices = False
        num_float_properties = 0
        num_double_properties = 0
        num_uchar_properties = 0
        line_format = ''
        scalar_field_index = 0
        scalar_field_index_found = False
        while True:
            line = f.readline().decode('utf-8')
            if 'format ascii' in line:
                format = 'ascii'
            elif 'format binary' in line:
                format = 'binary'
            elif 'element vertex' in line:
                n_vertices = int(line.split()[-1])
                reading_vertices = True
            elif 'element face' in line:
                reading_vertices = False
            elif 'end_header' in line:
                reading_vertices = False
                break
            if reading_vertices:
                if 'property' in line:
                    if 'float' in line:
                        num_float_properties += 1
                        line_format += 'f'
                    elif 'double' in line:
                        num_double_properties += 1
                        line_format += 'd'
                    elif 'uchar' in line:
                        num_uchar_properties += 1
                        line_format += 'B'
                    if 'scalar' in line:
                        scalar_field_index_found = True
                    if not scalar_field_index_found:
                        scalar_field_index += 1
        if not scalar_field_index_found:
            return np.array([])
       
        # Read the scalar field
        scalar_field = []
        for n in range(n_vertices):
            # Each vertex is represented by three floats (x, y, z) and one float (scalar field)
            if format == 'binary':
                try:
                    data = f.read(4*num_float_properties + 8*num_double_properties + num_uchar_properties)
                    if not data:
                        raise ValueError('Error in extracting scalar field from file.')
                    values = struct.unpack(line_format, data)
                    scalar_field.append(values[scalar_field_index])
                except struct.error:
                        raise ValueError('Error in extracting scalar field from file.')
            elif format == 'ascii':
                try:
                    line = f.readline().decode('utf-8')
                    values = list(map(float, line.split()))
                    scalar_field.append(values[scalar_field_index])
                except ValueError:
                        raise ValueError('Error in extracting scalar field from file.')
 
    return np.array(scalar_field)


def uniqueMaterialList(seq):
    """
    Unify objects from the input list while preserving their order.

    Generic identity-based deduplication; used both for lists of Material
    (Slope) and lists of Drag (Vegetation).

    Returns
    -------
    table :
        Unique objects in first-appearance order.
    ids : (E,) int32
        Per-element indices into the returned table.
    """
    table = []
    index = {}
    ids = np.empty(len(seq), dtype=np.int32)
    for i, mobj in enumerate(seq):
        key = id(mobj)  # identity-based; change to value-based if you prefer
        j = index.get(key)
        if j is None:
            j = len(table)
            table.append(mobj)
            index[key] = j
        ids[i] = j
    return table, ids


def getTriPoints(points: np.ndarray, triangle: np.ndarray, samples: np.ndarray) -> np.ndarray:
    i0, i1, i2 = triangle[:, 0], triangle[:, 1], triangle[:, 2]
    if points.shape[2] == 1:
        # same coordinates for all samples
        v0 = points[:, i0, 0]           # (3,K)
        v1 = points[:, i1, 0]
        v2 = points[:, i2, 0]
    else:
        # per-sample perturbed coordinates: gather pairwise (node,sample)
        v0 = points[:, i0, samples]         # (3,K)
        v1 = points[:, i1, samples]
        v2 = points[:, i2, samples]
    return np.stack((v0, v1, v2), axis=1)  # (3,3,K)


def isInsideTriangle(p: np.ndarray, points: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """
    Edge-function inside test using cross products in 3D.

    Inputs:
      p       : (3, S) points to test (columns are samples)
      points  : (3, 3, S) or (3, 3, 1) triangle vertices per sample:
                v0 = points[:, 0, :], v1 = points[:, 1, :], v2 = points[:, 2, :]
                If points.shape[2] == 1, it's broadcast to S.
      tol     : tolerance for inside test

    Returns:
      idx : (S,) int32
            -1 if inside (or on edges within tol),
             0 if outside “to the right” of edge v0->v1,
             1 if outside “to the right” of edge v1->v2,
             2 if outside “to the right” of edge v2->v0.

    Notes:
      - The triangle winding defines “right/left”. Winding is v0->v1->v2; the normal is n = (v1-v0)×(v2-v0).
      - We do not normalise n; signs are preserved and it’s faster.
    """
    p = np.asarray(p, dtype=float)
    assert p.shape[0] == 3
    num_samples = p.shape[1]

    points = np.asarray(points, dtype=float)
    if points.shape[:2] != (3, 3) or points.ndim != 3:
        raise ValueError("points must have shape (3, 3, S) or (3, 3, 1)")
    if points.shape[2] == 1:
        points = np.repeat(points, num_samples, axis=2)
    elif points.shape[2] != num_samples:
        raise ValueError("points.shape[2] must be 1 or S to match p")

    v0 = points[:, 0, :]   # (3, S)
    v1 = points[:, 1, :]
    v2 = points[:, 2, :]

    # Triangle normal (not normalised)
    n = np.cross((v1 - v0).T, (v2 - v0).T).T  # (3, S)

    # Edge vectors per the v0->v1->v2 winding
    e01 =  v1 - v0         # edge 0 (v0->v1)
    e12 =  v2 - v1         # edge 1 (v1->v2)
    e20 =  v0 - v2         # edge 2 (v2->v0)

    # Vectors from edge anchors to query point
    w0 = p - v0            # for edge 0 anchor v0
    w1 = p - v1            # for edge 1 anchor v1
    w2 = p - v2            # for edge 2 anchor v2

    # Signed edge tests: s = n · (edge × (p - anchor))
    c0 = np.cross(e01.T, w0.T).T
    c1 = np.cross(e12.T, w1.T).T
    c2 = np.cross(e20.T, w2.T).T

    s0 = np.einsum('ij,ij->j', n, c0)   # edge 0 (v0->v1)
    s1 = np.einsum('ij,ij->j', n, c1)   # edge 1 (v1->v2)
    s2 = np.einsum('ij,ij->j', n, c2)   # edge 2 (v2->v0)

    inside = (s0 >= -tol) & (s1 >= -tol) & (s2 >= -tol)

    # Choose the most-negative edge (the one to cross)
    edge = np.argmin(np.vstack((s0, s1, s2)), axis=0).astype(np.int32)
    edge[inside] = -1
    return edge


def timeParabolaPlane(
    p: np.ndarray, v: np.ndarray, a: np.ndarray,
    points: np.ndarray,
    tol: float = 1e-12,
    t_min: float = 0.0,
    t_max: float = np.inf,
) -> np.ndarray:
    """
    Vectorised intersection: x(t)=p + v t + 0.5 a t^2 with plane(points[:,0],points[:,1],points[:,2]).

    Inputs
      p, v, a   : (3, s)
      points    : (3, 3, s) or (3, 3, 1)
      tol       : tolerance used in plane/roots checks
      t_min     : lower time window (inclusive, with tolerance)
      t_max     : upper time window (inclusive, with tolerance)

    Returns
      t_hit : (s,) time to hit in [t_min, t_max] (np.inf if none)

    Quadratic selection when both roots acceptable:
      choose min(root) if min(root) > t_min + tol, else choose max(root).
    """
    p = np.asarray(p, dtype=float); v = np.asarray(v, dtype=float); a = np.asarray(a, dtype=float)
    assert p.shape[0] == v.shape[0] == a.shape[0] == 3
    s = p.shape[1]

    points = np.asarray(points, dtype=float)
    if points.shape[:2] != (3, 3) or points.ndim != 3:
        raise ValueError("points must have shape (3, 3, s) or (3, 3, 1)")
    if points.shape[2] == 1:
        points = np.repeat(points, s, axis=2)
    elif points.shape[2] != s:
        raise ValueError("points.shape[2] must be 1 or s")

    p1 = points[:, 0, :]  # (3, s)
    p2 = points[:, 1, :]
    p3 = points[:, 2, :]

    # plane n·x + d = 0
    e0 = p2 - p1
    e1 = p3 - p1
    n  = np.cross(e0.T, e1.T).T                  # (3, s)
    nn = np.linalg.norm(n, axis=0)
    valid_plane = nn > tol
    n[:, valid_plane] /= nn[valid_plane]
    d = -np.sum(n * p1, axis=0)

    # coefficients: alpha t^2 + beta t + gamma = 0
    alpha = 0.5 * np.einsum('ij,ij->j', a, n)    # (s,)
    beta  =        np.einsum('ij,ij->j', v, n)
    gamma =        np.einsum('ij,ij->j', p, n) + d

    v_small = np.linalg.norm(v, axis=0) <= tol
    a_small = np.linalg.norm(a, axis=0) <= tol
    plane_tol = max(10.0 * tol, 1e-12)

    t_hit = np.full(s, np.inf, dtype=float)

    # helper: accept times within window (with tolerance)
    def _in_window(t):
        return (t >= (t_min - tol)) & (t <= (t_max + tol))

    # ----- stationary: v≈0 & a≈0 -----
    stat = valid_plane & v_small & a_small
    if np.any(stat):
        on_plane = np.abs(gamma) <= plane_tol
        # whole trajectory lies on plane → choose t_min if within window
        if np.isfinite(t_min):
            t_hit[stat & on_plane] = t_min

    # ----- linear: alpha≈0 & beta!=0 -----
    lin = valid_plane & (np.abs(alpha) <= tol) & (np.abs(beta) > tol)
    if np.any(lin):
        t_lin = -gamma[lin] / beta[lin]
        acc = _in_window(t_lin)
        if np.any(acc):
            chosen = np.clip(t_lin[acc], t_min, t_max)
            t_hit[np.flatnonzero(lin)[acc]] = chosen

    # ----- constant/parallel (excluding stationary): alpha≈0 & beta≈0 -----
    const = valid_plane & (np.abs(alpha) <= tol) & (np.abs(beta) <= tol) & ~stat
    if np.any(const):
        on_plane = np.abs(gamma[const]) <= plane_tol
        if np.any(on_plane):
            # intersects for all t → choose t_min if within window
            if np.isfinite(t_min):
                t_hit[np.flatnonzero(const)[on_plane]] = t_min

    # ----- quadratic: alpha != 0 -----
    quad = valid_plane & (np.abs(alpha) > tol)
    if np.any(quad):
        a2   = alpha[quad]
        b    = beta[quad]
        c    = gamma[quad]
        disc = b*b - 4.0*a2*c
        real = disc >= -tol  # allow tiny negative due to round-off
        if np.any(real):
            idx   = np.flatnonzero(quad)[real]
            r     = np.sqrt(np.maximum(disc[real], 0.0))
            denom = 2.0 * a2[real]

            t1 = (-b[real] - r) / denom
            t2 = (-b[real] + r) / denom

            t_small = np.minimum(t1, t2)
            t_large = np.maximum(t1, t2)

            # Get larger root if element's normal points up
            t_final = np.where(n[-1] >= 0, t_large, t_small)
            
            acc = _in_window(t_final)
            t_hit[idx[acc]] = np.clip(t_final[acc], t_min, t_max)

    return t_hit


def timeRaySegment(
    p: np.ndarray, v: np.ndarray,
    points: np.ndarray,
    tol: float = 1e-12,
    t_min: float = 0.0,
    t_max: float = np.inf,
) -> np.ndarray:
    """
    Vectorised time t such that p + v*t intersects segment [p1, p2].

    Inputs
      p, v    : (3, S)
      points  : (3, 2, S) or (3, 2, 1)  with points[:,0,:]=p1, points[:,1,:]=p2
      tol     : absolute tolerance
      t_min   : lower time window (inclusive with tol)
      t_max   : upper time window (inclusive with tol)

    Returns
      t_hit   : (S,)  time of intersection; np.inf if no intersection.
                Parallel rule: if line ∥ segment and p at t=0 lies on the
                *finite* segment, return 0 (then windowed/clipped); else inf.
    """
    p = np.asarray(p, dtype=float); v = np.asarray(v, dtype=float)
    assert p.shape[0] == v.shape[0] == 3
    s = p.shape[1]

    pts = np.asarray(points, dtype=float)
    if pts.ndim != 3 or pts.shape[:2] != (3, 2):
        raise ValueError("points must be (3, 2, S) or (3, 2, 1)")
    if pts.shape[2] == 1:
        pts = np.repeat(pts, s, axis=2)
    elif pts.shape[2] != s:
        raise ValueError("points.shape[2] must be 1 or S")

    p1 = pts[:, 0, :]             # (3, S)
    p2 = pts[:, 1, :]
    e  = p2 - p1                  # (3, S) segment direction
    w  = p1 - p                   # (3, S) from line point to seg start

    # Helper: time window predicate and clipper
    def in_window(t):
        return (t >= (t_min - tol)) & (t <= (t_max + tol))
    def clamp_window(t):
        return np.clip(t, t_min, t_max)

    # Cross and triple products
    n   = np.cross(v.T, e.T).T            # (3, S)  n = v × e
    n2  = np.sum(n * n, axis=0)           # (S,)    ||v×e||^2

    # Parallel if ||v×e|| ~ 0
    parallel = n2 <= tol

    t_hit = np.full(s, np.inf, dtype=float)

    # --- Non-parallel branch ---
    np_mask = ~parallel
    if np.any(np_mask):
        nm   = n[:, np_mask]
        n2m  = n2[np_mask]
        wm   = w[:, np_mask]
        em   = e[:, np_mask]
        pm   = p[:, np_mask]
        vm   = v[:, np_mask]

        # Coplanarity: (p1 - p) · (v × e) ~ 0
        cop = np.abs(np.sum(wm * nm, axis=0)) <= tol
        if np.any(cop):
            idx = np.flatnonzero(np_mask)[cop]
            nm  = nm[:, cop]; n2c = n2m[cop]
            wm2 = wm[:, cop]; em2 = em[:, cop]; vm2 = vm[:, cop]

            # t = ((w × e) · n) / ||n||^2
            t_num = np.sum(np.cross(wm2.T, em2.T).T * nm, axis=0)
            t     = t_num / n2c

            # u = ((w × v) · n) / ||n||^2   (segment parameter)
            u_num = np.sum(np.cross(wm2.T, vm2.T).T * nm, axis=0)
            u     = u_num / n2c

            on_seg = (u >= -tol) & (u <= 1.0 + tol)
            ok_t   = in_window(t)
            acc    = on_seg & ok_t
            if np.any(acc):
                t_hit[idx[acc]] = clamp_window(t[acc])

    # --- Parallel branch ---
    if np.any(parallel):
        pr = np.flatnonzero(parallel)
        # Colinear if (w × e) ~ 0
        col = np.linalg.norm(np.cross(w[:, pr].T, e[:, pr].T), axis=1) <= tol
        if np.any(col):
            prc = pr[col]
            # Check if p (t=0) lies on the finite segment
            ee2 = np.sum(e[:, prc] * e[:, prc], axis=0)
            on_seg = np.zeros_like(ee2, dtype=bool)

            # Degenerate segment → treat as a point
            deg = ee2 <= tol
            if np.any(deg):
                on_seg[deg] = np.linalg.norm(p[:, prc][:, deg] - p1[:, prc][:, deg], axis=0) <= tol

            if np.any(~deg):
                u0 = np.sum((p[:, prc][:, ~deg] - p1[:, prc][:, ~deg]) * e[:, prc][:, ~deg], axis=0) / ee2[~deg]
                on_seg[~deg] = (u0 >= -tol) & (u0 <= 1.0 + tol)

            if np.any(on_seg):
                t0 = np.zeros(np.count_nonzero(on_seg))
                ok0 = in_window(t0)
                if np.any(ok0):
                    t_hit[prc[on_seg][ok0]] = clamp_window(t0[ok0])

    return t_hit


def timeClosest(
    p: np.ndarray,     # (3, S)
    v: np.ndarray,     # (3, S)
    a: np.ndarray,     # (3, S)
    pt: np.ndarray,    # (3, S) or (3, 1)
    *,
    t_min: float = 0.0,
    t_max: float | None = None,
    tol: float = 1e-12
) -> np.ndarray:
    """
    Vectorised closest-approach time for uniformly accelerated motion.

    Returns
    -------
    t_star : (S,) array
        Time that minimises ||p + v t + 0.5 a t^2 - pt|| within [t_min, t_max].

    Notes
    -----
    If a ≈ 0 => linear solution t* = -(r0·v)/(v·v) clamped to [t_min, t_max].
    If also v ≈ 0 => returns t* = t_min.
    """
    # Broadcast pt to (3, S)
    if pt.shape[1] == 1:
        pt = np.repeat(pt, p.shape[1], axis=1)

    r0 = p - pt                          # (3, S)
    rv = np.sum(r0 * v, axis=0)          # (S,)
    vv = np.sum(v * v, axis=0)           # (S,)
    ra = np.sum(r0 * a, axis=0)          # (S,)
    va = np.sum(v * a, axis=0)           # (S,)
    aa = np.sum(a * a, axis=0)           # (S,)

    S = p.shape[1]
    t_star = np.empty(S, dtype=float)

    # Masks
    has_accel = aa > tol
    no_accel  = ~has_accel

    # --- Case 1: a ≈ 0  (linear in t)
    if np.any(no_accel):
        mask = no_accel
        vv_m = vv[mask]
        rv_m = rv[mask]
        # If vv > 0: t* = -rv/vv; else (v≈0): fallback to boundary
        moving = vv_m > tol
        t_lin = np.full(vv_m.shape, t_min if t_max is None else np.clip(t_min, t_min, t_max))
        t_lin[moving] = -rv_m[moving] / vv_m[moving]
        if t_max is None:
            t_lin = np.maximum(t_lin, t_min)
        else:
            t_lin = np.clip(t_lin, t_min, t_max)
        t_star[mask] = t_lin

    # --- Case 2: a != 0  (cubic in t)
    if np.any(has_accel):
        idx = np.flatnonzero(has_accel)

        c3 = 0.5 * aa[idx]
        c2 = 1.5 * va[idx]
        c1 = vv[idx] + ra[idx]
        c0 = rv[idx]

        # Monic polynomial: t^3 + b2 t^2 + b1 t + b0 = 0
        b2 = c2 / c3
        b1 = c1 / c3
        b0 = c0 / c3

        # Build batched companion matrices (S3, 3, 3)
        S3 = b0.shape[0]
        C = np.zeros((S3, 3, 3), dtype=float)
        C[:, 0, 2] = -b0
        C[:, 1, 0] = 1.0
        C[:, 1, 2] = -b1
        C[:, 2, 1] = 1.0
        C[:, 2, 2] = -b2

        # Roots of cubic via eigenvalues of the companion matrix
        roots = np.linalg.eigvals(C)          # (S3, 3)
        # Keep real roots
        roots_real = np.real(roots)
        roots_imag = np.abs(np.imag(roots))
        real_mask = roots_imag < 1e-10
        t_candidates = np.where(real_mask, roots_real, np.nan)  # (S3, 3)

        # Append boundaries
        t_lo = np.full((S3, 1), t_min)
        if t_max is None or np.isinf(t_max):
            t_hi = None
        else:
            t_hi = np.full((S3, 1), t_max)

        # Clamp/cull candidate roots to window
        if t_max is None:
            t_candidates = np.where((t_candidates >= t_min - tol), t_candidates, np.nan)
        else:
            in_win = (t_candidates >= t_min - tol) & (t_candidates <= t_max + tol)
            t_candidates = np.where(in_win, t_candidates, np.nan)

        # Stack all candidates: real roots + t_min (+ t_max if finite)
        if t_hi is None:
            T = np.concatenate([t_candidates, t_lo], axis=1)    # (S3, K)
        else:
            T = np.concatenate([t_candidates, t_lo, t_hi], axis=1)

        # Evaluate f(t) for all candidates and pick the argmin
        # p_eval: broadcast to (3, S3, K)
        t_eval = T
        # Replace NaNs with a sentinel (we’ll set their cost to +inf)
        valid = np.isfinite(t_eval)
        t_eval = np.where(valid, t_eval, 0.0)

        # r(t) = r0 + v t + 0.5 a t^2
        r_eval = (r0[:, idx, None]
                  + v[:, idx, None] * t_eval[None, :, :]
                  + 0.5 * a[:, idx, None] * (t_eval[None, :, :] ** 2))
        d2 = np.sum(r_eval * r_eval, axis=0)           # (S3, K)
        d2 = np.where(valid, d2, np.inf)

        k_min = np.argmin(d2, axis=1)
        t_best = T[np.arange(S3), k_min]
        t_star[idx] = t_best

    # # Return also the minimum squared distances for convenience
    # r_best = r0 + v * t_star[None, :] + 0.5 * a * (t_star[None, :] ** 2)
    # d2_min = np.sum(r_best * r_best, axis=0)
    return t_star


def rotationAlign2x(vector: np.ndarray) -> np.ndarray:
    """Return R such that R @ vector == ex (to numerical tolerance)."""
    d = np.asarray(vector, dtype=float).reshape(3)
    n = np.linalg.norm(d)
    if n == 0.0:
        raise ValueError("direction cannot be zero.")
    d = d / n

    ex = np.array([1.0, 0.0, 0.0], dtype=float)
    c = float(np.dot(d, ex))               # cos(theta)
    if c > 1 - 1e-12:
        # Already aligned
        return np.eye(3)
    if c < -1 + 1e-12:
        # Opposite: rotate 180° about any axis ⟂ to ex (and d)
        # Choose a stable axis: prefer y unless d is ~y, then use z.
        cand = np.array([0.0, 1.0, 0.0])
        if abs(np.dot(d, cand)) > 0.9:
            cand = np.array([0.0, 0.0, 1.0])
        u = np.cross(d, cand)
        u /= np.linalg.norm(u)  # unit axis
        K = np.array([[0, -u[2], u[1]],
                      [u[2], 0, -u[0]],
                      [-u[1], u[0], 0]], dtype=float)
        # Rodrigues with theta=pi → R = I + 2 K^2  (since sin(pi)=0, 1-cos(pi)=2)
        return np.eye(3) + 2.0 * (K @ K)

    # General case: Rodrigues rotating d to ex around k = d × ex
    k = np.cross(d, ex)
    s = np.linalg.norm(k)                  # sin(theta) > 0 here
    u = k / s                              # unit axis
    K = np.array([[0, -u[2], u[1]],
                  [u[2], 0, -u[0]],
                  [-u[1], u[0], 0]], dtype=float)
    # Standard Rodrigues: R = I + K*sinθ + K^2*(1-cosθ).
    # But sinθ = s, cosθ = c, and K is built with unit axis.
    return np.eye(3) + K * s + (K @ K) * (1.0 - c)


def tri_plane_intersections_yz_per_triangle(
    tri_coords: np.ndarray,   # (E,3,3) → [[x,y,z] per vertex]
    xs: np.ndarray,           # (E,) plane x = xs[i] per triangle
    eps: float = 1e-9,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For each triangle i, intersect it with the plane x=xs[i].
    Returns exactly two intersections (y,z) and their edge ids (0,1,2).
    If a triangle does not intersect properly, outputs NaNs and -1s.

    Args:
        tri_coords : (E,3,3) float
        xs         : (E,) float, one slice-x per triangle
        eps        : float, tolerance

    Returns:
        Y   : (E,2) float, y coords of intersections (NaN if not cut)
        Z   : (E,2) float, z coords of intersections
        Eids: (E,2) int, edge ids (0,1,2) or -1 if invalid
    """
    E = tri_coords.shape[0]

    A = tri_coords[:, 0, :]   # (E,3)
    B = tri_coords[:, 1, :]
    C = tri_coords[:, 2, :]

    Ax, Ay, Az = A[:, 0], A[:, 1], A[:, 2]
    Bx, By, Bz = B[:, 0], B[:, 1], B[:, 2]
    Cx, Cy, Cz = C[:, 0], C[:, 1], C[:, 2]

    def edge_hits(Px, Py, Pz, Qx, Qy, Qz, edge_id):
        den = Qx - Px
        between = (np.minimum(Px, Qx) <= xs) & (xs <= np.maximum(Px, Qx))
        nonpar  = np.abs(den) > eps
        base = between & nonpar
        t = np.full(E, np.nan)
        t[base] = (xs[base] - Px[base]) / den[base]
        on = base & (t >= -eps) & (t <= 1.0 + eps)

        y = np.full(E, np.nan); z = np.full(E, np.nan); eid = np.full(E, -1, int)
        if np.any(on):
            tm = t[on]
            y[on] = Py[on] * (1.0 - tm) + Qy[on] * tm
            z[on] = Pz[on] * (1.0 - tm) + Qz[on] * tm
            eid[on] = edge_id
        return y, z, eid

    # Edge ids: 0:A->B, 1:B->C, 2:C->A
    y0, z0, e0 = edge_hits(Ax, Ay, Az, Bx, By, Bz, 0)
    y1, z1, e1 = edge_hits(Bx, By, Bz, Cx, Cy, Cz, 1)
    y2, z2, e2 = edge_hits(Cx, Cy, Cz, Ax, Ay, Az, 2)

    # Stack → (E,3)
    Y = np.stack([y0, y1, y2], axis=1)
    Z = np.stack([z0, z1, z2], axis=1)
    Eids = np.stack([e0, e1, e2], axis=1)

    # Output arrays
    Y_out = np.full((E, 2), np.nan)
    Z_out = np.full((E, 2), np.nan)
    E_out = np.full((E, 2), -1, int)

    # For each triangle, pick two valid intersections
    for i in range(E):
        hits = ~np.isnan(Y[i])
        idx = np.flatnonzero(hits)
        if idx.size < 2:
            continue
        if idx.size > 2:
            # Degenerate: vertex on plane; pick two with largest Δy
            yk = Y[i, idx]; zk = Z[i, idx]; ek = Eids[i, idx]
            d01 = abs(yk[0] - yk[1])
            d02 = abs(yk[0] - yk[2])
            d12 = abs(yk[1] - yk[2])
            if d01 >= d02 and d01 >= d12:
                idx = idx[[0,1]]
            elif d02 >= d01 and d02 >= d12:
                idx = idx[[0,2]]
            else:
                idx = idx[[1,2]]
        # sort by y (left→right)
        y_pair = Y[i, idx]; z_pair = Z[i, idx]; e_pair = Eids[i, idx]
        if y_pair[0] <= y_pair[1]:
            Y_out[i] = y_pair; Z_out[i] = z_pair; E_out[i] = e_pair
        else:
            Y_out[i] = y_pair[::-1]; Z_out[i] = z_pair[::-1]; E_out[i] = e_pair[::-1]

    return Y_out, Z_out, E_out


def pcWalk(p_init, p_end, slab_points, dlt_x, dlt_z):
    '''
    Algorithm to walk from p_init to p_end through slab_points.
    slab_points: (N,2) array of (x,y) points in the slab
    dlt_x: (N,) array of local spacing between points
    dlt_z: float, global step size
    Returns: (M,2) array of (x,y) points along the path
    1. Start at p_init
    2. Move in the direction of p_end, but only as far as dlt_z
    3. At each step, look for points in slab_points within dlt_z
       in the current direction and within dlt_x locally
    4. If points are found, move towards the average position of
       those points, weighted by 1/distance^2
    5. If no points are found, reset direction towards p_end
    6. Stop when p_end is reached or no progress can be made
    '''
    left_bound, bottom_bound = slab_points.min(axis=0)
    right_bound, top_bound = slab_points.max(axis=0)
    nodes = [np.array(p_init, copy=True)]
    cur = np.array(p_init, copy=True)
    def _plot_walk():
        # import matplotlib.pyplot as plt
        # plt.scatter(slab_points[:,0], slab_points[:,1], s=0.5)
        # plt.plot(np.array(nodes)[:,0], np.array(nodes)[:,1], '-o', color='red', markersize=2)
        # plt.savefig('walk.png')
        # plt.close()
        pass
    def _reset(cur_point):
        dir_init = p_end - cur_point
        dir_init = dir_init / np.linalg.norm(dir_init)
        dir = np.array(dir_init, copy=True)
        prev_dir = np.repeat(dir_init[None, :], 10, axis=0)
        return dir, prev_dir
    if np.linalg.norm(p_end - cur) < dlt_z:
        return np.array([p_init, p_end], dtype=float)
    dir, prev_dir = _reset(cur)
    break_next = False
    while True:
        _plot_walk()
        if len(nodes) > slab_points.shape[0]:
            # print('Too many nodes, giving up')
            break  # give up
        if np.linalg.norm(cur - p_end) < dlt_z:
            nodes += [np.array(p_end, copy=True)]
            break  # reached the end

        # Get points within dlt_z in the current direction
        dlt = slab_points - cur
        dist = np.linalg.norm(dlt, axis=1)
        
        # Blend with direction to endpoint
        dist_global = np.linalg.norm(p_end - cur)
        dir_global = (p_end - cur) / dist_global
        search_dir = dir * 1/dlt_z + dir_global * 1/dist_global
        search_dir /= 1/dlt_z + 1/dist_global

        dlt_proj = np.sum(dlt * search_dir, axis=1)

        in_direction = dlt_proj > 0
        within_reach = dist <= dlt_z

        # Calculate new direction
        if np.any(in_direction):
            new_dir_slab = np.average(dlt[in_direction], axis=0, weights=1/(dist[in_direction]**2 + 1e-12))
            new_dir_slab /= np.linalg.norm(new_dir_slab)
            alpha_slab = max(np.dot(dir, new_dir_slab), 0.1)

            if np.any(in_direction & within_reach):
                new_dir_local = np.average(dlt[in_direction & within_reach], axis=0, weights=1/(dlt_x[in_direction & within_reach]**2 + 1e-12))
                new_dir_local /= np.linalg.norm(new_dir_local)
                alpha_local = max(np.dot(dir, new_dir_local), 0.1)

                if alpha_local > alpha_slab:
                    new_dir = new_dir_local
                else:
                    new_dir = new_dir_slab
                alpha = min(alpha_local, alpha_slab)
            else:
                new_dir = new_dir_slab
                alpha = alpha_slab
        else:
            if break_next:
                break  # give up
            break_next = True
            if np.linalg.norm(p_end - cur) < dlt_z:
                nodes += [np.array(p_end, copy=True)]
                break  # reached the end
            dir, prev_dir = _reset(cur)
            continue

        if np.any(prev_dir @ new_dir < -0.75):
            # don't allow sharp turns: avoid loops
            break_next = True
            if break_next:
                break  # give up
            dir, prev_dir = _reset(cur)
            continue
        
        # Advance
        prev_dir[1:, :] = prev_dir[:-1, :]
        prev_dir[0, :] = dir
        dir = new_dir

        if np.any(within_reach):
            dlt_seg = np.max(dist[within_reach])
        else:
            dlt_seg = dlt_z

        cur += dir * dlt_seg * alpha

        cur[0] = np.clip(cur[0], left_bound, right_bound)
        cur[1] = np.clip(cur[1], bottom_bound, top_bound)

        nodes += [np.array(cur, copy=True)]
        break_next = False
    return np.array(nodes, dtype=float)


def unit(v: np.ndarray, fallback: np.ndarray | None = None, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)

    if n > eps:
        return v / n

    if fallback is not None:
        return unit(fallback, eps=eps)

    raise ValueError("Zero vector cannot be normalised.")


def normal2d(t: np.ndarray) -> np.ndarray:
    return np.array([-t[1], t[0]])


def limit_turn(
    e_new: np.ndarray,
    e_old: np.ndarray,
    max_turn_angle: float | None,
) -> np.ndarray:
    if max_turn_angle is None:
        return unit(e_new, e_old)

    e_new = unit(e_new, e_old)
    e_old = unit(e_old)

    cosang = np.clip(e_old @ e_new, -1.0, 1.0)
    angle = np.arccos(cosang)

    if angle <= max_turn_angle:
        return e_new

    cross = e_old[0] * e_new[1] - e_old[1] * e_new[0]
    sign = np.sign(cross) if cross != 0.0 else 1.0

    c = np.cos(sign * max_turn_angle)
    s = np.sin(sign * max_turn_angle)

    R = np.array([[c, -s], [s, c]])
    return R @ e_old


def weighted_centroid(
    xy: np.ndarray,
    centre: np.ndarray,
    power: float = 1.0,
    eps: float = 1e-6,
) -> np.ndarray:
    if power == 0.0:
        return xy.mean(axis=0)

    d = np.linalg.norm(xy - centre, axis=1)
    w = 1.0 / np.maximum(d, eps) ** power

    return np.average(xy, axis=0, weights=w)


def strip_indices_along_tangent(
    xy: np.ndarray,
    tree: cKDTree,
    centre: np.ndarray,
    tangent: np.ndarray,
    half_width: float,
    transverse_radius: float,
) -> np.ndarray:
    """
    Select points in a finite strip.

    tangent:
        Marching direction.

    half_width:
        Half-thickness of the strip along the marching direction.

    transverse_radius:
        Search extent along the section direction, i.e. normal to tangent.
    """
    tangent = unit(tangent)
    section_dir = normal2d(tangent)

    query_radius = np.hypot(half_width, transverse_radius)

    idx = np.asarray(tree.query_ball_point(centre, query_radius), dtype=int)

    if idx.size == 0:
        return idx

    rel = xy[idx] - centre

    s = rel @ tangent
    r = rel @ section_dir

    keep = (np.abs(s) <= half_width) & (np.abs(r) <= transverse_radius)

    return idx[keep]


def refine_next_node(
    xy: np.ndarray,
    tree: cKDTree,
    x_prev: np.ndarray,
    e_prev: np.ndarray,
    increment: float,
    transverse_radius: float,
    *,
    min_points: int = 20,
    max_iter: int = 20,
    tol: float = 1e-3,
    max_turn_angle: float | None = np.deg2rad(45.0),
    weight_power: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Find the next polyline node.

    The next node is constrained to be exactly `increment` away from x_prev.
    """
    e_prev = unit(e_prev)
    q = x_prev + increment * e_prev

    idx_final = np.empty(0, dtype=int)

    for _ in range(max_iter):
        idx = strip_indices_along_tangent(
            xy=xy,
            tree=tree,
            centre=q,
            tangent=e_prev,
            half_width=0.5 * increment,
            transverse_radius=transverse_radius,
        )

        if idx.size < min_points:
            break

        # Only use points ahead of the previous node.
        rel_prev = xy[idx] - x_prev
        ahead = rel_prev @ e_prev > 0.0
        idx = idx[ahead]

        if idx.size < min_points:
            break

        centroid = weighted_centroid(
            xy[idx],
            centre=q,
            power=weight_power,
        )

        v = centroid - x_prev

        if np.linalg.norm(v) < tol:
            break

        e_trial = unit(v, e_prev)
        e_trial = limit_turn(e_trial, e_prev, max_turn_angle)

        q_new = x_prev + increment * e_trial

        idx_final = idx

        if np.linalg.norm(q_new - q) < tol:
            q = q_new
            break

        q = q_new

    e_next = unit(q - x_prev, e_prev)

    return q, e_next, idx_final


def trace_nodes_from_seed(
    points: np.ndarray,
    increment: float,
    *,
    x0: np.ndarray,
    e0: np.ndarray,
    transverse_radius: float,
    min_points: int = 20,
    max_nodes: int | None = None,
    max_iter: int = 20,
    tol: float = 1e-3,
    max_turn_angle: float | None = np.deg2rad(45.0),
    weight_power: float = 1.0,
    min_new_fraction: float = 0.10,
) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    xy = points[:, :2]

    tree = cKDTree(xy)

    x = np.asarray(x0, dtype=float).copy()
    e = unit(np.asarray(e0, dtype=float))

    nodes = [x.copy()]
    used = np.zeros(points.shape[0], dtype=bool)

    while max_nodes is None or len(nodes) < max_nodes:
        idx_here = strip_indices_along_tangent(
            xy=xy,
            tree=tree,
            centre=x,
            tangent=e,
            half_width=0.5 * increment,
            transverse_radius=transverse_radius,
        )

        if idx_here.size >= min_points:
            n_new = np.count_nonzero(~used[idx_here])
            new_fraction = n_new / idx_here.size

            if new_fraction < min_new_fraction:
                break

            used[idx_here] = True

        x_next, e_next, idx_search = refine_next_node(
            xy=xy,
            tree=tree,
            x_prev=x,
            e_prev=e,
            increment=increment,
            transverse_radius=transverse_radius,
            min_points=min_points,
            max_iter=max_iter,
            tol=tol,
            max_turn_angle=max_turn_angle,
            weight_power=weight_power,
        )

        if idx_search.size < min_points:
            break

        x = x_next
        e = e_next

        nodes.append(x.copy())

    return np.asarray(nodes)


def trace_polyline_nodes(
    points: np.ndarray,
    increment: float,
    *,
    transverse_radius: float,
    buffer_steps: int = 5,
    initial_direction: tuple[float, float] = (1.0, 0.0),
    min_points: int = 20,
    max_nodes: int | None = None,
    max_iter: int = 20,
    tol: float = 1e-3,
    max_turn_angle: float | None = np.deg2rad(45.0),
    weight_power: float = 1.0,
    min_new_fraction: float = 0.10,
) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    xy = points[:, :2]

    x_start = xy[np.argmin(xy[:, 0])]
    e_start = unit(np.asarray(initial_direction, dtype=float))

    # First pass: deliberately allow bad initialisation.
    forward_nodes = trace_nodes_from_seed(
        points,
        increment,
        x0=x_start,
        e0=e_start,
        transverse_radius=transverse_radius,
        min_points=min_points,
        max_nodes=max_nodes,
        max_iter=max_iter,
        tol=tol,
        max_turn_angle=max_turn_angle,
        weight_power=weight_power,
        min_new_fraction=min_new_fraction,
    )

    if forward_nodes.shape[0] <= buffer_steps + 1:
        raise RuntimeError(
            "Not enough nodes were traced to apply the requested buffer_steps."
        )

    anchor = forward_nodes[buffer_steps]

    # Stable tangent at the anchor.
    e_anchor = unit(forward_nodes[buffer_steps + 1] - forward_nodes[buffer_steps])

    # Walk backwards from the stable anchor.
    backward_nodes = trace_nodes_from_seed(
        points,
        increment,
        x0=anchor,
        e0=-e_anchor,
        transverse_radius=transverse_radius,
        min_points=min_points,
        max_nodes=max_nodes,
        max_iter=max_iter,
        tol=tol,
        max_turn_angle=max_turn_angle,
        weight_power=weight_power,
        min_new_fraction=min_new_fraction,
    )

    # backward_nodes starts at anchor and goes backwards:
    # [anchor, b1, b2, ...]
    # Reverse it and remove duplicate anchor.
    backward_part = backward_nodes[::-1]

    # forward_nodes also starts at anchor.
    forward_part = forward_nodes[buffer_steps+1:]

    nodes = np.vstack([backward_part, forward_part])

    return nodes


def profile_geometry_from_nodes(nodes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert node polyline into profile centres and directions.

    Returns
    -------
    centres : (N, 2) ndarray
        Profile locations, placed at segment midpoints.

    tangents : (N, 2) ndarray
        Marching tangent of each segment.

    section_dirs : (N, 2) ndarray
        XY direction of the 2D section, normal to the marching tangent.
    """
    nodes = np.asarray(nodes, dtype=float)

    if nodes.shape[0] < 2:
        raise ValueError("At least two polyline nodes are required.")

    seg = nodes[1:] - nodes[:-1]
    length = np.linalg.norm(seg, axis=1)

    tangents = seg / length[:, None]
    centres = 0.5 * (nodes[:-1] + nodes[1:])
    section_dirs = np.column_stack([-tangents[:, 1], tangents[:, 0]])

    return centres, tangents, section_dirs


def assign_sections_from_nodes(
    points: np.ndarray,
    nodes: np.ndarray,
    *,
    transverse_radius: float,
    min_points: int = 20,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """
    Stage 2.

    Assign points to profiles located at segment midpoints.

    Each section i owns points whose projected coordinate along the segment
    tangent lies between the two segment endpoints:

        abs((p - centre_i) . tangent_i) <= length_i / 2

    The section direction is normal to the segment tangent.
    """
    points = np.asarray(points, dtype=float)
    xy = points[:, :2]

    centres, tangents, section_dirs = profile_geometry_from_nodes(nodes)

    tree = cKDTree(xy)

    seg_lengths = np.linalg.norm(nodes[1:] - nodes[:-1], axis=1)

    section_ids: list[np.ndarray] = []

    for c, t, L in zip(centres, tangents, seg_lengths):
        idx = strip_indices_along_tangent(
            xy=xy,
            tree=tree,
            centre=c,
            tangent=t,
            half_width=0.5 * L,
            transverse_radius=transverse_radius,
        )

        if idx.size < min_points:
            idx = np.empty(0, dtype=int)

        section_ids.append(idx)

    return section_ids, centres, tangents, section_dirs


def brentq(
    f: Callable[[np.ndarray], np.ndarray],
    a: np.ndarray,
    b: np.ndarray,
    xtol: float = 1e-10,
    rtol: float = 4 * np.finfo(float).eps,
    ftol: float = 0.0,
    maxiter: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorised Brent root finder for many independent 1D brackets.

    Parameters
    ----------
    f : callable
        Vectorised function. For input x with shape (N,), must return shape (N,).
    a, b : array_like
        Initial bracketing intervals. Must have the same shape.
    xtol : float, optional
        Absolute tolerance on x.
    rtol : float, optional
        Relative tolerance on x.
    ftol : float, optional
        Absolute tolerance on f(x). If |f(x)| <= ftol, root is accepted.
    maxiter : int, optional
        Maximum number of iterations.

    Returns
    -------
    root : np.ndarray
        Estimated roots. Entries that never had a valid bracket are NaN.
    converged : np.ndarray of bool
        True where convergence was achieved.
    iters : np.ndarray of int
        Number of iterations performed for each problem.

    Notes
    -----
    This is a batched version of Brent's method. Each root keeps its own state,
    but updates are performed with boolean masks.
    """
    a = np.asarray(a, dtype=float).copy()
    b = np.asarray(b, dtype=float).copy()

    if a.shape != b.shape:
        raise ValueError("a and b must have the same shape")

    shape = a.shape
    n = a.size

    a = a.ravel()
    b = b.ravel()

    fa = np.asarray(f(a), dtype=float).ravel()
    fb = np.asarray(f(b), dtype=float).ravel()

    if fa.shape != a.shape or fb.shape != b.shape:
        raise ValueError("f(x) must return an array with the same shape as x")

    root = np.full(n, np.nan, dtype=float)
    converged = np.zeros(n, dtype=bool)
    iters = np.zeros(n, dtype=int)
    bracketed = fa * fb < 0.0
    root[~bracketed] = b[~bracketed]
    converged[~bracketed] = True

    # Valid initial brackets only
    valid = np.isfinite(a) & np.isfinite(b) & np.isfinite(fa) & np.isfinite(fb) & bracketed

    if not np.any(valid):
        return root.reshape(shape), converged.reshape(shape), iters.reshape(shape)
    

    # Brent state
    c = a.copy()
    fc = fa.copy()
    d = np.empty_like(a)
    d[:] = np.nan

    mflag = np.ones(n, dtype=bool)

    # Ensure |f(a)| >= |f(b)| on valid entries
    swap = valid & (np.abs(fa) < np.abs(fb))
    if np.any(swap):
        a[swap], b[swap] = b[swap].copy(), a[swap].copy()
        fa[swap], fb[swap] = fb[swap].copy(), fa[swap].copy()

    active = valid.copy()

    for k in range(1, maxiter + 1):
        if not np.any(active):
            break

        tol = xtol + rtol * np.abs(b)
        done = active & ((np.abs(fb) <= ftol) | (np.abs(b - a) <= tol))
        if np.any(done):
            root[done] = np.where(fb[done] < 0, a[done], b[done])
            converged[done] = True
            iters[done] = k - 1
            active[done] = False

        if not np.any(active):
            break

        s = np.empty_like(a)

        # Inverse quadratic interpolation where possible
        iq = (
            active
            & (fa != fc)
            & (fb != fc)
        )

        if np.any(iq):
            ai, bi, ci = a[iq], b[iq], c[iq]
            fai, fbi, fci = fa[iq], fb[iq], fc[iq]

            s[iq] = (
                ai * fbi * fci / ((fai - fbi) * (fai - fci))
                + bi * fai * fci / ((fbi - fai) * (fbi - fci))
                + ci * fai * fbi / ((fci - fai) * (fci - fbi))
            )

        # Secant otherwise
        sec = active & ~iq
        if np.any(sec):
            as_, bs_ = a[sec], b[sec]
            fas, fbs = fa[sec], fb[sec]
            s[sec] = bs_ - fbs * (bs_ - as_) / (fbs - fas)

        # Acceptance / fallback tests
        mid_lo = (3.0 * a + b) / 4.0
        mid_hi = b

        between = ((s > np.minimum(mid_lo, mid_hi)) & (s < np.maximum(mid_lo, mid_hi)))

        cond1 = ~between
        cond2 = mflag & (np.abs(s - b) >= np.abs(b - c) / 2.0)
        cond3 = (~mflag) & (np.abs(s - b) >= np.abs(c - d) / 2.0)
        cond4 = mflag & (np.abs(b - c) < tol)
        cond5 = (~mflag) & (np.abs(c - d) < tol)

        use_bisect = active & (cond1 | cond2 | cond3 | cond4 | cond5)

        if np.any(use_bisect):
            s[use_bisect] = 0.5 * (a[use_bisect] + b[use_bisect])
            mflag[use_bisect] = True

        accepted = active & ~use_bisect
        if np.any(accepted):
            mflag[accepted] = False

        fs = np.full_like(fa, np.nan)
        if np.any(active):
            fs[active] = np.asarray(f(s), dtype=float)[active]

        # If function evaluation produced invalid values, deactivate those cases
        bad_fs = active & ~np.isfinite(fs)
        if np.any(bad_fs):
            active[bad_fs] = False
            iters[bad_fs] = k

        still = active.copy()
        if not np.any(still):
            break

        d[still] = c[still]
        c[still] = b[still]
        fc[still] = fb[still]

        left = still & (fa * fs < 0.0)
        right = still & ~left

        if np.any(left):
            b[left] = s[left]
            fb[left] = fs[left]

        if np.any(right):
            a[right] = s[right]
            fa[right] = fs[right]

        swap = still & (np.abs(fa) < np.abs(fb))
        if np.any(swap):
            a[swap], b[swap] = b[swap].copy(), a[swap].copy()
            fa[swap], fb[swap] = fb[swap].copy(), fa[swap].copy()

        iters[still] = k

    # Final convergence check after loop
    remaining = active & ((np.abs(fb) <= ftol) | (np.abs(b - a) <= (xtol + rtol * np.abs(b))))
    if np.any(remaining):
        root[remaining] = np.where(fb[remaining] < 0, a[remaining], b[remaining])
        converged[remaining] = True

    # For valid but unconverged cases, return current best estimate anyway
    unconverged_valid = valid & ~converged
    root[unconverged_valid] = b[unconverged_valid]

    return root.reshape(shape), converged.reshape(shape), iters.reshape(shape)


def rk4_step(
    dt: float| np.ndarray,
    x: np.ndarray,
    v: np.ndarray,
    accel, *args
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorised RK4 step for many blocks at once.

    Parameters
    ----------
    dt : float
        Time step.
    x : (3, n) ndarray
        Positions.
    v : (3, n) ndarray
        Velocities.
    accel : callable
        Function a = accel(x, v, *args), returning (3, n) acceleration.
    *args :
        Extra data passed to accel.

    Returns
    -------
    x_new : (3, n) ndarray
    v_new : (3, n) ndarray
    """
    k1x = v
    k1v = accel(x, v, *args)

    x2 = x + 0.5 * dt * k1x
    v2 = v + 0.5 * dt * k1v
    k2x = v2
    k2v = accel(x2, v2, *args)

    x3 = x + 0.5 * dt * k2x
    v3 = v + 0.5 * dt * k2v
    k3x = v3
    k3v = accel(x3, v3, *args)

    x4 = x + dt * k3x
    v4 = v + dt * k3v
    k4x = v4
    k4v = accel(x4, v4, *args)

    x_new = x + (dt / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
    v_new = v + (dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)

    return x_new, v_new

    
def getSubSamples(index: np.ndarray, *arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    """Get sub-samples."""
    return tuple(a if a.shape[-1] == 1 else a[...,index]  for a in arrays)


def setSubSamples(index: np.ndarray, values: Tuple, *arrays: np.ndarray) -> None:
    """Set sub-samples back into arrays."""
    for a, v in zip(arrays, values):
        if a.shape[-1] != 1:
            a[..., index] = v
        else:
            a[...] = v


def sampleNormals(position: np.ndarray, elementsIDs: np.ndarray, elementNormals: np.ndarray) -> np.ndarray:
    """Sample the normal vectors at the given positions."""
    S = position.shape[1]
    hit = elementsIDs >= 0
    ids_safe = np.where(hit, elementsIDs, 0)             # avoid OOB for -1

    # choose sample axis selector: 0 if element_norm has only one slice; else 0..S-1
    sel = np.arange(S) if elementNormals.shape[2] == S else np.zeros(S, dtype=int)

    # vectorised gather: result is (D, S)
    norm_per_sample = elementNormals[:, ids_safe, sel].copy()

    # clear out the non-touching samples
    norm_per_sample[:, ~hit] = np.nan
    return norm_per_sample


def decompose(
    vector: np.ndarray,  # (D, S)
    normal: np.ndarray,  # (D, S)  (not necessarily unit)
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Decompose `vector` into normal and tangential components w.r.t. `normal`.

    Args:
        vector: (D, S) vectors to decompose.
        normal: (D, S) normals; can be non-unit and vary per sample.
        eps:    small number to guard division by zero.

    Returns:
        vn: (1, S) normal velocity (scalar) vn_vec = n * vn.
        vt: (D, S) tangential component in Cartesian coordinates.

    Notes:
        - Works for D = 2 or 3, S arbitrary.
        - If a normal has ~zero length, the output for that sample is:
            vn_vec = 0, vt_vec = vector, vn = 0, vt = ||vector||.
    """
    # Scalar normal projection per sample (1, S), then expand to (D, S)
    vn_scalar = np.sum(vector * normal, axis=0)  # (S,)


    # Tangential component (Cartesian) (D,S) - (D,S)*(S,)
    vt_vec = vector - normal * vn_scalar         # (D, S)

    return vn_scalar, vt_vec
