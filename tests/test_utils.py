"""Tests for pyrockfall._utils: geometric/mesh primitives used throughout the
package (Geometry, Geometry3D, Slope, Vegetation, Mesh, Analysis all import
from here).

Two real bugs were found and fixed while writing these tests (confirmed with
the user before changing production code):

1. uniqueMaterialList hardcoded `isinstance(mobj, Material)`, even though the
   function is shared by Slope (Material) and Vegetation (Drag) -- see
   test_vegetation.py for the full story. Fixed by dropping the
   type-specific check (already covered there; re-tested directly here).

2. getSubMesh's signature/return didn't match how _mesh.py's Mesh.split()
   and segmentation actually call it: callers pass raw (V, T) vertex/
   triangle arrays and unpack a 3-tuple, but the function expected a mesh
   *object* with remove_triangles_by_index()/etc. and returned a single
   object. Fixed by reimplementing it as array-based submesh extraction
   (keep referenced vertices, remap triangle indices, return
   (V_sub, T_sub, vertex_ids)), matching every real call site.
"""
import numpy as np
import pytest

from pyrockfall._utils import (
    build_neighbours_mesh,
    build_neighbours_polygon,
    brentq,
    uniqueMaterialList,
    getTriPoints,
    isInsideTriangle,
    timeClosest,
    timeParabolaPlane,
    timeRaySegment,
    decompose,
    sampleNormals,
    getSubSamples,
    setSubSamples,
    getSubMesh,
    triangleCentroids,
    tri_plane_intersections_yz_per_triangle,
    rotationAlign2x,
    unit,
    normal2d,
    limit_turn,
    weighted_centroid,
    profile_geometry_from_nodes,
    pcWalk,
    trace_polyline_nodes,
    assign_sections_from_nodes,
)


# ---------------------------------------------------------------------------
# build_neighbours_polygon (used by Geometry, 2D polyline adjacency)
# ---------------------------------------------------------------------------

def test_build_neighbours_polygon_open_chain():
    elements = np.array([[0, 1], [1, 2], [2, 3]])
    nb = build_neighbours_polygon(elements)
    np.testing.assert_array_equal(nb, [[-1, 1], [0, 2], [1, -1]])


def test_build_neighbours_polygon_closed_loop_has_no_boundary():
    elements = np.array([[0, 1], [1, 2], [2, 0]])
    nb = build_neighbours_polygon(elements)
    assert not np.any(nb == -1)
    # each segment's neighbours must themselves list it back
    for seg in range(3):
        for loc in range(2):
            other = nb[seg, loc]
            assert seg in nb[other]


def test_build_neighbours_polygon_empty():
    nb = build_neighbours_polygon(np.empty((0, 2), dtype=int))
    assert nb.shape == (0, 2)


def test_build_neighbours_polygon_rejects_non_manifold():
    # node 1 is shared by three segments -> non-manifold
    elements = np.array([[0, 1], [1, 2], [1, 3]])
    with pytest.raises(ValueError, match="Non-manifold polygon"):
        build_neighbours_polygon(elements)


# ---------------------------------------------------------------------------
# build_neighbours_mesh (used by Geometry3D, Mesh)
# ---------------------------------------------------------------------------

def test_build_neighbours_mesh_two_triangle_square():
    # Same fixture as test_geometry3d.py's diagonal-split square.
    triangles = np.array([[0, 1, 2], [0, 2, 3]])
    nb = build_neighbours_mesh(triangles)
    assert nb.shape == (2, 3)
    assert nb[0, 2] == 1
    assert nb[1, 0] == 0
    assert nb[0, 0] == -1 and nb[0, 1] == -1
    assert nb[1, 1] == -1 and nb[1, 2] == -1


def test_build_neighbours_mesh_rejects_wrong_shape():
    with pytest.raises(ValueError, match=r"triangles must be \(M,3\)"):
        build_neighbours_mesh(np.array([[0, 1], [1, 2]]))


# ---------------------------------------------------------------------------
# brentq (vectorised batched Brent root finder; used by Geometry.intersectDamped)
# ---------------------------------------------------------------------------

def test_brentq_finds_known_roots_vectorised():
    # f1(x) = x^2 - 4 has a root at x=2 in [0, 10]
    # f2(x) = x - 3 has a root at x=3 in [0, 10]
    def f(x):
        out = np.empty_like(x)
        out[0] = x[0] ** 2 - 4.0
        out[1] = x[1] - 3.0
        return out

    a = np.array([0.0, 0.0])
    b = np.array([10.0, 10.0])
    roots, converged, _ = brentq(f, a, b, xtol=1e-12)
    assert np.all(converged)
    np.testing.assert_allclose(roots, [2.0, 3.0], atol=1e-8)


def test_brentq_matches_scipy_reference():
    from scipy.optimize import brentq as scipy_brentq

    f_scalar = lambda x: np.cos(x) - x
    root_ref = scipy_brentq(f_scalar, 0.0, 1.0, xtol=1e-12)

    f_vec = lambda x: np.cos(x) - x
    roots, converged, _ = brentq(f_vec, np.array([0.0]), np.array([1.0]), xtol=1e-12)
    assert converged[0]
    assert roots[0] == pytest.approx(root_ref, abs=1e-8)


def test_brentq_unbracketed_interval_reports_converged_with_b():
    # Characterization: when fa*fb >= 0 (no sign change), brentq does not
    # raise -- it just reports the interval as "converged" with root = b.
    f = lambda x: np.full_like(x, 5.0)  # never crosses zero
    roots, converged, _ = brentq(f, np.array([0.0]), np.array([1.0]))
    assert converged[0]
    assert roots[0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# uniqueMaterialList (used by Slope and Vegetation)
# ---------------------------------------------------------------------------

def test_unique_material_list_dedups_by_identity_preserving_order():
    class Obj:
        def __init__(self, name):
            self.name = name

    a, b = Obj("A"), Obj("B")
    table, ids = uniqueMaterialList([a, b, a, a, b])
    assert table == [a, b]
    np.testing.assert_array_equal(ids, [0, 1, 0, 0, 1])
    assert ids.dtype == np.int32


def test_unique_material_list_no_duplicates():
    items = [object() for _ in range(4)]
    table, ids = uniqueMaterialList(items)
    assert table == items
    np.testing.assert_array_equal(ids, [0, 1, 2, 3])


def test_unique_material_list_accepts_any_type_not_just_material():
    # Regression: this used to hardcode isinstance(mobj, Material).
    table, ids = uniqueMaterialList([1, 2, 1, "x"])
    assert table == [1, 2, "x"]
    np.testing.assert_array_equal(ids, [0, 1, 0, 2])


def test_unique_material_list_empty():
    table, ids = uniqueMaterialList([])
    assert table == []
    assert ids.shape == (0,)


# ---------------------------------------------------------------------------
# getTriPoints (used by Geometry3D.exitTime/intersectParabola)
# ---------------------------------------------------------------------------

def test_get_tri_points_single_realization_shared_across_samples():
    points = np.arange(3 * 4 * 1, dtype=float).reshape(3, 4, 1)
    triangle = np.array([[0, 1, 2], [1, 2, 3]])
    samples = np.array([0, 1])
    out = getTriPoints(points, triangle, samples)
    assert out.shape == (3, 3, 2)
    np.testing.assert_array_equal(out[:, :, 0], points[:, [0, 1, 2], 0])
    np.testing.assert_array_equal(out[:, :, 1], points[:, [1, 2, 3], 0])


def test_get_tri_points_per_sample_gather():
    points = np.arange(3 * 4 * 2, dtype=float).reshape(3, 4, 2)
    triangle = np.array([[0, 1, 2], [1, 2, 3]])
    samples = np.array([0, 1])
    out = getTriPoints(points, triangle, samples)
    assert out.shape == (3, 3, 2)
    np.testing.assert_array_equal(out[:, :, 0], points[:, [0, 1, 2], 0])
    np.testing.assert_array_equal(out[:, :, 1], points[:, [1, 2, 3], 1])


# ---------------------------------------------------------------------------
# isInsideTriangle (used by Geometry3D.intersectParabola)
# ---------------------------------------------------------------------------

RIGHT_TRIANGLE = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # v0,v1,v2


def _tri_points(*vertices):
    return np.stack(vertices, axis=0)[:, :, None].transpose(1, 0, 2)  # (3,3,1)


def test_is_inside_triangle_strictly_inside_point():
    p = np.array([[0.2], [0.2], [0.0]])
    tri = _tri_points(*RIGHT_TRIANGLE)
    idx = isInsideTriangle(p, tri)
    assert idx[0] == -1


@pytest.mark.parametrize(
    "point, expected_edge",
    [
        ((0.5, -0.5, 0.0), 0),  # below edge v0->v1 (the x-axis)
        ((1.0, 1.0, 0.0), 1),   # beyond the hypotenuse v1->v2
        ((-0.5, 0.5, 0.0), 2),  # left of edge v2->v0 (the y-axis)
    ],
)
def test_is_inside_triangle_outside_reports_crossed_edge(point, expected_edge):
    p = np.array([[point[0]], [point[1]], [point[2]]])
    tri = _tri_points(*RIGHT_TRIANGLE)
    idx = isInsideTriangle(p, tri)
    assert idx[0] == expected_edge


def test_is_inside_triangle_on_edge_counts_as_inside():
    p = np.array([[0.5], [0.0], [0.0]])  # exactly on edge v0->v1
    tri = _tri_points(*RIGHT_TRIANGLE)
    idx = isInsideTriangle(p, tri)
    assert idx[0] == -1


def test_is_inside_triangle_vectorised_multiple_points():
    pts = np.array([[0.2, 0.5, -0.5], [0.2, -0.5, 0.5], [0.0, 0.0, 0.0]])
    tri = _tri_points(*RIGHT_TRIANGLE)
    idx = isInsideTriangle(pts, tri)
    np.testing.assert_array_equal(idx, [-1, 0, 2])


# ---------------------------------------------------------------------------
# timeClosest (used by Geometry3D.intersectParabola for the neighbour walk)
# ---------------------------------------------------------------------------

def test_time_closest_linear_matches_analytical_projection():
    p = np.array([[0.0], [0.0], [0.0]])
    v = np.array([[1.0], [0.0], [0.0]])
    a = np.zeros((3, 1))
    target = np.array([[3.0], [1.0], [0.0]])
    t = timeClosest(p, v, a, target)
    assert t[0] == pytest.approx(3.0)


def test_time_closest_stationary_returns_t_min():
    p = np.array([[5.0], [5.0], [0.0]])
    v = np.zeros((3, 1))
    a = np.zeros((3, 1))
    target = np.array([[0.0], [0.0], [0.0]])
    t = timeClosest(p, v, a, target, t_min=2.0)
    assert t[0] == pytest.approx(2.0)


def test_time_closest_with_acceleration_matches_grid_search():
    p = np.array([[0.0], [10.0], [0.0]])
    v = np.array([[1.0], [0.0], [0.0]])
    a = np.array([[0.0], [-9.81], [0.0]])
    target = np.array([[5.0], [0.0], [0.0]])
    t = timeClosest(p, v, a, target, t_min=0.0, t_max=10.0)

    grid = np.linspace(0.0, 10.0, 200_001)
    pos = p + v * grid + 0.5 * a * grid ** 2
    d2 = np.sum((pos - target) ** 2, axis=0)
    t_ref = grid[np.argmin(d2)]
    assert t[0] == pytest.approx(t_ref, abs=5e-4)


# ---------------------------------------------------------------------------
# timeParabolaPlane (used by Geometry3D.intersectParabola)
# ---------------------------------------------------------------------------

def _horizontal_plane_points():
    # z=0 plane defined by three points: (0,0,0), (1,0,0), (0,1,0).
    points = np.zeros((3, 3, 1))
    points[:, 0, 0] = [0.0, 0.0, 0.0]
    points[:, 1, 0] = [1.0, 0.0, 0.0]
    points[:, 2, 0] = [0.0, 1.0, 0.0]
    return points


def test_time_parabola_plane_free_fall_matches_closed_form():
    H, g = 20.0, 9.81
    p = np.array([[0.0], [0.0], [H]])
    v = np.zeros((3, 1))
    a = np.array([[0.0], [0.0], [-g]])
    t = timeParabolaPlane(p, v, a, _horizontal_plane_points())
    assert t[0] == pytest.approx(np.sqrt(2 * H / g), rel=1e-8)


def test_time_parabola_plane_no_hit_returns_inf():
    p = np.array([[0.0], [0.0], [1.0]])
    v = np.array([[0.0], [0.0], [1.0]])  # moving away from the z=0 plane, no accel
    a = np.zeros((3, 1))
    t = timeParabolaPlane(p, v, a, _horizontal_plane_points())
    assert np.isinf(t[0])


# ---------------------------------------------------------------------------
# timeRaySegment (used by Geometry3D.exitTime)
# ---------------------------------------------------------------------------

def test_time_ray_segment_matches_independent_line_solve():
    p = np.array([[0.0], [0.0], [0.0]])
    v = np.array([[1.0], [1.0], [0.0]])
    seg = np.array([[[2.0], [0.0]], [[0.0], [2.0]], [[0.0], [0.0]]])  # (3,2,1): p1=(2,0,0), p2=(0,2,0)

    t = timeRaySegment(p, v, seg)

    # independent reference: solve p + v*t on the line x+y=2
    t_ref = 2.0 / (v[0, 0] + v[1, 0])
    assert t[0] == pytest.approx(t_ref, rel=1e-8)


def test_time_ray_segment_misses_returns_inf():
    p = np.array([[0.0], [0.0], [0.0]])
    v = np.array([[1.0], [0.0], [0.0]])
    # segment far away, not aligned with ray direction (parallel case, off the ray line)
    seg = np.array([[[0.0], [0.0]], [[5.0], [6.0]], [[0.0], [0.0]]])
    t = timeRaySegment(p, v, seg)
    assert np.isinf(t[0])


# ---------------------------------------------------------------------------
# decompose (used by Analysis._impacts / _addRoughness)
# ---------------------------------------------------------------------------

def test_decompose_reconstructs_original_vector():
    vector = np.array([[3.0], [4.0], [0.0]])
    normal = np.array([[0.0], [1.0], [0.0]])
    vn, vt = decompose(vector, normal)
    assert vn[0] == pytest.approx(4.0)
    np.testing.assert_allclose(vt[:, 0], [3.0, 0.0, 0.0])
    reconstructed = vt + normal * vn
    np.testing.assert_allclose(reconstructed, vector)


def test_decompose_zero_normal_puts_everything_in_tangential():
    vector = np.array([[3.0], [4.0], [5.0]])
    normal = np.zeros((3, 1))
    vn, vt = decompose(vector, normal)
    assert vn[0] == pytest.approx(0.0)
    np.testing.assert_allclose(vt, vector)


def test_decompose_vectorised_multiple_samples():
    vector = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    normal = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    vn, vt = decompose(vector, normal)
    np.testing.assert_allclose(vn, [1.0, 1.0])
    np.testing.assert_allclose(vt, np.zeros((3, 2)))


# ---------------------------------------------------------------------------
# sampleNormals (used by Analysis)
# ---------------------------------------------------------------------------

def test_sample_normals_gathers_by_element_id_shared_across_samples():
    element_normals = np.array([[[1.0], [0.0]], [[0.0], [1.0]], [[0.0], [0.0]]])  # (3, E=2, 1)
    element_ids = np.array([0, 1, 0])
    out = sampleNormals(np.zeros((3, 3)), element_ids, element_normals)
    assert out.shape == (3, 3)
    np.testing.assert_allclose(out[:, 0], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(out[:, 1], [0.0, 1.0, 0.0])
    np.testing.assert_allclose(out[:, 2], [1.0, 0.0, 0.0])


def test_sample_normals_marks_non_touching_samples_as_nan():
    element_normals = np.array([[[1.0]], [[0.0]], [[0.0]]])  # (3, E=1, 1)
    element_ids = np.array([0, -1])
    out = sampleNormals(np.zeros((3, 2)), element_ids, element_normals)
    assert not np.isnan(out[:, 0]).any()
    assert np.isnan(out[:, 1]).all()


def test_sample_normals_per_sample_slice():
    element_normals = np.array([[[1.0, 2.0]], [[0.0, 0.0]], [[0.0, 0.0]]])  # (3, E=1, S=2)
    element_ids = np.array([0, 0])
    out = sampleNormals(np.zeros((3, 2)), element_ids, element_normals)
    np.testing.assert_allclose(out[0], [1.0, 2.0])


# ---------------------------------------------------------------------------
# getSubSamples / setSubSamples (used by Analysis)
# ---------------------------------------------------------------------------

def test_get_sub_samples_indexes_full_arrays_and_broadcasts_singletons():
    full = np.arange(10.0).reshape(2, 5)
    shared = np.ones((2, 1))
    mask = np.array([True, False, True, False, True])
    sub_full, sub_shared = getSubSamples(mask, full, shared)
    np.testing.assert_array_equal(sub_full, full[:, mask])
    np.testing.assert_array_equal(sub_shared, shared)  # unaffected: shape[-1] == 1


def test_set_sub_samples_writes_back_into_full_arrays():
    full = np.zeros((2, 5))
    mask = np.array([True, False, True, False, True])
    values = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    setSubSamples(mask, (values,), full)
    np.testing.assert_array_equal(full[:, mask], values)
    np.testing.assert_array_equal(full[:, ~mask], np.zeros((2, 2)))


def test_set_sub_samples_broadcasts_into_singleton_array():
    shared = np.zeros((2, 1))
    mask = np.array([True, False, True])
    setSubSamples(mask, (np.array([[9.0], [9.0]]),), shared)
    np.testing.assert_array_equal(shared, [[9.0], [9.0]])


# ---------------------------------------------------------------------------
# getSubMesh (used by Mesh.split / Mesh's segment-by-x)
# ---------------------------------------------------------------------------

def test_get_sub_mesh_keeps_only_referenced_vertices_and_remaps_indices():
    V = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [2.0, 0.0, 0.0]])
    T = np.array([[0, 1, 2], [0, 2, 3], [1, 4, 2]])

    V_sub, T_sub, vertex_ids = getSubMesh(V, T, [0, 1])

    assert V_sub.shape == (4, 3)  # triangles 0,1 only touch vertices 0,1,2,3
    np.testing.assert_array_equal(vertex_ids, [0, 1, 2, 3])
    np.testing.assert_allclose(V_sub, V[vertex_ids])
    np.testing.assert_array_equal(V_sub[T_sub], V[T[[0, 1]]])


def test_get_sub_mesh_single_triangle():
    V = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [5.0, 5.0, 5.0]])
    T = np.array([[0, 1, 2], [1, 2, 3]])
    V_sub, T_sub, vertex_ids = getSubMesh(V, T, [1])
    assert V_sub.shape == (3, 3)
    np.testing.assert_array_equal(vertex_ids, [1, 2, 3])
    np.testing.assert_array_equal(T_sub, [[0, 1, 2]])


# ---------------------------------------------------------------------------
# triangleCentroids (used by Mesh.split / segmentation)
# ---------------------------------------------------------------------------

def test_triangle_centroids_matches_analytical_mean():
    coords = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [1.0, 1.0, 1.0]])
    connect = np.array([[0, 1, 2], [0, 1, 3]])
    centroids = triangleCentroids(coords, connect)
    np.testing.assert_allclose(centroids[0], [1.0, 1.0, 0.0])
    np.testing.assert_allclose(centroids[1], [4.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])


# ---------------------------------------------------------------------------
# tri_plane_intersections_yz_per_triangle (used by Mesh 2D slicing)
# ---------------------------------------------------------------------------

def test_tri_plane_intersections_two_valid_edges():
    tri = np.array([[[0.0, 0.0, 0.0], [2.0, 2.0, 0.0], [2.0, 0.0, 2.0]]])
    xs = np.array([1.0])
    Y, Z, E = tri_plane_intersections_yz_per_triangle(tri, xs)
    pairs = sorted(zip(np.round(Y[0], 6), np.round(Z[0], 6)))
    assert pairs == [(0.0, 1.0), (1.0, 0.0)]
    assert set(E[0]) == {0, 2}  # edges A->B and A->C, not the x-constant edge B->C


def test_tri_plane_intersections_no_crossing_gives_nan():
    tri = np.array([[[5.0, 0.0, 0.0], [6.0, 1.0, 0.0], [6.0, 0.0, 1.0]]])
    xs = np.array([1.0])
    Y, Z, E = tri_plane_intersections_yz_per_triangle(tri, xs)
    assert np.isnan(Y).all()
    assert np.isnan(Z).all()
    assert np.all(E == -1)


# ---------------------------------------------------------------------------
# rotationAlign2x (used by PointCloud, Model, Mesh)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("vector", [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [3.0, 4.0, 5.0]])
def test_rotation_align_2x_maps_vector_to_ex(vector):
    v = np.array(vector)
    R = rotationAlign2x(v)
    d = v / np.linalg.norm(v)
    np.testing.assert_allclose(R @ d, [1.0, 0.0, 0.0], atol=1e-8)
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-8)  # orthogonal
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-8)  # proper rotation


def test_rotation_align_2x_rejects_zero_vector():
    with pytest.raises(ValueError):
        rotationAlign2x(np.zeros(3))


# ---------------------------------------------------------------------------
# unit / normal2d / limit_turn / weighted_centroid
# (small helpers underpinning trace_polyline_nodes / assign_sections_from_nodes)
# ---------------------------------------------------------------------------

def test_unit_normalizes_vector():
    np.testing.assert_allclose(unit(np.array([3.0, 4.0])), [0.6, 0.8])


def test_unit_zero_vector_uses_fallback():
    result = unit(np.array([0.0, 0.0]), fallback=np.array([2.0, 0.0]))
    np.testing.assert_allclose(result, [1.0, 0.0])


def test_unit_zero_vector_no_fallback_raises():
    with pytest.raises(ValueError):
        unit(np.array([0.0, 0.0]))


def test_normal2d_is_perpendicular():
    t = np.array([1.0, 0.0])
    n = normal2d(t)
    assert np.dot(t, n) == pytest.approx(0.0)


def test_limit_turn_within_budget_returns_unclamped_direction():
    e_new = limit_turn(np.array([1.0, 0.1]), np.array([1.0, 0.0]), max_turn_angle=np.deg2rad(30))
    np.testing.assert_allclose(e_new, unit(np.array([1.0, 0.1])))


def test_limit_turn_clamps_to_max_angle():
    e_old = np.array([1.0, 0.0])
    e_new = limit_turn(np.array([0.0, 1.0]), e_old, max_turn_angle=np.deg2rad(30))
    cosang = np.dot(e_old, e_new)
    assert np.arccos(np.clip(cosang, -1, 1)) == pytest.approx(np.deg2rad(30), abs=1e-6)
    assert np.linalg.norm(e_new) == pytest.approx(1.0)


def test_weighted_centroid_power_zero_is_plain_mean():
    xy = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0]])
    c = weighted_centroid(xy, centre=np.array([0.0, 0.0]), power=0.0)
    np.testing.assert_allclose(c, [1.0, 1.0])


# ---------------------------------------------------------------------------
# profile_geometry_from_nodes (used by _pointcloud.py)
# ---------------------------------------------------------------------------

def test_profile_geometry_from_nodes_midpoints_and_perpendicular_dirs():
    nodes = np.array([[0.0, 0.0], [3.0, 0.0], [3.0, 4.0]])
    centres, tangents, section_dirs = profile_geometry_from_nodes(nodes)

    np.testing.assert_allclose(centres, [[1.5, 0.0], [3.0, 2.0]])
    np.testing.assert_allclose(tangents, [[1.0, 0.0], [0.0, 1.0]])
    # section directions must be unit and perpendicular to their tangent
    for t, s in zip(tangents, section_dirs):
        assert np.dot(t, s) == pytest.approx(0.0, abs=1e-8)
        assert np.linalg.norm(s) == pytest.approx(1.0)


def test_profile_geometry_from_nodes_requires_at_least_two_nodes():
    with pytest.raises(ValueError, match="At least two polyline nodes"):
        profile_geometry_from_nodes(np.array([[0.0, 0.0]]))


# ---------------------------------------------------------------------------
# pcWalk / trace_polyline_nodes / assign_sections_from_nodes
# (used by PointCloud; smoke tests on a synthetic, noisy straight-line cloud)
# ---------------------------------------------------------------------------

def _noisy_line_xy(n=400, length=20.0, noise=0.1, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, length, n)
    return np.column_stack([t, noise * rng.standard_normal(n)])


def test_pcwalk_follows_a_noisy_straight_line():
    xy = _noisy_line_xy()
    p_init = np.array([0.0, 0.0])
    p_end = np.array([20.0, 0.0])
    path = pcWalk(p_init, p_end, xy, dlt_x=np.full(xy.shape[0], 0.5), dlt_z=1.0)

    assert path.shape[1] == 2
    np.testing.assert_allclose(path[0], p_init, atol=1e-8)
    np.testing.assert_allclose(path[-1], p_end, atol=1e-8)
    # progress is broadly monotonic in x and stays near the line (y small)
    assert np.all(np.diff(path[:, 0]) > -1e-6)
    assert np.max(np.abs(path[:, 1])) < 1.0


def test_pcwalk_returns_direct_pair_when_already_close():
    p_init = np.array([0.0, 0.0])
    p_end = np.array([0.5, 0.0])
    xy = _noisy_line_xy()
    path = pcWalk(p_init, p_end, xy, dlt_x=np.full(xy.shape[0], 0.5), dlt_z=1.0)
    np.testing.assert_array_equal(path, [p_init, p_end])


def test_trace_polyline_nodes_follows_a_noisy_straight_line():
    xy = _noisy_line_xy()
    nodes = trace_polyline_nodes(
        xy, increment=1.0, transverse_radius=2.0, buffer_steps=2, initial_direction=(1.0, 0.0)
    )
    assert nodes.shape[1] == 2
    assert nodes.shape[0] > 5
    # consecutive nodes advance roughly `increment` along x and stay near y=0
    step_lengths = np.linalg.norm(np.diff(nodes, axis=0), axis=1)
    np.testing.assert_allclose(step_lengths, 1.0, atol=0.05)
    assert np.max(np.abs(nodes[:, 1])) < 1.0


def test_assign_sections_from_nodes_splits_points_by_projected_position():
    xy = _noisy_line_xy(n=500, length=10.0, seed=1)
    points3d = np.column_stack([xy, np.zeros(xy.shape[0])])
    nodes = np.array([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]])

    section_ids, centres, tangents, section_dirs = assign_sections_from_nodes(
        points3d, nodes, transverse_radius=1.0, min_points=5
    )

    assert len(section_ids) == 2
    np.testing.assert_allclose(centres, [[2.5, 0.0], [7.5, 0.0]])
    for idx, (lo, hi) in zip(section_ids, [(0.0, 5.0), (5.0, 10.0)]):
        assert idx.size > 0
        xs = xy[idx, 0]
        assert xs.min() >= lo - 0.5
        assert xs.max() <= hi + 0.5
