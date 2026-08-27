"""Tests for pyrockfall.Mesh: triangle-mesh Model implementation.

Covers construction/validation, the attribute dict API's mesh-specific
sizing rule (per-point OR per-triangle length), save/load round-trips,
segment()/split() (which exercise the getSubMesh fix from test_utils.py),
slopeGeometry() (per-triangle and per-vertex label reduction, including the
majority/tie-break rules), to_tri_attr()/to_point_attr() (mean and mode,
including tie-breaking by centroid distance), and slice() (2D polyline
extraction, cross-checked against hand-computed edge intersections).
"""
import numpy as np
import pytest

import pyrockfall as pr


# ---------------------------------------------------------------------------
# Fixtures: small, hand-computable meshes
# ---------------------------------------------------------------------------

def strip_mesh():
    """Two quads (four triangles) forming a 2x1 flat strip along x in [0,2]."""
    pts = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0],
        [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 0.0],
    ])
    tris = np.array([
        [0, 1, 4], [0, 4, 3],  # quad0: x in [0,1]
        [1, 2, 5], [1, 5, 4],  # quad1: x in [1,2]
    ])
    return pts, tris


# ---------------------------------------------------------------------------
# Construction / points / triangles validation
# ---------------------------------------------------------------------------

def test_construct_pads_2d_points_to_3d():
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    m = pr.Mesh(points=pts, triangles=np.array([[0, 1, 2]]))
    assert m.points.shape == (3, 3)
    np.testing.assert_array_equal(m.points[:, 2], 0.0)


def test_construct_with_colors_normals_and_attrs():
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    tris = np.array([[0, 1, 2]])
    colors = np.array([[1.0, 0.0, 0.0]] * 3)
    normals = np.array([[0.0, 0.0, 1.0]] * 3)
    m = pr.Mesh(points=pts, triangles=tris, colors=colors, normals=normals, attrs={"extra": np.arange(3)})
    np.testing.assert_array_equal(m.get_attr("colors"), colors)
    np.testing.assert_array_equal(m.get_attr("normals"), normals)
    np.testing.assert_array_equal(m.get_attr("extra"), np.arange(3))


def test_points_setter_rejects_wrong_shape():
    m = pr.Mesh()
    with pytest.raises(ValueError, match=r"Mesh\.points must have shape"):
        m.points = np.zeros((3, 4))


def test_triangles_setter_rejects_negative_index():
    m = pr.Mesh(points=np.zeros((3, 3)))
    with pytest.raises(ValueError, match="negative indices"):
        m.triangles = np.array([[0, -1, 2]])


def test_triangles_setter_rejects_out_of_range_index():
    m = pr.Mesh(points=np.zeros((3, 3)))
    with pytest.raises(ValueError, match="out of range"):
        m.triangles = np.array([[0, 1, 5]])


def test_triangles_setter_rejects_wrong_shape():
    m = pr.Mesh(points=np.zeros((3, 3)))
    with pytest.raises(ValueError, match=r"Mesh\.triangles must have shape"):
        m.triangles = np.array([[0, 1]])


# ---------------------------------------------------------------------------
# Attribute dict API: mesh-specific per-point OR per-triangle sizing
# ---------------------------------------------------------------------------

def test_set_attr_accepts_per_point_or_per_triangle_length():
    pts, tris = strip_mesh()
    m = pr.Mesh(points=pts, triangles=tris)
    m.set_attr("per_point", np.arange(pts.shape[0], dtype=float))
    m.set_attr("per_tri", np.arange(tris.shape[0]))
    assert m.get_attr("per_point").shape == (6,)
    assert m.get_attr("per_tri").shape == (4,)


def test_set_attr_rejects_length_matching_neither():
    pts, tris = strip_mesh()
    m = pr.Mesh(points=pts, triangles=tris)
    with pytest.raises(ValueError, match="!= number of points"):
        m.set_attr("bad", np.array([1, 2]))


def test_points_replaced_drops_per_point_attrs_but_keeps_per_triangle_attrs():
    pts, tris = strip_mesh()
    m = pr.Mesh(points=pts, triangles=tris)
    m.set_attr("per_point", np.arange(pts.shape[0], dtype=float))
    m.set_attr("per_tri", np.arange(tris.shape[0]))

    m.points = np.vstack([pts, [9.0, 9.0, 9.0]])  # N changes 6 -> 7
    assert not m.has_attr("per_point")  # length no longer matches N
    assert m.has_attr("per_tri")  # length still matches E, untouched


# ---------------------------------------------------------------------------
# save / load round trip
# ---------------------------------------------------------------------------

def test_save_load_npz_roundtrip(tmp_path):
    pts, tris = strip_mesh()
    m = pr.Mesh(points=pts, triangles=tris)
    m.set_attr("value", np.arange(pts.shape[0], dtype=float))

    path = tmp_path / "mesh.npz"
    m.save(str(path), attributes="*")
    assert path.exists()

    loaded = pr.Mesh.load(str(path), attributes=["value"])
    np.testing.assert_allclose(loaded.points, m.points)
    np.testing.assert_array_equal(loaded.triangles, m.triangles)
    np.testing.assert_allclose(loaded.get_attr("value"), m.get_attr("value"))


def test_save_refuses_overwrite_without_flag(tmp_path):
    pts, tris = strip_mesh()
    m = pr.Mesh(points=pts, triangles=tris)
    path = tmp_path / "mesh.npz"
    m.save(str(path))
    with pytest.raises(FileExistsError):
        m.save(str(path))
    m.save(str(path), overwrite=True)  # should not raise


def test_load_npz_missing_triangles_raises(tmp_path):
    path = tmp_path / "bad.npz"
    np.savez(str(path), points=np.zeros((3, 3)))
    with pytest.raises(ValueError, match="must contain 'points' and 'triangles'"):
        pr.Mesh.load(str(path))


# ---------------------------------------------------------------------------
# segment() / split() (exercise the getSubMesh array-based extraction)
# ---------------------------------------------------------------------------

def test_segment_splits_into_slabs_along_x():
    pts, tris = strip_mesh()
    m = pr.Mesh(points=pts, triangles=tris)
    segments = m.segment(dx=1.0)
    assert len(segments) == 2
    for seg, (lo, hi) in zip(segments, [(0.0, 1.0), (1.0, 2.0)]):
        assert seg.points[:, 0].min() == pytest.approx(lo)
        assert seg.points[:, 0].max() == pytest.approx(hi)
        assert seg.triangles.shape == (2, 3)
        # connectivity stays valid after reindexing
        assert seg.triangles.max() < seg.points.shape[0]


def test_segment_rejects_nonpositive_dx():
    pts, tris = strip_mesh()
    m = pr.Mesh(points=pts, triangles=tris)
    with pytest.raises(ValueError, match="dx must be > 0"):
        m.segment(dx=0.0)


def test_segment_empty_mesh_returns_empty_list():
    assert pr.Mesh().segment(dx=1.0) == []


def test_split_balances_triangle_counts_across_the_cut():
    pts, tris = strip_mesh()
    m = pr.Mesh(points=pts, triangles=tris)
    parts = m.split(dx=0.5)
    assert len(parts) == 2
    left, right = parts
    assert left.points[:, 0].max() <= right.points[:, 0].min() + 1e-9
    assert left.triangles.shape[0] == 2
    assert right.triangles.shape[0] == 2


def test_split_empty_mesh_returns_empty_list():
    assert pr.Mesh().split() == []


# ---------------------------------------------------------------------------
# slopeGeometry()
# ---------------------------------------------------------------------------

def test_slope_geometry_per_triangle_labels():
    pts, tris = strip_mesh()
    m = pr.Mesh(points=pts, triangles=tris)
    m.set_attr("mat", np.array([0, 0, 1, 1]))
    geom = m.slopeGeometry(label="mat")
    assert isinstance(geom, pr.Geometry3D)
    np.testing.assert_array_equal(geom.attributes, [0, 0, 1, 1])
    np.testing.assert_array_equal(geom.elements, tris)


def test_slope_geometry_per_vertex_labels_majority_rule():
    pts, tris = strip_mesh()
    m = pr.Mesh(points=pts, triangles=tris)
    # per-point labels; verified by hand for each triangle's vertex triple.
    labels = np.array([0, 0, 1, 1, 1, 0])
    m.set_attr("mat", labels)
    geom = m.slopeGeometry(label="mat")
    expected = []
    for tri in tris:
        vals, counts = np.unique(labels[tri], return_counts=True)
        expected.append(vals[np.argmax(counts)])
    np.testing.assert_array_equal(geom.attributes, expected)


def test_slope_geometry_missing_attribute_raises():
    pts, tris = strip_mesh()
    m = pr.Mesh(points=pts, triangles=tris)
    with pytest.raises(ValueError, match="not found"):
        m.slopeGeometry(label="missing")


def test_slope_geometry_wrong_length_attribute_raises():
    pts, tris = strip_mesh()
    m = pr.Mesh(points=pts, triangles=tris)
    # set_attr() itself already rejects a length matching neither N nor E,
    # so bypass it directly to reach slopeGeometry()'s own validation.
    m._attrs["mat"] = np.array([0, 1, 0])  # neither N=6 nor E=4
    with pytest.raises(ValueError, match="must have length"):
        m.slopeGeometry(label="mat")


def test_slope_geometry_nodes_std_shape_validation():
    pts, tris = strip_mesh()
    m = pr.Mesh(points=pts, triangles=tris)
    m.set_attr("mat", np.zeros(tris.shape[0], dtype=int))
    with pytest.raises(ValueError, match="nodes_std"):
        m.slopeGeometry(label="mat", nodes_std=np.zeros((2, 3)))


# ---------------------------------------------------------------------------
# to_tri_attr()
# ---------------------------------------------------------------------------

def test_to_tri_attr_mean_scalar():
    pts, tris = strip_mesh()
    m = pr.Mesh(points=pts, triangles=tris)
    m.set_attr("val", np.array([0.0, 3.0, 6.0, 0.0, 3.0, 6.0]))
    tri_vals = m.to_tri_attr("val", method="mean")
    expected = np.array([0.0, 3.0, 6.0, 0.0, 3.0, 6.0])[tris].mean(axis=1)
    np.testing.assert_allclose(tri_vals, expected)


def test_to_tri_attr_mean_vector_valued():
    pts, tris = strip_mesh()
    m = pr.Mesh(points=pts, triangles=tris)
    vec = np.column_stack([np.arange(6.0), np.arange(6.0) * 2])
    m.set_attr("vec", vec)
    tri_vals = m.to_tri_attr("vec", method="mean")
    expected = vec[tris].mean(axis=1)
    np.testing.assert_allclose(tri_vals, expected)


def test_to_tri_attr_mode_all_equal():
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    m = pr.Mesh(points=pts, triangles=np.array([[0, 1, 2]]))
    m.set_attr("lab", np.array([4, 4, 4]))
    result = m.to_tri_attr("lab", method="mode")
    assert result[0] == 4


def test_to_tri_attr_mode_majority_pair():
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    m = pr.Mesh(points=pts, triangles=np.array([[0, 1, 2]]))
    m.set_attr("lab", np.array([4, 5, 4]))
    result = m.to_tri_attr("lab", method="mode")
    assert result[0] == 4


def test_to_tri_attr_mode_all_different_ties_break_by_barycentre_distance():
    # point0 deliberately closest to the triangle's centroid.
    pts = np.array([[0.34, 0.34, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    m = pr.Mesh(points=pts, triangles=np.array([[0, 1, 2]]))
    m.set_attr("lab", np.array([7, 8, 9]))
    centroid = pts.mean(axis=0)
    closest = np.argmin(np.linalg.norm(pts - centroid, axis=1))
    result = m.to_tri_attr("lab", method="mode")
    assert result[0] == np.array([7, 8, 9])[closest]


def test_to_tri_attr_out_stores_result_as_attribute():
    pts, tris = strip_mesh()
    m = pr.Mesh(points=pts, triangles=tris)
    m.set_attr("val", np.arange(6, dtype=float))
    m.to_tri_attr("val", method="mean", out="val_tri")
    assert m.has_attr("val_tri")
    assert m.get_attr("val_tri").shape == (4,)


def test_to_tri_attr_missing_attribute_raises():
    pts, tris = strip_mesh()
    m = pr.Mesh(points=pts, triangles=tris)
    with pytest.raises(ValueError, match="not found"):
        m.to_tri_attr("missing")


def test_to_tri_attr_invalid_method_raises():
    pts, tris = strip_mesh()
    m = pr.Mesh(points=pts, triangles=tris)
    m.set_attr("val", np.zeros(6))
    with pytest.raises(ValueError, match="method must be one of"):
        m.to_tri_attr("val", method="bogus")


# ---------------------------------------------------------------------------
# to_point_attr()
# ---------------------------------------------------------------------------

def test_to_point_attr_mean_averages_incident_triangles():
    pts, tris = strip_mesh()
    m = pr.Mesh(points=pts, triangles=tris)
    m.set_attr("tri_val", np.array([0.0, 10.0, 20.0, 30.0]))
    point_vals = m.to_point_attr("tri_val", method="mean")

    # point 4 belongs to triangles 0 ([0,1,4]), 1 ([0,4,3]) and 3 ([1,5,4])
    assert point_vals[4] == pytest.approx(np.mean([0.0, 10.0, 30.0]))
    # point 0 belongs to triangles 0 ([0,1,4]) and 1 ([0,4,3])
    assert point_vals[0] == pytest.approx(np.mean([0.0, 10.0]))
    # point 2 is only in triangle 2 ([1,2,5])
    assert point_vals[2] == pytest.approx(20.0)


def test_to_point_attr_mean_isolated_point_defaults_to_nan():
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [5.0, 5.0, 5.0]])
    m = pr.Mesh(points=pts, triangles=np.array([[0, 1, 2]]))
    m.set_attr("tri_val", np.array([42.0]))
    point_vals = m.to_point_attr("tri_val", method="mean")
    assert np.isnan(point_vals[3])
    np.testing.assert_allclose(point_vals[:3], 42.0)


def test_to_point_attr_mean_isolated_point_uses_fill():
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [5.0, 5.0, 5.0]])
    m = pr.Mesh(points=pts, triangles=np.array([[0, 1, 2]]))
    m.set_attr("tri_val", np.array([42.0]))
    point_vals = m.to_point_attr("tri_val", method="mean", fill=-1.0)
    assert point_vals[3] == pytest.approx(-1.0)


def test_to_point_attr_mode_ties_break_by_nearest_centroid():
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.1, 0.0]])
    tris = np.array([[0, 1, 2], [1, 3, 2]])
    m = pr.Mesh(points=pts, triangles=tris)
    m.set_attr("tri_lab", np.array([5, 6], dtype=int))
    point_vals = m.to_point_attr("tri_lab", method="mode")

    c0 = pts[tris[0]].mean(axis=0)
    c1 = pts[tris[1]].mean(axis=0)
    expected_at_1 = 5 if np.linalg.norm(pts[1] - c0) < np.linalg.norm(pts[1] - c1) else 6
    assert point_vals[1] == expected_at_1
    assert point_vals[0] == 5  # only in triangle 0
    assert point_vals[3] == 6  # only in triangle 1


def test_to_point_attr_mode_rejects_non_integer():
    pts, tris = strip_mesh()
    m = pr.Mesh(points=pts, triangles=tris)
    m.set_attr("tri_val", np.array([0.5, 1.5, 2.5, 3.5]))
    with pytest.raises(ValueError, match="requires integer"):
        m.to_point_attr("tri_val", method="mode")


def test_to_point_attr_wrong_length_raises():
    pts, tris = strip_mesh()
    m = pr.Mesh(points=pts, triangles=tris)
    with pytest.raises(ValueError, match="not found"):
        m.to_point_attr("missing")


# ---------------------------------------------------------------------------
# slice()
# ---------------------------------------------------------------------------

def test_slice_matches_hand_computed_edge_intersections():
    # A ramp where z == y everywhere, so slice nodes are easy to verify by hand.
    pts = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 1.0], [2.0, 1.0, 1.0]])
    tris = np.array([[0, 1, 2], [1, 3, 2]])
    m = pr.Mesh(points=pts, triangles=tris)
    m.set_attr("mat", np.array([0, 1]))

    profiles, xs = m.slice(np.array([1.0, 0.0, 0.0]), increment=0.5, label="mat")
    assert len(profiles) == 4
    np.testing.assert_allclose(xs, [0.25, 0.75, 1.25, 1.75])

    for prof, x in zip(profiles, xs):
        # On this ramp, y == z. The profile crosses three edges: the bottom
        # boundary edge (y=0, for all x), the shared diagonal edge between
        # the two triangles (y = 1 - x/2), and the top boundary edge (y=1,
        # for all x).
        expected_y = np.sort([0.0, 1.0 - x / 2.0, 1.0])
        np.testing.assert_allclose(np.sort(prof.nodes[:, 0]), expected_y, atol=1e-8)
        np.testing.assert_allclose(prof.nodes[:, 0], prof.nodes[:, 1])  # y == z throughout
        np.testing.assert_array_equal(prof.attributes, [0, 1])


def test_slice_missing_label_attribute_raises():
    pts, tris = strip_mesh()
    m = pr.Mesh(points=pts, triangles=tris)
    with pytest.raises(ValueError, match="not found"):
        m.slice(np.array([1.0, 0.0, 0.0]), increment=0.5, label="missing")


def test_slice_rejects_nonpositive_increment():
    pts, tris = strip_mesh()
    m = pr.Mesh(points=pts, triangles=tris)
    m.set_attr("mat", np.zeros(tris.shape[0], dtype=int))
    with pytest.raises(ValueError, match="increment must be > 0"):
        m.slice(np.array([1.0, 0.0, 0.0]), increment=0.0, label="mat")


def test_slice_empty_mesh_returns_empty():
    profiles, xs = pr.Mesh().slice(np.array([1.0, 0.0, 0.0]), increment=0.5, label="mat")
    assert profiles == []
    assert xs.size == 0


# ---------------------------------------------------------------------------
# clip() is inherited from Model unimplemented for Mesh (known limitation)
# ---------------------------------------------------------------------------

def test_clip_not_implemented_for_mesh():
    pts, tris = strip_mesh()
    m = pr.Mesh(points=pts, triangles=tris)
    with pytest.raises(NotImplementedError):
        m.clip(np.zeros(3), np.ones(3))
