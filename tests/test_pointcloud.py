"""Tests for pyrockfall.PointCloud: unstructured point-set Model implementation.

Covers construction/validation, the attribute dict API, clip(), resolution(),
save/load round-trips, and the higher-level geometric operations
(segment(), split(), slice(), section()) using synthetic point clouds shaped
so the underlying algorithms (cone fitting, plane fitting, KNN
classification, polyline tracing) behave in a well-defined, checkable way
rather than degenerating.
"""
import numpy as np
import pytest

import pyrockfall as pr


# ---------------------------------------------------------------------------
# Construction / points validation
# ---------------------------------------------------------------------------

def test_construct_pads_2d_points_to_3d():
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    pc = pr.PointCloud(points=pts)
    assert pc.points.shape == (3, 3)
    np.testing.assert_array_equal(pc.points[:, 2], 0.0)


def test_construct_with_colors_normals_and_attrs():
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    colors = np.array([[1.0, 0.0, 0.0]] * 3)
    normals = np.array([[0.0, 0.0, 1.0]] * 3)
    pc = pr.PointCloud(points=pts, colors=colors, normals=normals, attrs={"extra": np.arange(3)})
    np.testing.assert_array_equal(pc.get_attr("colors"), colors)
    np.testing.assert_array_equal(pc.get_attr("normals"), normals)
    np.testing.assert_array_equal(pc.get_attr("extra"), np.arange(3))
    assert pc.has_colors() and pc.has_normals()


def test_empty_pointcloud_has_no_colors_or_normals():
    pc = pr.PointCloud()
    assert not pc.has_colors()
    assert not pc.has_normals()


def test_points_setter_rejects_wrong_shape():
    pc = pr.PointCloud()
    with pytest.raises(ValueError, match=r"PointCloud\.points must be"):
        pc.points = np.zeros((3, 4))


def test_points_replaced_drops_stale_attrs():
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    pc = pr.PointCloud(points=pts)
    pc.set_attr("value", np.array([1.0, 2.0, 3.0]))
    pc.points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])  # N: 3 -> 2
    assert not pc.has_attr("value")


# ---------------------------------------------------------------------------
# clip()
# ---------------------------------------------------------------------------

def test_clip_keeps_points_within_aabb_and_reindexes_attrs():
    xs, ys = np.meshgrid(np.arange(5), np.arange(5))
    pts = np.column_stack([xs.ravel(), ys.ravel(), np.zeros(25)]).astype(float)
    pc = pr.PointCloud(points=pts)
    pc.set_attr("id", np.arange(25))

    pc.clip(np.array([1.0, 1.0, -1.0]), np.array([3.0, 3.0, 1.0]))

    assert pc.points.shape == (9, 3)
    assert np.all((pc.points[:, 0] >= 1) & (pc.points[:, 0] <= 3))
    assert np.all((pc.points[:, 1] >= 1) & (pc.points[:, 1] <= 3))
    np.testing.assert_array_equal(pc.get_attr("id"), [6, 7, 8, 11, 12, 13, 16, 17, 18])


def test_clip_empty_pointcloud_is_a_no_op():
    pc = pr.PointCloud()
    pc.clip(np.zeros(3), np.ones(3))
    assert pc.points.shape == (0, 3)


def test_clip_rejects_wrong_shaped_bounds():
    pc = pr.PointCloud(points=np.zeros((3, 3)))
    with pytest.raises(ValueError, match=r"shape \(3,\)"):
        pc.clip(np.zeros(2), np.ones(3))


# ---------------------------------------------------------------------------
# resolution()
# ---------------------------------------------------------------------------

def test_resolution_matches_regular_grid_spacing():
    xs, ys = np.meshgrid(np.arange(6), np.arange(6))
    pts = np.column_stack([xs.ravel(), ys.ravel(), np.zeros(36)]).astype(float)
    pc = pr.PointCloud(points=pts)
    assert pc.resolution() == pytest.approx(1.0, rel=1e-6)


def test_resolution_scales_with_spacing():
    xs, ys = np.meshgrid(np.arange(6) * 2.0, np.arange(6) * 2.0)
    pts = np.column_stack([xs.ravel(), ys.ravel(), np.zeros(36)])
    pc = pr.PointCloud(points=pts)
    assert pc.resolution() == pytest.approx(2.0, rel=1e-6)


def test_resolution_single_point_is_zero():
    pc = pr.PointCloud(points=np.array([[0.0, 0.0, 0.0]]))
    assert pc.resolution() == 0.0


def test_resolution_is_cached():
    pts = np.random.default_rng(0).uniform(size=(20, 3))
    pc = pr.PointCloud(points=pts)
    r1 = pc.resolution()
    r2 = pc.resolution()
    assert r1 == r2


# ---------------------------------------------------------------------------
# save / load round trip
# ---------------------------------------------------------------------------

def test_save_load_npz_roundtrip(tmp_path):
    pts = np.random.default_rng(0).uniform(size=(10, 3))
    pc = pr.PointCloud(points=pts)
    pc.set_attr("colors", np.random.default_rng(1).uniform(size=(10, 3)))
    pc.set_attr("normals", np.random.default_rng(2).uniform(size=(10, 3)))

    path = tmp_path / "cloud.npz"
    pc.save(str(path), attributes="*")
    assert path.exists()

    loaded = pr.PointCloud.load(str(path), attributes=["colors", "normals"])
    np.testing.assert_allclose(loaded.points, pc.points)
    assert loaded.has_colors() and loaded.has_normals()
    np.testing.assert_allclose(loaded.get_attr("colors"), pc.get_attr("colors"))
    np.testing.assert_allclose(loaded.get_attr("normals"), pc.get_attr("normals"))


def test_save_load_ply_roundtrip(tmp_path):
    pts = np.random.default_rng(0).uniform(size=(10, 3)).astype(np.float32).astype(float)
    pc = pr.PointCloud(points=pts)

    path = tmp_path / "cloud.ply"
    pc.save(str(path))
    assert path.exists()

    loaded = pr.PointCloud.load(str(path))
    np.testing.assert_allclose(loaded.points, pc.points, atol=1e-5)


def test_save_refuses_overwrite_without_flag(tmp_path):
    pc = pr.PointCloud(points=np.zeros((3, 3)))
    path = tmp_path / "cloud.npz"
    pc.save(str(path))
    with pytest.raises(FileExistsError):
        pc.save(str(path))
    pc.save(str(path), overwrite=True)  # should not raise


def test_save_unsupported_extension_raises(tmp_path):
    pc = pr.PointCloud(points=np.zeros((3, 3)))
    with pytest.raises(ValueError, match="Unsupported extension"):
        pc.save(str(tmp_path / "cloud.foo"))


# ---------------------------------------------------------------------------
# _from_indices (exercised directly and via segment()/split())
# ---------------------------------------------------------------------------

def test_from_indices_preserves_selected_points_and_attrs():
    pts = np.arange(15.0).reshape(5, 3)
    pc = pr.PointCloud(points=pts)
    pc.set_attr("id", np.arange(5))
    sub = pc._from_indices(np.array([0, 2, 4]))
    np.testing.assert_allclose(sub.points, pts[[0, 2, 4]])
    np.testing.assert_array_equal(sub.get_attr("id"), [0, 2, 4])
    assert sub is not pc


# ---------------------------------------------------------------------------
# segment(): requires a genuinely cone-shaped point cloud for coneApproximation
# to converge to a finite, nonzero radius.
# ---------------------------------------------------------------------------

def cone_wall_points(seed=0, n_theta=60, n_rho=30):
    """Points on a true cone (z linear in radial distance from an apex)."""
    theta = np.linspace(-0.6, 0.6, n_theta)
    rho = np.linspace(20.0, 40.0, n_rho)
    TH, RHO = np.meshgrid(theta, rho)
    apex = np.array([5.0, -3.0])
    x = apex[0] + RHO * np.cos(TH)
    y = apex[1] + RHO * np.sin(TH)
    z = 2.0 * RHO
    return np.column_stack([x.ravel(), y.ravel(), z.ravel()])


def test_segment_partitions_a_cone_shaped_wall():
    pts = cone_wall_points()
    pc = pr.PointCloud(points=pts)
    segments = pc.segment(ds=5.0)
    assert len(segments) > 1
    total = sum(s.points.shape[0] for s in segments)
    # nearly all points are assigned to some segment (boundary points of the
    # arc-length windowing may be dropped)
    assert total >= 0.9 * pts.shape[0]
    for seg in segments:
        assert seg.points.shape[0] > 0


def test_segment_zero_radius_raises():
    # A flat, non-conical point cloud: z has no linear relationship with
    # radial distance from any (x0,y0), so the cone fit's slope stays ~0
    # and radius blows up/degenerates -- exercised here as a documented
    # failure mode, not asserted to be "correct" behaviour.
    xs, ys = np.meshgrid(np.linspace(0, 10, 20), np.linspace(0, 10, 20))
    pts = np.column_stack([xs.ravel(), ys.ravel(), np.zeros(400)])
    pc = pr.PointCloud(points=pts)
    with pytest.raises((ValueError, ZeroDivisionError)):
        pc.segment(ds=1.0)


# ---------------------------------------------------------------------------
# split()
# ---------------------------------------------------------------------------

def kinked_wall_points(seed=1, n=1500):
    """A wall with a visible kink at x=5, so the two halves' dip directions differ."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 10, n)
    z = rng.uniform(0, 5, n)
    y = np.where(x < 5, 0.3 * x, 0.3 * (10 - x)) + 0.02 * rng.standard_normal(n)
    return np.column_stack([x, y, z])


def test_split_produces_two_nonoverlapping_x_ranges():
    pts = kinked_wall_points()
    pc = pr.PointCloud(points=pts)
    parts = pc.split(dx=0.5)
    assert len(parts) == 2
    left, right = parts
    assert left.points.shape[0] > 0 and right.points.shape[0] > 0
    assert left.points[:, 0].max() <= right.points[:, 0].min() + 1e-9
    assert left.points.shape[0] + right.points.shape[0] == pts.shape[0]


def test_split_empty_pointcloud_returns_empty_list():
    assert pr.PointCloud().split() == []


# ---------------------------------------------------------------------------
# slice(): end-to-end on a synthetic two-material vertical wall
# ---------------------------------------------------------------------------

def two_material_wall(seed=0, n=2000, x_extent=10.0, z_extent=5.0, split_z=2.5):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, x_extent, n)
    z = rng.uniform(0, z_extent, n)
    y = 0.02 * rng.standard_normal(n)
    pts = np.column_stack([x, y, z])
    mat = (z > split_z).astype(int)
    return pts, mat


def test_slice_produces_profiles_spanning_full_height_with_correct_materials():
    pts, mat = two_material_wall()
    pc = pr.PointCloud(points=pts)
    pc.set_attr("mat", mat)

    profiles, xs = pc.slice(np.array([1.0, 0.0, 0.0]), increment=1.0, label="mat")
    assert len(profiles) > 0
    np.testing.assert_allclose(np.diff(xs), 1.0, atol=1e-6)

    for prof in profiles:
        heights = prof.nodes[:, 1]  # column 1 = z (height) in the (y,z) slice plane
        assert heights.min() < 0.5  # reaches near the bottom
        assert heights.max() > two_material_wall.__defaults__[3] - 0.5  # reaches near the top

        # segment midpoint heights, matched against their assigned material
        mid_heights = 0.5 * (heights[:-1] + heights[1:])
        upper = prof.attributes == 1
        lower = prof.attributes == 0
        if np.any(upper) and np.any(lower):
            assert mid_heights[upper].mean() > mid_heights[lower].mean()


def test_slice_missing_label_raises():
    pts, mat = two_material_wall()
    pc = pr.PointCloud(points=pts)
    with pytest.raises(ValueError, match="not found"):
        pc.slice(np.array([1.0, 0.0, 0.0]), increment=1.0, label="mat")


def test_slice_rejects_nonpositive_increment():
    pts, mat = two_material_wall()
    pc = pr.PointCloud(points=pts)
    pc.set_attr("mat", mat)
    with pytest.raises(ValueError, match="increment must be > 0"):
        pc.slice(np.array([1.0, 0.0, 0.0]), increment=0.0, label="mat")


def test_slice_empty_pointcloud_returns_empty():
    profiles, xs = pr.PointCloud().slice(np.array([1.0, 0.0, 0.0]), increment=1.0, label="mat")
    assert profiles == []
    assert xs.size == 0


# ---------------------------------------------------------------------------
# section(): end-to-end smoke test (heuristic strike-tracing pipeline)
# ---------------------------------------------------------------------------

def test_section_produces_profiles_and_restores_original_state():
    pts, mat = two_material_wall(n=3000, x_extent=20.0)
    pc = pr.PointCloud(points=pts)
    pc.set_attr("mat", mat)
    original_points = pts.copy()

    profiles, sections_pc = pc.section(increment=2.0, label="mat", transverse_radius=1.0, min_points=10)

    assert len(profiles) > 0
    assert sections_pc.points.shape[0] > 0
    assert sections_pc.has_attr("mat")
    assert sections_pc.has_attr("Profile")
    for prof in profiles:
        assert prof.nodes.shape[0] >= 2
        assert len(prof.attributes) == prof.nodes.shape[0] - 1

    # the original point cloud (points + attribute) must be restored after
    # the internal rotate/translate round trip
    np.testing.assert_allclose(pc.points, original_points, atol=1e-6)
    np.testing.assert_array_equal(pc.get_attr("mat"), mat)


def test_section_degenerate_input_fails_cleanly():
    # A tiny, unstructured point cloud can't satisfy an unreasonably large
    # min_points requirement anywhere in the pipeline (polyline tracing or
    # section assignment); either failure is an acceptable, clean rejection
    # of a degenerate request.
    pc = pr.PointCloud(points=np.random.default_rng(0).uniform(size=(50, 3)))
    pc.set_attr("mat", np.zeros(50, dtype=int))
    with pytest.raises((ValueError, RuntimeError)):
        pc.section(increment=1.0, label="mat", transverse_radius=0.01, min_points=1000)
