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
# segment(): windows along the strike (s coordinate of the curvilinear frame)
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


def test_segment_flat_cloud_is_partitioned_exactly():
    # A flat point cloud has no cone-like shape; windows along the strike
    # still partition it without losing or duplicating points.
    xs, ys = np.meshgrid(np.linspace(0, 10, 20), np.linspace(0, 4, 8))
    pts = np.column_stack([xs.ravel(), ys.ravel(), np.zeros(160)])
    pc = pr.PointCloud(points=pts)
    pc.set_attr("id", np.arange(160))
    segments = pc.segment(ds=2.5)
    assert len(segments) == 4
    ids = np.concatenate([seg.get_attr("id") for seg in segments])
    np.testing.assert_array_equal(np.sort(ids), np.arange(160))
    for a, b in zip(segments[:-1], segments[1:]):
        assert a.points[:, 0].max() <= b.points[:, 0].min() + 1e-9


def test_segment_overlap_duplicates_points_between_windows():
    xs, ys = np.meshgrid(np.linspace(0, 10, 41), np.linspace(0, 4, 8))
    pts = np.column_stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)])
    pc = pr.PointCloud(points=pts)
    plain = sum(seg.points.shape[0] for seg in pc.segment(ds=2.5))
    overlapped = sum(seg.points.shape[0] for seg in pc.segment(ds=2.5, du=1.0))
    assert plain == pts.shape[0]
    assert overlapped > plain


def test_segment_rejects_invalid_window():
    pc = pr.PointCloud(points=np.random.default_rng(0).uniform(size=(50, 3)))
    with pytest.raises(ValueError, match="ds must be > 0"):
        pc.segment(ds=0.0)
    with pytest.raises(ValueError, match="du must satisfy"):
        pc.segment(ds=1.0, du=1.0)


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


# ---------------------------------------------------------------------------
# Strike-based slicing (curvilinear frame): a wall with a 90 degree corner
# ---------------------------------------------------------------------------

def corner_wall(n=12000, seed=0, leg=30.0, height=12.0, dip_deg=60.0):
    """Wall dipping `dip_deg` that turns 90 degrees half-way, with two layers."""
    rng = np.random.default_rng(seed)
    truth = pr.CurvilinearFrame([[0.0, 0.0], [leg, 0.0], [leg, leg]])
    s = rng.uniform(0.0, 2 * leg, n)
    e = rng.uniform(0.0, height, n)
    d = -(e - 0.5 * height) / np.tan(np.radians(dip_deg)) + 0.02 * rng.standard_normal(n)
    pts = truth.sde_to_xyz(np.column_stack([s, d, e]))
    mat = (e > 0.5 * height).astype(int)
    return pts, mat


def test_slice_along_strike_follows_the_wall_around_the_corner():
    pts, mat = corner_wall()
    pc = pr.PointCloud(points=pts)
    pc.set_attr("mat", mat)
    original = pts.copy()

    profiles, s_pos = pc.slice(increment=2.0, label="mat", slice_height_increment=0.5)
    assert len(profiles) == pytest.approx(30, abs=2)        # 60 m of wall every 2 m
    np.testing.assert_allclose(np.diff(s_pos), 2.0, atol=1e-9)
    np.testing.assert_array_equal(pc.points, original)      # the model is not moved

    frame = pc.frame
    widths = []
    for prof, s_k in zip(profiles, s_pos):
        d, e = prof.nodes[:, 0], prof.nodes[:, 1]
        assert e[0] > e[-1]                                  # crest to toe
        assert d[-1] > d[0]                                  # d grows down-dip
        assert e.max() > 11.0 and e.min() < 1.0
        widths.append(np.ptp(d))
        mid = 0.5 * (e[:-1] + e[1:])
        clear = np.abs(mid - 6.0) > 1.0                      # away from the layer boundary
        np.testing.assert_array_equal(prof.attributes[clear], (mid[clear] > 6.0).astype(int))
        # back in world coordinates the nodes lie on the wall
        xyz = frame.sde_to_xyz(np.column_stack([np.full(d.shape, s_k), d, e]))
        dist = np.min(np.linalg.norm(pts[None, ::7, :] - xyz[:, None, :], axis=-1), axis=1)
        assert np.median(dist) < 0.6
    # the same profile everywhere, including through the corner
    np.testing.assert_allclose(widths, 12.0 / np.tan(np.radians(60.0)), atol=0.6)


def test_slice_with_an_explicit_frame_matches_a_fixed_direction():
    pts, mat = two_material_wall()
    pc = pr.PointCloud(points=pts)
    pc.set_attr("mat", mat)
    fixed, xs = pc.slice(np.array([1.0, 0.0, 0.0]), increment=1.0, label="mat", slice_height_increment=0.3)
    frame = pr.CurvilinearFrame.straight((0.0, 0.0), (1.0, 0.0))
    framed, ss = pc.slice(None, increment=1.0, label="mat", slice_height_increment=0.3, frame=frame)
    np.testing.assert_allclose(ss, xs, atol=1e-9)
    assert len(fixed) == len(framed)
    for a, b in zip(fixed, framed):
        np.testing.assert_allclose(a.nodes, b.nodes, atol=1e-9)
        np.testing.assert_array_equal(a.attributes, b.attributes)


def test_slice_min_points_and_transverse_radius():
    pts, mat = two_material_wall()
    pc = pr.PointCloud(points=pts)
    pc.set_attr("mat", mat)
    assert pc.slice(increment=1.0, label="mat", slice_height_increment=0.3, min_points=10**6)[0] == []
    stray = np.array([[5.0, 3.0, 2.5]] * 3)                  # off the wall, in one slab
    pc2 = pr.PointCloud(points=np.vstack([pts, stray]))
    pc2.set_attr("mat", np.r_[mat, [0, 0, 0]])
    profiles, _ = pc2.slice(np.array([1.0, 0.0, 0.0]), increment=1.0, label="mat",
                            slice_height_increment=0.3, transverse_radius=0.5)
    assert max(np.abs(p.nodes[:, 0]).max() for p in profiles) < 0.5


def test_section_follows_the_corner_and_returns_world_points():
    pts, mat = corner_wall()
    pc = pr.PointCloud(points=pts)
    pc.set_attr("mat", mat)
    profiles, sections_pc = pc.section(increment=3.0, label="mat")
    assert len(profiles) == pytest.approx(20, abs=2)
    ids = sections_pc.get_attr("Profile")
    assert len(np.unique(ids)) == len(profiles)
    # sections on both legs of the wall
    assert sections_pc.points[:, 1].max() > 20.0 and sections_pc.points[:, 0].min() < 5.0
    for prof in profiles:
        assert prof.nodes[:, 0].min() == pytest.approx(0.0)  # distance from the crest node
        assert prof.nodes[:, 1].min() == pytest.approx(0.0)  # height above the base


def test_segment_and_split_follow_the_corner():
    pts, mat = corner_wall()
    pc = pr.PointCloud(points=pts)
    pc.set_attr("mat", mat)
    segments = pc.segment(ds=10.0)
    assert len(segments) in (6, 7)                           # 60 m of wall; the tail may spill over
    assert sum(seg.points.shape[0] for seg in segments) == pts.shape[0]
    assert all(seg.has_attr("mat") and seg._frame is pc._frame for seg in segments)

    first, second = pc.split()
    assert first.points.shape[0] + second.points.shape[0] == pts.shape[0]
    # one part per leg: their dip directions differ by about 90 degrees
    diff = abs(first.dipDirection() - second.dipDirection()) % 360
    assert min(diff, 360 - diff) == pytest.approx(90.0, abs=10.0)


# ---------------------------------------------------------------------------
# removePoints() / fillHoles()
# ---------------------------------------------------------------------------

def test_remove_points_by_attribute_value_list_and_mask():
    pts, mat = corner_wall(n=3000)
    labels = np.where(pts[:, 2] > 9.0, 5, mat).astype(float)
    pc = pr.PointCloud(points=pts.copy())
    pc.set_attr("mat", labels)
    pc.set_attr("id", np.arange(3000))

    n = pc.removePoints("mat", 5)
    assert n == np.count_nonzero(labels == 5) > 0
    assert pc.points.shape[0] == 3000 - n
    np.testing.assert_array_equal(pc.get_attr("id"), np.flatnonzero(labels != 5))
    np.testing.assert_allclose(pc._removed, pts[labels == 5])

    n2 = pc.removePoints("mat", [0, 7])                      # 7 does not occur
    assert n2 == np.count_nonzero(labels == 0)
    assert set(np.unique(pc.get_attr("mat"))) == {1.0}
    assert pc._removed.shape[0] == n + n2                    # remembered across calls

    n3 = pc.removePoints(pc.points[:, 2] > 8.0)
    assert n3 > 0 and pc.points[:, 2].max() <= 8.0
    assert pc.removePoints("mat", 99) == 0


def test_remove_points_rejects_bad_input():
    pts, mat = corner_wall(n=500)
    pc = pr.PointCloud(points=pts)
    pc.set_attr("mat", mat)
    with pytest.raises(ValueError, match="not found"):
        pc.removePoints("missing", 0)
    with pytest.raises(ValueError, match="value"):
        pc.removePoints("mat")
    with pytest.raises(ValueError, match="Mask must be boolean"):
        pc.removePoints(np.zeros(10, dtype=bool))
    with pytest.raises(ValueError, match="Mask must be boolean"):
        pc.removePoints(np.zeros(500, dtype=int))


def holed_wall(seed=0):
    """Corner wall with a painted patch (label 9) on the first leg and one on the
    corner; returns the cloud, the true labels and the patch mask."""
    pts, mat = corner_wall(n=30000, seed=seed)
    truth = pr.CurvilinearFrame([[0.0, 0.0], [30.0, 0.0], [30.0, 30.0]])
    sde = truth.xyz_to_sde(pts)
    patch = (np.hypot(sde[:, 0] - 12.0, sde[:, 2] - 6.0) < 2.5) | (np.hypot(sde[:, 0] - 30.0, sde[:, 2] - 6.0) < 2.5)
    labels = np.where(patch, 9, mat)
    pc = pr.PointCloud(points=pts.copy())
    pc.set_attr("mat", labels.astype(float))
    pc.set_attr("colors", np.tile([0.2, 0.4, 0.6], (len(pts), 1)))
    return pc, pts, mat, patch


def test_fill_holes_after_removing_a_class():
    pc, pts, mat, patch = holed_wall()
    n_removed = pc.removePoints("mat", 9)
    assert n_removed == np.count_nonzero(patch)
    kept = pc.points.copy()

    filled, info = pc.fillHoles(attributes=["mat"], return_info=True)
    np.testing.assert_array_equal(pc.points, kept)                    # the source is untouched
    assert len(info["holes"]) == 2
    new = filled.get_attr("filled") == 1
    assert new.sum() == info["n_new"] == pytest.approx(n_removed, rel=0.4)
    np.testing.assert_array_equal(filled.points[~new], kept)
    assert set(np.unique(filled.get_attr("hole_id")[new])) == {1, 2}
    assert np.all(filled.get_attr("kriging_std")[~new] == 0)

    # the new points lie on the wall: near the removed ones, and on the same plane
    removed = pts[patch]
    d = np.min(np.linalg.norm(removed[None, ::3, :] - filled.points[new][:, None, :], axis=-1), axis=1)
    assert np.percentile(d, 95) < 0.5
    truth = pr.CurvilinearFrame([[0.0, 0.0], [30.0, 0.0], [30.0, 30.0]])
    sde = truth.xyz_to_sde(filled.points[new])
    on_plane = sde[:, 1] + (sde[:, 2] - 6.0) / np.tan(np.radians(60.0))
    assert np.sqrt(np.mean(on_plane ** 2)) < 0.1

    # labels: the two layers are recovered, the removed class is gone
    lab = filled.get_attr("mat")[new]
    assert set(np.unique(lab)) <= {0.0, 1.0}
    clear = np.abs(sde[:, 2] - 6.0) > 0.5
    assert np.mean(lab[clear] == (sde[clear, 2] > 6.0)) > 0.97
    P = np.column_stack([filled.get_attr("mat_P0"), filled.get_attr("mat_P1")])
    np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-6)
    assert np.all(filled.get_attr("mat_Pmax")[~new] == 1.0)           # one-hot for existing points
    # attributes that are not kriged follow the nearest existing point
    np.testing.assert_allclose(filled.get_attr("colors")[new], np.tile([0.2, 0.4, 0.6], (new.sum(), 1)))


def test_fill_holes_at_the_edge_uses_the_removed_positions():
    pts, mat = corner_wall(n=20000)
    crest_bite = (np.abs(pts[:, 0] - 12.0) < 2.5) & (pts[:, 2] > 8.0) & (pts[:, 1] < 10.0)   # open at the crest
    pc = pr.PointCloud(points=pts.copy())
    pc.set_attr("mat", mat)
    pc.removePoints(crest_bite)
    assert pc.fillHoles(use_removed=False).get_attr("filled").sum() == 0
    filled = pc.fillHoles()
    new = filled.get_attr("filled") == 1
    assert new.sum() == pytest.approx(crest_bite.sum(), rel=0.5)
    assert filled.points[new][:, 2].min() > 7.0


def test_fill_holes_without_holes_returns_a_copy_and_validates_input():
    pts, mat = corner_wall(n=8000)
    pc = pr.PointCloud(points=pts)
    pc.set_attr("mat", mat)
    out, info = pc.fillHoles(attributes="mat", return_info=True)
    assert info["n_new"] == 0 and out is not pc
    np.testing.assert_array_equal(out.points, pts)
    assert out.get_attr("filled").sum() == 0
    with pytest.raises(ValueError, match="not found"):
        pc.fillHoles(attributes=["missing"])
    with pytest.raises(ValueError, match="others must be"):
        pc.fillHoles(others="zero")


def test_fill_holes_kinds_override_and_nan_for_others():
    pc, pts, mat, patch = holed_wall(seed=1)
    pc.set_attr("height", pc.points[:, 2].copy())
    pc.removePoints("mat", 9)
    filled = pc.fillHoles(attributes=["mat", "height"], kinds={"mat": "continuous"}, others="nan")
    new = filled.get_attr("filled") == 1
    assert filled.has_attr("mat_std") and not filled.has_attr("mat_Pmax")
    assert filled.has_attr("height_std")
    # a continuous attribute equal to z is recovered from the kriged surface
    np.testing.assert_allclose(filled.get_attr("height")[new], filled.points[new][:, 2], atol=0.5)
    assert np.isnan(filled.get_attr("colors")[new]).all()
    assert not np.isnan(filled.get_attr("colors")[~new]).any()


def test_slice_ignores_a_sparse_layer_of_stray_points():
    pts, mat = two_material_wall(n=6000)
    rng = np.random.default_rng(7)
    # sparse second layer 0.6 in front of the wall, plus isolated points above the crest
    stray = np.column_stack([rng.uniform(0, 10, 120), np.full(120, 0.6), rng.uniform(0, 5, 120)])
    above = np.column_stack([rng.uniform(0, 10, 10), np.zeros(10), np.full(10, 6.0)])
    pc = pr.PointCloud(points=np.vstack([pts, stray, above]))
    pc.set_attr("mat", np.r_[mat, np.zeros(130, dtype=int)])

    profiles, xs = pc.slice(np.array([1.0, 0.0, 0.0]), increment=1.0, label="mat",
                            slice_height_increment=0.3, method="walk",
                            outlier_fraction=0.3, outlier_radius=0.15)
    assert len(profiles) == 10
    for prof in profiles:
        assert np.abs(prof.nodes[:, 0]).max() < 0.2         # stays on the wall
        assert prof.nodes[:, 1].max() < 5.1                 # does not start at a stray point
        assert prof.nodes[:, 1].min() < 0.5

    # with the filter off the walk starts from the stray points above the crest
    raw, _ = pc.slice(np.array([1.0, 0.0, 0.0]), increment=1.0, label="mat",
                      slice_height_increment=0.3, method="walk", outlier_fraction=0.0)
    assert max(p.nodes[:, 1].max() for p in raw) > 5.5

    # the default interpolation needs no filter
    interp, _ = pc.slice(np.array([1.0, 0.0, 0.0]), increment=1.0, label="mat", slice_height_increment=0.3)
    assert len(interp) == 10
    for prof in interp:
        assert np.abs(prof.nodes[:, 0]).max() < 0.2
        assert prof.nodes[:, 1].max() < 5.6


def test_slice_methods_agree_and_invalid_method_raises():
    pts, mat = corner_wall()
    pc = pr.PointCloud(points=pts)
    pc.set_attr("mat", mat)
    interp, s_i = pc.slice(increment=3.0, label="mat", slice_height_increment=0.5)
    walk, s_w = pc.slice(increment=3.0, label="mat", slice_height_increment=0.5, method="walk")
    np.testing.assert_allclose(s_i, s_w)
    assert len(interp) == len(walk)
    for a, b in zip(interp, walk):
        # same wall: same extent in d and e, to within a node spacing
        np.testing.assert_allclose(np.ptp(a.nodes, axis=0), np.ptp(b.nodes, axis=0), atol=0.5)
        assert a.nodes[0, 1] > a.nodes[-1, 1]
        assert len(a.attributes) == a.nodes.shape[0] - 1
    with pytest.raises(ValueError, match="method must be"):
        pc.slice(increment=3.0, label="mat", method="spline")


# ---------------------------------------------------------------------------
# slice(positions=...): several models cut at the same places
# ---------------------------------------------------------------------------

def test_slice_at_given_positions_matches_the_default_grid():
    pts, mat = corner_wall()
    pc = pr.PointCloud(points=pts)
    pc.set_attr("mat", mat)
    grid, s_grid = pc.slice(increment=2.0, label="mat", slice_height_increment=0.5)
    given, s_given = pc.slice(increment=2.0, label="mat", slice_height_increment=0.5, positions=s_grid)
    np.testing.assert_array_equal(s_given, s_grid)
    for a, b in zip(grid, given):
        np.testing.assert_array_equal(a.nodes, b.nodes)
        np.testing.assert_array_equal(a.attributes, b.attributes)


def test_slice_two_clouds_of_one_wall_at_the_same_positions():
    pts, mat = corner_wall()
    full = pr.PointCloud(points=pts)
    full.set_attr("mat", mat)
    # a part of the wall that starts elsewhere: its own grid would not line up
    keep = full.sde()[:, 0] > 13.3
    part = pr.PointCloud(points=pts[keep])
    part.set_attr("mat", mat[keep])
    part.frame = full.frame
    stations = np.array([-5.0, 7.0, 20.0, 41.5, 100.0])  # the first and last fall outside the wall
    _, s_full = full.slice(increment=1.0, label="mat", positions=stations)
    prof_part, s_part = part.slice(increment=1.0, label="mat", positions=stations)
    np.testing.assert_array_equal(s_full, [7.0, 20.0, 41.5])
    np.testing.assert_array_equal(s_part, [20.0, 41.5])
    _, s_own = part.slice(increment=1.0, label="mat")
    assert not np.isin(s_part, s_own).all()               # the default grid starts at the cloud
    assert len(prof_part) == 2


def test_slice_rejects_invalid_positions():
    pts, mat = two_material_wall()
    pc = pr.PointCloud(points=pts)
    pc.set_attr("mat", mat)
    with pytest.raises(ValueError, match="finite"):
        pc.slice(increment=1.0, label="mat", positions=[1.0, np.nan])
    with pytest.raises(ValueError, match="1D"):
        pc.slice(increment=1.0, label="mat", positions=[[1.0, 2.0]])


def test_section_at_given_positions():
    pts, mat = corner_wall()
    pc = pr.PointCloud(points=pts)
    pc.set_attr("mat", mat)
    profiles, sections_pc = pc.section(increment=1.0, label="mat", positions=[10.0, 35.0])
    assert len(profiles) == 2
    s = pc.frame.xyz_to_sde(sections_pc.points)[:, 0]
    np.testing.assert_allclose(np.unique(np.round(s, 6)), [10.0, 35.0], atol=1e-6)


def test_fill_holes_cover_side_on_a_vertical_wall():
    # On a vertical face "outward" cannot be read from the frame: normals or an
    # explicit side are needed
    rng = np.random.default_rng(0)
    x, z = rng.uniform(0, 30, 6000), rng.uniform(0, 12, 6000)
    pts = np.column_stack([x, 0.01 * rng.standard_normal(6000), z])
    mat = np.where(np.hypot(x - 15, z - 6) < 2.5, 0, 1)
    pc = pr.PointCloud(points=pts)
    pc.set_attr("mat", mat)
    pc.removePoints("mat", 0)
    with pytest.raises(ValueError, match="nearly vertical"):
        pc.fillHoles(attributes=["mat"], cover="removed")
    with pytest.raises(ValueError, match="'removed'"):
        pc.fillHoles(attributes=["mat"], cover="vegetation")
    with pytest.raises(ValueError, match="cover_side must be"):
        pc.fillHoles(attributes=["mat"], cover="removed", cover_side=2)
    filled, info = pc.fillHoles(attributes=["mat"], cover="removed", cover_side=-1, return_info=True)
    assert info["cover_side"] == -1.0 and filled.has_attr("cover_bound")
    # normals pointing to -y give the same side as the w axis along -y, or the opposite
    pc.set_attr("normals", np.tile([0.0, -1.0, 0.0], (pc.points.shape[0], 1)))
    w_y = pc.frame.uvw_to_xyz(np.array([[0.0, 0.0, 1.0]]))[0] - pc.frame.uvw_to_xyz(np.zeros((1, 3)))[0]
    _, info = pc.fillHoles(attributes=["mat"], cover="removed", return_info=True)
    assert info["cover_side"] == (1.0 if w_y[1] < 0 else -1.0)
