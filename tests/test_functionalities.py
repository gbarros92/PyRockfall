"""Tests for pyrockfall._functionalities: user-facing convenience functions.

Three real bugs were found and fixed while writing these tests (confirmed
with the user before changing production code):

1. removeMaterial(slope, material) crashed (ValueError, mismatched node and
   material counts) whenever the material to remove was a contiguous run in
   the *middle* of the profile (removing the first or last material worked
   fine). Fixed by correctly bridging the gap left by a removed middle run:
   the new connecting edge gets its own material entry (taking on the
   material of the segment immediately after the gap), instead of silently
   being left unaccounted for.

2. combinePercentiles' final inversion step (np.interp(ranks, F_mix,
   percentiles_grid)) picked the *last* matching x whenever F_mix had
   duplicate/flat values -- which happens whenever the combined rows have
   different value ranges (the normal case, not an edge case). E.g.
   weighting [1.0, 0.0] over two rows with different maxima returned the
   *other* row's max instead of the intended row's own value. Fixed by
   deduplicating F_mix before inversion, taking the smallest x for the
   F=100 plateau (true end of support) and the largest x for the F=0
   plateau (true start of support).

3. interpAttributes crashed (IndexError) whenever exactly one categorical
   attribute was requested, because sklearn's KNeighborsClassifier squeezes
   predict() to 1D for a single-column target, but the code always indexed
   it as 2D. Fixed by reshaping predict() output to 2D when needed (for
   both the continuous and categorical branches).

Not covered in this pass: `runWallProfiles`. It orchestrates a full
model-slicing / per-profile-analysis pipeline (dip alignment, clipping,
segmenting, slicing, talus handling) that would need substantial synthetic
3D wall/mesh scaffolding and a working `runner` callback to exercise
meaningfully; recommended as a follow-up with dedicated fixtures.
"""
import numpy as np
import pytest

import pyrockfall as pr
from pyrockfall._functionalities import (
    singleMaterialSlope,
    removeMaterial,
    findClosest,
    extrudePolyline,
    grid2mesh,
    mesh2PointCloud,
    pointCloud2Mesh,
    sampleRealisations,
    slopeBelow,
    materialLayers,
    interpPercentiles,
    combinePercentiles,
    _is_categorical,
    interpAttributes,
    aggregateOnFloor,
    rocksSeeders,
    slopeFeatures,
)


def make_layered_slope():
    # A 5-node profile, y decreasing left to right (a typical slope
    # cross-section), with materials [A, A, B, C] across its 4 elements.
    nodes = np.array([[0.0, 4.0], [1.0, 3.0], [2.0, 2.0], [3.0, 1.0], [4.0, 0.0]])
    geometry = pr.Geometry(nodes=nodes)
    mat_a = pr.Material(name="A")
    mat_b = pr.Material(name="B")
    mat_c = pr.Material(name="C")
    slope = pr.Slope(geometry, materials=[mat_a, mat_a, mat_b, mat_c])
    return slope, mat_a, mat_b, mat_c


# ---------------------------------------------------------------------------
# singleMaterialSlope
# ---------------------------------------------------------------------------

def test_single_material_slope_extracts_contiguous_run():
    slope, mat_a, mat_b, mat_c = make_layered_slope()
    result = singleMaterialSlope(slope, mat_a)
    assert len(result) == 1
    np.testing.assert_allclose(result[0].nodes, [[0.0, 4.0], [1.0, 3.0], [2.0, 2.0]])
    assert result[0].materialTable == [mat_a]


def test_single_material_slope_by_index():
    slope, mat_a, mat_b, mat_c = make_layered_slope()
    result = singleMaterialSlope(slope, 2)  # index 2 == mat_c
    assert len(result) == 1
    np.testing.assert_allclose(result[0].nodes, [[3.0, 1.0], [4.0, 0.0]])


def test_single_material_slope_not_present_returns_empty():
    slope, mat_a, mat_b, mat_c = make_layered_slope()
    other = pr.Material(name="Other")
    assert singleMaterialSlope(slope, other) == []


def test_single_material_slope_discontiguous_runs_returns_multiple_slopes():
    nodes = np.array([[0.0, 4.0], [1.0, 3.0], [2.0, 2.0], [3.0, 1.0], [4.0, 0.0]])
    geometry = pr.Geometry(nodes=nodes)
    mat_a = pr.Material(name="A")
    mat_b = pr.Material(name="B")
    slope = pr.Slope(geometry, materials=[mat_a, mat_b, mat_a, mat_a])  # A appears twice

    result = singleMaterialSlope(slope, mat_a)
    assert len(result) == 2
    np.testing.assert_allclose(result[0].nodes, [[0.0, 4.0], [1.0, 3.0]])
    np.testing.assert_allclose(result[1].nodes, [[2.0, 2.0], [3.0, 1.0], [4.0, 0.0]])


def test_single_material_slope_invalid_type_raises():
    slope, mat_a, mat_b, mat_c = make_layered_slope()
    with pytest.raises(TypeError, match="material must be an instance of Material or an integer"):
        singleMaterialSlope(slope, "not a material")


# ---------------------------------------------------------------------------
# removeMaterial
# ---------------------------------------------------------------------------

def test_remove_material_at_start():
    slope, mat_a, mat_b, mat_c = make_layered_slope()
    result = removeMaterial(slope, mat_a)
    np.testing.assert_allclose(result.nodes, [[2.0, 2.0], [3.0, 1.0], [4.0, 0.0]])
    np.testing.assert_array_equal(result.materialIDs, [0, 1])
    assert [m.name for m in result.materialTable] == ["B", "C"]


def test_remove_material_at_end():
    slope, mat_a, mat_b, mat_c = make_layered_slope()
    result = removeMaterial(slope, mat_c)
    np.testing.assert_allclose(result.nodes, [[0.0, 4.0], [1.0, 3.0], [2.0, 2.0], [3.0, 1.0]])
    np.testing.assert_array_equal(result.materialIDs, [0, 0, 1])
    assert [m.name for m in result.materialTable] == ["A", "B"]


def test_remove_material_in_the_middle_bridges_the_gap():
    # Regression test: this used to raise ValueError (mismatched node and
    # material counts) before the fix.
    slope, mat_a, mat_b, mat_c = make_layered_slope()
    result = removeMaterial(slope, mat_b)
    np.testing.assert_allclose(
        result.nodes, [[0.0, 4.0], [1.0, 3.0], [2.0, 2.0], [3.0, 1.0], [4.0, 0.0]]
    )
    np.testing.assert_array_equal(result.materialIDs, [0, 0, 1, 1])
    assert [m.name for m in result.materialTable] == ["A", "C"]
    # every element is consistently assigned: materials line up with materialIDs
    assert len(result.materialIDs) == len(result.elements)


def test_remove_material_by_index():
    slope, mat_a, mat_b, mat_c = make_layered_slope()
    result = removeMaterial(slope, 1)  # index 1 == mat_b
    assert [m.name for m in result.materialTable] == ["A", "C"]


def test_remove_material_not_present_returns_original_slope():
    slope, mat_a, mat_b, mat_c = make_layered_slope()
    other = pr.Material(name="Other")
    assert removeMaterial(slope, other) is slope


def test_remove_material_leaving_empty_slope_raises():
    nodes = np.array([[0.0, 1.0], [1.0, 0.0]])
    geometry = pr.Geometry(nodes=nodes)
    mat = pr.Material(name="Only")
    slope = pr.Slope(geometry, materials=mat)
    with pytest.raises(ValueError, match="Cannot remove the material"):
        removeMaterial(slope, mat)


# ---------------------------------------------------------------------------
# findClosest
# ---------------------------------------------------------------------------

def test_find_closest_interpolates_within_a_segment():
    slope, *_ = make_layered_slope()
    x, e = findClosest(slope, height=2.5)
    assert x == pytest.approx(1.5)
    assert e == 1


def test_find_closest_exact_node_height():
    slope, *_ = make_layered_slope()
    x, e = findClosest(slope, height=0.0)
    assert x == pytest.approx(4.0)
    assert e == 3


def test_find_closest_rejects_non_2d_slope():
    nodes = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 0.0, 0.0]])
    elements = np.array([[0, 1, 2]])
    geometry = pr.Geometry(nodes=nodes, elements=elements)
    mat = pr.Material(name="A")
    slope = pr.Slope(geometry, materials=mat)
    with pytest.raises(ValueError, match="only implemented for 2D slopes"):
        findClosest(slope, height=0.5)


# ---------------------------------------------------------------------------
# extrudePolyline
# ---------------------------------------------------------------------------

def test_extrude_polyline_shapes_and_values():
    nodes_2d = np.array([[0.0, 0.0], [1.0, 0.5], [2.0, 0.0]])
    nodes_3d, tris = extrudePolyline(nodes_2d, dy=2.0)

    assert nodes_3d.shape == (6, 3)
    assert tris.shape == (4, 3)
    np.testing.assert_allclose(nodes_3d[:3], np.column_stack([nodes_2d[:, 0], np.zeros(3), nodes_2d[:, 1]]))
    np.testing.assert_allclose(nodes_3d[3:], np.column_stack([nodes_2d[:, 0], np.full(3, 2.0), nodes_2d[:, 1]]))
    np.testing.assert_array_equal(tris, [[0, 1, 3], [3, 1, 4], [1, 2, 4], [4, 2, 5]])


def test_extrude_polyline_triangles_reference_valid_vertices():
    nodes_2d = np.random.default_rng(0).uniform(size=(5, 2))
    nodes_3d, tris = extrudePolyline(nodes_2d, dy=1.0)
    assert tris.min() >= 0
    assert tris.max() < nodes_3d.shape[0]


# ---------------------------------------------------------------------------
# grid2mesh
# ---------------------------------------------------------------------------

def test_grid2mesh_shapes_and_z_plane():
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 1.0])
    points, triangles = grid2mesh(x, y, z0=5.0)
    assert points.shape == (6, 3)
    assert triangles.shape == (2 * 2 * 1, 3)
    np.testing.assert_allclose(points[:, 2], 5.0)
    assert triangles.min() >= 0
    assert triangles.max() < points.shape[0]


def test_grid2mesh_triangle_area_matches_cell_size():
    x = np.array([0.0, 2.0])
    y = np.array([0.0, 3.0])
    points, triangles = grid2mesh(x, y)
    total_area = 0.0
    for tri in triangles:
        a, b, c = points[tri, :2]
        e1 = b - a
        e2 = c - a
        total_area += 0.5 * abs(e1[0] * e2[1] - e1[1] * e2[0])
    assert total_area == pytest.approx(2.0 * 3.0)


# ---------------------------------------------------------------------------
# mesh2PointCloud / pointCloud2Mesh
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


def flat_grid_points():
    """3x2 grid of points on the z=0 plane."""
    xs, ys = np.meshgrid(np.arange(3.0), np.arange(2.0))
    return np.column_stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)])


def test_mesh2_point_cloud_keeps_points_and_node_attrs():
    pts, tris = strip_mesh()
    m = pr.Mesh(points=pts, triangles=tris)
    m.set_attr("elevation", pts[:, 2].copy())        # per-node (N=6)
    m.set_attr("slope", np.arange(tris.shape[0]))    # per-triangle (E=4)

    pc = mesh2PointCloud(m)

    assert isinstance(pc, pr.PointCloud)
    np.testing.assert_array_equal(pc.points, pts)
    np.testing.assert_array_equal(pc.get_attr("elevation"), pts[:, 2])
    assert not pc.has_attr("slope")


def test_mesh2_point_cloud_with_no_extra_attrs():
    pts, tris = strip_mesh()
    m = pr.Mesh(points=pts, triangles=tris)
    pc = mesh2PointCloud(m)
    np.testing.assert_array_equal(pc.points, pts)
    assert pc.list_attrs() == []


def test_point_cloud2_mesh_triangulates_flat_grid():
    pts = flat_grid_points()
    pc = pr.PointCloud(points=pts, attrs={"elevation": pts[:, 2].copy()})

    m = pointCloud2Mesh(pc)

    assert isinstance(m, pr.Mesh)
    np.testing.assert_array_equal(m.points, pts)
    assert m.triangles.ndim == 2 and m.triangles.shape[1] == 3
    assert m.triangles.shape[0] > 0
    assert m.triangles.min() >= 0
    assert m.triangles.max() < pts.shape[0]
    np.testing.assert_array_equal(m.get_attr("elevation"), pts[:, 2])


def test_mesh_to_point_cloud_to_mesh_roundtrip_preserves_points():
    pts, tris = strip_mesh()
    m = pr.Mesh(points=pts, triangles=tris)

    m2 = pointCloud2Mesh(mesh2PointCloud(m))

    np.testing.assert_array_equal(m2.points, m.points)


def _corner_wall_points(n=4000, seed=0, leg=30.0, height=12.0):
    """Wall dipping 60 degrees that turns 90 degrees half-way along."""
    rng = np.random.default_rng(seed)
    truth = pr.CurvilinearFrame([[0.0, 0.0], [leg, 0.0], [leg, leg]])
    s = rng.uniform(0.0, 2 * leg, n)
    e = rng.uniform(0.0, height, n)
    d = -(e - 0.5 * height) / np.tan(np.radians(60.0))
    return truth.sde_to_xyz(np.column_stack([s, d, e])), e


def _triangle_edges(m):
    tri = m.points[m.triangles]
    return np.linalg.norm(tri - np.roll(tri, -1, axis=1), axis=2)


def test_point_cloud2_mesh_uv_follows_a_corner():
    pts, e = _corner_wall_points()
    pc = pr.PointCloud(points=pts, attrs={"mat": (e > 6.0).astype(int)})
    m = pointCloud2Mesh(pc)

    np.testing.assert_array_equal(m.points, pts)
    np.testing.assert_array_equal(m.get_attr("mat"), (e > 6.0).astype(int))
    assert m._frame is pc._frame                                # shares the frame
    assert np.unique(m.triangles).size == pts.shape[0]          # every point is used
    # triangles stay on the wall: none cuts across the inside of the corner
    assert np.percentile(_triangle_edges(m).max(axis=1), 99) < 3.0
    # one rotation cannot flatten a wall with a corner: it leaves long triangles
    plane = pointCloud2Mesh(pc, method="plane")
    assert np.percentile(_triangle_edges(plane).max(axis=1), 99) > np.percentile(_triangle_edges(m).max(axis=1), 99)

    # the mesh can be sectioned along the strike straight away
    profiles, sections_pc = m.section(increment=2.3, label="mat")
    assert len(profiles) > 20
    heights = np.array([np.ptp(p.nodes[:, 1]) for p in profiles])
    assert np.median(heights) > 11.0


def test_point_cloud2_mesh_max_edge_and_invalid_method():
    pts, e = _corner_wall_points()
    gap = (np.abs(pts[:, 0] - 12.0) < 3.0) & (pts[:, 1] < 6.0)   # 6 m wide strip removed from one leg
    pc = pr.PointCloud(points=pts[~gap])
    full = pointCloud2Mesh(pc)
    cut = pointCloud2Mesh(pc, max_edge=3.0)
    assert cut.triangles.shape[0] < full.triangles.shape[0]
    assert _triangle_edges(full).max() > 5.0                     # the gap is bridged
    assert _triangle_edges(cut).max() <= 3.0                     # ... unless long triangles are dropped
    np.testing.assert_array_equal(cut.points, pc.points)         # nodes are kept either way
    with pytest.raises(ValueError, match="method must be"):
        pointCloud2Mesh(pc, method="xy")


def _filled_wall(seed=0):
    """Corner wall with two layers and a patch removed and refilled by kriging."""
    pts, e = _corner_wall_points(n=12000, seed=seed)
    rng = np.random.default_rng(seed)
    truth = pr.CurvilinearFrame([[0.0, 0.0], [30.0, 0.0], [30.0, 30.0]])
    sde = truth.xyz_to_sde(pts)
    # smooth undulations of the face (spatially correlated) plus a little noise
    sde[:, 1] += 0.4 * np.sin(sde[:, 0] / 2.0) * np.cos(sde[:, 2] / 1.5) + 0.01 * rng.standard_normal(len(sde))
    pts = truth.sde_to_xyz(sde)
    patch = np.hypot(sde[:, 0] - 15.0, sde[:, 2] - 6.0) < 3.0
    pc = pr.PointCloud(points=pts, attrs={"mat": np.where(patch, 9, (e > 6.0).astype(int))})
    pc.removePoints("mat", 9)
    return pc.fillHoles(attributes=["mat"], return_info=True)


def test_sample_realisations_coordinates_follow_std_and_materials_follow_probabilities():
    filled, info = _filled_wall()
    new = filled.get_attr("filled") == 1
    sigma = filled.get_attr("kriging_std")
    P = np.column_stack([filled.get_attr("mat_P0"), filled.get_attr("mat_P1")])
    assert sigma[new].min() > 0 and (P[new].max(axis=1) < 1).any()

    n = 400
    reals = sampleRealisations(filled, n, "mat", correlated=False, seed=1)
    assert len(reals) == n
    frame = filled.frame
    w0 = frame.xyz_to_uvw(filled.points)

    W = np.stack([frame.xyz_to_uvw(r.points)[:, 2] for r in reals])          # (n, N)
    K = np.stack([r.get_attr("mat") for r in reals])
    DW = np.stack([r.get_attr("dw") for r in reals])

    # certain points never change
    for r in reals[:5]:
        np.testing.assert_array_equal(r.points[~new], filled.points[~new])
        np.testing.assert_array_equal(r.get_attr("mat")[~new], filled.get_attr("mat")[~new])
    # points move only along w, by the recorded displacement
    uv_moved = np.stack([frame.xyz_to_uvw(r.points[new])[:, :2] for r in reals[:5]])
    np.testing.assert_allclose(uv_moved, np.broadcast_to(w0[new, :2], uv_moved.shape), atol=1e-6)
    np.testing.assert_allclose(W[:, new] - w0[new, 2], DW[:, new], atol=1e-6)

    # coordinates: per point, mean w and std of w match the kriged w and kriging_std
    dw = W[:, new] - w0[new, 2]
    assert np.abs(np.median(dw.mean(axis=0) / sigma[new])) < 0.05
    ratio = dw.std(axis=0) / sigma[new]
    assert np.median(ratio) == pytest.approx(1.0, abs=0.05)
    # materials: per point, the frequency of each class matches its probability
    freq = np.column_stack([(K[:, new] == 0).mean(axis=0), (K[:, new] == 1).mean(axis=0)])
    assert np.abs(freq - P[new]).mean() < 0.02
    assert np.abs(freq - P[new]).max() < 0.15


def test_sample_realisations_correlated_fields_keep_the_per_point_distributions():
    filled, info = _filled_wall()
    new = filled.get_attr("filled") == 1
    sigma = filled.get_attr("kriging_std")[new]
    P1 = filled.get_attr("mat_P1")[new]
    frame = filled.frame
    w0 = frame.xyz_to_uvw(filled.points[new])
    from scipy.spatial import cKDTree
    _, nn = cKDTree(w0[:, :2]).query(w0[:, :2], k=2)
    nn = nn[:, 1]

    stats = {}
    for name, kw in (("independent", dict(correlated=False)),
                     ("fitted", dict()),
                     ("from fillHoles", dict(info=info))):
        reals = sampleRealisations(filled, 300, "mat", uncertain_only=True, seed=2, **kw)
        assert reals[0].points.shape[0] == new.sum()
        DW = np.stack([r.get_attr("dw") for r in reals])
        K = np.stack([r.get_attr("mat") for r in reals])
        z = DW / sigma
        stats[name] = dict(
            std_ratio=np.median(DW.std(axis=0) / sigma),
            freq_err=np.abs((K == 1).mean(axis=0) - P1).mean(),
            rough=np.abs(z - z[:, nn]).mean(),                       # neighbour difference of z
            flips=(K != K[:, nn]).mean(),
        )
    for name, st in stats.items():
        # same per-point distributions in all modes (a correlated field over a small
        # hole has fewer independent values, so the tolerance is wider)
        assert st["std_ratio"] == pytest.approx(1.0, abs=0.15), name
        assert st["freq_err"] < 0.06, name
    for name in ("fitted", "from fillHoles"):
        # but neighbours are alike: smooth surfaces and coherent layers
        assert stats[name]["rough"] < 0.5 * stats["independent"]["rough"], name
        assert stats[name]["flips"] < 0.5 * stats["independent"]["flips"], name


def test_sample_realisations_validates_input():
    filled, _ = _filled_wall()
    with pytest.raises(ValueError, match="n_realisations"):
        sampleRealisations(filled, 0, "mat")
    with pytest.raises(ValueError, match="not found"):
        sampleRealisations(filled, 2, "mat", std="missing")
    plain = pr.PointCloud(points=filled.points, attrs={"mat": filled.get_attr("mat"),
                                                      "kriging_std": filled.get_attr("kriging_std")})
    with pytest.raises(ValueError, match="No probability attributes"):
        sampleRealisations(plain, 2, "mat")
    bad = pr.PointCloud(points=filled.points, attrs=dict(filled.attrs))
    bad.set_attr("mat_P0", bad.get_attr("mat_P0") * 0.5)
    with pytest.raises(ValueError, match="sum to 1"):
        sampleRealisations(bad, 2, "mat")


# ---------------------------------------------------------------------------
# slopeBelow
# ---------------------------------------------------------------------------

def test_slope_below_trims_profile_to_seeder_height():
    slope, *_ = make_layered_slope()
    rock = pr.Rock(mass=1.0, density=2500.0)
    seeder = pr.Seeder(points=np.array([2.5, 2.5]), rocks=[rock])
    result = slopeBelow(slope, seeder)
    np.testing.assert_allclose(
        result.nodes, [[1.5, 2.5], [2.0, 2.0], [3.0, 1.0], [4.0, 0.0]]
    )


def test_slope_below_seeder_already_below_returns_same_slope():
    slope, *_ = make_layered_slope()
    rock = pr.Rock(mass=1.0, density=2500.0)
    seeder = pr.Seeder(points=np.array([2.5, -10.0]), rocks=[rock])
    result = slopeBelow(slope, seeder)
    assert result is slope


def test_slope_below_rejects_multi_point_seeder():
    slope, *_ = make_layered_slope()
    rock = pr.Rock(mass=1.0, density=2500.0)
    seeder = pr.Seeder(points=np.array([[0.0, 1.0], [1.0, 1.0]]), rocks=[rock])
    with pytest.raises(ValueError, match="Must be a point seeder."):
        slopeBelow(slope, seeder)


def test_slope_below_rejects_non_seeder():
    slope, *_ = make_layered_slope()
    with pytest.raises(TypeError, match="seeder must be an instance of Seeder"):
        slopeBelow(slope, "not a seeder")


# ---------------------------------------------------------------------------
# materialLayers
# ---------------------------------------------------------------------------

def test_material_layers_reports_bottom_to_top_order_and_heights():
    slope, mat_a, mat_b, mat_c = make_layered_slope()
    materials, heights = materialLayers(slope)
    assert [m.name for m in materials] == ["A", "B", "C"]
    np.testing.assert_allclose(heights, [4.0, 2.0, 1.0])


# ---------------------------------------------------------------------------
# interpPercentiles
# ---------------------------------------------------------------------------

def test_interp_percentiles_linear_case():
    data = np.array([0.0, 2.5, 5.0, 10.0])
    percentiles = np.array([0.0, 5.0, 10.0])  # values at ranks 0,50,100
    ranks = interpPercentiles(data, percentiles)
    np.testing.assert_allclose(ranks, [0.0, 25.0, 50.0, 100.0])


def test_interp_percentiles_clips_outside_range():
    data = np.array([-5.0, 15.0])
    percentiles = np.array([0.0, 10.0])
    ranks = interpPercentiles(data, percentiles)
    np.testing.assert_allclose(ranks, [0.0, 100.0])


def test_interp_percentiles_rejects_wrong_ranks_shape():
    data = np.array([1.0])
    percentiles = np.array([0.0, 5.0, 10.0])
    with pytest.raises(ValueError, match="`ranks` must have shape"):
        interpPercentiles(data, percentiles, ranks=np.array([0.0, 100.0]))


# ---------------------------------------------------------------------------
# combinePercentiles
# ---------------------------------------------------------------------------

def test_combine_percentiles_identical_rows_returns_same_curve():
    ranks = np.array([0.0, 50.0, 100.0])
    p = np.array([[0.0, 5.0, 10.0], [0.0, 5.0, 10.0]])
    result = combinePercentiles(p, ranks)
    np.testing.assert_allclose(result, [0.0, 5.0, 10.0])


def test_combine_percentiles_extreme_weighting_reduces_to_single_row():
    # Regression test: this used to return the *other* row's endpoint
    # (11.0) instead of the correct 10.0 for full weight on row 0, because
    # of a tie-breaking bug in the inversion step.
    ranks = np.array([0.0, 25.0, 50.0, 75.0, 100.0])
    p = np.array([[0.0, 2.5, 5.0, 7.5, 10.0], [1.0, 3.5, 6.0, 8.5, 11.0]])

    result_row0 = combinePercentiles(p, ranks, likelihood=np.array([1.0, 0.0]))
    np.testing.assert_allclose(result_row0, p[0])

    result_row1 = combinePercentiles(p, ranks, likelihood=np.array([0.0, 1.0]))
    np.testing.assert_allclose(result_row1, p[1])


def test_combine_percentiles_equal_weight_is_between_rows():
    ranks = np.array([0.0, 50.0, 100.0])
    p = np.array([[0.0, 5.0, 10.0], [2.0, 6.0, 12.0]])
    result = combinePercentiles(p, ranks)
    assert np.all(result >= p.min(axis=0))
    assert np.all(result <= p.max(axis=0))


def test_combine_percentiles_default_ranks_and_uniform_weights():
    p = np.array([[0.0, 10.0], [0.0, 10.0]])
    result = combinePercentiles(p)  # default ranks = linspace(0,100,2), uniform weights
    np.testing.assert_allclose(result, [0.0, 10.0])


def test_combine_percentiles_rejects_wrong_ranks_shape():
    p = np.array([[0.0, 10.0], [0.0, 10.0]])
    with pytest.raises(ValueError, match="`ranks` must have shape"):
        combinePercentiles(p, ranks=np.array([0.0, 50.0, 100.0]))


def test_combine_percentiles_rejects_negative_likelihood():
    p = np.array([[0.0, 10.0], [0.0, 10.0]])
    with pytest.raises(ValueError, match="`likelihood` must be nonnegative"):
        combinePercentiles(p, ranks=np.array([0.0, 100.0]), likelihood=np.array([1.0, -1.0]))


def test_combine_percentiles_all_zero_likelihood_returns_nan():
    p = np.array([[0.0, 10.0], [0.0, 10.0]])
    result = combinePercentiles(p, ranks=np.array([0.0, 100.0]), likelihood=np.array([0.0, 0.0]))
    assert np.all(np.isnan(result))


# ---------------------------------------------------------------------------
# _is_categorical
# ---------------------------------------------------------------------------

def test_is_categorical_strings_are_categorical():
    assert _is_categorical(np.array(["a", "b", "c"])) is True


def test_is_categorical_many_unique_ints_are_continuous():
    assert _is_categorical(np.arange(1000)) is False


def test_is_categorical_few_unique_ints_are_categorical():
    assert _is_categorical(np.array([0, 1, 0, 1, 0, 1] * 20)) is True


def test_is_categorical_floats_are_never_categorical():
    assert _is_categorical(np.linspace(0.0, 1.0, 5)) is False


# ---------------------------------------------------------------------------
# interpAttributes
# ---------------------------------------------------------------------------

def make_source_destination_clouds():
    src = pr.PointCloud(
        points=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    )
    dst = pr.PointCloud(points=np.array([[0.1, 0.0, 0.0], [0.9, 1.0, 0.0]]))
    return src, dst


def test_interp_attributes_continuous_nearest_neighbor():
    src, dst = make_source_destination_clouds()
    src.set_attr("value", np.array([0.0, 10.0, 0.0, 10.0]))
    interpAttributes(src, dst, ["value"], method="nearest")
    np.testing.assert_allclose(dst.get_attr("value"), [0.0, 10.0])


def test_interp_attributes_single_categorical_attribute():
    # Regression test: this used to raise IndexError before the fix.
    src, dst = make_source_destination_clouds()
    src.set_attr("label", np.array(["A", "B", "A", "B"]))
    interpAttributes(src, dst, ["label"], method="nearest")
    np.testing.assert_array_equal(dst.get_attr("label"), ["A", "B"])


def test_interp_attributes_mixed_categorical_and_continuous():
    src, dst = make_source_destination_clouds()
    src.set_attr("value", np.array([0.0, 10.0, 0.0, 10.0]))
    src.set_attr("label", np.array(["A", "B", "A", "B"]))
    interpAttributes(src, dst, ["value", "label"], method="nearest")
    np.testing.assert_allclose(dst.get_attr("value"), [0.0, 10.0])
    np.testing.assert_array_equal(dst.get_attr("label"), ["A", "B"])


def test_interp_attributes_two_categorical_attributes():
    src, dst = make_source_destination_clouds()
    src.set_attr("label", np.array(["A", "B", "A", "B"]))
    src.set_attr("label2", np.array(["X", "Y", "X", "Y"]))
    interpAttributes(src, dst, ["label", "label2"], method="nearest")
    np.testing.assert_array_equal(dst.get_attr("label"), ["A", "B"])
    np.testing.assert_array_equal(dst.get_attr("label2"), ["X", "Y"])


class _StubModel:
    """Minimal duck-typed stand-in exposing exactly what interpAttributes needs.

    PointCloud always pads 2D points to 3D internally, so a genuine
    dimension mismatch can't be constructed through it -- a plain stub is
    used here instead.
    """

    def __init__(self, points, attrs=None):
        self.points = np.asarray(points, dtype=float)
        self._attrs = dict(attrs or {})

    def get_attr(self, name):
        return self._attrs[name]

    def set_attr(self, name, value):
        self._attrs[name] = value


def test_interp_attributes_rejects_mismatched_dimensions():
    src = _StubModel(points=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), attrs={"value": np.array([1.0, 2.0])})
    dst = _StubModel(points=np.array([[0.0, 0.0], [1.0, 0.0]]))  # 2D destination
    with pytest.raises(ValueError, match="must be \\(N,D\\) with same D"):
        interpAttributes(src, dst, ["value"], method="nearest")


# ---------------------------------------------------------------------------
# aggregateOnFloor
# ---------------------------------------------------------------------------

def make_wall_point_cloud():
    pts = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 1.0],
    ])
    wall = pr.PointCloud(points=pts)
    wall.set_attr("profile", np.array([0, 0, 1, 1]))
    wall.set_attr("E1_5", np.array([0.0, 2.0, 10.0, 12.0]))
    wall.set_attr("E1_50", np.array([5.0, 7.0, 15.0, 17.0]))
    wall.set_attr("E1_95", np.array([10.0, 12.0, 20.0, 22.0]))
    return wall, pts


def test_aggregate_on_floor_matches_manual_combine_percentiles():
    wall, pts = make_wall_point_cloud()
    percentiles = np.array([5.0, 50.0, 95.0])

    out = aggregateOnFloor(wall, "profile", ["E1"], percentiles)

    expected_p0 = combinePercentiles(np.array([[0.0, 5.0, 10.0], [2.0, 7.0, 12.0]]), percentiles)
    expected_p1 = combinePercentiles(np.array([[10.0, 15.0, 20.0], [12.0, 17.0, 22.0]]), percentiles)

    assert set(out.keys()) == {"E1_5", "E1_50", "E1_95"}
    np.testing.assert_allclose(out["E1_5"], [expected_p0[0], expected_p1[0]])
    np.testing.assert_allclose(out["E1_50"], [expected_p0[1], expected_p1[1]])
    np.testing.assert_allclose(out["E1_95"], [expected_p0[2], expected_p1[2]])


def test_aggregate_on_floor_restores_original_wall_orientation():
    wall, pts = make_wall_point_cloud()
    percentiles = np.array([5.0, 50.0, 95.0])
    aggregateOnFloor(wall, "profile", ["E1"], percentiles)
    np.testing.assert_allclose(wall.points, pts, atol=1e-8)


def test_aggregate_on_floor_custom_likelihood_matches_manual_combine():
    wall, pts = make_wall_point_cloud()
    percentiles = np.array([5.0, 50.0, 95.0])

    likelihood = lambda points: np.array([2.0, 1.0])  # weight the lower point twice as much
    out = aggregateOnFloor(wall, "profile", ["E1"], percentiles, likelihood=likelihood)

    w = np.array([2.0, 1.0]) / 3.0
    expected_p0 = combinePercentiles(np.array([[0.0, 5.0, 10.0], [2.0, 7.0, 12.0]]), percentiles, likelihood=w)
    np.testing.assert_allclose(
        [out["E1_5"][0], out["E1_50"][0], out["E1_95"][0]], expected_p0
    )


# ---------------------------------------------------------------------------
# rocksSeeders
# ---------------------------------------------------------------------------

def test_rocks_seeders_collects_unique_rocks_preserving_order():
    r1 = pr.Rock(name="r1")
    r2 = pr.Rock(name="r2")
    r3 = pr.Rock(name="r3")
    seeder1 = pr.Seeder(points=np.array([0.0, 0.0]), rocks=[r1, r2])
    seeder2 = pr.Seeder(points=np.array([1.0, 1.0]), rocks=[r2, r3])

    result = rocksSeeders([seeder1, seeder2])
    assert [r.name for r in result] == ["r1", "r2", "r3"]


def test_rocks_seeders_empty_list():
    assert rocksSeeders([]) == []


# ---------------------------------------------------------------------------
# slopeFeatures
# ---------------------------------------------------------------------------

def test_slope_features_straight_45_degree_slope():
    nodes = np.array([[0.0, 10.0], [5.0, 5.0], [10.0, 0.0]])
    geometry = pr.Geometry(nodes=nodes)
    mat = pr.Material(name="A")
    slope = pr.Slope(geometry, materials=mat)

    height, slope_angle, roughness = slopeFeatures(slope)
    assert height == pytest.approx(10.0)
    assert slope_angle == pytest.approx(45.0)
    assert roughness == pytest.approx(0.0, abs=1e-8)


def test_slope_features_height_matches_range():
    nodes = np.array([[0.0, 0.0], [2.0, -3.0], [4.0, 7.0], [6.0, 1.0]])
    geometry = pr.Geometry(nodes=nodes)
    mat = pr.Material(name="A")
    slope = pr.Slope(geometry, materials=mat)
    height, _, _ = slopeFeatures(slope)
    assert height == pytest.approx(7.0 - (-3.0))


def test_slope_features_roughness_is_nonzero_for_irregular_profile():
    nodes = np.array([[0.0, 10.0], [3.0, 8.0], [5.0, 6.0], [7.0, 1.0], [10.0, 0.0]])
    geometry = pr.Geometry(nodes=nodes)
    mat = pr.Material(name="A")
    slope = pr.Slope(geometry, materials=mat)
    height, slope_angle, roughness = slopeFeatures(slope)
    assert roughness > 0.0
