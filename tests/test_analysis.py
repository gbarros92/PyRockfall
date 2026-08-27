"""Tests for pyrockfall.Analysis: the rockfall simulation orchestration engine.

Two real bugs were found and fixed while writing these tests (confirmed
with the user before changing production code):

1. postprocess() set `self.trajectories.floor = np.min(self.slope.nodes[-1])`
   -- indexing the *last node's* coordinates instead of the last *column*
   (all y/z values across every node), which is what `Slope.floor` already
   computes correctly. For any profile where the last node isn't the lowest
   point, this gave the wrong floor elevation. Fixed by using
   `self.slope.floor` directly.

2. _slide()'s `is_sticking` mask included a pre-step term
   `(vt_norm <= self.stoppedVelocity)`, so a block starting a slide step
   *at rest* had its freshly computed (possibly large, physically correct)
   post-step velocity forcibly zeroed -- even on a steep frictionless slope
   where it should accelerate away from rest. Reproduced: 10 iterations on
   a 45deg frictionless slope kept velocity at exactly 0.0 forever while
   position crept by a fixed tiny amount each step. Fixed by deciding
   "sticking" from the step's outcome only (`stop_in_step` or
   `vt_new_norm <= stoppedVelocity`), not from how the block started.

Given the size and complexity of this module, tests are organized as:
  * Direct unit tests for each "pure kernel" method (validation, sampling
    setup, and the physics kernels), using hand-computable inputs.
  * A couple of true end-to-end `run()` tests on small synthetic slopes,
    checking physically sensible outcomes (free-fall timing, landing at the
    correct floor, a block settling to rest on a frictional run-out) rather
    than pinning exact numeric trajectories.
"""
import numpy as np
import pytest

import pyrockfall as pr
from pyrockfall import stats
from pyrockfall._analysis import Analysis, AnalysisRocksThrown, Sampling, FLOOR_ELEMENT_ID
from pyrockfall._seeder import SeederRocksThrown


def make_flat_slope(x0=-50.0, x1=50.0, y=0.0, **material_kwargs):
    nodes = np.array([[x0, y], [x1, y]])
    geometry = pr.Geometry(nodes=nodes)
    material = pr.Material(name="Flat", **material_kwargs)
    return pr.Slope(geometry, materials=material)


def make_analysis(slope=None, seeders=None):
    a = Analysis()
    a.slope = slope if slope is not None else make_flat_slope()
    a.seeders = seeders if seeders is not None else []
    return a


# ---------------------------------------------------------------------------
# Construction / copy semantics
# ---------------------------------------------------------------------------

def test_default_construction():
    a = Analysis()
    assert a.seeders == []
    assert a.rockThrowMode == AnalysisRocksThrown.IndividuallyPerSeeder
    assert a.samplingMethod == Sampling.LatinHypercube
    assert a.gravity == pytest.approx(-9.80665)


def test_copy_constructor_carries_over_configuration():
    original = Analysis()
    original.numberOfRocks = 42
    original.gravity = -1.0
    original.timeStep = 0.5
    original.slope = make_flat_slope()

    copy = Analysis(copy=original)
    assert copy.numberOfRocks == 42
    assert copy.gravity == pytest.approx(-1.0)
    assert copy.timeStep == pytest.approx(0.5)
    assert copy.slope is original.slope


def test_copy_constructor_carries_over_slope_and_vegetation():
    # Regression test: the whitelist used to check for "slope"/"vegetation",
    # but the instance actually stores these as _slope/_vegetation (the
    # properties' backing fields), so they were silently never copied.
    nodes = np.array([[0.0, 0.0], [1.0, 0.0]])
    slope = pr.Slope(pr.Geometry(nodes=nodes), materials=pr.Material(name="A"))
    veg = pr.Vegetation(pr.Geometry(nodes=nodes), drag=pr.Drag(name="D"))

    original = Analysis()
    original.slope = slope
    original.vegetation = veg

    copy = Analysis(copy=original)
    assert copy.slope is slope
    assert copy.vegetation is veg


def test_copy_constructor_does_not_carry_trajectories():
    original = Analysis()
    original.trajectories.mass = 99.0
    copy = Analysis(copy=original)
    assert copy.trajectories is not original.trajectories
    assert copy.trajectories.mass == 0.0


# ---------------------------------------------------------------------------
# slope / vegetation / maps properties
# ---------------------------------------------------------------------------

def test_slope_unset_raises_runtime_error():
    a = Analysis()
    with pytest.raises(RuntimeError, match="Slope not set"):
        _ = a.slope


def test_vegetation_unset_raises_runtime_error():
    a = Analysis()
    assert a.hasVegetation is False
    with pytest.raises(RuntimeError, match="no vegetation"):
        _ = a.vegetation


def test_maps_unset_raises_runtime_error():
    a = Analysis()
    with pytest.raises(RuntimeError, match="Sample maps not set"):
        _ = a.maps


# ---------------------------------------------------------------------------
# _checkBeforeRun
# ---------------------------------------------------------------------------

def test_check_before_run_no_seeders_raises():
    a = Analysis()
    a.slope = make_flat_slope()
    with pytest.raises(ValueError, match="No seeders defined"):
        a._checkBeforeRun()


def test_check_before_run_rejects_non_seeder():
    a = Analysis()
    a.slope = make_flat_slope()
    a.seeders = ["not a seeder"]
    with pytest.raises(ValueError, match="All seeders must be of type Seeder"):
        a._checkBeforeRun()


def test_check_before_run_no_slope_raises():
    rock = pr.Rock(mass=1.0, density=2500.0)
    seeder = pr.Seeder(points=np.array([0.0, 10.0]), rocks=[rock])
    a = Analysis()
    a.seeders = [seeder]
    with pytest.raises(RuntimeError, match="Slope not set"):
        a._checkBeforeRun()


def test_check_before_run_dimension_mismatch_raises():
    rock = pr.Rock(mass=1.0, density=2500.0)
    seeder_3d = pr.Seeder(points=np.array([0.0, 10.0, 0.0]), rocks=[rock])
    a = Analysis()
    a.seeders = [seeder_3d]
    a.slope = make_flat_slope()  # 2D
    with pytest.raises(ValueError, match="dimensionality must match"):
        a._checkBeforeRun()


def test_check_before_run_distributed_mode_requires_positive_rocks():
    rock = pr.Rock(mass=1.0, density=2500.0)
    seeder = pr.Seeder(points=np.array([0.0, 10.0]), rocks=[rock])
    a = Analysis()
    a.seeders = [seeder]
    a.slope = make_flat_slope()
    a.rockThrowMode = AnalysisRocksThrown.DistributedFromNumberOfRocks
    a.numberOfRocks = 0
    with pytest.raises(ValueError, match="Number of rocks thrown must be greater than 0"):
        a._checkBeforeRun()


def test_check_before_run_invalid_velocity_scale_raises():
    rock = pr.Rock(mass=1.0, density=2500.0)
    seeder = pr.Seeder(points=np.array([0.0, 10.0]), rocks=[rock])
    a = Analysis()
    a.seeders = [seeder]
    a.slope = make_flat_slope()
    a.scaleByVelocity = True
    a.K = 0.0
    with pytest.raises(ValueError, match="Scaling parameter for velocity"):
        a._checkBeforeRun()


def test_check_before_run_invalid_mass_scale_raises():
    rock = pr.Rock(mass=1.0, density=2500.0)
    seeder = pr.Seeder(points=np.array([0.0, 10.0]), rocks=[rock])
    a = Analysis()
    a.seeders = [seeder]
    a.slope = make_flat_slope()
    a.scaleByMass = True
    a.C = -1.0
    with pytest.raises(ValueError, match="Scaling parameter for mass"):
        a._checkBeforeRun()


def test_check_before_run_valid_configuration_passes():
    rock = pr.Rock(mass=1.0, density=2500.0)
    seeder = pr.Seeder(points=np.array([0.0, 10.0]), rocks=[rock])
    a = Analysis()
    a.seeders = [seeder]
    a.slope = make_flat_slope()
    a._checkBeforeRun()  # should not raise
    assert a._ndim == 2


# ---------------------------------------------------------------------------
# _distribute_rocks
# ---------------------------------------------------------------------------

def test_distribute_rocks_splits_evenly_across_seeders():
    rock = pr.Rock(mass=1.0, density=2500.0)
    s1 = pr.Seeder(points=np.array([0.0, 10.0]), rocks=[rock])
    s2 = pr.Seeder(points=np.array([5.0, 10.0]), rocks=[rock])
    a = make_analysis(seeders=[s1, s2])
    a.rockThrowMode = AnalysisRocksThrown.DistributedFromNumberOfRocks
    a.numberOfRocks = 100
    a._distribute_rocks()
    assert s1.numberOfRocks == 50
    assert s2.numberOfRocks == 50
    assert s1.rockThrowMode == SeederRocksThrown.Overall
    assert s2.rockThrowMode == SeederRocksThrown.Overall


def test_distribute_rocks_no_op_when_individually_per_seeder():
    rock = pr.Rock(mass=1.0, density=2500.0)
    s1 = pr.Seeder(points=np.array([0.0, 10.0]), rocks=[rock])
    s1.numberOfRocks = 7
    a = make_analysis(seeders=[s1])
    a.rockThrowMode = AnalysisRocksThrown.IndividuallyPerSeeder
    a._distribute_rocks()
    assert s1.numberOfRocks == 7


# ---------------------------------------------------------------------------
# _prepare_sample_maps
# ---------------------------------------------------------------------------

def test_prepare_sample_maps_overall_mode():
    rock1 = pr.Rock(name="r1", mass=1.0, density=2500.0)
    rock2 = pr.Rock(name="r2", mass=2.0, density=2500.0)
    s1 = pr.Seeder(points=np.array([0.0, 10.0]), rocks=[rock1, rock2])
    s1.numberOfRocks = 50
    s1.rockThrowMode = SeederRocksThrown.Overall
    s2 = pr.Seeder(points=np.array([5.0, 10.0]), rocks=[rock1])
    s2.numberOfRocks = 50
    s2.rockThrowMode = SeederRocksThrown.Overall

    a = make_analysis(seeders=[s1, s2])
    a._prepare_sample_maps()

    assert a.maps.numSamples == 100
    assert len(a.maps.rockSamples[rock1]) == 25 + 50  # from s1 and s2
    assert len(a.maps.rockSamples[rock2]) == 25  # only from s1
    assert len(a.maps.seederSamples[s1]) == 50
    assert len(a.maps.seederSamples[s2]) == 50
    # partitions are disjoint and cover all samples
    all_ids = set(a.maps.seederSamples[s1]) | set(a.maps.seederSamples[s2])
    assert all_ids == set(range(100))


def test_prepare_sample_maps_per_rock_type_mode():
    rock1 = pr.Rock(name="r1", mass=1.0, density=2500.0)
    rock2 = pr.Rock(name="r2", mass=2.0, density=2500.0)
    s1 = pr.Seeder(points=np.array([0.0, 10.0]), rocks=[rock1, rock2])
    s1.numberOfRocks = 10
    s1.rockThrowMode = SeederRocksThrown.PerRockType

    a = make_analysis(seeders=[s1])
    a._prepare_sample_maps()

    assert a.maps.numSamples == 20  # 10 samples per rock type
    assert len(a.maps.rockSamples[rock1]) == 10
    assert len(a.maps.rockSamples[rock2]) == 10
    assert len(a.maps.seederSamples[s1]) == 20


# ---------------------------------------------------------------------------
# _draw_percentiles
# ---------------------------------------------------------------------------

def test_draw_percentiles_monte_carlo_shape_range_and_reproducibility():
    a = Analysis()
    a.samplingMethod = Sampling.MonteCarlo
    a.useSpecificSeed = True
    a.specificSeed = 42
    p1 = a._draw_percentiles(3, 10)
    p2 = a._draw_percentiles(3, 10)
    assert p1.shape == (3, 10)
    assert np.all((p1 >= 0.0) & (p1 < 1.0))
    np.testing.assert_array_equal(p1, p2)


def test_draw_percentiles_latin_hypercube_shape_range_and_reproducibility():
    a = Analysis()
    a.samplingMethod = Sampling.LatinHypercube
    a.useSpecificSeed = True
    a.specificSeed = 42
    p1 = a._draw_percentiles(3, 10)
    p2 = a._draw_percentiles(3, 10)
    assert p1.shape == (3, 10)
    assert np.all((p1 >= 0.0) & (p1 <= 1.0))
    np.testing.assert_array_equal(p1, p2)


def test_draw_percentiles_without_specific_seed_varies():
    a = Analysis()
    a.samplingMethod = Sampling.MonteCarlo
    a.useSpecificSeed = False
    p1 = a._draw_percentiles(2, 50)
    p2 = a._draw_percentiles(2, 50)
    assert not np.array_equal(p1, p2)


# ---------------------------------------------------------------------------
# _calcSlopeNormals
# ---------------------------------------------------------------------------

def test_calc_slope_normals_2d_horizontal_segment_points_up():
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    slope = pr.Slope(pr.Geometry(nodes=nodes), materials=pr.Material(name="A"))
    a = make_analysis(slope=slope)
    a._ndim = 2
    profiles = slope.nodes.T
    n = a._calcSlopeNormals(profiles)
    np.testing.assert_allclose(n, [[0.0, 0.0], [1.0, 1.0]], atol=1e-12)


def test_calc_slope_normals_3d_triangle_faces_up():
    nodes = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    elements = np.array([[0, 1, 2]])
    slope = pr.Slope(pr.Geometry3D(nodes=nodes, elements=elements), materials=pr.Material(name="A"))
    a = make_analysis(slope=slope)
    a._ndim = 3
    n = a._calcSlopeNormals(slope.nodes.T)
    np.testing.assert_allclose(n, [[0.0], [0.0], [1.0]], atol=1e-12)


def test_calc_slope_normals_rejects_dimension_mismatch():
    nodes = np.array([[0.0, 0.0], [1.0, 0.0]])
    slope = pr.Slope(pr.Geometry(nodes=nodes), materials=pr.Material(name="A"))
    a = make_analysis(slope=slope)
    a._ndim = 2
    profiles_3d = np.zeros((3, 2))
    with pytest.raises(ValueError, match="2-node elements require D=2"):
        a._calcSlopeNormals(profiles_3d)


# ---------------------------------------------------------------------------
# _addRoughness
# ---------------------------------------------------------------------------

def test_add_roughness_perturbs_angle_and_advances_impact_counters():
    mat0 = pr.Material(name="M0")
    mat1 = pr.Material(name="M1")
    slope = pr.Slope(pr.Geometry(nodes=np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])), materials=[mat0, mat1])
    a = make_analysis(slope=slope)

    normal = np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])  # samples on elem0(mat0), elem0(mat0), elem1(mat1)
    element_id = np.array([0, 0, 1])
    roughness_samples = np.array([[10.0, 20.0, 30.0], [0.0, 0.0, 0.0]])
    cum_impacts = np.zeros(2, dtype=int)

    out = a._addRoughness(normal, element_id, roughness_samples, cum_impacts)

    theta0 = np.degrees(np.arctan2(normal[1], normal[0]))
    expected_theta = theta0 + np.array([10.0, 20.0, 0.0])
    actual_theta = np.degrees(np.arctan2(out[1], out[0]))
    np.testing.assert_allclose(actual_theta, expected_theta, atol=1e-8)
    np.testing.assert_allclose(np.linalg.norm(out, axis=0), 1.0)  # unit normals
    np.testing.assert_array_equal(cum_impacts, [2, 1])


def test_add_roughness_wraps_around_when_out_of_samples():
    mat0 = pr.Material(name="M0")
    slope = pr.Slope(pr.Geometry(nodes=np.array([[0.0, 0.0], [1.0, 0.0]])), materials=mat0)
    a = make_analysis(slope=slope)

    normal = np.array([[1.0, 1.0, 1.0]], dtype=float)
    normal = np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]])
    element_id = np.array([0, 0, 0])
    roughness_samples = np.array([[5.0, -5.0]])  # only 2 samples for 3 impacts
    cum_impacts = np.zeros(1, dtype=int)

    out = a._addRoughness(normal, element_id, roughness_samples, cum_impacts)
    theta = np.degrees(np.arctan2(out[1], out[0]))
    np.testing.assert_allclose(theta, [5.0, -5.0, 5.0], atol=1e-8)  # wraps back to sample 0
    assert cum_impacts[0] == 3


# ---------------------------------------------------------------------------
# _impactMaterial / _impactDrag
# ---------------------------------------------------------------------------

def test_impact_material_gathers_per_sample_params_and_nans_misses():
    mat0 = pr.Material(name="M0")
    mat1 = pr.Material(name="M1")
    slope = pr.Slope(pr.Geometry(nodes=np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])), materials=[mat0, mat1])
    a = make_analysis(slope=slope)

    material_params = np.zeros((2, 2, 3))
    material_params[:, 0, :] = [[1, 2, 3], [4, 5, 6]]
    material_params[:, 1, :] = [[7, 8, 9], [10, 11, 12]]
    element_id = np.array([0, -1, 1])

    out = a._impactMaterial(element_id, material_params)
    np.testing.assert_allclose(out[:, 0], [1.0, 4.0])
    assert np.all(np.isnan(out[:, 1]))
    np.testing.assert_allclose(out[:, 2], [9.0, 12.0])


def test_impact_drag_gathers_per_sample_params_and_nans_misses():
    d0 = pr.Drag(name="D0")
    d1 = pr.Drag(name="D1")
    veg = pr.Vegetation(pr.Geometry(nodes=np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])), drag=[d0, d1])
    a = Analysis()
    a.vegetation = veg

    drag_params = np.zeros((1, 2, 3))
    drag_params[:, 0, :] = [1.5, 2.5, 3.5]
    drag_params[:, 1, :] = [7.0, 8.0, 9.0]
    segment_id = np.array([0, -1, 1])

    out = a._impactDrag(segment_id, drag_params)
    np.testing.assert_allclose(out[0], [1.5, np.nan, 9.0])


# ---------------------------------------------------------------------------
# _isImpacting
# ---------------------------------------------------------------------------

def test_is_impacting_negative_normal_velocity():
    a = Analysis()
    result = a._isImpacting(np.array([-1.0, 0.0, 1.0]))
    np.testing.assert_array_equal(result, [True, False, False])


# ---------------------------------------------------------------------------
# _impacts
# ---------------------------------------------------------------------------

def test_impacts_without_rotation_scales_by_restitution_only():
    a = Analysis()
    a._ndim = 2
    vn = np.array([-5.0])
    vt = np.array([[2.0], [0.0]])
    w = np.array([[0.0]])
    mass = np.array([10.0])
    density = np.array([2500.0])
    rn = np.array([0.5])
    rt = np.array([0.8])
    normal = np.array([[0.0], [1.0]])

    vn_post, vt_post, w_post = a._impacts(vn, vt, w, mass, density, rn, rt, normal)
    assert vn_post[0] == pytest.approx(2.5)  # -rn*vn = -0.5*-5.0
    np.testing.assert_allclose(vt_post.ravel(), [1.6, 0.0])  # vt*rt
    np.testing.assert_allclose(w_post, w)  # angular velocity untouched without rotation


def test_impacts_scale_by_velocity():
    a = Analysis()
    a._ndim = 2
    a.scaleByVelocity = True
    a.K = 10.0
    vn = np.array([-5.0])
    vt = np.array([[0.0], [0.0]])
    w = np.array([[0.0]])
    mass = np.array([10.0])
    density = np.array([2500.0])
    rn = np.array([0.5])
    rt = np.array([0.8])
    normal = np.array([[0.0], [1.0]])

    vn_post, _, _ = a._impacts(vn, vt, w, mass, density, rn, rt, normal)
    scln = 1 + (vn / a.K) ** 2
    expected = -rn / scln * vn
    np.testing.assert_allclose(vn_post, expected)


def test_impacts_scale_by_mass():
    a = Analysis()
    a._ndim = 2
    a.scaleByMass = True
    a.C = 5.0
    vn = np.array([-3.0])
    vt = np.array([[0.0], [0.0]])
    w = np.array([[0.0]])
    mass = np.array([20.0])
    density = np.array([2500.0])
    rn = np.array([0.4])
    rt = np.array([0.8])
    normal = np.array([[0.0], [1.0]])

    vn_post, _, _ = a._impacts(vn, vt, w, mass, density, rn, rt, normal)
    scln = 1 + (mass / a.C) ** 2
    expected = -rn / scln * vn
    np.testing.assert_allclose(vn_post, expected)


# ---------------------------------------------------------------------------
# _isSliding
# ---------------------------------------------------------------------------

def test_is_sliding_stopped_on_flat_ground_stays_stopped():
    a = Analysis()
    a._ndim = 2
    a.gravity = -10.0
    a.stoppedVelocity = 1e-5
    normal = np.array([[0.0], [1.0]])
    vt_stopped = np.array([[0.0], [0.0]])
    phi = np.array([30.0])
    result = a._isSliding(vt_stopped, normal, phi)
    assert result[0] == False  # no tangential gravity on flat ground -> friction holds


def test_is_sliding_stopped_on_steep_low_friction_slope_starts_sliding():
    a = Analysis()
    a._ndim = 2
    a.gravity = -10.0
    a.stoppedVelocity = 1e-5
    theta = np.radians(60)
    normal = np.array([[np.sin(theta)], [np.cos(theta)]])
    vt_stopped = np.array([[0.0], [0.0]])
    result = a._isSliding(vt_stopped, normal, np.array([10.0]))
    assert result[0] == True


def test_is_sliding_already_moving_is_always_sliding():
    a = Analysis()
    a._ndim = 2
    a.gravity = -10.0
    a.stoppedVelocity = 1e-5
    normal = np.array([[0.0], [1.0]])
    vt_moving = np.array([[1.0], [0.0]])
    result = a._isSliding(vt_moving, normal, np.array([89.0]))
    assert result[0] == True


# ---------------------------------------------------------------------------
# _isFalling
# ---------------------------------------------------------------------------

def test_is_falling_threshold_and_downward_normal():
    a = Analysis()
    a.normalVelocityThreshold = 0.1
    vn = np.array([0.05, 0.5, 0.05])
    normals = np.array([[0.0, 0.0, 0.0], [1.0, -0.5, -0.5]])
    result = a._isFalling(vn, normals)
    np.testing.assert_array_equal(result, [False, True, True])


# ---------------------------------------------------------------------------
# _slide (regression test for the is_sticking fix)
# ---------------------------------------------------------------------------

def test_slide_block_starting_at_rest_accelerates_on_frictionless_slope():
    nodes = np.array([[-10.0, 10.0], [10.0, -10.0]])  # 45-degree line
    slope = pr.Slope(pr.Geometry(nodes=nodes), materials=pr.Material(name="A"))
    a = make_analysis(slope=slope)
    a._ndim = 2
    a.gravity = -10.0
    a.timeStep = 0.01
    a.stoppedVelocity = 1e-5

    normal = np.array([[1 / np.sqrt(2)], [1 / np.sqrt(2)]])
    pos = np.array([[0.0], [0.0]])
    vt = np.array([[0.0], [0.0]])
    phi = np.array([0.0])  # frictionless
    profiles = slope.nodes.T[:, :, None]
    elem_id = np.array([0])

    speeds = []
    for _ in range(6):
        pos, vt, a0, dt, elem_id = a._slide(pos, vt, normal, phi, profiles, elem_id)
        speeds.append(float(np.linalg.norm(vt)))

    # must accelerate away from rest -- this used to stay stuck at exactly 0
    assert speeds[0] > 0.0
    assert all(b > a_ for a_, b in zip(speeds, speeds[1:]))  # monotonically increasing
    # matches constant-acceleration physics: |gt| = g*sin(45deg)
    expected_speed_gain_per_step = 10.0 * np.sin(np.radians(45)) * a.timeStep
    np.testing.assert_allclose(speeds[0], expected_speed_gain_per_step, rtol=1e-6)


def test_slide_with_friction_eventually_stops():
    nodes = np.array([[-10.0, 0.0], [10.0, 0.0]])  # flat ground
    slope = pr.Slope(pr.Geometry(nodes=nodes), materials=pr.Material(name="A"))
    a = make_analysis(slope=slope)
    a._ndim = 2
    a.gravity = -10.0
    a.timeStep = 0.01
    a.stoppedVelocity = 1e-5

    normal = np.array([[0.0], [1.0]])
    pos = np.array([[0.0], [0.0]])
    vt = np.array([[2.0], [0.0]])  # already moving
    phi = np.array([30.0])  # friction on flat ground must decelerate it
    profiles = slope.nodes.T[:, :, None]
    elem_id = np.array([0])

    for _ in range(200):
        pos, vt, a0, dt, elem_id = a._slide(pos, vt, normal, phi, profiles, elem_id)
        if np.linalg.norm(vt) == 0.0:
            break

    assert np.linalg.norm(vt) == 0.0  # friction brought it to a full stop


# ---------------------------------------------------------------------------
# _fall
# ---------------------------------------------------------------------------

def test_fall_matches_analytical_free_fall_time():
    slope = make_flat_slope(x0=-50.0, x1=50.0, y=0.0)
    a = make_analysis(slope=slope)
    a._ndim = 2
    a.gravity = -9.81
    a.tolerance = 1e-7

    H = 20.0
    pos = np.array([[0.0], [H]])
    vel = np.zeros((2, 1))
    profiles = slope.nodes.T[:, :, None]
    element_id = np.array([-1])
    segment_id = np.array([-1])
    canopy = np.empty((1, 0, 1))

    pos2, vel2, dt, elem2, seg2 = a._fall(pos, vel, profiles, element_id, segment_id, canopy=canopy)
    t_ref = np.sqrt(2 * H / 9.81)
    assert dt[0] == pytest.approx(t_ref, rel=1e-6)
    np.testing.assert_allclose(pos2[:, 0], [0.0, 0.0], atol=1e-8)
    assert vel2[1, 0] == pytest.approx(-9.81 * t_ref, rel=1e-6)
    # lands on the actual floor segment (element 0), not the FLOOR_ELEMENT_ID
    # fallback -- that sentinel is only used when no real segment is hit.
    assert elem2[0] == 0


# ---------------------------------------------------------------------------
# postprocess (regression test for the floor fix)
# ---------------------------------------------------------------------------

def test_postprocess_sets_floor_mass_and_inertia():
    # A profile where the last node is deliberately NOT the lowest point,
    # to distinguish the fix from the old (broken) nodes[-1] indexing.
    nodes = np.array([[0.0, 4.0], [1.0, 0.5], [2.0, 2.0], [3.0, 3.0]])
    slope = pr.Slope(pr.Geometry(nodes=nodes), materials=pr.Material(name="A"))
    a = make_analysis(slope=slope)

    rock_params = np.array([[10.0, 20.0], [2500.0, 2500.0]])  # mass, density
    a.postprocess(rock_params)

    assert a.trajectories.floor == pytest.approx(0.5)
    assert a.trajectories.floor == pytest.approx(slope.floor)
    np.testing.assert_allclose(a.trajectories.mass, [10.0, 20.0])
    r = np.cbrt(3 * rock_params[0] / (4 * np.pi * rock_params[1]))
    expected_inertia = 2 / 5 * rock_params[0] * r ** 2
    np.testing.assert_allclose(a.trajectories.inertia, expected_inertia)


# ---------------------------------------------------------------------------
# End-to-end run()
# ---------------------------------------------------------------------------

def test_run_vertical_drop_with_full_absorption_lands_at_floor():
    slope = make_flat_slope(x0=-50.0, x1=50.0, y=0.0, normalRestitution=0.0, tangentialRestitution=0.5, frictionAngle=30.0)
    rock = pr.Rock(name="R", mass=10.0, density=2500.0)
    seeder = pr.Seeder(points=np.array([0.0, 20.0]), rocks=[rock])
    seeder.numberOfRocks = 3
    seeder.translationalVelocity = [0.0, 0.0]
    seeder.angularVelocity = [0.0]

    a = Analysis()
    a.seeders = [seeder]
    a.slope = slope
    a.samplingMethod = Sampling.MonteCarlo
    a.useSpecificSeed = True
    a.specificSeed = 1
    a.maxIter = 500
    a.gravity = -9.81
    a.stoppedVelocity = 1e-3
    a.normalVelocityThreshold = 0.1

    a.run()

    assert a.trajectories.floor == pytest.approx(slope.floor)
    np.testing.assert_allclose(a.trajectories.mass, [10.0, 10.0, 10.0])
    final_pos = a.trajectories._position_history[-1]
    np.testing.assert_allclose(final_pos[1], 0.0, atol=1e-6)  # settled at the floor


def test_run_block_slides_down_incline_and_settles_on_frictional_runout():
    # Regression test for the _slide fix: before it, a block starting a
    # slide step at rest never gained velocity and would not settle in a
    # bounded number of iterations.
    nodes = np.array([[0.0, 10.0], [10.0, 0.0], [30.0, 0.0]])
    material = pr.Material(name="A", normalRestitution=0.2, tangentialRestitution=0.6, frictionAngle=20.0)
    slope = pr.Slope(pr.Geometry(nodes=nodes), materials=material)

    rock = pr.Rock(name="R", mass=10.0, density=2500.0)
    seeder = pr.Seeder(points=np.array([2.0, 8.5]), rocks=[rock])  # small drop above the incline
    seeder.numberOfRocks = 2
    seeder.translationalVelocity = [0.0, 0.0]
    seeder.angularVelocity = [0.0]

    a = Analysis()
    a.seeders = [seeder]
    a.slope = slope
    a.samplingMethod = Sampling.MonteCarlo
    a.useSpecificSeed = True
    a.specificSeed = 1
    a.maxIter = 3000
    a.gravity = -9.81
    a.stoppedVelocity = 1e-3
    a.normalVelocityThreshold = 0.1
    a.timeStep = 0.01

    a.run()

    num_iterations = a.trajectories._position_history.shape[0]
    assert num_iterations < a.maxIter  # terminated on its own, not by hitting the cap

    final_pos = a.trajectories._position_history[-1]
    final_vel = a.trajectories._velocity_history[-1]
    # settled somewhere on the flat run-out (between the toe of the incline and its end)
    assert np.all((final_pos[0] >= 10.0 - 1e-6) & (final_pos[0] <= 30.0))
    np.testing.assert_allclose(final_pos[1], 0.0, atol=1e-6)
    np.testing.assert_allclose(final_vel, 0.0, atol=1e-6)


def test_run_raises_when_not_configured():
    a = Analysis()
    with pytest.raises(ValueError, match="No seeders defined"):
        a.run()
