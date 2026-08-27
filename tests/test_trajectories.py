"""Tests for pyrockfall.Trajectories: interpolating position/velocity/
acceleration between recorded simulation snapshots, in 2D and 3D.

Trajectories.addData() records instantaneous "event" snapshots (time,
position, velocity, angular velocity, acceleration) -- e.g. one per impact,
matching how Analysis._saveTrajectories() is actually called. Between two
consecutive snapshots, motion is assumed to have constant acceleration
(equal to the value stored at the *later* snapshot), so
position(t)/velocity(t) reconstruct the start-of-interval velocity via
`v_start = v_end - a * Dt` and then integrate forward with the standard
constant-acceleration kinematics:

    p(t) = p_start + v_start * dt + 0.5 * a * dt**2
    v(t) = v_start + a * dt

The core of these tests builds free-fall (and fall-then-bounce) fixtures by
hand -- with known closed-form positions/velocities at every instant -- and
checks that querying intermediate times reproduces that closed form, for
both 2D and 3D trajectories and for multiple blocks at once.
"""
import numpy as np
import pytest

from pyrockfall._trajectories import Trajectories

DIMS = [2, 3]


# ---------------------------------------------------------------------------
# Fixture builders: hand-computable free-fall (and fall-then-bounce) motion.
#
# Convention (matching the rest of the package): the *last* axis is vertical
# and subject to gravity; all other axes carry constant horizontal velocity.
# ---------------------------------------------------------------------------

def _zero_angvel(ndim, n_blocks):
    return np.zeros((1 if ndim == 2 else ndim, n_blocks))


def make_free_fall_trajectory(ndim, heights, horiz_vel, g=9.81):
    """Two-snapshot single free-fall segment for len(heights) blocks."""
    n_blocks = len(heights)
    H = np.asarray(heights, dtype=float)
    v_horiz = np.asarray(horiz_vel, dtype=float)  # (ndim-1, n_blocks)
    T = np.sqrt(2 * H / g)

    p0 = np.zeros((ndim, n_blocks))
    p0[-1] = H
    v0 = np.zeros((ndim, n_blocks))
    v0[:-1] = v_horiz
    a0 = np.zeros((ndim, n_blocks))
    a0[-1] = -g

    p1 = np.zeros((ndim, n_blocks))
    p1[:-1] = v_horiz * T
    v1 = v0.copy()
    v1[-1] = -g * T
    a1 = a0.copy()

    traj = Trajectories()
    traj.gravity = g
    traj.mass = 1.0
    traj.addData(np.zeros(n_blocks), p0, v0, _zero_angvel(ndim, n_blocks), a0)
    traj.addData(T, p1, v1, _zero_angvel(ndim, n_blocks), a1)
    traj.simulationDone()
    return traj, H, v_horiz, T


def make_bounce_trajectory(ndim, height, horiz_vel1, horiz_vel2, restitution, g=9.81):
    """Three-snapshot fall -> bounce -> fall trajectory, single block."""
    H = float(height)
    v1 = np.asarray(horiz_vel1, dtype=float)
    v2 = np.asarray(horiz_vel2, dtype=float)
    T1 = np.sqrt(2 * H / g)
    vy0_2 = restitution * g * T1  # post-bounce upward speed
    T2 = 2 * vy0_2 / g  # time to return to the floor from the bounce

    p0 = np.zeros((ndim, 1))
    p0[-1, 0] = H
    v0 = np.zeros((ndim, 1))
    v0[:-1, 0] = v1
    a0 = np.zeros((ndim, 1))
    a0[-1, 0] = -g

    p1 = np.zeros((ndim, 1))
    p1[:-1, 0] = v1 * T1
    v1_snap = v0.copy()
    v1_snap[-1, 0] = -g * T1
    a1 = a0.copy()

    p2 = np.zeros((ndim, 1))
    p2[:-1, 0] = p1[:-1, 0] + v2 * T2
    v2_snap = np.zeros((ndim, 1))
    v2_snap[:-1, 0] = v2
    v2_snap[-1, 0] = vy0_2 - g * T2
    a2 = a0.copy()

    traj = Trajectories()
    traj.gravity = g
    traj.mass = 1.0
    traj.addData(np.zeros(1), p0, v0, _zero_angvel(ndim, 1), a0)
    traj.addData(np.array([T1]), p1, v1_snap, _zero_angvel(ndim, 1), a1)
    traj.addData(np.array([T1 + T2]), p2, v2_snap, _zero_angvel(ndim, 1), a2)
    traj.simulationDone()
    return traj, H, v1, v2, T1, T2, vy0_2


# ---------------------------------------------------------------------------
# Single free-fall segment: intermediate position/velocity, single block
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ndim", DIMS)
def test_position_at_segment_midpoint_matches_free_fall(ndim):
    horiz_vel = np.array([[3.0]]) if ndim == 2 else np.array([[3.0], [1.5]])
    traj, H, v_horiz, T = make_free_fall_trajectory(ndim, heights=[20.0], horiz_vel=horiz_vel, g=9.81)

    t_mid = float(T[0]) / 2.0
    pos = traj.position(t_mid)
    vel = traj.velocity(t_mid)

    expected_pos = np.zeros(ndim)
    expected_pos[:-1] = v_horiz[:, 0] * t_mid
    expected_pos[-1] = H[0] - 0.5 * traj.gravity * t_mid ** 2

    expected_vel = np.zeros(ndim)
    expected_vel[:-1] = v_horiz[:, 0]
    expected_vel[-1] = -traj.gravity * t_mid

    np.testing.assert_allclose(pos[:, 0], expected_pos, rtol=1e-8)
    np.testing.assert_allclose(vel[:, 0], expected_vel, rtol=1e-8)


@pytest.mark.parametrize("ndim", DIMS)
def test_position_and_velocity_at_several_intermediate_fractions(ndim):
    horiz_vel = np.array([[4.0]]) if ndim == 2 else np.array([[4.0], [-2.0]])
    traj, H, v_horiz, T = make_free_fall_trajectory(ndim, heights=[12.0], horiz_vel=horiz_vel, g=9.81)

    fractions = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
    for frac in fractions:
        t = float(T[0]) * frac
        pos = traj.position(t)
        vel = traj.velocity(t)

        expected_pos = np.zeros(ndim)
        expected_pos[:-1] = v_horiz[:, 0] * t
        expected_pos[-1] = H[0] - 0.5 * traj.gravity * t ** 2
        expected_vel = np.zeros(ndim)
        expected_vel[:-1] = v_horiz[:, 0]
        expected_vel[-1] = -traj.gravity * t

        np.testing.assert_allclose(pos[:, 0], expected_pos, rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(vel[:, 0], expected_vel, rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("ndim", DIMS)
def test_position_at_exact_recorded_snapshots(ndim):
    horiz_vel = np.array([[3.0]]) if ndim == 2 else np.array([[3.0], [1.0]])
    traj, H, v_horiz, T = make_free_fall_trajectory(ndim, heights=[20.0], horiz_vel=horiz_vel, g=9.81)

    pos_start = traj.position(0.0)
    expected_start = np.zeros(ndim)
    expected_start[-1] = H[0]
    np.testing.assert_allclose(pos_start[:, 0], expected_start, atol=1e-8)

    pos_end = traj.position(float(T[0]))
    expected_end = np.zeros(ndim)
    expected_end[:-1] = v_horiz[:, 0] * T[0]
    np.testing.assert_allclose(pos_end[:, 0], expected_end, atol=1e-8)


@pytest.mark.parametrize("ndim", DIMS)
def test_vectorised_time_query_matches_scalar_queries(ndim):
    horiz_vel = np.array([[3.0]]) if ndim == 2 else np.array([[3.0], [1.0]])
    traj, H, v_horiz, T = make_free_fall_trajectory(ndim, heights=[20.0], horiz_vel=horiz_vel, g=9.81)

    times = np.array([0.2, 0.6, 1.2]) * float(T[0]) / 1.2  # a few points within [0, T]
    pos_vec = traj.position(times)
    assert pos_vec.shape == (ndim, times.size, 1)
    for i, t in enumerate(times):
        pos_scalar = traj.position(float(t))
        np.testing.assert_allclose(pos_vec[:, i, 0], pos_scalar[:, 0], atol=1e-10)


# ---------------------------------------------------------------------------
# Multiple blocks at once, with per-block distinct query times
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ndim", DIMS)
def test_multiple_blocks_interpolated_independently(ndim):
    heights = [20.0, 5.0, 45.0]
    horiz_vel = np.array([[3.0, 1.0, 2.0]]) if ndim == 2 else np.array([[3.0, 1.0, 2.0], [-1.0, 0.5, 0.0]])
    traj, H, v_horiz, T = make_free_fall_trajectory(ndim, heights=heights, horiz_vel=horiz_vel, g=9.81)

    assert traj.numberOfBlocks == 3

    # per-block distinct query times: each block queried at its own T/3
    t_query = (T / 3.0).reshape(1, -1)
    pos = traj.position(t_query)
    vel = traj.velocity(t_query)
    assert pos.shape == (ndim, 3)

    for b in range(3):
        t = T[b] / 3.0
        expected_pos = np.zeros(ndim)
        expected_pos[:-1] = v_horiz[:, b] * t
        expected_pos[-1] = H[b] - 0.5 * traj.gravity * t ** 2
        expected_vel = np.zeros(ndim)
        expected_vel[:-1] = v_horiz[:, b]
        expected_vel[-1] = -traj.gravity * t
        np.testing.assert_allclose(pos[:, b], expected_pos, rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(vel[:, b], expected_vel, rtol=1e-8, atol=1e-10)


def test_common_scalar_time_applies_to_every_block():
    heights = [20.0, 5.0]
    horiz_vel = np.array([[3.0, 1.0]])
    traj, H, v_horiz, T = make_free_fall_trajectory(2, heights=heights, horiz_vel=horiz_vel, g=9.81)

    t_common = float(min(T)) * 0.5  # within both blocks' flight time
    pos = traj.position(t_common)
    for b in range(2):
        expected_x = v_horiz[0, b] * t_common
        expected_y = H[b] - 0.5 * traj.gravity * t_common ** 2
        np.testing.assert_allclose(pos[:, b], [expected_x, expected_y], rtol=1e-8)


# ---------------------------------------------------------------------------
# Multi-segment (fall -> bounce -> fall) trajectory
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ndim", DIMS)
def test_bounce_trajectory_first_segment_matches_free_fall(ndim):
    horiz_vel1 = np.array([3.0]) if ndim == 2 else np.array([3.0, 1.0])
    horiz_vel2 = np.array([2.7]) if ndim == 2 else np.array([2.7, 0.9])
    traj, H, v1, v2, T1, T2, vy0_2 = make_bounce_trajectory(
        ndim, height=20.0, horiz_vel1=horiz_vel1, horiz_vel2=horiz_vel2, restitution=0.6
    )

    t_mid = T1 / 2.0
    pos = traj.position(t_mid)
    expected_pos = np.zeros(ndim)
    expected_pos[:-1] = v1 * t_mid
    expected_pos[-1] = H - 0.5 * traj.gravity * t_mid ** 2
    np.testing.assert_allclose(pos[:, 0], expected_pos, rtol=1e-8)


@pytest.mark.parametrize("ndim", DIMS)
def test_bounce_trajectory_second_segment_matches_post_bounce_projectile(ndim):
    horiz_vel1 = np.array([3.0]) if ndim == 2 else np.array([3.0, 1.0])
    horiz_vel2 = np.array([2.7]) if ndim == 2 else np.array([2.7, 0.9])
    traj, H, v1, v2, T1, T2, vy0_2 = make_bounce_trajectory(
        ndim, height=20.0, horiz_vel1=horiz_vel1, horiz_vel2=horiz_vel2, restitution=0.6
    )

    # apex of the bounce: vertical velocity should be ~0
    t_apex = T1 + T2 / 2.0
    vel_apex = traj.velocity(t_apex)
    assert vel_apex[-1, 0] == pytest.approx(0.0, abs=1e-8)
    np.testing.assert_allclose(vel_apex[:-1, 0], v2, rtol=1e-8)

    expected_apex_height = vy0_2 * (T2 / 2.0) - 0.5 * traj.gravity * (T2 / 2.0) ** 2
    pos_apex = traj.position(t_apex)
    assert pos_apex[-1, 0] == pytest.approx(expected_apex_height, rel=1e-8)

    # landing back at the floor at the end of the second segment
    pos_land = traj.position(T1 + T2)
    assert pos_land[-1, 0] == pytest.approx(0.0, abs=1e-8)


# ---------------------------------------------------------------------------
# acceleration()
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ndim", DIMS)
def test_acceleration_matches_stored_constant_acceleration(ndim):
    horiz_vel = np.array([[3.0]]) if ndim == 2 else np.array([[3.0], [1.0]])
    traj, H, v_horiz, T = make_free_fall_trajectory(ndim, heights=[20.0], horiz_vel=horiz_vel, g=9.81)

    acc = traj.acceleration(float(T[0]) / 2.0)
    expected = np.zeros(ndim)
    expected[-1] = -traj.gravity
    np.testing.assert_allclose(acc[:, 0], expected, atol=1e-10)


# ---------------------------------------------------------------------------
# NaN handling and out-of-range characterization
# ---------------------------------------------------------------------------

def test_nan_time_propagates_to_nan_output():
    traj, H, v_horiz, T = make_free_fall_trajectory(2, heights=[20.0], horiz_vel=np.array([[3.0]]), g=9.81)
    pos = traj.position(np.nan)
    vel = traj.velocity(np.nan)
    assert np.all(np.isnan(pos))
    assert np.all(np.isnan(vel))


def test_query_beyond_stop_time_clamps_to_final_state():
    traj, H, v_horiz, T = make_free_fall_trajectory(2, heights=[20.0], horiz_vel=np.array([[3.0]]), g=9.81)
    pos_at_stop = traj.position(float(T[0]))
    pos_beyond = traj.position(float(T[0]) + 100.0)
    np.testing.assert_allclose(pos_beyond, pos_at_stop, atol=1e-8)


def test_query_before_start_time_extrapolates_backward():
    # Characterization, not a correctness claim: querying before the first
    # recorded time is NOT clamped (unlike querying beyond the last time) --
    # it extrapolates backward using the first segment's constant
    # acceleration/velocity, which can give a "position" before t=0.
    traj, H, v_horiz, T = make_free_fall_trajectory(2, heights=[20.0], horiz_vel=np.array([[3.0]]), g=9.81)
    pos_before = traj.position(-1.0)
    expected = np.array([3.0 * -1.0, 20.0 - 0.5 * traj.gravity * 1.0 ** 2])
    np.testing.assert_allclose(pos_before[:, 0], expected, atol=1e-8)


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ndim", DIMS)
def test_history_properties_have_expected_shapes(ndim):
    horiz_vel = np.array([[3.0, 1.0]]) if ndim == 2 else np.array([[3.0, 1.0], [0.5, -0.5]])
    traj, H, v_horiz, T = make_free_fall_trajectory(ndim, heights=[20.0, 5.0], horiz_vel=horiz_vel, g=9.81)

    assert traj.timeHistory.shape == (2, 2)
    assert traj.positionHistory.shape == (2, ndim, 2)
    assert traj.velocityHistory.shape == (2, ndim, 2)
    assert traj.accelerationHistory.shape == (2, ndim, 2)
    assert traj.angularVelocityHistory.shape == (2, 1 if ndim == 2 else ndim, 2)
    assert traj.numberOfBlocks == 2


@pytest.mark.parametrize("ndim", DIMS)
def test_impact_arrays_are_transposed_history(ndim):
    horiz_vel = np.array([[3.0]]) if ndim == 2 else np.array([[3.0], [1.0]])
    traj, H, v_horiz, T = make_free_fall_trajectory(ndim, heights=[20.0], horiz_vel=horiz_vel, g=9.81)

    np.testing.assert_allclose(traj.impactPoints, np.transpose(traj.positionHistory, (1, 0, 2)))
    np.testing.assert_allclose(traj.impactVelocities, np.transpose(traj.velocityHistory, (1, 0, 2)))
    np.testing.assert_allclose(
        traj.impactAngularVelocities, np.transpose(traj.angularVelocityHistory, (1, 0, 2))
    )


def test_start_stop_time_and_endpoints():
    traj, H, v_horiz, T = make_free_fall_trajectory(2, heights=[20.0, 5.0], horiz_vel=np.array([[3.0, 1.0]]), g=9.81)
    np.testing.assert_allclose(traj.startTime, [0.0, 0.0])
    np.testing.assert_allclose(traj.stopTime, T)
    np.testing.assert_allclose(traj.endpoints, traj.positionHistory[-1])


def test_reason_stopped_defaults_to_stopped_per_block():
    traj, H, v_horiz, T = make_free_fall_trajectory(2, heights=[20.0, 5.0], horiz_vel=np.array([[3.0, 1.0]]), g=9.81)
    assert traj.reasonStopped() == ["Stopped", "Stopped"]


def test_call_interpolates_between_start_and_stop():
    traj, H, v_horiz, T = make_free_fall_trajectory(2, heights=[20.0], horiz_vel=np.array([[3.0]]), g=9.81)
    result = traj(numPoints=50)
    assert result.shape == (2, 50, 1)
    np.testing.assert_allclose(result[:, 0, 0], traj.position(0.0)[:, 0], atol=1e-8)
    np.testing.assert_allclose(result[:, -1, 0], traj.position(float(T[0]))[:, 0], atol=1e-8)


# ---------------------------------------------------------------------------
# collector(): first-crossing-time detection
# ---------------------------------------------------------------------------

def test_collector_vertical_line_matches_horizontal_travel_time():
    traj, H, v_horiz, T = make_free_fall_trajectory(2, heights=[20.0], horiz_vel=np.array([[3.0]]), g=9.81)
    x_c = float(v_horiz[0, 0] * T[0] / 2.0)
    points = np.array([[x_c], [0.0]])
    normals = np.array([[1.0], [0.0]])
    t_cross = traj.collector(points, normals)
    assert t_cross[0, 0] == pytest.approx(x_c / v_horiz[0, 0], rel=1e-6)


def test_collector_horizontal_line_matches_fall_time():
    traj, H, v_horiz, T = make_free_fall_trajectory(2, heights=[20.0], horiz_vel=np.array([[3.0]]), g=9.81)
    target_height = H[0] / 2.0
    points = np.array([[0.0], [target_height]])
    normals = np.array([[0.0], [1.0]])
    t_cross = traj.collector(points, normals)
    expected = np.sqrt(2 * (H[0] - target_height) / traj.gravity)
    assert t_cross[0, 0] == pytest.approx(expected, rel=1e-6)


def test_collector_no_crossing_returns_nan():
    traj, H, v_horiz, T = make_free_fall_trajectory(2, heights=[20.0], horiz_vel=np.array([[3.0]]), g=9.81)
    points = np.array([[0.0], [1000.0]])  # far above the trajectory's reach
    normals = np.array([[0.0], [1.0]])
    t_cross = traj.collector(points, normals)
    assert np.isnan(t_cross[0, 0])


# ---------------------------------------------------------------------------
# writeNPZ / readNPZ round trip
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ndim", DIMS)
def test_write_npz_read_npz_roundtrip(tmp_path, ndim):
    horiz_vel = np.array([[3.0]]) if ndim == 2 else np.array([[3.0], [1.0]])
    traj, H, v_horiz, T = make_free_fall_trajectory(ndim, heights=[20.0], horiz_vel=horiz_vel, g=9.81)
    traj.mass = 2.0
    traj.inertia = 0.5
    traj.floor = 0.0

    path = traj.writeNPZ(str(tmp_path / "traj"))
    assert path.endswith(".npz")

    loaded = Trajectories()
    loaded.readNPZ(path)

    np.testing.assert_allclose(loaded.timeHistory, traj.timeHistory)
    np.testing.assert_allclose(loaded.positionHistory, traj.positionHistory)
    np.testing.assert_allclose(loaded.velocityHistory, traj.velocityHistory)
    assert loaded.mass == pytest.approx(traj.mass)
    assert loaded.inertia == pytest.approx(traj.inertia)
    assert loaded.gravity == pytest.approx(traj.gravity)

    # loaded data still interpolates the same way
    t_mid = float(T[0]) / 2.0
    np.testing.assert_allclose(loaded.position(t_mid), traj.position(t_mid), atol=1e-8)
