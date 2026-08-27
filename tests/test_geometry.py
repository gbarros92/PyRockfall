"""Tests for pyrockfall.Geometry: node-perturbation sampling and trajectory
intersection methods (intersectFloor, intersectParabolaMatrix,
intersectParabola, intersectDamped).

Sampling
--------
Geometry.rvs()/.ppf() perturb each node independently with a Normal(node,
node_std) distribution. We verify this two ways, consistent with the other
test files in this repo:
  * rvs() Monte Carlo sample mean/std (large S) approximate `nodes`/`nodes_std`.
  * ppf() via a deterministic midpoint quantile grid (quadrature of the
    inverse-CDF) matches `nodes`/`nodes_std` tightly, with no randomness.

Intersections
-------------
Rather than re-deriving the implementation's own algebra, each method is
checked against an independently computed reference:
  * intersectFloor: closed-form free-fall/linear-motion times, plus a
    position-formula residual check for the general ballistic case.
  * intersectParabolaMatrix/intersectParabola: a flat floor is cross-checked
    against intersectFloor (same physics, must agree exactly); a sloped
    polyline with zero acceleration (pure straight-line motion) is
    cross-checked against an independent 2x2 line-segment intersection solve.
  * intersectDamped: cross-checked against scipy.optimize.brentq applied to
    the closed-form damped-ODE solution (the same physics described in the
    method's own docstring, solved independently of the implementation).
"""
import numpy as np
import pytest
from scipy.optimize import brentq

import pyrockfall as pr


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def make_polyline_geometry():
    nodes = np.array([[0.0, 0.0], [1.0, 0.5], [2.0, 0.2], [3.0, 1.0]])  # (N=4, D=2)
    nodes_std = np.array([[0.1, 0.0], [0.2, 0.05], [0.0, 0.0], [0.3, 0.1]])
    return pr.Geometry(nodes=nodes, nodes_std=nodes_std)


def test_nodes_and_nodes_std_properties():
    g = make_polyline_geometry()
    np.testing.assert_array_equal(g.nodes, np.array([[0.0, 0.0], [1.0, 0.5], [2.0, 0.2], [3.0, 1.0]]))
    assert bool(g.hasUncertainty) is True
    assert g.numRandomVariables == g.nodes_std.size


def test_deterministic_geometry_has_no_uncertainty():
    nodes = np.array([[0.0, 0.0], [1.0, 1.0]])
    g = pr.Geometry(nodes=nodes)
    assert bool(g.hasUncertainty) is False
    assert g.numRandomVariables == 0
    np.testing.assert_array_equal(g.nodes_std, np.zeros_like(nodes))


def test_rvs_shape_and_deterministic_fast_path():
    nodes = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
    g = pr.Geometry(nodes=nodes)
    s = g.rvs(S=50)
    assert s.shape == (3, 2, 1)  # all-zero std: single deterministic realization
    np.testing.assert_array_equal(s[:, :, 0], g.nodes)


def test_ppf_shape_and_deterministic_fast_path_ignores_q():
    nodes = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
    g = pr.Geometry(nodes=nodes)
    q = np.array([0.01, 0.5, 0.99])
    p = g.ppf(q)
    assert p.shape == (3, 2, 1)
    np.testing.assert_array_equal(p[:, :, 0], g.nodes)


def test_rvs_sample_mean_and_std_match_nodes_and_nodes_std():
    g = make_polyline_geometry()
    np.random.seed(0)
    s = g.rvs(S=200_000)
    assert s.shape == (4, 2, 200_000)

    mean = s.mean(axis=2)
    std = s.std(axis=2)
    np.testing.assert_allclose(mean, g.nodes, atol=5e-2)
    np.testing.assert_allclose(std, g.nodes_std, atol=5e-3)


def test_ppf_numerical_mean_and_std_match_nodes_and_nodes_std():
    g = make_polyline_geometry()
    n = 100_000
    # Midpoint quantile grid: mean/std of ppf(q) over a fine, symmetric grid
    # approximate the true mean/std via quadrature, without any randomness.
    grid = (np.arange(n) + 0.5) / n
    q = grid  # (S,) broadcasts across all nodes/dims
    p = g.ppf(q)
    assert p.shape == (4, 2, n)

    mean = p.mean(axis=2)
    std = p.std(axis=2)
    np.testing.assert_allclose(mean, g.nodes, atol=1e-3)
    np.testing.assert_allclose(std, g.nodes_std, atol=1e-3)


def test_ppf_zero_std_entries_are_exact_even_when_mixed_with_nonzero_std():
    # Regression test: ppf() used to return NaN (via norm.ppf(..., scale=0))
    # for a deterministic node/coordinate whenever *other* nodes had nonzero
    # std, because only the fully-deterministic (all std == 0) case was
    # fast-pathed. Fixed in _geometry.py by substituting a safe placeholder
    # scale for zero-std entries and overwriting them with the exact node
    # coordinate afterward.
    g = make_polyline_geometry()
    q = np.array([0.1, 0.3, 0.7, 0.9])
    p = g.ppf(q)
    assert not np.isnan(p).any()
    zero_std_mask = g.nodes_std == 0.0
    for i in range(p.shape[2]):
        np.testing.assert_array_equal(p[:, :, i][zero_std_mask], g.nodes[zero_std_mask])


def test_rvs_zero_std_entries_are_exact_even_when_mixed_with_nonzero_std():
    g = make_polyline_geometry()
    s = g.rvs(S=100, random_state=0)
    zero_std_mask = g.nodes_std == 0.0
    for i in range(s.shape[2]):
        np.testing.assert_array_equal(s[:, :, i][zero_std_mask], g.nodes[zero_std_mask])


def test_rvs_reproducible_with_random_state():
    g = make_polyline_geometry()
    a = g.rvs(S=20, random_state=42)
    b = g.rvs(S=20, random_state=42)
    np.testing.assert_array_equal(a, b)


def test_ppf_3d_q_matches_1d_q_broadcast():
    g = make_polyline_geometry()
    q1d = np.array([0.2, 0.6])
    p1d = g.ppf(q1d)
    q3d = np.broadcast_to(q1d[None, None, :], (4, 2, 2)).copy()
    p3d = g.ppf(q3d)
    np.testing.assert_allclose(p1d, p3d)


def test_ppf_rejects_wrong_q_ndim():
    g = make_polyline_geometry()
    with pytest.raises(ValueError, match=r"`q` must be shape \(S,\) or \(D, M, S\)"):
        g.ppf(np.zeros((2, 2)))


def test_ppf_rejects_mismatched_3d_leading_shape():
    g = make_polyline_geometry()
    with pytest.raises(ValueError, match="leading shape must be"):
        g.ppf(np.zeros((99, 99, 5)))


def test_nodes_std_shape_mismatch_raises():
    nodes = np.array([[0.0, 0.0], [1.0, 1.0]])
    with pytest.raises(ValueError, match="nodes_std must match nodes shape."):
        pr.Geometry(nodes=nodes, nodes_std=np.zeros((3, 2)))


def test_nodes_wrong_ndim_raises():
    with pytest.raises(ValueError, match=r"`nodes` must have shape \(N, D\)."):
        pr.Geometry(nodes=np.zeros((2, 2, 2)))


# ---------------------------------------------------------------------------
# intersectFloor
# ---------------------------------------------------------------------------

def test_intersect_floor_free_fall_from_rest():
    g = pr.Geometry(nodes=np.array([[-1.0, 0.0], [1.0, 0.0]]))  # unused by intersectFloor
    y0, gvt = 20.0, 9.81
    p = np.array([[0.0], [y0]])
    v = np.zeros((2, 1))
    a = np.array([[0.0], [-gvt]])
    t = g.intersectFloor(p, v, a, floor=0.0)
    assert t[0] == pytest.approx(np.sqrt(2 * y0 / gvt), rel=1e-8)


def test_intersect_floor_linear_motion():
    g = pr.Geometry(nodes=np.array([[-1.0, 0.0], [1.0, 0.0]]))
    p = np.array([[0.0], [10.0]])
    v = np.array([[0.0], [-2.0]])
    a = np.zeros((2, 1))
    t = g.intersectFloor(p, v, a, floor=0.0)
    assert t[0] == pytest.approx(5.0, rel=1e-8)


def test_intersect_floor_stationary_on_floor_returns_t_min():
    g = pr.Geometry(nodes=np.array([[-1.0, 0.0], [1.0, 0.0]]))
    p = np.array([[3.0], [0.0]])
    v = np.zeros((2, 1))
    a = np.zeros((2, 1))
    t = g.intersectFloor(p, v, a, floor=0.0, t_min=0.5)
    assert t[0] == pytest.approx(0.5)


def test_intersect_floor_moving_away_returns_inf():
    g = pr.Geometry(nodes=np.array([[-1.0, 0.0], [1.0, 0.0]]))
    p = np.array([[0.0], [5.0]])
    v = np.array([[0.0], [3.0]])  # moving up, no gravity: never returns
    a = np.zeros((2, 1))
    t = g.intersectFloor(p, v, a, floor=0.0)
    assert np.isinf(t[0])


def test_intersect_floor_respects_t_min_choosing_later_root():
    # Ballistic arc starting exactly at the floor, moving upward: roots at
    # t=0 (launch) and t=2*vy/g (landing). With t_min past the launch, the
    # landing root must be selected.
    gvt = 9.81
    vy = 8.0
    g = pr.Geometry(nodes=np.array([[-1.0, 0.0], [1.0, 0.0]]))
    p = np.array([[0.0], [0.0]])
    v = np.array([[0.0], [vy]])
    a = np.array([[0.0], [-gvt]])
    landing = 2 * vy / gvt
    t = g.intersectFloor(p, v, a, floor=0.0, t_min=landing / 2)
    assert t[0] == pytest.approx(landing, rel=1e-8)


def test_intersect_floor_vectorised_mixed_cases():
    g = pr.Geometry(nodes=np.array([[-1.0, 0.0], [1.0, 0.0]]))
    gvt = 9.81
    p = np.array([[0.0, 0.0, 0.0], [20.0, 10.0, 0.0]])
    v = np.array([[0.0, 0.0, 0.0], [0.0, -2.0, 0.0]])
    a = np.array([[0.0, 0.0, 0.0], [-gvt, 0.0, 0.0]])
    t = g.intersectFloor(p, v, a, floor=0.0)
    np.testing.assert_allclose(
        t, [np.sqrt(2 * 20.0 / gvt), 5.0, 0.0], rtol=1e-8, atol=1e-8
    )


def test_intersect_floor_ballistic_residual_is_self_consistent():
    g = pr.Geometry(nodes=np.array([[-1.0, 0.0], [1.0, 0.0]]))
    rng = np.random.default_rng(3)
    p = np.vstack([np.zeros(20), rng.uniform(5.0, 20.0, size=20)])
    v = np.vstack([np.zeros(20), rng.uniform(-5.0, 5.0, size=20)])
    a = np.vstack([np.zeros(20), np.full(20, -9.81)])
    t = g.intersectFloor(p, v, a, floor=0.0)
    assert np.all(np.isfinite(t))  # gravity guarantees a landing for all of these
    y_at_t = p[1] + v[1] * t + 0.5 * a[1] * t ** 2
    np.testing.assert_allclose(y_at_t, 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# intersectParabolaMatrix / intersectParabola
# ---------------------------------------------------------------------------

def flat_floor_geometry():
    return pr.Geometry(nodes=np.array([[-50.0, 0.0], [50.0, 0.0]]))


def valley_geometry():
    # Two segments forming a "V": left slope (-10,-5)->(0,0), right slope (0,0)->(10,-5).
    return pr.Geometry(nodes=np.array([[-10.0, -5.0], [0.0, 0.0], [10.0, -5.0]]))


def _line_line_intersection(p, v, A, B):
    """Solve p + v*t = A + w*(B - A) for (t, w); reference independent of the implementation."""
    Dv = B - A
    Mm = np.array([[v[0], -Dv[0]], [v[1], -Dv[1]]])
    rhs = A - p
    return np.linalg.solve(Mm, rhs)  # (t, w)


def test_intersect_parabola_matrix_shape():
    g = valley_geometry()
    p = np.array([[0.0], [8.0]])
    v = np.array([[0.0], [3.0]])
    a = np.zeros((2, 1))
    t_mat = g.intersectParabolaMatrix(p, v, a)
    assert t_mat.shape == (2, 1)  # M-1=2 segments, S=1


def test_intersect_parabola_matrix_is_inf_when_motion_parallel_to_segment():
    # If v is parallel to a segment's direction (and a=0) and p is not on
    # that segment's supporting line, the line is never crossed: the
    # candidate matrix entry must stay +inf for that segment.
    g = valley_geometry()
    left_dir = np.array([10.0, 5.0])  # (0,0) - (-10,-5)
    p = np.array([[-5.0], [10.0]])  # off the left segment's line
    v = left_dir.reshape(2, 1)
    a = np.zeros((2, 1))
    t_mat = g.intersectParabolaMatrix(p, v, a)
    assert np.isinf(t_mat[0, 0])  # left segment (index 0): never crossed


def test_intersect_parabola_flat_floor_matches_intersect_floor():
    g = flat_floor_geometry()
    p = np.array([[0.0], [10.0]])
    v = np.array([[5.0], [0.0]])
    a = np.array([[0.0], [-9.81]])
    elem_id = np.array([0])

    seg, t = g.intersectParabola(p, v, a, elem_id)
    t_floor = g.intersectFloor(p, v, a, floor=0.0)

    assert seg[0] == 0
    assert t[0] == pytest.approx(t_floor[0], rel=1e-8)
    x_land = p[0] + v[0] * t
    np.testing.assert_allclose(x_land, p[0] + v[0] * t_floor)


def test_intersect_parabola_straight_line_hits_left_slope_matches_reference():
    g = valley_geometry()
    p = np.array([-8.0, 8.0])
    v = np.array([3.0, -6.0])
    a = np.zeros(2)
    A0, B0 = np.array([-10.0, -5.0]), np.array([0.0, 0.0])
    t_ref, w_ref = _line_line_intersection(p, v, A0, B0)
    assert 0.0 <= w_ref <= 1.0  # sanity: reference hit lands on the segment

    P, V, Av = p.reshape(2, 1), v.reshape(2, 1), a.reshape(2, 1)
    seg, t = g.intersectParabola(P, V, Av, np.array([0]))
    assert seg[0] == 0
    assert t[0] == pytest.approx(t_ref, rel=1e-8)


def test_intersect_parabola_straight_line_hits_right_slope_matches_reference():
    g = valley_geometry()
    p = np.array([2.0, 8.0])
    v = np.array([4.0, -8.0])
    a = np.zeros(2)
    A1, B1 = np.array([0.0, 0.0]), np.array([10.0, -5.0])
    t_ref, w_ref = _line_line_intersection(p, v, A1, B1)
    assert 0.0 <= w_ref <= 1.0

    P, V, Av = p.reshape(2, 1), v.reshape(2, 1), a.reshape(2, 1)
    seg, t = g.intersectParabola(P, V, Av, np.array([1]))
    assert seg[0] == 1
    assert t[0] == pytest.approx(t_ref, rel=1e-8)


def test_intersect_parabola_no_hit_returns_minus_one_and_nan():
    g = valley_geometry()
    p = np.array([[0.0], [8.0]])
    v = np.array([[0.0], [3.0]])
    a = np.zeros((2, 1))
    seg, t = g.intersectParabola(p, v, a, np.array([0]))
    assert seg[0] == -1
    assert np.isnan(t[0])


def test_intersect_parabola_respects_t_max_window():
    g = flat_floor_geometry()
    p = np.array([[0.0], [10.0]])
    v = np.array([[5.0], [0.0]])
    a = np.array([[0.0], [-9.81]])
    t_land = g.intersectFloor(p, v, a, floor=0.0)[0]

    seg_ok, t_ok = g.intersectParabola(p, v, a, np.array([0]), t_max=t_land + 1.0)
    assert seg_ok[0] == 0

    seg_blocked, t_blocked = g.intersectParabola(p, v, a, np.array([0]), t_max=t_land - 0.5)
    assert seg_blocked[0] == -1
    assert np.isnan(t_blocked[0])


def test_intersect_parabola_vectorised_multiple_samples():
    g = flat_floor_geometry()
    p = np.array([[0.0, -5.0], [10.0, 5.0]])
    v = np.array([[5.0, 0.0], [0.0, 0.0]])
    a = np.array([[0.0, 0.0], [-9.81, -9.81]])
    elem_id = np.array([0, 0])
    seg, t = g.intersectParabola(p, v, a, elem_id)
    t_floor = g.intersectFloor(p, v, a, floor=0.0)
    np.testing.assert_allclose(t, t_floor, rtol=1e-8)
    assert np.all(seg == 0)


# ---------------------------------------------------------------------------
# intersectDamped
# ---------------------------------------------------------------------------

def _r_damped_reference(t, p, v, a, D):
    """Closed-form solution of v' = a - D v, independently evaluated
    (the same physics documented in Geometry.intersectDamped's docstring)."""
    drift = a / D
    transient = (v - drift) / D
    return p + drift * t + transient * (1.0 - np.exp(-D * t))


def test_intersect_damped_flat_floor_matches_independent_brentq():
    g = flat_floor_geometry()
    p = np.array([0.0, 10.0])
    v = np.array([5.0, 0.0])
    a = np.array([0.0, -9.81])
    damping = 0.5

    f = lambda t: _r_damped_reference(t, p, v, a, damping)[1] - 0.0
    t_ref = brentq(f, 1e-6, 30.0, xtol=1e-12)

    P, V, Av = p.reshape(2, 1), v.reshape(2, 1), a.reshape(2, 1)
    seg, t = g.intersectDamped(P, V, Av, np.array([damping]), np.array([0]))
    assert seg[0] == 0
    assert t[0] == pytest.approx(t_ref, rel=1e-6)


def test_intersect_damped_position_at_result_is_on_floor_and_within_segment():
    g = flat_floor_geometry()
    p = np.array([0.0, 10.0])
    v = np.array([8.0, 2.0])
    a = np.array([0.0, -9.81])
    damping = 1.5

    P, V, Av = p.reshape(2, 1), v.reshape(2, 1), a.reshape(2, 1)
    seg, t = g.intersectDamped(P, V, Av, np.array([damping]), np.array([0]))
    assert seg[0] == 0
    pos = _r_damped_reference(t[0], p, v, a, damping)
    assert pos[1] == pytest.approx(0.0, abs=1e-6)
    assert -50.0 <= pos[0] <= 50.0


def test_intersect_damped_rejects_nonpositive_damping():
    g = flat_floor_geometry()
    p = np.array([[0.0], [10.0]])
    v = np.array([[5.0], [0.0]])
    a = np.array([[0.0], [-9.81]])
    with pytest.raises(ValueError, match="`damping` must be positive"):
        g.intersectDamped(p, v, a, np.array([0.0]), np.array([0]))


def test_intersect_damped_no_motion_off_segment_returns_no_hit():
    g = flat_floor_geometry()
    p = np.array([[0.0], [10.0]])  # above the floor, never moves
    v = np.zeros((2, 1))
    a = np.zeros((2, 1))
    seg, t = g.intersectDamped(p, v, a, np.array([1.0]), np.array([0]))
    assert seg[0] == -1
    assert np.isnan(t[0])


def test_intersect_damped_vectorised_matches_per_sample_calls():
    g = flat_floor_geometry()
    p = np.array([[0.0, -10.0], [10.0, 6.0]])
    v = np.array([[5.0, -3.0], [0.0, 1.0]])
    a = np.array([[0.0, 0.0], [-9.81, -9.81]])
    damping = np.array([0.5, 2.0])
    elem_id = np.array([0, 0])

    seg_batch, t_batch = g.intersectDamped(p, v, a, damping, elem_id)

    for i in range(2):
        seg_single, t_single = g.intersectDamped(
            p[:, i:i + 1], v[:, i:i + 1], a[:, i:i + 1], damping[i:i + 1], elem_id[i:i + 1]
        )
        assert seg_batch[i] == seg_single[0]
        assert t_batch[i] == pytest.approx(t_single[0], rel=1e-8, nan_ok=True)
