"""Tests for pyrockfall.Geometry3D: a 3D triangular-mesh subclass of Geometry.

Sampling
--------
Geometry3D reuses Geometry's node-perturbation logic verbatim (rvs/ppf are
not overridden). We re-verify the same mean/std properties as
test_geometry.py, but now with genuinely 3D nodes/nodes_std, to confirm the
dimension-agnostic implementation behaves correctly for D=3 too (not just
D=2).

exitTime / intersectParabola
-----------------------------
Geometry3D overrides these two methods with mesh-specific (triangle
neighbour-walk) implementations, distinct from Geometry's 2D polyline
versions. A simple two-triangle flat mesh (a 10x10 square split along its
diagonal into two coplanar triangles) is used so that intersection times can
be cross-checked against independent, hand-derived geometry:
  * exitTime: an independent line/diagonal intersection (t such that
    p + v*t lies on the shared diagonal edge).
  * intersectParabola: free-fall time (H = 0.5 g t^2) for a purely vertical
    drop, and a horizontal-landing-position cross-check for an oblique
    ballistic arc that lands in the *other* triangle than it starts in
    (forcing the neighbour-walk to actually walk).
"""
import numpy as np
import pytest

import pyrockfall as pr


# ---------------------------------------------------------------------------
# Mesh fixture: two coplanar triangles forming a 10x10 square at z=0,
# split along the diagonal from (0,0,0) to (10,10,0).
#
#   n3(0,10,0) ---- n2(10,10,0)
#      |  T1      / |
#      |        /   |
#      |      /  T0 |
#      |    /       |
#   n0(0,0,0) ---- n1(10,0,0)
#
# T0 = (n0, n1, n2): "lower-right" half (y < x)
# T1 = (n0, n2, n3): "upper-left" half (y > x)
# Shared edge: the diagonal n0-n2.
# ---------------------------------------------------------------------------

SQUARE_NODES = np.array([
    [0.0, 0.0, 0.0],
    [10.0, 0.0, 0.0],
    [10.0, 10.0, 0.0],
    [0.0, 10.0, 0.0],
])
SQUARE_ELEMENTS = np.array([[0, 1, 2], [0, 2, 3]])


def make_square_mesh():
    return pr.Geometry3D(nodes=SQUARE_NODES, elements=SQUARE_ELEMENTS)


# ---------------------------------------------------------------------------
# Sampling (dimension-agnostic logic, exercised for D=3)
# ---------------------------------------------------------------------------

def make_mesh_with_std():
    nodes = np.array([
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [10.0, 10.0, 2.0],
        [0.0, 10.0, 1.0],
        [5.0, 5.0, 3.0],
    ])
    nodes_std = np.array([
        [0.2, 0.1, 0.0],
        [0.0, 0.0, 0.0],
        [0.3, 0.2, 0.05],
        [0.1, 0.0, 0.1],
        [0.0, 0.15, 0.2],
    ])
    elements = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]])
    return pr.Geometry3D(nodes=nodes, nodes_std=nodes_std, elements=elements)


def test_nodes_and_nodes_std_are_3d():
    g = make_mesh_with_std()
    assert g.nodes.shape == (5, 3)
    assert g.nodes_std.shape == (5, 3)
    assert bool(g.hasUncertainty) is True
    assert g.numRandomVariables == g.nodes_std.size


def test_deterministic_3d_mesh_has_no_uncertainty():
    g = make_square_mesh()
    assert bool(g.hasUncertainty) is False
    assert g.numRandomVariables == 0
    np.testing.assert_array_equal(g.nodes_std, np.zeros_like(g.nodes))


def test_rvs_shape_is_3d():
    g = make_mesh_with_std()
    s = g.rvs(S=10)
    assert s.shape == (5, 3, 10)


def test_rvs_sample_mean_and_std_match_nodes_and_nodes_std_in_3d():
    g = make_mesh_with_std()
    np.random.seed(0)
    s = g.rvs(S=200_000)
    mean = s.mean(axis=2)
    std = s.std(axis=2)
    np.testing.assert_allclose(mean, g.nodes, atol=5e-2)
    np.testing.assert_allclose(std, g.nodes_std, atol=5e-3)


def test_ppf_numerical_mean_and_std_match_nodes_and_nodes_std_in_3d():
    g = make_mesh_with_std()
    n = 100_000
    grid = (np.arange(n) + 0.5) / n
    p = g.ppf(grid)
    assert p.shape == (5, 3, n)
    mean = p.mean(axis=2)
    std = p.std(axis=2)
    np.testing.assert_allclose(mean, g.nodes, atol=1e-3)
    np.testing.assert_allclose(std, g.nodes_std, atol=1e-3)


def test_ppf_zero_std_entries_exact_in_3d_mixed_geometry():
    g = make_mesh_with_std()
    q = np.array([0.1, 0.4, 0.6, 0.9])
    p = g.ppf(q)
    assert not np.isnan(p).any()
    zero_std_mask = g.nodes_std == 0.0
    for i in range(p.shape[2]):
        np.testing.assert_array_equal(p[:, :, i][zero_std_mask], g.nodes[zero_std_mask])


def test_deterministic_fast_path_returns_exact_nodes_in_3d():
    g = make_square_mesh()
    s = g.rvs(S=25)
    p = g.ppf(np.linspace(0.01, 0.99, 7))
    assert s.shape == (4, 3, 1)
    assert p.shape == (4, 3, 1)
    np.testing.assert_array_equal(s[:, :, 0], g.nodes)
    np.testing.assert_array_equal(p[:, :, 0], g.nodes)


# ---------------------------------------------------------------------------
# Mesh structure
# ---------------------------------------------------------------------------

def test_neighbours_shape_and_shared_diagonal_edge():
    g = make_square_mesh()
    nb = g.neighbours
    assert nb.shape == (2, 3)
    # T0's edge index 2 is (v2->v0) = (n2->n0), the reverse of T1's edge
    # index 0 (n0->n2): they must be mutual neighbours across the diagonal.
    assert nb[0, 2] == 1
    assert nb[1, 0] == 0
    # The other four edges are all mesh boundaries.
    assert nb[0, 0] == -1 and nb[0, 1] == -1
    assert nb[1, 1] == -1 and nb[1, 2] == -1


# ---------------------------------------------------------------------------
# exitTime
# ---------------------------------------------------------------------------

def test_exit_time_through_shared_diagonal_matches_independent_line_solve():
    g = make_square_mesh()
    p = np.array([[20.0 / 3.0], [10.0 / 3.0], [0.0]])
    v = np.array([[-1.0], [1.0], [0.0]])
    next_el, t = g.exitTime(p, v, np.array([0]))

    # Independent reference: solve p_xy + t*v_xy on the line y = x.
    t_ref = (p[1, 0] - p[0, 0]) / (v[0, 0] - v[1, 0])
    assert t[0] == pytest.approx(t_ref, rel=1e-8)
    assert next_el[0] == 1  # crosses into T1

    pos = p + v * t
    np.testing.assert_allclose(pos[:2, 0], [5.0, 5.0], atol=1e-8)
    assert pos[2, 0] == pytest.approx(0.0)


def test_exit_time_reaching_mesh_boundary_returns_no_neighbour():
    g = make_square_mesh()
    p = np.array([[5.0], [3.0], [0.0]])
    v = np.array([[0.0], [-1.0], [0.0]])  # heads straight for the y=0 boundary edge
    next_el, t = g.exitTime(p, v, np.array([0]))
    assert next_el[0] == -1
    assert t[0] == pytest.approx(3.0)


def test_exit_time_no_motion_returns_no_exit():
    g = make_square_mesh()
    p = np.array([[5.0], [3.0], [0.0]])
    v = np.zeros((3, 1))
    next_el, t = g.exitTime(p, v, np.array([0]))
    assert next_el[0] == -1
    assert np.isnan(t[0])


def test_exit_time_out_of_plane_velocity_is_projected_away():
    # Velocity's z-component must be projected onto the triangle's plane
    # before computing the exit; the result should match the purely
    # in-plane case exactly.
    g = make_square_mesh()
    p = np.array([[20.0 / 3.0], [10.0 / 3.0], [0.0]])
    v_planar = np.array([[-1.0], [1.0], [0.0]])
    v_with_z = np.array([[-1.0], [1.0], [5.0]])

    next_el_planar, t_planar = g.exitTime(p, v_planar, np.array([0]))
    next_el_z, t_z = g.exitTime(p, v_with_z, np.array([0]))

    assert next_el_z[0] == next_el_planar[0]
    assert t_z[0] == pytest.approx(t_planar[0], rel=1e-8)


def test_exit_time_vectorised_multiple_samples():
    g = make_square_mesh()
    p = np.array([[20.0 / 3.0, 5.0], [10.0 / 3.0, 3.0], [0.0, 0.0]])
    v = np.array([[-1.0, 0.0], [1.0, -1.0], [0.0, 0.0]])
    next_el, t = g.exitTime(p, v, np.array([0, 0]))
    assert next_el.tolist() == [1, -1]
    np.testing.assert_allclose(t, [5.0 / 3.0, 3.0], rtol=1e-8)


def test_exit_time_rejects_shape_mismatch():
    g = make_square_mesh()
    p = np.zeros((3, 2))
    v = np.zeros((3, 3))
    with pytest.raises(ValueError, match=r"`p` and `v` must be \(3, S\)"):
        g.exitTime(p, v, np.array([0, 0]))


# ---------------------------------------------------------------------------
# intersectParabola
# ---------------------------------------------------------------------------

def test_intersect_parabola_vertical_drop_matches_free_fall_time():
    g = make_square_mesh()
    H, grav = 20.0, 9.81
    p = np.array([[20.0 / 3.0], [10.0 / 3.0], [H]])
    v = np.zeros((3, 1))
    a = np.array([[0.0], [0.0], [-grav]])

    hit, t = g.intersectParabola(p, v, a, np.array([0]))
    t_ref = np.sqrt(2 * H / grav)

    assert hit[0] == 0
    assert t[0] == pytest.approx(t_ref, rel=1e-6)


def test_intersect_parabola_result_independent_of_starting_triangle_guess():
    # The two triangles are coplanar, so the free-fall time to the shared
    # plane is identical regardless of which triangle's plane is used
    # first; a wrong initial guess must still converge to the same answer
    # via the neighbour walk.
    g = make_square_mesh()
    H, grav = 20.0, 9.81
    p = np.array([[20.0 / 3.0], [10.0 / 3.0], [H]])
    v = np.zeros((3, 1))
    a = np.array([[0.0], [0.0], [-grav]])

    hit_direct, t_direct = g.intersectParabola(p, v, a, np.array([0]))
    hit_walked, t_walked = g.intersectParabola(p, v, a, np.array([1]))

    assert hit_direct[0] == hit_walked[0] == 0
    assert t_walked[0] == pytest.approx(t_direct[0], rel=1e-6)


def test_intersect_parabola_oblique_landing_crosses_into_other_triangle():
    # Launched from inside T0, moving toward T1's region: the walk must
    # cross the shared diagonal to land in the correct triangle.
    g = make_square_mesh()
    H, grav = 10.0, 9.81
    p = np.array([[8.0], [2.0], [H]])
    v = np.array([[-4.0], [3.0], [0.0]])
    a = np.array([[0.0], [0.0], [-grav]])

    hit, t = g.intersectParabola(p, v, a, np.array([0]))
    t_ref = np.sqrt(2 * H / grav)
    x_land = 8.0 - 4.0 * t_ref
    y_land = 2.0 + 3.0 * t_ref

    assert t[0] == pytest.approx(t_ref, rel=1e-6)
    assert hit[0] == (1 if y_land > x_land else 0)

    pos_xy = p[:2, 0] + v[:2, 0] * t[0]
    np.testing.assert_allclose(pos_xy, [x_land, y_land], atol=1e-6)


def test_intersect_parabola_no_hit_when_moving_away_forever():
    g = make_square_mesh()
    p = np.array([[5.0], [3.0], [1.0]])
    v = np.array([[0.0], [0.0], [2.0]])  # straight up, no gravity: never lands
    a = np.zeros((3, 1))
    hit, t = g.intersectParabola(p, v, a, np.array([0]), t_max=100.0)
    assert hit[0] == -1
    assert np.isnan(t[0])


def test_intersect_parabola_respects_t_min_and_t_max_window():
    g = make_square_mesh()
    H, grav = 20.0, 9.81
    p = np.array([[20.0 / 3.0], [10.0 / 3.0], [H]])
    v = np.zeros((3, 1))
    a = np.array([[0.0], [0.0], [-grav]])
    t_ref = np.sqrt(2 * H / grav)

    hit_ok, t_ok = g.intersectParabola(p, v, a, np.array([0]), t_max=t_ref + 1.0)
    assert hit_ok[0] == 0

    hit_blocked, t_blocked = g.intersectParabola(p, v, a, np.array([0]), t_max=t_ref - 0.5)
    assert hit_blocked[0] == -1
    assert np.isnan(t_blocked[0])


def test_intersect_parabola_vectorised_multiple_samples():
    g = make_square_mesh()
    H, grav = 20.0, 9.81
    p = np.array([[20.0 / 3.0, 8.0], [10.0 / 3.0, 2.0], [H, 10.0]])
    v = np.array([[0.0, -4.0], [0.0, 3.0], [0.0, 0.0]])
    a = np.array([[0.0, 0.0], [0.0, 0.0], [-grav, -grav]])
    hit, t = g.intersectParabola(p, v, a, np.array([0, 0]))

    hit0, t0 = g.intersectParabola(p[:, 0:1], v[:, 0:1], a[:, 0:1], np.array([0]))
    hit1, t1 = g.intersectParabola(p[:, 1:2], v[:, 1:2], a[:, 1:2], np.array([0]))

    np.testing.assert_array_equal(hit, [hit0[0], hit1[0]])
    np.testing.assert_allclose(t, [t0[0], t1[0]], rtol=1e-8)


def test_intersect_parabola_rejects_shape_mismatch():
    g = make_square_mesh()
    p = np.zeros((3, 2))
    v = np.zeros((3, 2))
    a = np.zeros((3, 3))
    with pytest.raises(ValueError, match=r"`p`, `v`, `a` must be \(3, S\)"):
        g.intersectParabola(p, v, a, np.array([0, 0]))
