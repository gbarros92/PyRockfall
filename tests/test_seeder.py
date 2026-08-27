"""Tests for pyrockfall._seeder: Seeder (point), LineSeeder, AreaSeeder, in 2D and 3D.

Geometry is the focus:
  * Seeder (point seeder, single point): every sampled position must equal
    exactly that point.
  * Seeder with multiple points ("point seeder with multiple points"):
    every sampled position must be exactly one of the given points.
  * LineSeeder: every sampled position must lie on the given polyline.
  * AreaSeeder: every sampled position must be a convex combination
    (non-negative barycentric weights summing to 1) of the given points --
    tested exactly by using a simplex (triangle in 2D, tetrahedron in 3D) so
    the barycentric decomposition is unique and solvable directly.

Velocity sampling (translational/angular) is exercised the same way as
Rock/Material: rvs() sample moments and ppf() quadrature moments are checked
against the prescribed per-component distributions, using a point seeder so
position sampling is deterministic and doesn't interfere.
"""
import numpy as np
import pytest

import pyrockfall as pr
from pyrockfall import stats


# ---------------------------------------------------------------------------
# Geometry helpers (used to verify sample positions independently of the
# implementation's own sampling formulas)
# ---------------------------------------------------------------------------

def _point_on_segment(sample, a, b, tol=1e-8):
    """True if `sample` (D,) lies on the closed segment [a, b] (each (D,))."""
    ab = b - a
    seg_len2 = np.dot(ab, ab)
    if seg_len2 == 0:
        return np.allclose(sample, a, atol=tol)
    t = np.dot(sample - a, ab) / seg_len2
    if t < -tol or t > 1 + tol:
        return False
    closest = a + t * ab
    return np.linalg.norm(sample - closest) < tol


def _point_on_polyline(sample, P, tol=1e-6):
    """P is (D, N); True if `sample` (D,) lies on any segment of the polyline."""
    N = P.shape[1]
    return any(_point_on_segment(sample, P[:, i], P[:, i + 1], tol=tol) for i in range(N - 1))


def _barycentric_weights(simplex_points, sample):
    """Solve for unique barycentric weights of `sample` w.r.t. a (D, D+1) simplex."""
    D, N = simplex_points.shape
    assert N == D + 1
    A = np.vstack([simplex_points, np.ones((1, N))])
    b = np.concatenate([sample, [1.0]])
    return np.linalg.solve(A, b)


def make_rocks():
    return [pr.Rock(name="R1", mass=1.0, density=2500.0)]


# ---------------------------------------------------------------------------
# Geometry fixtures per dimension
# ---------------------------------------------------------------------------

def single_point(D):
    return np.array([1.0, 2.0]) if D == 2 else np.array([1.0, 2.0, 3.0])


def multi_points(D):
    if D == 2:
        return np.array([[0.0, 4.0, -3.0], [0.0, 0.0, 5.0]])
    return np.array([[0.0, 4.0, -3.0], [0.0, 0.0, 5.0], [0.0, 1.0, 2.0]])


def straight_line(D):
    if D == 2:
        return np.array([[0.0, 4.0], [0.0, 3.0]])
    return np.array([[0.0, 3.0], [0.0, 4.0], [0.0, 5.0]])


def polyline(D):
    if D == 2:
        return np.array([[0.0, 4.0, 4.0], [0.0, 0.0, 3.0]])
    return np.array([[0.0, 3.0, 3.0], [0.0, 4.0, 4.0], [0.0, 0.0, 5.0]])


def simplex(D):
    if D == 2:
        return np.array([[0.0, 4.0, 1.0], [0.0, 0.0, 5.0]])
    return np.array([[0.0, 4.0, 0.0, 0.0], [0.0, 0.0, 4.0, 0.0], [0.0, 0.0, 0.0, 4.0]])


DIMS = [2, 3]


# ---------------------------------------------------------------------------
# Point seeder: all samples at the same point
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("D", DIMS)
def test_point_seeder_flags_and_counts(D):
    s = pr.Seeder(points=single_point(D), rocks=make_rocks())
    assert s.D == D
    assert s.isPointSeeder is True
    assert s.isLineSeeder is False
    assert s.isAreaSeeder is False
    assert s.numRVsPosition == 0
    expected_angular = 1 if D == 2 else 3
    assert s.numRandomVariables == D + expected_angular


@pytest.mark.parametrize("D", DIMS)
def test_point_seeder_rvs_all_samples_at_same_point(D):
    s = pr.Seeder(points=single_point(D), rocks=make_rocks())
    positions, _, _ = s.rvs(200)
    assert positions.shape == (D, 200)
    expected = single_point(D)
    for i in range(200):
        np.testing.assert_array_equal(positions[:, i], expected)


@pytest.mark.parametrize("D", DIMS)
def test_point_seeder_ppf_all_samples_at_same_point(D):
    s = pr.Seeder(points=single_point(D), rocks=make_rocks())
    M = 25
    q = np.random.default_rng(0).uniform(0.0, 1.0, size=(s.numRandomVariables, M))
    positions, _, _ = s.ppf(q)
    assert positions.shape == (D, M)
    expected = single_point(D)
    for i in range(M):
        np.testing.assert_array_equal(positions[:, i], expected)


@pytest.mark.parametrize("D", DIMS)
def test_point_seeder_with_multiple_points_samples_are_among_given_points(D):
    pts = multi_points(D)
    s = pr.Seeder(points=pts, rocks=make_rocks())
    assert s.numRVsPosition == 1
    positions, _, _ = s.rvs(300)
    for i in range(positions.shape[1]):
        matches_any = any(np.allclose(positions[:, i], pts[:, j]) for j in range(pts.shape[1]))
        assert matches_any


# ---------------------------------------------------------------------------
# Point seeder velocities: rvs/ppf moments vs. prescribed distributions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("D", DIMS)
def test_point_seeder_velocity_ppf_moments_match_prescribed_distributions(D):
    s = pr.Seeder(points=single_point(D), rocks=make_rocks())
    trans_dists = [stats.Normal(1.0, 2.0), stats.Normal(-1.0, 1.5)] + ([stats.Normal(0.5, 1.0)] if D == 3 else [])
    ang_dists = [stats.Normal(0.0, 0.2)] if D == 2 else [stats.Normal(0.0, 0.1), stats.Normal(0.1, 0.2), stats.Normal(-0.1, 0.3)]
    s.translationalVelocity = trans_dists
    s.angularVelocity = ang_dists

    n = 100_000
    grid = (np.arange(n) + 0.5) / n
    q = np.vstack([grid] * s.numRandomVariables)
    _, trans_samples, ang_samples = s.ppf(q)

    for i, d in enumerate(trans_dists):
        assert trans_samples[i].mean() == pytest.approx(d.mean(), rel=1e-3)
        assert trans_samples[i].var() == pytest.approx(d.var(), rel=1e-2)
    for i, d in enumerate(ang_dists):
        assert ang_samples[i].mean() == pytest.approx(d.mean(), rel=1e-3)
        assert ang_samples[i].var() == pytest.approx(d.var(), rel=1e-2)


@pytest.mark.parametrize("D", DIMS)
def test_point_seeder_velocity_rvs_moments_match_prescribed_distributions(D):
    s = pr.Seeder(points=single_point(D), rocks=make_rocks())
    trans_dists = [stats.Normal(1.0, 2.0), stats.Normal(-1.0, 1.5)] + ([stats.Normal(0.5, 1.0)] if D == 3 else [])
    ang_dists = [stats.Normal(0.0, 0.2)] if D == 2 else [stats.Normal(0.0, 0.1), stats.Normal(0.1, 0.2), stats.Normal(-0.1, 0.3)]
    s.translationalVelocity = trans_dists
    s.angularVelocity = ang_dists

    np.random.seed(0)
    _, trans_samples, ang_samples = s.rvs(100_000)

    for i, d in enumerate(trans_dists):
        assert trans_samples[i].mean() == pytest.approx(d.mean(), rel=5e-2)
        assert trans_samples[i].var() == pytest.approx(d.var(), rel=5e-2)
    for i, d in enumerate(ang_dists):
        assert ang_samples[i].mean() == pytest.approx(d.mean(), rel=5e-2, abs=1e-3)
        assert ang_samples[i].var() == pytest.approx(d.var(), rel=5e-2)


@pytest.mark.parametrize("D", DIMS)
def test_velocity_setter_rejects_wrong_length_vector(D):
    s = pr.Seeder(points=single_point(D), rocks=make_rocks())
    with pytest.raises(ValueError):
        s.translationalVelocity = [stats.Normal(0.0, 1.0)] * (D + 1)
    with pytest.raises(ValueError):
        s.angularVelocity = [stats.Normal(0.0, 1.0)] * 99


# ---------------------------------------------------------------------------
# Line seeder: all samples lie on the polyline
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("D", DIMS)
def test_line_seeder_flags_and_counts(D):
    P = straight_line(D)
    s = pr.LineSeeder(points=P, rocks=make_rocks())
    assert s.isPointSeeder is False
    assert s.isLineSeeder is True
    assert s.isAreaSeeder is False
    assert s.numRVsPosition == 1
    expected_angular = 1 if D == 2 else 3
    assert s.numRandomVariables == D + expected_angular + 1


@pytest.mark.parametrize("D", DIMS)
def test_line_seeder_rvs_returns_documented_shape(D):
    # Regression test: LineSeeder._sample_positions used to index P/seg with
    # a non-raveled (1, M) idx array, producing an extra singleton axis
    # (D, 1, M) instead of the documented (D, M) for any line with >= 2
    # points (fixed in _seeder.py by raveling idx before indexing).
    P = straight_line(D)
    s = pr.LineSeeder(points=P, rocks=make_rocks())
    positions, _, _ = s.rvs(500)
    assert positions.shape == (D, 500)


@pytest.mark.parametrize("D", DIMS)
def test_line_seeder_rvs_samples_lie_on_straight_line(D):
    P = straight_line(D)
    s = pr.LineSeeder(points=P, rocks=make_rocks())
    positions, _, _ = s.rvs(500)
    for i in range(positions.shape[1]):
        assert _point_on_segment(positions[:, i], P[:, 0], P[:, 1], tol=1e-6)


@pytest.mark.parametrize("D", DIMS)
def test_line_seeder_ppf_samples_lie_on_straight_line(D):
    P = straight_line(D)
    s = pr.LineSeeder(points=P, rocks=make_rocks())
    M = 40
    q = np.random.default_rng(1).uniform(0.0, 1.0, size=(s.numRandomVariables, M))
    positions, _, _ = s.ppf(q)
    for i in range(M):
        assert _point_on_segment(positions[:, i], P[:, 0], P[:, 1], tol=1e-6)


@pytest.mark.parametrize("D", DIMS)
def test_line_seeder_rvs_samples_lie_on_multisegment_polyline(D):
    P = polyline(D)
    s = pr.LineSeeder(points=P, rocks=make_rocks())
    positions, _, _ = s.rvs(500)
    for i in range(positions.shape[1]):
        assert _point_on_polyline(positions[:, i], P, tol=1e-6)


@pytest.mark.parametrize("D", DIMS)
def test_line_seeder_endpoints_map_to_polyline_endpoints(D):
    P = straight_line(D)
    s = pr.LineSeeder(points=P, rocks=make_rocks())
    q = np.zeros((s.numRandomVariables, 2))
    q[-1, :] = [0.0, 1.0]  # position quantile row is last
    positions, _, _ = s.ppf(q)
    np.testing.assert_allclose(positions[:, 0], P[:, 0], atol=1e-8)
    np.testing.assert_allclose(positions[:, 1], P[:, -1], atol=1e-8)


# ---------------------------------------------------------------------------
# Area seeder: all samples are convex combinations of the given points
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("D", DIMS)
def test_area_seeder_flags_and_counts(D):
    P = simplex(D)
    s = pr.AreaSeeder(points=P, rocks=make_rocks())
    assert s.isPointSeeder is False
    assert s.isLineSeeder is False
    assert s.isAreaSeeder is True
    assert s.numRVsPosition == P.shape[1]
    expected_angular = 1 if D == 2 else 3
    assert s.numRandomVariables == D + expected_angular + P.shape[1]


@pytest.mark.parametrize("D", DIMS)
def test_area_seeder_rvs_samples_are_convex_combinations(D):
    P = simplex(D)
    s = pr.AreaSeeder(points=P, rocks=make_rocks())
    positions, _, _ = s.rvs(500)
    for i in range(positions.shape[1]):
        w = _barycentric_weights(P, positions[:, i])
        assert w.sum() == pytest.approx(1.0, abs=1e-8)
        assert np.all(w >= -1e-8)
        np.testing.assert_allclose(P @ w, positions[:, i], atol=1e-8)


@pytest.mark.parametrize("D", DIMS)
def test_area_seeder_ppf_samples_are_convex_combinations(D):
    P = simplex(D)
    s = pr.AreaSeeder(points=P, rocks=make_rocks())
    M = 60
    q = np.random.default_rng(2).uniform(1e-3, 1.0, size=(s.numRandomVariables, M))
    positions, _, _ = s.ppf(q)
    for i in range(M):
        w = _barycentric_weights(P, positions[:, i])
        assert w.sum() == pytest.approx(1.0, abs=1e-8)
        assert np.all(w >= -1e-8)


@pytest.mark.parametrize("D", DIMS)
def test_area_seeder_vertex_quantiles_reproduce_vertices(D):
    P = simplex(D)
    N = P.shape[1]
    s = pr.AreaSeeder(points=P, rocks=make_rocks())
    # Position quantile rows are the last N rows; setting only the j-th one
    # to a large positive value should recover vertex j after normalization
    # (u_norm = u / sum(u), so a dominant component pulls the combination
    # toward that vertex).
    for j in range(N):
        q = np.zeros((s.numRandomVariables, 1))
        q[-N + j, 0] = 1.0
        q[-N:, 0] += 1e-9  # avoid an exact all-zero column
        positions, _, _ = s.ppf(q)
        np.testing.assert_allclose(positions[:, 0], P[:, j], atol=1e-6)


def test_area_seeder_requires_at_least_two_points():
    s = pr.AreaSeeder(points=np.array([1.0, 2.0]), rocks=make_rocks())
    with pytest.raises(ValueError, match="AreaSeeder requires at least 2 points."):
        s.rvs(5)


# ---------------------------------------------------------------------------
# Construction validation (shared base-class behavior)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("D", [1, 4])
def test_seeder_rejects_invalid_dimension(D):
    pts = np.zeros(D) if D > 0 else np.zeros(0)
    with pytest.raises(ValueError, match=r"D must be 2 or 3"):
        pr.Seeder(points=pts, rocks=make_rocks())


def test_seeder_rejects_wrong_ndim_points():
    with pytest.raises(ValueError, match=r"points must be shape \(D,\) or \(D, N\)"):
        pr.Seeder(points=np.zeros((2, 2, 2)), rocks=make_rocks())


def test_ppf_rejects_wrong_number_of_rows():
    s = pr.Seeder(points=single_point(2), rocks=make_rocks())
    with pytest.raises(ValueError, match="q must have shape"):
        s.ppf(np.zeros((s.numRandomVariables + 1, 10)))
