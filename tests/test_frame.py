"""Tests for pyrockfall.CurvilinearFrame and the coordinate methods of Model.

Covers the polyline fit, the (x,y,z) <-> (s,d,e) <-> (u,v,w) transforms and
their round trips on a wall with a 90 degree corner, the orientation
convention (d increases down-dip), corner refinement, the straight frame,
persistence, and the Model-level API (fitFrame/frame/sde/uvw and cache
invalidation).
"""
import numpy as np
import pytest

import pyrockfall as pr
from pyrockfall import CurvilinearFrame, fit_polyline


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

LEG = 40.0
HEIGHT = 20.0
DIP = np.radians(60.0)


def l_wall(n=8000, seed=0, noise=0.02):
    """Wall with a 90 degree corner, dipping 60 degrees, built by unrolling a
    planar face along the polyline (0,0) -> (LEG,0) -> (LEG,LEG).

    Returns xyz, the (s, d, e) used to build it and a two-class label that is
    layered with elevation.
    """
    rng = np.random.default_rng(seed)
    truth = CurvilinearFrame([[0.0, 0.0], [LEG, 0.0], [LEG, LEG]])
    s = rng.uniform(0.0, 2 * LEG, n)
    e = rng.uniform(0.0, HEIGHT, n)
    d = -(e - 0.5 * HEIGHT) / np.tan(DIP) + noise * rng.standard_normal(n)   # lower = further out
    xyz = truth.sde_to_xyz(np.column_stack([s, d, e]))
    mat = (e > 0.5 * HEIGHT).astype(int)
    return xyz, np.column_stack([s, d, e]), mat


# ---------------------------------------------------------------------------
# fit_polyline
# ---------------------------------------------------------------------------

def test_fit_polyline_straight_strip_is_one_segment():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 30, 3000)
    y = 0.5 * x + 0.05 * rng.standard_normal(3000)
    V, _ = fit_polyline(x, y, tol=0.5, res=0.5)
    assert V.shape == (2, 2)
    direction = (V[1] - V[0]) / np.linalg.norm(V[1] - V[0])
    np.testing.assert_allclose(direction, np.array([1.0, 0.5]) / np.hypot(1.0, 0.5), atol=1e-2)


def test_fit_polyline_finds_the_corner():
    rng = np.random.default_rng(1)
    t = rng.uniform(0, 2 * LEG, 4000)
    x = np.where(t < LEG, t, LEG) + 0.05 * rng.standard_normal(4000)
    y = np.where(t < LEG, 0.0, t - LEG) + 0.05 * rng.standard_normal(4000)
    V, _ = fit_polyline(x, y, tol=0.5, res=0.5)
    assert V.shape == (3, 2)
    np.testing.assert_allclose(V[1], [LEG, 0.0], atol=0.5)


# ---------------------------------------------------------------------------
# CurvilinearFrame: transforms
# ---------------------------------------------------------------------------

def test_known_polyline_gives_expected_coordinates():
    f = CurvilinearFrame([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]])
    xyz = np.array([
        [5.0, 0.0, 1.0],     # on the first leg
        [5.0, 2.0, 1.0],     # first leg, 2 to the left
        [5.0, -3.0, 4.0],    # first leg, 3 to the right
        [8.0, 5.0, 0.0],     # second leg, 2 to the left
        [10.0, 0.0, 7.0],    # the corner itself
        [10.0, 4.0, 2.0],    # on the second leg
    ])
    sde = f.xyz_to_sde(xyz)
    # s is the arclength on the wall line; d is the distance from it, e is z
    np.testing.assert_allclose(sde[0], [5.0, 0.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(sde[1, 1:], [2.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(sde[2, 1:], [-3.0, 4.0], atol=1e-12)
    np.testing.assert_allclose(sde[3, 1:], [2.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(sde[4], [10.0, 0.0, 7.0], atol=1e-12)
    np.testing.assert_allclose(sde[5], [14.0, 0.0, 2.0], atol=1e-12)
    # off the line, lines of constant s lean towards the corner's bisector
    assert sde[1, 0] > 5.0 and sde[2, 0] < 5.0
    assert f.length == pytest.approx(20.0)
    np.testing.assert_allclose(np.degrees(f.turning_angles()), [90.0])


def test_round_trips_on_corner_wall_are_exact():
    xyz, _, _ = l_wall()
    f = CurvilinearFrame.fit(xyz)
    sde, n_nan, n_amb = f.xyz_to_sde(xyz, return_info=True)
    assert n_nan == 0 and n_amb == 0
    np.testing.assert_allclose(f.sde_to_xyz(sde), xyz, atol=1e-9)
    uvw = f.sde_to_uvw(sde)
    np.testing.assert_allclose(f.uvw_to_sde(uvw), sde, atol=1e-9)
    np.testing.assert_allclose(f.uvw_to_xyz(f.xyz_to_uvw(xyz)), xyz, atol=1e-9)
    np.testing.assert_array_equal(sde[:, 2], xyz[:, 2])  # e is z


def test_fit_recovers_the_wall_and_orients_d_down_dip():
    xyz, truth_sde, _ = l_wall()
    f = CurvilinearFrame.fit(xyz, corner_length=None)
    assert f.n_segments == 2
    assert np.degrees(f.turning_angles()[0]) == pytest.approx(90.0, abs=3.0)
    assert f.length == pytest.approx(2 * LEG, rel=0.05)
    sde = f.xyz_to_sde(xyz)
    assert np.corrcoef(sde[:, 1], sde[:, 2])[0, 1] < -0.9      # d grows as elevation falls
    # distance from the wall line is recovered up to a constant offset
    off = sde[:, 1] - truth_sde[:, 1]
    assert np.std(off) < 0.2
    info = f.diagnostics(xyz)
    assert info["n_nan"] == 0 and info["n_ambiguous"] == 0 and info["n_past_fold"] == 0


def test_pca_axes_are_right_handed_and_w_is_small():
    xyz, _, _ = l_wall()
    f = CurvilinearFrame.fit(xyz)
    A = f.pca_axes
    np.testing.assert_allclose(A @ A.T, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(np.cross(A[0], A[1]), A[2], atol=1e-12)
    assert A[0, 0] > 0.99          # U along +s
    assert A[1, 2] > 0             # V upward
    assert f.pca_var[0] > f.pca_var[1] > f.pca_var[2]
    uvw = f.xyz_to_uvw(xyz)
    # the unrolled face is a plane: W is noise, far below the corner's bulge
    c = xyz - xyz.mean(axis=0)
    plain_w = np.sqrt(np.linalg.eigvalsh(np.cov(c.T))[0])
    assert uvw[:, 2].std() < 0.1 * plain_w
    assert abs(f.horizontal_angle_uv()) < 2.0


def test_uvw_requires_pca():
    f = CurvilinearFrame([[0.0, 0.0], [1.0, 0.0]])
    with pytest.raises(ValueError, match="fit_pca"):
        f.sde_to_uvw(np.zeros((1, 3)))


def test_fold_limits_and_points_past_them():
    f = CurvilinearFrame([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]])
    lim = f.fold_limits()
    assert np.isfinite(lim[:, 0]).all() and np.isinf(lim[:, 1]).all()   # inner side is the left
    np.testing.assert_allclose(lim[:, 0], 10.0)
    sde = f.xyz_to_sde(np.array([[5.0, 9.0, 0.0], [-1.0, 11.0, 0.0]]))
    assert np.isfinite(sde[0]).all()                                   # within the limit
    # past segment 0's fold limit, but behind the start: the wall line runs straight on
    np.testing.assert_allclose(sde[1, :2], [-1.0, 11.0])


def test_xyz_to_sde_is_bounded_past_the_ends():
    # Regression: past an end vertex the mitred offset direction was extrapolated, so
    # it kept turning until the offset lines folded and s ran off to +-infinity.
    f = CurvilinearFrame([[0.0, 0.0], [40.0, 0.0], [45.0, 8.0]])        # short, sharp end
    g = np.mgrid[-20:70:0.5, -30:30:0.5].reshape(2, -1).T
    xyz = np.column_stack([g, np.zeros(len(g))])
    sde = f.xyz_to_sde(xyz)
    ok = np.isfinite(sde[:, 0])
    s = sde[ok, 0]
    assert s.min() >= -np.hypot(*(g - f.polyline[0]).T).max()
    assert s.max() <= f.length + np.hypot(*(g - f.polyline[-1]).T).max()
    np.testing.assert_allclose(f.sde_to_xyz(sde[ok]), xyz[ok], atol=1e-8)
    # beyond the last vertex s is the distance along the last segment's line
    p = f.polyline[-1] + 5.0 * f.Eh[-1] + 3.0 * f.N[-1]
    np.testing.assert_allclose(f.xyz_to_sde(np.array([[*p, 0.0]]))[0, :2], [f.length + 5.0, 3.0])


# ---------------------------------------------------------------------------
# Corner refinement
# ---------------------------------------------------------------------------

def _skew(frame, sde):
    """Angle (degrees) between each point's line of constant s and the wall normal."""
    seg = np.clip(np.searchsorted(frame.S0, sde[:, 0], side="right") - 1, 0, frame.n_segments - 1)
    t = (sde[:, 0] - frame.S0[seg]) / frame.L[seg]
    off = (1 - t)[:, None] * frame.M[seg] + t[:, None] * frame.M[seg + 1]
    return np.degrees(np.arctan(np.abs(np.einsum("ij,ij->i", off, frame.Eh[seg]))))


def test_refine_corners_keeps_the_wall_line_and_confines_the_turn():
    xyz, _, _ = l_wall()
    plain = CurvilinearFrame.fit(xyz, corner_length=None)
    fine = plain.refine_corners(8.0)
    assert fine.n_segments == plain.n_segments + 2
    assert fine.length == pytest.approx(plain.length)
    np.testing.assert_allclose(np.sort(np.degrees(fine.turning_angles()))[-1],
                               np.degrees(plain.turning_angles()).max(), atol=1e-9)
    # away from the corner the slices are perpendicular to the wall
    sde = fine.xyz_to_sde(xyz)
    corner = fine.S0[np.argmax(fine.turning_angles()) + 1]
    far = np.abs(sde[:, 0] - corner) > 8.0 + 1e-9
    assert _skew(fine, sde[far]).max() < 1e-6
    # without refinement they fan out along the whole wall
    assert np.median(_skew(plain, plain.xyz_to_sde(xyz))) > 15.0
    # still exact
    np.testing.assert_allclose(fine.sde_to_xyz(sde), xyz, atol=1e-9)
    assert not np.isnan(sde).any()


def test_refine_corners_auto_needs_d_max_and_respects_min_turn():
    f = CurvilinearFrame([[0.0, 0.0], [10.0, 0.0], [20.0, 0.2]])   # ~1 degree bend
    with pytest.raises(ValueError, match="d_max"):
        f.refine_corners("auto")
    assert f.refine_corners("auto", d_max=5.0).n_segments == 2      # below min_turn: untouched
    g = CurvilinearFrame([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]]).refine_corners("auto", d_max=2.0)
    assert g.n_segments == 4
    np.testing.assert_allclose(g.L, [7.0, 3.0, 3.0, 7.0])           # 1.5 * 2 * tan(45 deg) = 3


def test_default_fit_refines_corners():
    xyz, _, _ = l_wall()
    assert CurvilinearFrame.fit(xyz).n_segments == 4
    assert CurvilinearFrame.fit(xyz, corner_length=None).n_segments == 2


# ---------------------------------------------------------------------------
# Straight frame, degenerate input, persistence
# ---------------------------------------------------------------------------

def test_straight_frame_is_a_plain_rotation():
    rng = np.random.default_rng(3)
    xyz = rng.uniform(-5, 5, size=(200, 3))
    f = CurvilinearFrame.straight((1.0, 2.0), (3.0, 4.0))
    sde = f.xyz_to_sde(xyz)
    u = np.array([0.6, 0.8])
    np.testing.assert_allclose(sde[:, 0], (xyz[:, :2] - [1.0, 2.0]) @ u, atol=1e-12)
    np.testing.assert_allclose(sde[:, 1], (xyz[:, :2] - [1.0, 2.0]) @ [-0.8, 0.6], atol=1e-12)
    np.testing.assert_allclose(f.sde_to_xyz(sde), xyz, atol=1e-12)
    with pytest.raises(ValueError, match="zero"):
        CurvilinearFrame.straight((0, 0), (0, 0))


def test_invalid_polylines_raise():
    with pytest.raises(ValueError, match="at least two"):
        CurvilinearFrame([[0.0, 0.0]])
    with pytest.raises(ValueError, match="zero-length"):
        CurvilinearFrame([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
    with pytest.raises(ValueError, match="turns back"):
        CurvilinearFrame([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]])


def test_fit_falls_back_to_a_straight_line_on_unstructured_points():
    xyz = np.random.default_rng(0).uniform(size=(40, 3))
    f = CurvilinearFrame.fit(xyz)
    assert f.n_segments >= 1
    assert not np.isnan(f.xyz_to_sde(xyz)).any()


def test_save_load_roundtrip(tmp_path):
    xyz, _, _ = l_wall(n=2000)
    f = CurvilinearFrame.fit(xyz)
    path = tmp_path / "frame.json"
    f.save(path)
    g = CurvilinearFrame.load(path)
    np.testing.assert_allclose(g.polyline, f.polyline)
    np.testing.assert_allclose(g.xyz_to_uvw(xyz), f.xyz_to_uvw(xyz), atol=1e-12)


# ---------------------------------------------------------------------------
# Model API
# ---------------------------------------------------------------------------

def test_model_sde_uvw_and_stored_attributes():
    xyz, _, mat = l_wall()
    pc = pr.PointCloud(points=xyz)
    pc.set_attr("mat", mat)
    sde = pc.sde(store=True)
    uvw = pc.uvw(store=True)
    assert sde.shape == uvw.shape == xyz.shape
    for name in ("strike", "dip_direction", "elevation", "U", "V", "W"):
        assert pc.has_attr(name)
    np.testing.assert_array_equal(pc.get_attr("elevation"), xyz[:, 2])
    np.testing.assert_array_equal(pc.get_attr("V"), uvw[:, 1])
    assert pc.frame is pc.frame            # fitted once, cached


def test_model_uses_normals_to_ignore_the_floor():
    xyz, _, _ = l_wall(n=6000)
    rng = np.random.default_rng(5)
    # a wide floor in front of the wall, at its toe
    floor = np.column_stack([rng.uniform(0, LEG + 20, 6000), rng.uniform(-20, LEG, 6000), np.zeros(6000)])
    floor = floor[(floor[:, 1] < -6) | (floor[:, 0] > LEG + 6)]
    pts = np.vstack([xyz, floor])
    normals = np.vstack([np.tile([0.0, -0.87, 0.5], (len(xyz), 1)), np.tile([0.0, 0.0, 1.0], (len(floor), 1))])
    pc = pr.PointCloud(points=pts, normals=normals)
    f = pc.fitFrame(corner_length=None)
    assert f.n_segments == 2
    assert f.length == pytest.approx(2 * LEG, rel=0.08)


def wide_wall(leg=60.0, height=50.0, dip=50.0, n=20000, seed=0):
    """Laid-back version of :func:`l_wall`: 50 m high at 50 degrees, so the face is
    ~42 m wide in plan, as wide as the legs are long. Normals are exact; every
    face point passes the steep mask (|nz| = cos(dip) = 0.64 < 0.7)."""
    rng = np.random.default_rng(seed)
    truth = CurvilinearFrame([[0.0, 0.0], [leg, 0.0], [leg, leg]])
    s = rng.uniform(0.0, 2 * leg, n)
    e = rng.uniform(0.0, height, n)
    d = -(e - 0.5 * height) / np.tan(np.radians(dip)) + 0.05 * rng.standard_normal(n)
    xyz = truth.sde_to_xyz(np.column_stack([s, d, e]))
    normals = np.tile([np.sin(np.radians(dip)), 0.0, np.cos(np.radians(dip))], (n, 1))
    return xyz, normals


def test_model_normals_on_a_laid_back_wall_match_the_fit_without():
    # Regression: the steep mask kept the whole face, a strip wider than the corner
    # can take, and the fit zigzagged (39 segments, every point ambiguous).
    xyz, normals = wide_wall()
    plain = pr.PointCloud(points=xyz).fitFrame(corner_length=None)
    f = pr.PointCloud(points=xyz, normals=normals).fitFrame(corner_length=None)
    assert plain.n_segments == f.n_segments == 2
    assert f.length == pytest.approx(120.0, rel=0.05)
    assert np.degrees(f.turning_angles()[0]) == pytest.approx(90.0, abs=3.0)
    info = f.diagnostics(xyz)
    assert info["n_nan"] == 0 and info["n_ambiguous"] == 0 and info["n_past_fold"] == 0


def hooked_wall(n=20000, seed=0, length=100.0, height=50.0, dip=50.0, hook=9.0, turn=65.0):
    """Straight laid-back wall along x plus a stub of face at mid elevation that runs
    ``hook`` m off its end at ``turn`` degrees, like the face along a clipped border."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, length, n)
    e = rng.uniform(0.0, height, n)
    y = (e - 0.5 * height) / np.tan(np.radians(dip)) + 0.05 * rng.standard_normal(n)
    m = n // 40
    a, u = np.radians(turn), rng.uniform(0.0, hook, m)
    stub = np.column_stack([length + u * np.cos(a), u * np.sin(a), 0.5 * height + rng.uniform(-1, 1, m)])
    xyz = np.vstack([np.column_stack([x, y, e]), stub])
    normals = np.tile([0.0, -np.sin(np.radians(dip)), np.cos(np.radians(dip))], (n + m, 1))
    return xyz, normals


def test_fit_drops_a_hooked_end_segment_that_folds():
    # Regression (susan.ply with normals): the band followed a short stub of face off
    # the end of the wall, giving a 9 m end segment turned back 65 degrees whose fold
    # limit (~14 m) is narrower than the wall (+-21 m), and s values of tens of km.
    xyz, normals = hooked_wall()
    for corner_length in (None, "auto"):
        f = pr.PointCloud(points=xyz, normals=normals).fitFrame(corner_length=corner_length)
        assert np.degrees(f.turning_angles()).max(initial=0.0) < 5.0
        assert f.length == pytest.approx(100.0, abs=5.0)
        s = f.xyz_to_sde(xyz)[:, 0]
        assert not np.isnan(s).any()
        assert -1.0 < s.min() and s.max() < f.length + 1.0


def test_fit_polyline_refits_do_not_diverge_on_a_wide_strip():
    # Re-binning along the previous polyline must not make a too-wide strip worse.
    xyz, _ = wide_wall()
    V0, _ = fit_polyline(xyz[:, 0], xyz[:, 1], tol=1.0, res=0.5, n_iter=0)
    V, _ = fit_polyline(xyz[:, 0], xyz[:, 1], tol=1.0, res=0.5, n_iter=2)
    assert len(V) <= len(V0)
    folded = lambda P: sum(CurvilinearFrame(P).xyz_to_sde(xyz, return_info=True)[1:])
    assert folded(V) <= folded(V0)


def test_frame_is_invalidated_by_moves_and_kept_by_subsets():
    xyz, _, _ = l_wall(n=3000)
    pc = pr.PointCloud(points=xyz.copy())
    f = pc.frame
    pc.clip(xyz.min(axis=0), xyz.max(axis=0) - [0, 0, 5.0])
    assert pc._frame is f                  # a subset keeps the frame
    sub = pc._from_indices(np.arange(10))
    assert sub._frame is f
    pc.translate(1.0, "x")
    assert pc._frame is None               # moved: refit on demand
    pc.frame
    pc.rotate(10.0)
    assert pc._frame is None
    g = CurvilinearFrame.straight()
    pc.frame = g
    assert pc.frame is g
    with pytest.raises(TypeError):
        pc.frame = "not a frame"


def test_fit_frame_needs_points():
    with pytest.raises(ValueError, match=">=3 points"):
        pr.PointCloud().fitFrame()


def test_mesh_frame_uses_triangle_normals():
    # steep strip (the face) next to a flat strip (the floor)
    n = 30
    x = np.linspace(0, 30, n)
    rows = [np.column_stack([x, np.full(n, y), np.full(n, z)]) for y, z in ((0.0, 10.0), (3.0, 0.0), (12.0, 0.0))]
    pts = np.vstack(rows)
    tris = []
    for r in range(2):
        for i in range(n - 1):
            a, b, c, d = r * n + i, r * n + i + 1, (r + 1) * n + i, (r + 1) * n + i + 1
            tris += [[a, b, d], [a, d, c]]
    m = pr.Mesh(points=pts, triangles=np.array(tris))
    steep = m._steep_mask()
    assert steep[:n].all()                 # crest row: only face triangles
    assert not steep[2 * n:].any()         # far floor row: only floor triangles
    sde = m.sde()
    assert not np.isnan(sde).any()
    assert np.ptp(sde[:, 0]) == pytest.approx(30.0, rel=0.01)


def test_outward_sign_points_to_the_air_on_a_dipping_face():
    rng = np.random.default_rng(0)
    truth = pr.CurvilinearFrame([[0.0, 0.0], [30.0, 0.0]])
    e = rng.uniform(0, 10, 3000)
    sde = np.column_stack([rng.uniform(0, 30, 3000), -(e - 5) / np.tan(np.radians(50.0)), e])
    pc = pr.PointCloud(points=truth.sde_to_xyz(sde))
    frame = pc.fitFrame()
    pc.uvw()
    out = frame.outward_sign()
    # moving along +out*w from the face goes up (into the air above a dipping face)
    p0 = frame.uvw_to_xyz(np.array([[0.0, 0.0, 0.0]]))[0]
    p1 = frame.uvw_to_xyz(np.array([[0.0, 0.0, out]]))[0]
    assert p1[2] > p0[2]
