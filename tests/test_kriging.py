"""Tests for pyrockfall._kriging: variogram model and fit, ordinary kriging
(single system and moving neighbourhood), hole detection on an occupancy grid
and hole filling in the (u, v) plane.

The cross-check against PyKrige runs only if PyKrige is installed.
"""
import numpy as np
import pytest

from pyrockfall._kriging import (
    Variogram,
    NestedVariogram,
    variogram_from_dict,
    fit_variogram,
    fit_nested_variogram,
    surface_mask,
    ordinary_kriging,
    poly_design,
    find_holes,
    fill_holes_uv,
    GaussianField,
    RandomField,
    conditional_simulation,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def gaussian_field(xy, variogram, seed=0):
    """One realisation of a zero-mean field with the covariance of `variogram`."""
    rng = np.random.default_rng(seed)
    p = variogram.transform(xy)
    h = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)
    cov = variogram.psill + variogram.nugget - variogram.gamma(h)
    L = np.linalg.cholesky(cov + 1e-10 * np.eye(len(xy)))
    return L @ rng.standard_normal(len(xy))


def plane_with_hole(seed=0, n=12000, hole_centre=(30.0, 10.0), hole_radius=4.0):
    """Points on a gently curved face, layered classes, and a disc to cut out."""
    rng = np.random.default_rng(seed)
    uv = np.column_stack([rng.uniform(0, 60, n), rng.uniform(0, 20, n)])
    w = 0.3 * np.sin(uv[:, 0] / 6.0) + 0.02 * (uv[:, 1] - 10.0) ** 2 / 10.0
    label = np.digitize(uv[:, 1] + 0.5 * np.sin(uv[:, 0] / 5.0), [6.0, 13.0])   # 3 wavy layers
    in_hole = np.hypot(uv[:, 0] - hole_centre[0], uv[:, 1] - hole_centre[1]) < hole_radius
    return np.column_stack([uv, w]), label, in_hole


# ---------------------------------------------------------------------------
# Variogram
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model", ["spherical", "exponential", "gaussian", "linear"])
def test_variogram_is_zero_at_origin_and_increasing(model):
    v = Variogram(model=model, psill=2.0, range=5.0, nugget=0.5)
    h = np.linspace(0, 20, 200)
    g = v.gamma(h)
    assert g[0] == 0.0
    assert g[1] >= 0.5                       # nugget right after the origin
    assert np.all(np.diff(g[1:]) >= -1e-12)


def test_spherical_reaches_sill_at_range():
    v = Variogram("spherical", psill=2.0, range=5.0, nugget=0.5)
    np.testing.assert_allclose(v.gamma([5.0, 9.0]), 2.5)
    assert v.gamma(2.5) < 2.5


def test_unknown_model_raises():
    with pytest.raises(ValueError, match="Unknown variogram model"):
        Variogram(model="cubic")


def test_anisotropy_transform_stretches_across_the_major_axis():
    v = Variogram(aniso=4.0, angle=90.0)       # major axis along y
    p = v.transform(np.array([[0.0, 1.0], [1.0, 0.0]]))
    np.testing.assert_allclose(np.linalg.norm(p, axis=1), [1.0, 4.0], atol=1e-12)


def test_fit_variogram_recovers_a_known_model():
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 60, size=(900, 2))
    true = Variogram("spherical", psill=1.5, range=12.0, nugget=0.0)
    z = gaussian_field(xy, true, seed=1)
    fit = fit_variogram(xy, z, "spherical")
    assert 0.6 * true.range < fit.range < 1.6 * true.range
    assert 0.5 * true.psill < fit.psill + fit.nugget < 1.8 * true.psill
    assert fit.nugget < 0.3


def test_fit_variogram_constant_data_is_harmless():
    xy = np.random.default_rng(0).uniform(size=(50, 2))
    fit = fit_variogram(xy, np.ones(50))
    assert np.isfinite([fit.psill, fit.range, fit.nugget]).all()


# ---------------------------------------------------------------------------
# Ordinary kriging
# ---------------------------------------------------------------------------

def test_kriging_is_exact_at_the_data_with_zero_variance():
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 10, size=(80, 2))
    z = np.sin(xy[:, 0]) + xy[:, 1]
    v = Variogram("spherical", psill=1.0, range=6.0, nugget=0.1)
    est, var = ordinary_kriging(xy, z, xy, v)
    np.testing.assert_allclose(est, z, atol=1e-8)
    np.testing.assert_allclose(var, 0.0, atol=1e-8)


def test_kriging_weights_sum_to_one():
    # A constant field is returned unchanged wherever it is estimated
    rng = np.random.default_rng(1)
    xy = rng.uniform(0, 10, size=(60, 2))
    new = rng.uniform(-5, 15, size=(40, 2))
    v = Variogram("exponential", psill=1.0, range=4.0, nugget=0.05)
    for kw in ({}, {"n_closest": 12}):
        est, var = ordinary_kriging(xy, np.full(60, 3.7), new, v, **kw)
        np.testing.assert_allclose(est, 3.7, atol=1e-9)
        assert np.all(var >= 0)


def test_kriging_variance_grows_away_from_the_data():
    xy = np.random.default_rng(2).uniform(0, 10, size=(100, 2))
    z = xy[:, 0]
    v = Variogram("spherical", psill=1.0, range=8.0, nugget=0.0)
    _, var = ordinary_kriging(xy, z, np.array([[5.0, 5.0], [30.0, 30.0]]), v)
    assert var[1] > var[0]
    assert var[1] == pytest.approx(v.psill, rel=0.2)      # beyond the range: about the sill


def test_kriging_several_columns_match_one_at_a_time():
    rng = np.random.default_rng(3)
    xy = rng.uniform(0, 10, size=(70, 2))
    vals = rng.standard_normal((70, 3))
    new = rng.uniform(0, 10, size=(25, 2))
    v = Variogram("spherical", psill=1.0, range=5.0, nugget=0.02)
    both, var = ordinary_kriging(xy, vals, new, v)
    assert both.shape == (25, 3)
    for j in range(3):
        one, var_j = ordinary_kriging(xy, vals[:, j], new, v)
        np.testing.assert_allclose(both[:, j], one, atol=1e-9)
        np.testing.assert_allclose(var, var_j, atol=1e-9)


def test_moving_neighbourhood_is_close_to_the_single_system():
    rng = np.random.default_rng(4)
    xy = rng.uniform(0, 30, size=(600, 2))
    v = Variogram("spherical", psill=1.0, range=6.0, nugget=0.0)
    z = gaussian_field(xy, v, seed=5)
    new = rng.uniform(5, 25, size=(100, 2))
    full, _ = ordinary_kriging(xy, z, new, v)
    local, _ = ordinary_kriging(xy, z, new, v, n_closest=40)
    assert np.sqrt(np.mean((full - local) ** 2)) < 0.15 * z.std()


def test_kriging_handles_duplicate_locations_and_empty_targets():
    xy = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    z = np.array([1.0, 1.0, 2.0, 3.0])
    v = Variogram("spherical", psill=1.0, range=3.0, nugget=0.0)
    est, var = ordinary_kriging(xy, z, np.array([[0.5, 0.5]]), v)
    assert np.isfinite(est).all() and np.isfinite(var).all()
    est, var = ordinary_kriging(xy, z, np.empty((0, 2)), v)
    assert est.shape == (0,) and var.shape == (0,)
    with pytest.raises(ValueError, match="same number of rows"):
        ordinary_kriging(xy, z[:2], xy, v)


def test_anisotropic_kriging_follows_the_layers():
    # Value depends on y only. Data along two columns; estimate between them.
    y = np.linspace(0, 10, 41)
    xy = np.vstack([np.column_stack([np.zeros(41), y]), np.column_stack([np.full(41, 20.0), y])])
    z = np.r_[np.sin(y), np.sin(y)]
    new = np.column_stack([np.full(41, 10.0), y])
    iso = Variogram("spherical", psill=1.0, range=6.0, nugget=0.0)
    lay = Variogram("spherical", psill=1.0, range=60.0, nugget=0.0, aniso=10.0, angle=0.0)
    e_iso, _ = ordinary_kriging(xy, z, new, iso)
    e_lay, _ = ordinary_kriging(xy, z, new, lay)
    err_iso = np.sqrt(np.mean((e_iso - np.sin(y)) ** 2))
    err_lay = np.sqrt(np.mean((e_lay - np.sin(y)) ** 2))
    assert err_lay < 0.1                 # sin(y) has amplitude 1
    assert err_lay < 0.3 * err_iso


@pytest.mark.parametrize("model", ["spherical", "exponential", "gaussian"])
def test_matches_pykrige(model):
    pykrige_ok = pytest.importorskip("pykrige.ok")
    rng = np.random.default_rng(6)
    xy = rng.uniform(0, 20, size=(120, 2))
    z = np.sin(xy[:, 0] / 3.0) + 0.3 * xy[:, 1] + 0.1 * rng.standard_normal(120)
    new = rng.uniform(0, 20, size=(60, 2))
    v = Variogram(model, psill=1.3, range=7.0, nugget=0.15)
    ref = pykrige_ok.OrdinaryKriging(
        xy[:, 0], xy[:, 1], z, variogram_model=model,
        variogram_parameters={"psill": v.psill, "range": v.range, "nugget": v.nugget},
    )
    z_ref, ss_ref = ref.execute("points", new[:, 0], new[:, 1])
    est, var = ordinary_kriging(xy, z, new, v)
    np.testing.assert_allclose(est, np.asarray(z_ref), atol=1e-6)
    np.testing.assert_allclose(var, np.asarray(ss_ref), atol=1e-6)


def test_poly_design_columns():
    uv = np.array([[1.0, 2.0], [3.0, 5.0]])
    A = poly_design(uv, 1, np.zeros(2), 1.0)
    np.testing.assert_allclose(A, [[1.0, 2.0, 1.0], [1.0, 5.0, 3.0]])     # 1, v, u
    assert poly_design(uv, 0, np.zeros(2), 1.0).shape == (2, 1)
    assert poly_design(uv, 2, np.zeros(2), 1.0).shape == (2, 6)


# ---------------------------------------------------------------------------
# Holes
# ---------------------------------------------------------------------------

def test_find_holes_enclosed_gap():
    uvw, _, in_hole = plane_with_hole()
    labels, n, origin, counts = find_holes(uvw[~in_hole, :2], cell=0.8)
    assert n == 1
    area = np.count_nonzero(labels) * 0.8 ** 2
    assert area == pytest.approx(np.pi * 4.0 ** 2, rel=0.25)
    ci, cj = np.argwhere(labels == 1).mean(axis=0)
    np.testing.assert_allclose(origin + (np.array([ci, cj]) + 0.5) * 0.8, [30.0, 10.0], atol=0.6)
    assert counts.sum() == np.count_nonzero(~in_hole)


def test_find_holes_no_gap_and_min_cells():
    uvw, _, _ = plane_with_hole()
    assert find_holes(uvw[:, :2], cell=0.8)[1] == 0
    # a one-cell gap is ignored as sampling noise
    one = np.hypot(uvw[:, 0] - 30.0, uvw[:, 1] - 10.0) < 0.6
    assert find_holes(uvw[~one, :2], cell=0.8, min_cells=6)[1] == 0
    with pytest.raises(ValueError, match="cell must be > 0"):
        find_holes(uvw[:, :2], cell=0.0)


def test_find_holes_at_the_edge_needs_the_removed_points():
    uvw, _, _ = plane_with_hole()
    bite = np.hypot(uvw[:, 0] - 30.0, uvw[:, 1] - 0.0) < 4.0      # half disc on the bottom edge
    kept = uvw[~bite, :2]
    assert find_holes(kept, cell=0.8)[1] == 0
    labels, n, _, _ = find_holes(kept, cell=0.8, extra_uv=uvw[bite, :2])
    assert n == 1
    assert np.count_nonzero(labels) * 0.8 ** 2 == pytest.approx(0.5 * np.pi * 16.0, rel=0.3)


def test_fill_holes_uv_recovers_height_and_classes():
    uvw, label, in_hole = plane_with_hole()
    res = fill_holes_uv(
        uvw[~in_hole], {"label": (label[~in_hole], "categorical"), "w_copy": (uvw[~in_hole, 2], "continuous")},
        cell=0.8,
    )
    new = res["uvw"]
    assert len(res["holes"]) == 1 and res["holes"][0]["n_new"] == new.shape[0] > 0
    # the new points are inside the disc, at about the density of the others
    r = np.hypot(new[:, 0] - 30.0, new[:, 1] - 10.0)
    assert r.max() < 4.0 + 1.5 * 0.8
    assert new.shape[0] == pytest.approx(np.count_nonzero(in_hole), rel=0.35)
    # height
    w_true = 0.3 * np.sin(new[:, 0] / 6.0) + 0.02 * (new[:, 1] - 10.0) ** 2 / 10.0
    assert np.sqrt(np.mean((new[:, 2] - w_true) ** 2)) < 0.06     # w spans about 0.8
    assert np.all(res["std"] >= 0)
    # classes
    lab = res["attrs"]["label"]
    true_label = np.digitize(new[:, 1] + 0.5 * np.sin(new[:, 0] / 5.0), [6.0, 13.0])
    assert np.mean(lab["value"] == true_label) > 0.93
    P = lab["probs"]
    assert P.shape == (new.shape[0], 3)
    assert P.min() >= 0.0 and P.max() <= 1.0
    np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-9)
    np.testing.assert_array_equal(lab["value"], lab["classes"][P.argmax(axis=1)])
    # a continuous attribute is kriged too
    cont = res["attrs"]["w_copy"]
    assert np.sqrt(np.mean((cont["value"] - w_true) ** 2)) < 0.08
    assert cont["std"].shape == (new.shape[0],)
    # every new point knows its nearest existing point
    kept = uvw[~in_hole]
    d = np.hypot(*(kept[res["nearest"], :2] - new[:, :2]).T)
    assert d.max() < 4.0 + 0.8


def test_fill_holes_uv_options_and_empty_result():
    uvw, label, in_hole = plane_with_hole(n=8000)
    kept = uvw[~in_hole]
    spec = {"label": (label[~in_hole], "categorical")}
    for kw in ({"shared_variogram": False}, {"n_closest": 32}, {"spacing": 0.5}, {"trend_order": 2}):
        res = fill_holes_uv(kept, spec, cell=0.9, **kw)
        assert res["uvw"].shape[0] > 0
        np.testing.assert_allclose(res["attrs"]["label"]["probs"].sum(axis=1), 1.0, atol=1e-9)
    coarse = fill_holes_uv(kept, spec, cell=0.9, spacing=0.5)["uvw"].shape[0]
    fine = fill_holes_uv(kept, spec, cell=0.9, spacing=0.25)["uvw"].shape[0]
    assert fine > 3 * coarse
    # nothing missing: empty, well-shaped result
    res = fill_holes_uv(uvw, {"label": (label, "categorical")}, cell=0.9)
    assert res["uvw"].shape == (0, 3) and res["holes"] == []
    assert res["attrs"]["label"]["probs"].shape == (0, 3)
    with pytest.raises(ValueError, match="kind must be"):
        fill_holes_uv(kept, {"label": (label[~in_hole], "ordinal")}, cell=0.9)
    with pytest.raises(ValueError, match="one row per point"):
        fill_holes_uv(kept, {"label": (label, "categorical")}, cell=0.9)


# ---------------------------------------------------------------------------
# GaussianField
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model", ["spherical", "exponential", "gaussian"])
def test_gaussian_field_has_unit_variance_and_the_model_correlation(model):
    rng = np.random.default_rng(0)
    xy = np.column_stack([np.linspace(0, 30, 61), np.zeros(61)])          # points 0.5 apart on a line
    v = Variogram(model, psill=1.0, range=6.0, nugget=0.0)
    z = GaussianField(xy, v).sample(rng, 4000)
    assert z.shape == (4000, 61)
    np.testing.assert_allclose(z.var(axis=0).mean(), 1.0, atol=0.05)
    assert np.abs(z.mean(axis=0)).max() < 0.1
    for lag in (2, 6, 12):                                               # 1, 3 and 6 apart
        h = 0.5 * lag
        rho = np.mean([np.corrcoef(z[:, i], z[:, i + lag])[0, 1] for i in range(0, 61 - lag, 5)])
        assert rho == pytest.approx(1.0 - v.gamma(h), abs=0.07)


def test_gaussian_field_nugget_and_anisotropy():
    rng = np.random.default_rng(1)
    xy = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 3.0]])
    noisy = GaussianField(xy[:2], Variogram("spherical", psill=0.5, range=60.0, nugget=0.5)).sample(rng, 6000)
    assert np.corrcoef(noisy[:, 0], noisy[:, 1])[0, 1] == pytest.approx(0.5 * (1 - Variogram("spherical", 1, 60.0).gamma(3.0)), abs=0.05)
    layered = GaussianField(xy, Variogram("spherical", psill=1.0, range=20.0, aniso=10.0, angle=0.0)).sample(rng, 6000)
    along = np.corrcoef(layered[:, 0], layered[:, 1])[0, 1]               # along the layers
    across = np.corrcoef(layered[:, 0], layered[:, 2])[0, 1]              # across them: range 2
    assert along > 0.7 and abs(across) < 0.07


def test_gaussian_field_resolution_keeps_the_short_range_roughness():
    rng = np.random.default_rng(2)
    xy = np.column_stack([np.arange(0, 20, 0.1), np.zeros(200)])            # points 0.1 apart
    v = Variogram("spherical", psill=1.0, range=20.0)
    for res, tol in ((None, None), (0.1, 0.2)):
        z = GaussianField(xy, v, resolution=res).sample(rng, 3000)
        g1 = 0.5 * np.mean((z[:, 1:] - z[:, :-1]) ** 2)                    # semivariance at 0.1
        if res is None:
            assert abs(g1 / v.gamma(0.1) - 1.0) > 0.5                      # grid of range/20 = 1 m: wrong at 0.1 m
        else:
            assert g1 == pytest.approx(v.gamma(0.1), rel=tol)
        np.testing.assert_allclose(z.var(axis=0).mean(), 1.0, atol=0.05)


# ---------------------------------------------------------------------------
# Nested variograms, surface texture and conditional simulation
# ---------------------------------------------------------------------------

def test_nested_variogram_sums_its_structures():
    nv = NestedVariogram([("spherical", 0.2, 1.0), ("exponential", 1.0, 10.0)], nugget=0.05, aniso=2.0, angle=30.0)
    h = np.array([0.0, 0.5, 2.0, 30.0])
    expected = Variogram("spherical", 0.2, 1.0).gamma(h) + Variogram("exponential", 1.0, 10.0).gamma(h) + np.where(h > 0, 0.05, 0)
    np.testing.assert_allclose(nv.gamma(h), expected)
    assert nv.psill == pytest.approx(1.2)
    xy = np.array([[1.0, 2.0], [-3.0, 0.5]])
    np.testing.assert_allclose(nv.transform(xy), Variogram(aniso=2.0, angle=30.0).transform(xy))
    assert variogram_from_dict(nv.to_dict()) == nv
    single = Variogram("gaussian", 0.7, 4.0, 0.1, 3.0, 5.0)
    assert variogram_from_dict(dict(single.to_dict(), from_class="2")) == single
    with pytest.raises(ValueError, match="Unknown variogram model"):
        NestedVariogram([("cubic", 1.0, 1.0)])
    # kriging accepts it and stays exact at the data
    rng = np.random.default_rng(0)
    pts = rng.uniform(0, 10, (60, 2))
    vals = np.sin(pts[:, 0])
    est, var = ordinary_kriging(pts, vals, pts[:5], nv)
    np.testing.assert_allclose(est, vals[:5], atol=1e-8)
    np.testing.assert_allclose(var, 0.0, atol=1e-8)


def test_fit_nested_variogram_sees_the_short_range_roughness():
    rng = np.random.default_rng(3)
    gu, gv = np.meshgrid(np.arange(0, 40, 0.15), np.arange(0, 40, 0.15), indexing="ij")
    xy = np.column_stack([gu.ravel(), gv.ravel()])                         # 71 000 points
    truth = NestedVariogram([("spherical", 0.05, 1.0), ("spherical", 1.0, 15.0)], nugget=0.0)
    z = RandomField(xy, truth, resolution=0.15).sample(rng, 1)[0]
    nested = fit_nested_variogram(xy, z, short_model="spherical", dense_lag=3.0, seed=1)
    (m1, c1, a1), (m2, c2, a2) = nested.structures
    assert (m1, m2) == ("spherical", "spherical")
    # the short structure and the slope at the origin are recovered ...
    for h in (0.15, 0.5, 1.0):
        assert nested.gamma(h) == pytest.approx(truth.gamma(h), rel=0.35)
    assert 0.3 < a1 < 3.0
    # ... which a fit on a thinned subset cannot see: it extrapolates them, far off
    thinned = fit_variogram(xy, z, max_points=2000, seed=1)
    err = lambda m: abs(float(m.gamma(0.15)) - float(truth.gamma(0.15)))
    assert err(thinned) > 2 * err(nested)
    with pytest.raises(ValueError, match="bounded models"):
        fit_nested_variogram(xy[:100], z[:100], "linear")
    flat = fit_nested_variogram(xy[:50], np.ones(50))
    assert flat.gamma(1.0) == pytest.approx(0.0, abs=1e-9)


def test_surface_mask_flags_second_layers_and_strays():
    rng = np.random.default_rng(4)
    uv = rng.uniform(0, 20, (8000, 2))
    w = 0.05 * np.sin(uv[:, 0]) + 0.01 * rng.standard_normal(8000)
    layer = (uv[:, 0] < 5) & (rng.random(8000) < 0.25)                     # a sparse second layer
    stray = rng.random(8000) < 0.005
    w = w + np.where(layer, 0.6, 0.0) + np.where(stray & ~layer, -1.0, 0.0)
    on = surface_mask(uv, w, radius=0.5, tol=0.15)
    assert on[~layer & ~stray].mean() > 0.99
    assert on[layer].mean() < 0.02
    assert not on[stray & ~layer].any()
    assert surface_mask(uv, w, radius=0.5, n_iter=1)[layer].mean() > on[layer].mean()   # later passes help
    assert surface_mask(uv, w)[layer].mean() < 0.01                       # 24 nearest: more robust
    assert surface_mask(np.empty((0, 2)), np.empty(0)).shape == (0,)


def test_random_field_variance_is_the_sill():
    rng = np.random.default_rng(5)
    xy = rng.uniform(0, 10, (300, 2))
    nv = NestedVariogram([("spherical", 0.3, 1.0), ("gaussian", 0.5, 6.0)], nugget=0.2, aniso=2.0)
    z = RandomField(xy, nv, resolution=0.2).sample(rng, 4000)
    assert z.shape == (4000, 300)
    assert z.var(axis=0).mean() == pytest.approx(1.0, abs=0.05)
    i, j = 0, 1
    h = np.linalg.norm(nv.transform(xy[[i]]) - nv.transform(xy[[j]]))
    cov = np.mean(z[:, i] * z[:, j])
    assert cov == pytest.approx(1.0 - nv.gamma(h), abs=0.06)


def test_conditional_simulation_honours_the_data_and_the_kriging_variance():
    rng = np.random.default_rng(6)
    v = NestedVariogram([("spherical", 0.2, 1.0), ("spherical", 1.0, 8.0)])
    data = rng.uniform(0, 20, (150, 2))
    vals = gaussian_field(data, Variogram("spherical", 1.2, 8.0), seed=6)
    new = np.column_stack([rng.uniform(5, 15, 400), rng.uniform(5, 15, 400)])
    new[:3] = data[:3]                                                     # some at the data
    real, est, var = conditional_simulation(data, vals, new, v, 600, rng=rng, resolution=0.1)
    assert real.shape == (600, 400)
    np.testing.assert_allclose(real[:, :3], np.broadcast_to(vals[:3], (600, 3)), atol=1e-6)
    e2, v2 = ordinary_kriging(data, vals, new, v)
    np.testing.assert_allclose(est, e2, atol=1e-10)
    np.testing.assert_allclose(var, v2, atol=1e-10)
    far = var > 0.2
    assert np.abs(real.mean(axis=0) - est)[far].max() < 4 * np.sqrt(var[far].max() / 600)
    assert np.median(real.var(axis=0)[far] / var[far]) == pytest.approx(1.0, abs=0.08)
    empty, *_ = conditional_simulation(data, vals, np.empty((0, 2)), v, 3, rng=rng)
    assert empty.shape == (3, 0)


def test_fill_holes_uv_ignores_a_second_layer_and_records_its_training_points():
    uvw, label, in_hole = plane_with_hole()
    rng = np.random.default_rng(7)
    near = np.hypot(uvw[:, 0] - 30.0, uvw[:, 1] - 10.0) < 9.0
    layer = near & ~in_hole & (uvw[:, 1] > 10.0) & (rng.random(len(uvw)) < 0.3)   # debris above the face
    uvw = uvw.copy()
    uvw[layer, 2] += 0.8
    kept = uvw[~in_hole]
    w_true = lambda p: 0.3 * np.sin(p[:, 0] / 6.0) + 0.02 * (p[:, 1] - 10.0) ** 2 / 10.0

    clean = fill_holes_uv(kept, cell=0.8)
    raw = fill_holes_uv(kept, cell=0.8, surface_tol=None)
    rmse = lambda r: np.sqrt(np.mean((r["uvw"][:, 2] - w_true(r["uvw"])) ** 2))
    assert rmse(clean) < 0.06
    assert rmse(raw) > 2 * rmse(clean)
    h = clean["holes"][0]
    assert 0.03 < h["off_surface"] < 0.3                                   # the layer's share of the ring
    assert not layer[~in_hole][h["train_idx"]].any()                       # no debris among the data
    assert h["n_train"] == len(h["train_idx"])
    assert "structures" in h["w_variogram"]
    single = fill_holes_uv(kept, cell=0.8, dense_variogram=False)["holes"][0]["w_variogram"]
    assert "model" in single
