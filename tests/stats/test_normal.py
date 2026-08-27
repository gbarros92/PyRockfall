"""Tests for pyrockfall.stats.Normal against analytical values and scipy."""
import numpy as np
import pytest
from scipy import stats as scipy_stats

import pyrockfall.stats as stats


@pytest.mark.parametrize("sigma", [0.0, -1.0, -0.001])
def test_invalid_sigma_raises(sigma):
    with pytest.raises(ValueError, match="sigma must be > 0."):
        stats.Normal(0.0, sigma)


def test_standard_normal_cdf_at_mean_is_half():
    d = stats.Normal(0.0, 1.0)
    assert d.cdf(0.0) == pytest.approx(0.5)


def test_standard_normal_pdf_at_zero():
    d = stats.Normal(0.0, 1.0)
    assert d.pdf(0.0) == pytest.approx(1.0 / np.sqrt(2 * np.pi))


def test_mean_var_std_match_constructor_params():
    d = stats.Normal(3.0, 2.0)
    assert d.mean() == pytest.approx(3.0)
    assert d.var() == pytest.approx(4.0)
    assert d.std() == pytest.approx(2.0)


def test_median_equals_mu():
    d = stats.Normal(5.0, 2.0)
    assert d.median() == pytest.approx(5.0)


def test_cdf_ppf_scalar_and_array_shapes():
    d = stats.Normal(0.0, 1.0)
    x = np.array([-1.0, 0.0, 1.0, 2.0])
    assert np.shape(d.cdf(1.0)) == ()
    assert d.cdf(x).shape == x.shape
    assert d.ppf(np.array([0.1, 0.5, 0.9])).shape == (3,)


def test_ppf_cdf_roundtrip():
    d = stats.Normal(1.0, 2.0)
    q = np.linspace(0.01, 0.99, 25)
    np.testing.assert_allclose(d.cdf(d.ppf(q)), q, rtol=1e-6)


def test_cross_check_against_scipy_norm():
    mu, sigma = 1.5, 2.5
    d = stats.Normal(mu, sigma)
    ref = scipy_stats.norm(loc=mu, scale=sigma)
    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 11)
    np.testing.assert_allclose(d.pdf(x), ref.pdf(x))
    np.testing.assert_allclose(d.cdf(x), ref.cdf(x))
    q = np.linspace(0.05, 0.95, 11)
    np.testing.assert_allclose(d.ppf(q), ref.ppf(q))


def test_rvs_shape_and_reproducibility():
    d = stats.Normal(0.0, 1.0)
    a = d.rvs(size=1000, random_state=123)
    b = d.rvs(size=1000, random_state=123)
    c = d.rvs(size=1000, random_state=456)
    assert a.shape == (1000,)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_rvs_accepts_generator_instance():
    d = stats.Normal(0.0, 1.0)
    samples = d.rvs(size=10, random_state=np.random.default_rng(0))
    assert samples.shape == (10,)


def test_interval_matches_scipy():
    d = stats.Normal(0.0, 1.0)
    lo, hi = d.interval(0.95)
    ref_lo, ref_hi = scipy_stats.norm(loc=0.0, scale=1.0).interval(0.95)
    assert lo == pytest.approx(ref_lo)
    assert hi == pytest.approx(ref_hi)
    assert lo < 0.0 < hi


def test_native_params_and_generic_params():
    d = stats.Normal(1.0, 2.0)
    assert d.native_params() == {"mu": 1.0, "sigma": 2.0}
    g = d.generic_params()
    assert g.id == int(stats.DistributionID.NORMAL)
    assert g.loc == pytest.approx(1.0)
    assert g.std == pytest.approx(2.0)
    assert g.rel_min == np.inf
    assert g.rel_max == np.inf


def test_expect_identity_function_equals_mean():
    d = stats.Normal(2.0, 1.0)
    assert d.expect(lambda x: x) == pytest.approx(2.0)
