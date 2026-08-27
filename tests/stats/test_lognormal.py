"""Tests for pyrockfall.stats.Lognormal (mu, sigma of underlying normal) vs analytical values and scipy."""
import numpy as np
import pytest
from scipy import stats as scipy_stats

import pyrockfall.stats as stats


@pytest.mark.parametrize("sigma", [0.0, -1.0])
def test_invalid_sigma_raises(sigma):
    with pytest.raises(ValueError, match="sigma must be > 0."):
        stats.Lognormal(0.0, sigma)


def test_mean_analytical():
    mu, sigma = 0.0, 0.5
    d = stats.Lognormal(mu, sigma)
    assert d.mean() == pytest.approx(np.exp(mu + 0.5 * sigma ** 2))


def test_var_analytical():
    mu, sigma = 0.0, 0.5
    expected = (np.exp(sigma ** 2) - 1.0) * np.exp(2 * mu + sigma ** 2)
    d = stats.Lognormal(mu, sigma)
    assert d.var() == pytest.approx(expected)


def test_median_equals_exp_mu():
    mu = 1.0
    d = stats.Lognormal(mu, 0.3)
    assert d.median() == pytest.approx(np.exp(mu))


def test_pdf_zero_for_nonpositive_x():
    d = stats.Lognormal(0.0, 0.5)
    assert d.pdf(0.0) == 0.0
    assert d.pdf(-1.0) == 0.0


def test_ppf_cdf_roundtrip():
    d = stats.Lognormal(0.0, 0.5)
    q = np.linspace(0.01, 0.99, 25)
    np.testing.assert_allclose(d.cdf(d.ppf(q)), q, rtol=1e-6)


def test_cross_check_against_scipy_lognorm():
    mu, sigma = 0.0, 0.5
    d = stats.Lognormal(mu, sigma)
    ref = scipy_stats.lognorm(s=sigma, scale=np.exp(mu))
    x = np.linspace(0.1, 3.0, 11)
    np.testing.assert_allclose(d.pdf(x), ref.pdf(x))
    np.testing.assert_allclose(d.cdf(x), ref.cdf(x))
    q = np.linspace(0.05, 0.95, 11)
    np.testing.assert_allclose(d.ppf(q), ref.ppf(q))


def test_rvs_positive_and_reproducible():
    d = stats.Lognormal(0.0, 0.5)
    a = d.rvs(size=2000, random_state=1)
    b = d.rvs(size=2000, random_state=1)
    assert np.all(a > 0.0)
    np.testing.assert_array_equal(a, b)


def test_native_and_generic_params():
    mu, sigma = 0.0, 0.5
    d = stats.Lognormal(mu, sigma)
    assert d.native_params() == {"mu": mu, "sigma": sigma}
    g = d.generic_params()
    assert g.id == int(stats.DistributionID.LOGNORMAL)
    assert g.loc == pytest.approx(d.mean())
