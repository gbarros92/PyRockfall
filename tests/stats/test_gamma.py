"""Tests for pyrockfall.stats.Gamma (shape alpha, rate lam) against analytical values and scipy."""
import numpy as np
import pytest
from scipy import stats as scipy_stats

import pyrockfall.stats as stats


@pytest.mark.parametrize("alpha, lam", [(0.0, 1.0), (1.0, 0.0), (-1.0, 2.0), (2.0, -1.0)])
def test_invalid_params_raises(alpha, lam):
    with pytest.raises(ValueError, match=r"alpha and lam \(rate λ\) must be > 0."):
        stats.Gamma(alpha, lam)


def test_mean_analytical():
    d = stats.Gamma(2.0, 3.0)
    assert d.mean() == pytest.approx(2.0 / 3.0)


def test_var_analytical():
    d = stats.Gamma(2.0, 3.0)
    assert d.var() == pytest.approx(2.0 / 9.0)


def test_pdf_zero_for_negative_x():
    d = stats.Gamma(2.0, 3.0)
    assert d.pdf(-1.0) == 0.0


def test_cdf_at_zero_is_zero():
    d = stats.Gamma(2.0, 3.0)
    assert d.cdf(0.0) == pytest.approx(0.0)


def test_ppf_cdf_roundtrip():
    d = stats.Gamma(2.0, 3.0)
    q = np.linspace(0.01, 0.99, 25)
    np.testing.assert_allclose(d.cdf(d.ppf(q)), q, rtol=1e-6)


def test_cross_check_against_scipy_gamma():
    alpha, lam = 2.0, 3.0
    d = stats.Gamma(alpha, lam)
    ref = scipy_stats.gamma(a=alpha, scale=1.0 / lam)
    x = np.linspace(0.01, 3.0, 11)
    np.testing.assert_allclose(d.pdf(x), ref.pdf(x))
    np.testing.assert_allclose(d.cdf(x), ref.cdf(x))
    q = np.linspace(0.05, 0.95, 11)
    np.testing.assert_allclose(d.ppf(q), ref.ppf(q))


def test_rvs_nonnegative_and_reproducible():
    d = stats.Gamma(2.0, 3.0)
    a = d.rvs(size=2000, random_state=1)
    b = d.rvs(size=2000, random_state=1)
    assert np.all(a >= 0.0)
    np.testing.assert_array_equal(a, b)


def test_native_and_generic_params():
    d = stats.Gamma(2.0, 3.0)
    assert d.native_params() == {"alpha": 2.0, "lam": 3.0}
    g = d.generic_params()
    assert g.id == int(stats.DistributionID.GAMMA)
    assert g.loc == pytest.approx(2.0 / 3.0)
    assert g.scale == pytest.approx(np.sqrt(2.0) / 3.0)


def test_gamma_shape_one_matches_exponential():
    lam = 2.0
    gamma_d = stats.Gamma(1.0, lam)
    expo_d = stats.Exponential(lam)
    x = np.linspace(0.01, 3.0, 11)
    np.testing.assert_allclose(gamma_d.pdf(x), expo_d.pdf(x), rtol=1e-8)
    np.testing.assert_allclose(gamma_d.cdf(x), expo_d.cdf(x), rtol=1e-8)
