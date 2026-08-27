"""Tests for pyrockfall.stats.Exponential (rate lam) against analytical values and scipy."""
import numpy as np
import pytest
from scipy import stats as scipy_stats

import pyrockfall.stats as stats


@pytest.mark.parametrize("lam", [0.0, -1.0, -0.5])
def test_invalid_lam_raises(lam):
    with pytest.raises(ValueError, match=r"lam \(rate λ\) must be > 0."):
        stats.Exponential(lam)


def test_mean_analytical():
    d = stats.Exponential(2.0)
    assert d.mean() == pytest.approx(0.5)


def test_var_analytical():
    d = stats.Exponential(2.0)
    assert d.var() == pytest.approx(0.25)


def test_pdf_at_zero_analytical():
    d = stats.Exponential(2.0)
    assert d.pdf(0.0) == pytest.approx(2.0)


def test_cdf_analytical_at_x():
    lam = 2.0
    d = stats.Exponential(lam)
    assert d.cdf(1.0) == pytest.approx(1.0 - np.exp(-lam * 1.0))


def test_pdf_zero_for_negative_x():
    d = stats.Exponential(2.0)
    assert d.pdf(-1.0) == 0.0


def test_ppf_cdf_roundtrip():
    d = stats.Exponential(2.0)
    q = np.linspace(0.01, 0.99, 25)
    np.testing.assert_allclose(d.cdf(d.ppf(q)), q, rtol=1e-6)


def test_cross_check_against_scipy_expon():
    lam = 2.0
    d = stats.Exponential(lam)
    ref = scipy_stats.expon(loc=0.0, scale=1.0 / lam)
    x = np.linspace(0.0, 3.0, 11)
    np.testing.assert_allclose(d.pdf(x), ref.pdf(x))
    np.testing.assert_allclose(d.cdf(x), ref.cdf(x))
    q = np.linspace(0.05, 0.95, 11)
    np.testing.assert_allclose(d.ppf(q), ref.ppf(q))


def test_rvs_nonnegative_and_reproducible():
    d = stats.Exponential(2.0)
    a = d.rvs(size=2000, random_state=1)
    b = d.rvs(size=2000, random_state=1)
    assert np.all(a >= 0.0)
    np.testing.assert_array_equal(a, b)


def test_native_and_generic_params():
    d = stats.Exponential(2.0)
    assert d.native_params() == {"lam": 2.0}
    g = d.generic_params()
    assert g.id == int(stats.DistributionID.EXPONENTIAL)
    assert g.rel_min == pytest.approx(0.5)
    assert g.rel_max == np.inf
