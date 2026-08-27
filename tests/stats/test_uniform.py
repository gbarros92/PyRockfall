"""Tests for pyrockfall.stats.Uniform against analytical values and scipy."""
import numpy as np
import pytest
from scipy import stats as scipy_stats

import pyrockfall.stats as stats


@pytest.mark.parametrize("lower, upper", [(5.0, 5.0), (5.0, 2.0)])
def test_invalid_bounds_raises(lower, upper):
    with pytest.raises(ValueError, match="Uniform requires lower < upper."):
        stats.Uniform(lower, upper)


def test_mean_var_analytical():
    d = stats.Uniform(2.0, 5.0)
    assert d.mean() == pytest.approx(3.5)
    assert d.var() == pytest.approx((5.0 - 2.0) ** 2 / 12.0)


def test_pdf_constant_inside_support():
    d = stats.Uniform(2.0, 5.0)
    assert d.pdf(3.5) == pytest.approx(1.0 / 3.0)
    x = np.array([2.5, 3.5, 4.5])
    np.testing.assert_allclose(d.pdf(x), np.full(3, 1.0 / 3.0))


def test_pdf_zero_outside_support():
    d = stats.Uniform(2.0, 5.0)
    assert d.pdf(1.0) == 0.0
    assert d.pdf(6.0) == 0.0


def test_cdf_boundary_and_midpoint():
    d = stats.Uniform(2.0, 5.0)
    assert d.cdf(2.0) == pytest.approx(0.0)
    assert d.cdf(5.0) == pytest.approx(1.0)
    assert d.cdf(3.5) == pytest.approx(0.5)


def test_ppf_linear_interpolation():
    d = stats.Uniform(2.0, 5.0)
    assert d.ppf(0.0) == pytest.approx(2.0)
    assert d.ppf(1.0) == pytest.approx(5.0)
    assert d.ppf(0.5) == pytest.approx(3.5)


def test_ppf_cdf_roundtrip():
    d = stats.Uniform(2.0, 5.0)
    q = np.linspace(0.01, 0.99, 25)
    np.testing.assert_allclose(d.cdf(d.ppf(q)), q, rtol=1e-6)


def test_cross_check_against_scipy_uniform():
    lower, upper = 2.0, 5.0
    d = stats.Uniform(lower, upper)
    ref = scipy_stats.uniform(loc=lower, scale=upper - lower)
    x = np.linspace(lower, upper, 11)
    np.testing.assert_allclose(d.pdf(x), ref.pdf(x))
    np.testing.assert_allclose(d.cdf(x), ref.cdf(x))
    q = np.linspace(0.05, 0.95, 11)
    np.testing.assert_allclose(d.ppf(q), ref.ppf(q))


def test_rvs_within_support_and_reproducible():
    d = stats.Uniform(2.0, 5.0)
    a = d.rvs(size=2000, random_state=1)
    b = d.rvs(size=2000, random_state=1)
    assert np.all((a >= 2.0) & (a <= 5.0))
    np.testing.assert_array_equal(a, b)


def test_native_and_generic_params():
    d = stats.Uniform(2.0, 5.0)
    assert d.native_params() == {"lower": 2.0, "upper": 5.0}
    g = d.generic_params()
    assert g.id == int(stats.DistributionID.UNIFORM)
    assert g.a == pytest.approx(2.0)
    assert g.b == pytest.approx(5.0)
