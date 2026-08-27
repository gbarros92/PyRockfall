"""Tests for pyrockfall.stats.Triangular against analytical values and scipy."""
import numpy as np
import pytest
from scipy import stats as scipy_stats

import pyrockfall.stats as stats


@pytest.mark.parametrize(
    "lower, mode, upper",
    [(1.0, 1.0, 3.0), (1.0, 3.0, 3.0), (3.0, 2.0, 1.0), (1.0, 4.0, 3.0)],
)
def test_invalid_ordering_raises(lower, mode, upper):
    with pytest.raises(ValueError, match="Triangular requires lower < mode < upper."):
        stats.Triangular(lower, mode, upper)


def test_mean_analytical():
    d = stats.Triangular(0.0, 1.0, 3.0)
    assert d.mean() == pytest.approx((0.0 + 1.0 + 3.0) / 3.0)


def test_var_analytical():
    a, c, b = 0.0, 1.0, 3.0
    expected = (a ** 2 + b ** 2 + c ** 2 - a * b - a * c - b * c) / 18.0
    d = stats.Triangular(a, c, b)
    assert d.var() == pytest.approx(expected)


def test_pdf_peak_at_mode():
    d = stats.Triangular(0.0, 1.0, 3.0)
    x = np.linspace(0.0, 3.0, 301)
    peak_x = x[np.argmax(d.pdf(x))]
    assert peak_x == pytest.approx(1.0, abs=0.02)


def test_pdf_zero_outside_support():
    d = stats.Triangular(0.0, 1.0, 3.0)
    assert d.pdf(-1.0) == 0.0
    assert d.pdf(4.0) == 0.0


def test_cdf_boundary_and_mode():
    lower, mode, upper = 0.0, 1.0, 3.0
    d = stats.Triangular(lower, mode, upper)
    assert d.cdf(lower) == pytest.approx(0.0)
    assert d.cdf(upper) == pytest.approx(1.0)
    assert d.cdf(mode) == pytest.approx((mode - lower) / (upper - lower))


def test_ppf_cdf_roundtrip():
    d = stats.Triangular(0.0, 1.0, 3.0)
    q = np.linspace(0.01, 0.99, 25)
    np.testing.assert_allclose(d.cdf(d.ppf(q)), q, rtol=1e-6)


def test_cross_check_against_scipy_triang():
    lower, mode, upper = 0.0, 1.0, 3.0
    d = stats.Triangular(lower, mode, upper)
    c = (mode - lower) / (upper - lower)
    ref = scipy_stats.triang(c=c, loc=lower, scale=upper - lower)
    x = np.linspace(lower, upper, 11)
    np.testing.assert_allclose(d.pdf(x), ref.pdf(x))
    np.testing.assert_allclose(d.cdf(x), ref.cdf(x))
    q = np.linspace(0.05, 0.95, 11)
    np.testing.assert_allclose(d.ppf(q), ref.ppf(q))


def test_rvs_within_support_and_reproducible():
    d = stats.Triangular(0.0, 1.0, 3.0)
    a = d.rvs(size=2000, random_state=1)
    b = d.rvs(size=2000, random_state=1)
    assert np.all((a >= 0.0) & (a <= 3.0))
    np.testing.assert_array_equal(a, b)


def test_native_and_generic_params():
    d = stats.Triangular(0.0, 1.0, 3.0)
    assert d.native_params() == {"lower": 0.0, "mode": 1.0, "upper": 3.0}
    g = d.generic_params()
    assert g.id == int(stats.DistributionID.TRIANGULAR)
    assert g.a == pytest.approx(0.0)
    assert g.b == pytest.approx(3.0)
