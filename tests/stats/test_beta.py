"""Tests for pyrockfall.stats.Beta against analytical values and scipy.

Note: the Beta docstring in _beta.py claims support "[0, +inf)" but the
implementation wraps scipy.stats.beta(a=alpha, b=beta), whose actual support
is [0, 1]. test_support_is_zero_one below characterizes the real (correct)
behavior; the docstring inaccuracy is reported separately, not asserted here.
"""
import numpy as np
import pytest
from scipy import stats as scipy_stats

import pyrockfall.stats as stats


@pytest.mark.parametrize("alpha, beta", [(0.0, 1.0), (1.0, 0.0), (-1.0, 1.0), (1.0, -2.0)])
def test_invalid_params_raises(alpha, beta):
    with pytest.raises(ValueError, match="alpha and beta must be > 0."):
        stats.Beta(alpha, beta)


def test_symmetric_beta_mean_is_half():
    d = stats.Beta(2.0, 2.0)
    assert d.mean() == pytest.approx(0.5)


def test_mean_analytical_asymmetric():
    d = stats.Beta(2.0, 3.0)
    assert d.mean() == pytest.approx(2.0 / 5.0)


def test_var_analytical():
    alpha, beta = 2.0, 3.0
    expected = alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1))
    d = stats.Beta(alpha, beta)
    assert d.var() == pytest.approx(expected)


def test_support_is_zero_one():
    d = stats.Beta(2.0, 3.0)
    assert d.cdf(0.0) == pytest.approx(0.0)
    assert d.cdf(1.0) == pytest.approx(1.0)


def test_pdf_zero_outside_zero_one():
    d = stats.Beta(2.0, 3.0)
    assert d.pdf(-0.5) == 0.0
    assert d.pdf(1.5) == 0.0


def test_ppf_cdf_roundtrip():
    d = stats.Beta(2.0, 3.0)
    q = np.linspace(0.01, 0.99, 25)
    np.testing.assert_allclose(d.cdf(d.ppf(q)), q, rtol=1e-6)


def test_cross_check_against_scipy_beta():
    alpha, beta = 2.0, 3.0
    d = stats.Beta(alpha, beta)
    ref = scipy_stats.beta(a=alpha, b=beta)
    x = np.linspace(0.05, 0.95, 11)
    np.testing.assert_allclose(d.pdf(x), ref.pdf(x))
    np.testing.assert_allclose(d.cdf(x), ref.cdf(x))
    q = np.linspace(0.05, 0.95, 11)
    np.testing.assert_allclose(d.ppf(q), ref.ppf(q))


def test_rvs_within_zero_one_and_reproducible():
    d = stats.Beta(2.0, 3.0)
    a = d.rvs(size=2000, random_state=1)
    b = d.rvs(size=2000, random_state=1)
    assert np.all((a >= 0.0) & (a <= 1.0))
    np.testing.assert_array_equal(a, b)


def test_native_and_generic_params():
    d = stats.Beta(2.0, 3.0)
    assert d.native_params() == {"alpha": 2.0, "beta": 3.0}
    g = d.generic_params()
    assert g.id == int(stats.DistributionID.BETA)
    assert g.rel_min == pytest.approx(g.loc)
    assert g.rel_max == pytest.approx(1.0 - g.loc)
