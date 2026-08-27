"""Characterization tests for pyrockfall.stats.Deterministic (no constructor validation)."""
import math

import numpy as np
import pytest

import pyrockfall.stats as stats


def test_mean_var_median_equal_value():
    d = stats.Deterministic(4.2)
    assert d.mean() == 4.2
    assert d.median() == 4.2
    assert d.var() == 0.0


def test_rvs_ignores_random_state_returns_constant_array():
    d = stats.Deterministic(4.2)
    a = d.rvs(size=5, random_state=999)
    b = d.rvs(size=5, random_state=12345)
    np.testing.assert_array_equal(a, np.full(5, 4.2))
    np.testing.assert_array_equal(a, b)


def test_pdf_is_always_zero():
    d = stats.Deterministic(4.2)
    assert d.pdf(4.2) == 0.0
    x = np.array([0.0, 4.2, 10.0])
    np.testing.assert_array_equal(d.pdf(x), np.zeros(3))


def test_cdf_is_step_function():
    d = stats.Deterministic(4.2)
    assert d.cdf(4.1) == 0.0
    assert d.cdf(4.2) == 1.0
    assert d.cdf(4.3) == 1.0


@pytest.mark.parametrize("q", [-1.0, 0.0, 0.5, 1.0, 2.0])
def test_ppf_returns_constant_ignoring_q(q):
    d = stats.Deterministic(4.2)
    assert d.ppf(np.array([q]))[0] == pytest.approx(4.2)


def test_expect_inside_and_outside_bounds():
    d = stats.Deterministic(4.2)
    assert d.expect(lambda x: x ** 2) == pytest.approx(4.2 ** 2)
    assert d.expect(lambda x: x, lb=5.0, ub=10.0) == 0.0
    assert d.expect(lambda x: x, lb=4.2, ub=4.2) == pytest.approx(4.2)


@pytest.mark.parametrize("confidence", [0.0, -0.1, 1.1])
def test_interval_rejects_invalid_confidence(confidence):
    d = stats.Deterministic(4.2)
    with pytest.raises(ValueError, match="confidence must be in \\(0, 1\\]."):
        d.interval(confidence)


def test_interval_valid_confidence_returns_point():
    d = stats.Deterministic(4.2)
    assert d.interval(0.9) == (4.2, 4.2)


def test_accepts_nan_and_inf_without_validation():
    d_nan = stats.Deterministic(float("nan"))
    assert math.isnan(d_nan.mean())
    d_inf = stats.Deterministic(float("inf"))
    assert d_inf.mean() == float("inf")


def test_native_and_generic_params():
    d = stats.Deterministic(4.2)
    assert d.native_params() == {"value": 4.2}
    g = d.generic_params()
    assert g.rel_min == 0.0
    assert g.rel_max == 0.0


def test_deterministic_registered_as_none_id():
    assert stats.Deterministic.DistID == stats.DistributionID.NONE
