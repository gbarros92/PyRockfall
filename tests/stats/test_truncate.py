"""Tests for pyrockfall.stats.Truncate: base distribution restricted to [lower, upper]."""
import numpy as np
import pytest

import pyrockfall.stats as stats


def test_no_bounds_raises():
    with pytest.raises(ValueError, match="Truncate needs at least one bound."):
        stats.Truncate(stats.Normal(0.0, 1.0))


def test_lower_ge_upper_raises():
    with pytest.raises(ValueError, match="Truncate requires lower < upper when both bounds are provided."):
        stats.Truncate(stats.Normal(0.0, 1.0), lower=2.0, upper=1.0)


def test_cdf_clamped_outside_bounds():
    d = stats.Truncate(stats.Normal(0.0, 1.0), lower=-1.0, upper=1.0)
    assert d.cdf(-2.0) == 0.0
    assert d.cdf(2.0) == 1.0


def test_cdf_normalized_at_endpoints():
    d = stats.Truncate(stats.Normal(0.0, 1.0), lower=-1.0, upper=1.0)
    assert d.cdf(1.0) == pytest.approx(1.0)
    assert d.cdf(-1.0) == pytest.approx(0.0)


def test_pdf_zero_outside_bounds():
    d = stats.Truncate(stats.Normal(0.0, 1.0), lower=-1.0, upper=1.0)
    assert d.pdf(-2.0) == 0.0
    assert d.pdf(2.0) == 0.0


def test_mean_between_bounds_and_symmetric():
    d = stats.Truncate(stats.Normal(0.0, 1.0), lower=-1.0, upper=1.0)
    mean = d.mean()
    assert -1.0 < mean < 1.0
    assert mean == pytest.approx(0.0, abs=1e-6)


def test_lower_only_bound_matches_shifted_exponential():
    lam = 1.0
    d = stats.Truncate(stats.Exponential(lam), lower=1.0, upper=None)
    x = np.linspace(1.0, 5.0, 11)
    np.testing.assert_allclose(d.cdf(x), 1.0 - np.exp(-(x - 1.0)), rtol=1e-6)


def test_upper_only_bound_reduces_to_smaller_uniform():
    d = stats.Truncate(stats.Uniform(0.0, 10.0), lower=None, upper=5.0)
    assert d.mean() == pytest.approx(2.5)
    assert d.var() == pytest.approx(25.0 / 12.0)


def test_ppf_cdf_roundtrip_within_truncated_support():
    d = stats.Truncate(stats.Normal(0.0, 1.0), lower=-2.0, upper=2.0)
    q = np.linspace(0.01, 0.99, 15)
    np.testing.assert_allclose(d.cdf(d.ppf(q)), q, rtol=1e-6)


def test_rvs_within_bounds_and_reproducible():
    d = stats.Truncate(stats.Normal(0.0, 1.0), lower=-0.5, upper=0.5)
    a = d.rvs(size=500, random_state=1)
    b = d.rvs(size=500, random_state=1)
    assert np.all((a >= -0.5) & (a <= 0.5))
    np.testing.assert_array_equal(a, b)


def test_zero_mass_range_falls_back_to_uniform_with_warning(capsys):
    d = stats.Truncate(stats.Normal(100.0, 1.0), lower=-10.0, upper=-5.0)
    assert isinstance(d.base, stats.Uniform)
    assert d.base.lower == -10.0
    assert d.base.upper == -5.0
    captured = capsys.readouterr()
    assert "Warning: Truncate has zero probability mass" in captured.out


def test_distid_preserved_from_base():
    d = stats.Truncate(stats.Beta(2.0, 3.0), lower=0.1, upper=0.9)
    assert d.DistID == stats.DistributionID.BETA


def test_native_params_reflects_base_family():
    d = stats.Truncate(stats.Beta(2.0, 3.0), lower=0.1, upper=0.9)
    assert d.native_params() == {"alpha": 2.0, "beta": 3.0}


@pytest.mark.parametrize("confidence", [0.0, -0.1, 1.1])
def test_interval_rejects_invalid_confidence(confidence):
    d = stats.Truncate(stats.Normal(0.0, 1.0), lower=-1.0, upper=1.0)
    with pytest.raises(ValueError, match="confidence must be in \\(0, 1\\]."):
        d.interval(confidence)


def test_expect_zero_when_requested_range_outside_truncated_range():
    d = stats.Truncate(stats.Normal(0.0, 1.0), lower=-1.0, upper=1.0)
    assert d.expect(lambda x: x, lb=5.0, ub=10.0) == 0.0
