"""Tests for pyrockfall.stats.Affine: Y = scale * X + translate."""
import numpy as np
import pytest

import pyrockfall.stats as stats


def test_zero_scale_raises():
    with pytest.raises(ValueError, match="Affine scale must be non-zero."):
        stats.Affine(stats.Normal(0.0, 1.0), scale=0.0)


def test_positive_scale_translate_mean_var():
    base = stats.Normal(0.0, 1.0)
    d = stats.Affine(base, scale=2.0, translate=3.0)
    assert d.mean() == pytest.approx(3.0)
    assert d.var() == pytest.approx(4.0 * base.var())


def test_negative_scale_flips_cdf():
    base = stats.Normal(0.0, 1.0)
    d = stats.Affine(base, scale=-1.0, translate=0.0)
    assert d.cdf(0.5) == pytest.approx(1.0 - base.cdf(-0.5))


def test_negative_scale_ppf_matches_identity():
    base = stats.Normal(0.0, 1.0)
    d = stats.Affine(base, scale=-1.0, translate=0.0)
    assert d.ppf(0.3) == pytest.approx(-base.ppf(0.7))


def test_pdf_uses_abs_scale():
    base = stats.Normal(0.0, 1.0)
    d = stats.Affine(base, scale=-2.0, translate=1.0)
    assert d.pdf(1.0) == pytest.approx(base.pdf(0.0) / 2.0)


def test_interval_ordering_preserved_for_negative_scale():
    base = stats.Normal(0.0, 1.0)
    d = stats.Affine(base, scale=-1.0, translate=0.0)
    lo, hi = d.interval(0.95)
    assert lo < hi
    base_lo, base_hi = base.interval(0.95)
    assert lo == pytest.approx(-base_hi)
    assert hi == pytest.approx(-base_lo)


def test_nested_affine_collapses_algebraically():
    base = stats.Normal(0.0, 1.0)
    inner = stats.Affine(base, 2.0, 1.0)
    outer = stats.Affine(inner, 3.0, 5.0)
    assert outer.base is base
    assert outer.scale == pytest.approx(6.0)
    assert outer.translate == pytest.approx(8.0)


def test_distid_preserved_from_base():
    d = stats.Affine(stats.Uniform(0.0, 1.0), scale=2.0, translate=0.0)
    assert d.DistID == stats.DistributionID.UNIFORM


def test_native_params_delegates_to_base_no_scale_translate():
    # Known limitation: native_params() reflects only the base family's
    # parameters, not the affine transform applied on top of it.
    d = stats.Affine(stats.Normal(2.0, 1.0), scale=3.0, translate=4.0)
    assert d.native_params() == {"mu": 2.0, "sigma": 1.0}


def test_generic_params_reflects_transform():
    d = stats.Affine(stats.Uniform(0.0, 1.0), scale=2.0, translate=1.0)
    g = d.generic_params()
    assert g.a == pytest.approx(1.0)
    assert g.b == pytest.approx(3.0)


def test_arithmetic_on_affine_produces_further_collapsed_affine():
    d = (stats.Normal(0.0, 1.0) + 1.0) * 2.0
    assert isinstance(d, stats.Affine)
    assert d.scale == pytest.approx(2.0)
    assert d.translate == pytest.approx(2.0)


def test_rvs_reproducible_with_negative_scale():
    base = stats.Normal(0.0, 1.0)
    d = stats.Affine(base, scale=-1.0, translate=0.0)
    a = d.rvs(size=100, random_state=1)
    b = d.rvs(size=100, random_state=1)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose(a, -1.0 * base.rvs(size=100, random_state=1))
