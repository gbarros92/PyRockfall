"""Shared Distribution ABC behavior: sf/isf, std, arithmetic overloads, registry."""
import numpy as np
import pytest

import pyrockfall.stats as stats
from pyrockfall.stats import (
    Affine,
    Deterministic,
    Distribution,
    DistributionID,
    Normal,
    Uniform,
    Triangular,
    Beta,
    Exponential,
    Lognormal,
    Gamma,
    registerDistribution,
)

REGISTERED_CLASSES = [Deterministic, Normal, Uniform, Triangular, Beta, Exponential, Lognormal, Gamma]

EXPECTED_DIST_IDS = {
    Deterministic: DistributionID.NONE,
    Normal: DistributionID.NORMAL,
    Uniform: DistributionID.UNIFORM,
    Triangular: DistributionID.TRIANGULAR,
    Beta: DistributionID.BETA,
    Exponential: DistributionID.EXPONENTIAL,
    Lognormal: DistributionID.LOGNORMAL,
    Gamma: DistributionID.GAMMA,
}


def test_sf_is_one_minus_cdf():
    d = Normal(0.0, 1.0)
    x = np.array([-1.0, 0.0, 0.5, 2.0])
    np.testing.assert_allclose(d.sf(x), 1.0 - d.cdf(x))


def test_isf_is_ppf_of_complement():
    d = Normal(0.0, 1.0)
    q = np.array([0.1, 0.25, 0.5, 0.75])
    np.testing.assert_allclose(d.isf(q), d.ppf(1.0 - q))


def test_std_is_sqrt_var():
    d = Normal(2.0, 3.0)
    assert d.std() == pytest.approx(np.sqrt(d.var()))
    assert d.std() == pytest.approx(3.0)


def test_add_scalar_returns_affine_with_translate():
    result = Normal(0.0, 1.0) + 5.0
    assert isinstance(result, Affine)
    assert result.mean() == pytest.approx(5.0)


def test_radd_scalar_symmetric():
    a = Normal(0.0, 1.0) + 5.0
    b = 5.0 + Normal(0.0, 1.0)
    assert a.mean() == pytest.approx(b.mean())
    assert a.var() == pytest.approx(b.var())


def test_sub_scalar():
    result = Normal(0.0, 1.0) - 2.0
    assert isinstance(result, Affine)
    assert result.mean() == pytest.approx(-2.0)


def test_rsub_scalar_negates():
    result = 5.0 - Normal(0.0, 1.0)
    assert isinstance(result, Affine)
    assert result.mean() == pytest.approx(5.0)
    assert result.var() == pytest.approx(Normal(0.0, 1.0).var())


def test_mul_scalar_positive():
    base = Normal(0.0, 1.0)
    result = base * 3.0
    assert isinstance(result, Affine)
    assert result.var() == pytest.approx(9.0 * base.var())


def test_mul_by_zero_returns_deterministic_zero():
    result = Normal(2.0, 1.0) * 0.0
    assert isinstance(result, Deterministic)
    assert result.value == 0.0


def test_rmul_symmetric():
    a = Normal(0.0, 1.0) * 3.0
    b = 3.0 * Normal(0.0, 1.0)
    assert a.mean() == pytest.approx(b.mean())
    assert a.var() == pytest.approx(b.var())


def test_truediv_scalar():
    result = Normal(0.0, 2.0) / 4.0
    assert isinstance(result, Affine)
    assert result.var() == pytest.approx((0.25 ** 2) * 4.0)


def test_truediv_by_zero_raises():
    with pytest.raises(ValueError, match="Division by zero."):
        Normal(0.0, 1.0) / 0.0


def test_neg_returns_affine_scale_negative_one():
    result = -Normal(2.0, 1.0)
    assert isinstance(result, Affine)
    assert result.mean() == pytest.approx(-2.0)


def test_pos_is_identity():
    d = Normal(0.0, 1.0)
    assert (+d) is d


def test_arithmetic_with_non_real_returns_typeerror():
    with pytest.raises(TypeError):
        Normal(0.0, 1.0) + "x"


def test_repr_contains_class_name_and_native_params():
    d = Normal(1.0, 2.0)
    r = repr(d)
    assert "Normal" in r
    assert "mu" in r
    assert "sigma" in r
    assert str(d) == repr(d)


@pytest.mark.parametrize("cls", REGISTERED_CLASSES)
def test_distribution_id_registered(cls):
    # registerDistribution stores DistID as a plain int (int(id_)), not the
    # DistributionID enum member itself -- compare by int value.
    assert int(cls.DistID) == int(EXPECTED_DIST_IDS[cls])
    assert int(cls.DistID) != int(DistributionID.NOTDEFINED)


def test_distribution_ids_are_unique():
    ids = [int(cls.DistID) for cls in REGISTERED_CLASSES]
    assert len(ids) == len(set(ids))


def test_generic_params_id_matches_distid(any_continuous_dist):
    d = any_continuous_dist
    assert d.generic_params().id == int(d.DistID)


def test_mean_matches_expect_identity(any_continuous_dist):
    d = any_continuous_dist
    assert d.mean() == pytest.approx(d.expect(lambda x: x), rel=1e-4, abs=1e-6)


def test_registerDistribution_rejects_duplicate_id():
    class Dummy(Distribution):
        def rvs(self, size=1, random_state=None):
            return np.zeros(size)

        def pdf(self, x):
            return np.zeros_like(np.asarray(x, dtype=float))

        def cdf(self, x):
            return np.zeros_like(np.asarray(x, dtype=float))

        def ppf(self, q):
            return np.zeros_like(np.asarray(q, dtype=float))

        def expect(self, func, lb=-np.inf, ub=np.inf):
            return 0.0

        def median(self):
            return 0.0

        def mean(self):
            return 0.0

        def var(self):
            return 0.0

        def interval(self, confidence=1.0):
            return (0.0, 0.0)

        def native_params(self):
            return {}

        def generic_params(self):
            return None

    with pytest.raises(ValueError, match="already registered"):
        registerDistribution(DistributionID.NORMAL)(Dummy)


def test_registerDistribution_rejects_abstract_class():
    class AbstractDummy(Distribution):
        pass

    with pytest.raises(TypeError, match="abstract"):
        registerDistribution(DistributionID.UNIFORM)(AbstractDummy)


def test_registerDistribution_rejects_notdefined_id():
    class Dummy2(Distribution):
        def rvs(self, size=1, random_state=None):
            return np.zeros(size)

        def pdf(self, x):
            return np.zeros_like(np.asarray(x, dtype=float))

        def cdf(self, x):
            return np.zeros_like(np.asarray(x, dtype=float))

        def ppf(self, q):
            return np.zeros_like(np.asarray(q, dtype=float))

        def expect(self, func, lb=-np.inf, ub=np.inf):
            return 0.0

        def median(self):
            return 0.0

        def mean(self):
            return 0.0

        def var(self):
            return 0.0

        def interval(self, confidence=1.0):
            return (0.0, 0.0)

        def native_params(self):
            return {}

        def generic_params(self):
            return None

    with pytest.raises(ValueError, match="NOTDEFINED"):
        registerDistribution(DistributionID.NOTDEFINED)(Dummy2)
