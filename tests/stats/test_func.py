"""Tests for asDistribution, makeDistribution, asDistributionVector, and the
public API surface of pyrockfall.stats.
"""
import re

import numpy as np
import pytest

import pyrockfall.stats as stats


PUBLIC_API_NAMES = [
    "Distribution",
    "DistributionID",
    "DistributionParameters",
    "registerDistribution",
    "Affine",
    "Truncate",
    "Deterministic",
    "Normal",
    "Uniform",
    "Triangular",
    "Beta",
    "Exponential",
    "Lognormal",
    "Gamma",
    "DistributionVector",
    "SupportsToDistribution",
    "DistributionLike",
    "asDistribution",
    "makeDistribution",
    "DistributionVectorLike",
    "asDistributionVector",
]


@pytest.mark.parametrize("name", PUBLIC_API_NAMES)
def test_public_api_names_exist(name):
    assert hasattr(stats, name)


# ---------------------------------------------------------------------------
# asDistribution
# ---------------------------------------------------------------------------

def test_asdistribution_passthrough_existing_distribution():
    d = stats.Normal(0.0, 1.0)
    assert stats.asDistribution(d) is d


def test_asdistribution_python_float_and_int_to_deterministic():
    assert isinstance(stats.asDistribution(3.5), stats.Deterministic)
    assert stats.asDistribution(3.5).value == 3.5
    assert isinstance(stats.asDistribution(3), stats.Deterministic)
    assert stats.asDistribution(3).value == 3.0


def test_asdistribution_numpy_scalar_types():
    d = stats.asDistribution(np.float64(2.5))
    assert isinstance(d, stats.Deterministic)
    assert d.value == 2.5
    d2 = stats.asDistribution(np.int32(4))
    assert isinstance(d2, stats.Deterministic)
    assert d2.value == 4.0


def test_asdistribution_bool_is_coerced_despite_exclude_bool_comment():
    # Characterization test: the source has a comment "(exclude bool)" on the
    # Real branch, but bool is a subclass of int in Python, so it still gets
    # coerced via the later `isinstance(value, (float, int))` fallback branch.
    d = stats.asDistribution(True)
    assert isinstance(d, stats.Deterministic)
    assert d.value == 1.0


def test_asdistribution_protocol_object_calls_to_distribution():
    class Wrapper:
        def to_distribution(self):
            return stats.Normal(1.0, 1.0)

    d = stats.asDistribution(Wrapper())
    assert isinstance(d, stats.Normal)
    assert d.mu == 1.0


def test_asdistribution_unsupported_type_raises_typeerror():
    with pytest.raises(TypeError, match="Cannot convert type"):
        stats.asDistribution("not a number")


# ---------------------------------------------------------------------------
# makeDistribution
# ---------------------------------------------------------------------------

def test_makedistribution_none_id_returns_deterministic():
    p = stats.DistributionParameters.absolute(
        stats.DistributionID.NONE, loc=4.2, scale=0.0, abs_min=4.2, abs_max=4.2
    )
    result = stats.makeDistribution(p)
    assert isinstance(result, stats.Deterministic)
    assert result.value == 4.2


def test_makedistribution_normal_scale_positive():
    p = stats.DistributionParameters.relative(
        stats.DistributionID.NORMAL, loc=2.0, scale=1.5, rel_min=np.inf, rel_max=np.inf
    )
    result = stats.makeDistribution(p)
    assert isinstance(result, stats.Normal)
    assert result.mu == pytest.approx(2.0)
    assert result.sigma == pytest.approx(1.5)


def test_makedistribution_normal_nonpositive_scale_returns_deterministic():
    p = stats.DistributionParameters.relative(
        stats.DistributionID.NORMAL, loc=2.0, scale=0.0, rel_min=np.inf, rel_max=np.inf
    )
    result = stats.makeDistribution(p)
    assert isinstance(result, stats.Deterministic)
    assert result.value == 2.0


def test_makedistribution_normal_with_finite_bounds_wraps_in_truncate():
    p = stats.DistributionParameters.absolute(
        stats.DistributionID.NORMAL, loc=0.0, scale=1.0, abs_min=-1.0, abs_max=1.0
    )
    result = stats.makeDistribution(p)
    assert isinstance(result, stats.Truncate)
    assert isinstance(result.base, stats.Normal)


def test_makedistribution_uniform_requires_finite_bounds():
    p = stats.DistributionParameters.relative(
        stats.DistributionID.UNIFORM, loc=3.5, scale=0.0, rel_min=np.inf, rel_max=np.inf
    )
    with pytest.raises(ValueError, match=re.escape("Uniform requires finite [a, b].")):
        stats.makeDistribution(p)


def test_makedistribution_uniform_requires_a_lt_b():
    p = stats.DistributionParameters.absolute(
        stats.DistributionID.UNIFORM, loc=3.5, scale=0.0, abs_min=5.0, abs_max=2.0
    )
    with pytest.raises(ValueError, match="Uniform needs a < b."):
        stats.makeDistribution(p)


def test_makedistribution_uniform_valid():
    p = stats.DistributionParameters.absolute(
        stats.DistributionID.UNIFORM, loc=3.5, scale=0.0, abs_min=2.0, abs_max=5.0
    )
    result = stats.makeDistribution(p)
    assert isinstance(result, stats.Uniform)
    assert result.native_params() == {"lower": 2.0, "upper": 5.0}


def test_makedistribution_triangular_requires_finite_bounds():
    p = stats.DistributionParameters.relative(
        stats.DistributionID.TRIANGULAR, loc=1.0, scale=0.0, rel_min=np.inf, rel_max=np.inf
    )
    with pytest.raises(ValueError, match=re.escape("Triangular requires finite [a, b].")):
        stats.makeDistribution(p)


def test_makedistribution_triangular_requires_mode_between_bounds():
    p = stats.DistributionParameters.absolute(
        stats.DistributionID.TRIANGULAR, loc=5.0, scale=0.0, abs_min=0.0, abs_max=3.0
    )
    with pytest.raises(ValueError, match=re.escape("Triangular needs a < mode(loc) < b.")):
        stats.makeDistribution(p)


def test_makedistribution_triangular_valid_roundtrip():
    original = stats.Triangular(0.0, 1.0, 3.0)
    p = original.generic_params()
    result = stats.makeDistribution(p)
    assert isinstance(result, stats.Triangular)
    assert result.native_params() == original.native_params()


def test_makedistribution_beta_requires_finite_interval():
    p = stats.DistributionParameters.relative(
        stats.DistributionID.BETA, loc=0.4, scale=0.1, rel_min=np.inf, rel_max=np.inf
    )
    with pytest.raises(ValueError, match=re.escape("Beta requires finite interval [a, b] with a < b.")):
        stats.makeDistribution(p)


def test_makedistribution_beta_loc0_out_of_range_raises():
    p = stats.DistributionParameters.absolute(
        stats.DistributionID.BETA, loc=0.0, scale=0.1, abs_min=0.0, abs_max=1.0
    )
    with pytest.raises(ValueError, match=re.escape("Beta: inferred base mean(loc0) must be in (0,1).")):
        stats.makeDistribution(p)


def test_makedistribution_beta_scale0_out_of_range_raises():
    p = stats.DistributionParameters.absolute(
        stats.DistributionID.BETA, loc=0.5, scale=0.6, abs_min=0.0, abs_max=1.0
    )
    with pytest.raises(
        ValueError,
        match=re.escape("Beta: inferred base std(scale0) must be in (0, sqrt(loc0*(1-loc0)))."),
    ):
        stats.makeDistribution(p)


def test_makedistribution_beta_valid_roundtrip():
    original = stats.Beta(2.0, 3.0)
    p = original.generic_params()
    result = stats.makeDistribution(p)
    assert isinstance(result, stats.Affine)
    assert isinstance(result.base, stats.Beta)
    assert result.base.native_params()["alpha"] == pytest.approx(2.0, rel=1e-6)
    assert result.base.native_params()["beta"] == pytest.approx(3.0, rel=1e-6)


def test_makedistribution_exponential_requires_positive_loc():
    p = stats.DistributionParameters.relative(
        stats.DistributionID.EXPONENTIAL, loc=0.0, scale=0.0, rel_min=0.0, rel_max=np.inf
    )
    with pytest.raises(ValueError, match=re.escape("Exponential requires loc > 0 (loc = 1/λ).")):
        stats.makeDistribution(p)


def test_makedistribution_exponential_valid():
    p = stats.DistributionParameters.relative(
        stats.DistributionID.EXPONENTIAL, loc=0.5, scale=0.0, rel_min=np.inf, rel_max=np.inf
    )
    result = stats.makeDistribution(p)
    assert isinstance(result, stats.Exponential)
    assert result.lam == pytest.approx(2.0)


def test_makedistribution_lognormal_requires_positive_loc_nonneg_scale():
    p = stats.DistributionParameters.relative(
        stats.DistributionID.LOGNORMAL, loc=0.0, scale=0.1, rel_min=np.inf, rel_max=np.inf
    )
    with pytest.raises(ValueError, match="Lognormal requires loc > 0 and scale >= 0."):
        stats.makeDistribution(p)


def test_makedistribution_lognormal_zero_scale_returns_deterministic():
    p = stats.DistributionParameters.relative(
        stats.DistributionID.LOGNORMAL, loc=2.0, scale=0.0, rel_min=np.inf, rel_max=np.inf
    )
    result = stats.makeDistribution(p)
    assert isinstance(result, stats.Deterministic)
    assert result.value == 2.0


def test_makedistribution_lognormal_valid_roundtrip():
    # Lognormal's natural generic_params() has rel_min == loc (its support
    # starts at 0), which is finite, so makeDistribution wraps the
    # reconstructed base in a Truncate(lower=0.0) -- a numerical no-op since
    # the lognormal already has ~0 mass below 0, but it does change the
    # returned type.
    original = stats.Lognormal(0.0, 0.5)
    p = original.generic_params()
    result = stats.makeDistribution(p)
    assert isinstance(result, stats.Truncate)
    base = result.base
    assert isinstance(base, stats.Lognormal)
    assert base.mu == pytest.approx(0.0, abs=1e-6)
    assert base.sigma == pytest.approx(0.5, rel=1e-6)


def test_makedistribution_gamma_requires_positive_loc_and_scale():
    p = stats.DistributionParameters.relative(
        stats.DistributionID.GAMMA, loc=0.0, scale=1.0, rel_min=0.0, rel_max=np.inf
    )
    with pytest.raises(ValueError, match="Gamma requires loc > 0 and scale > 0."):
        stats.makeDistribution(p)


def test_makedistribution_gamma_valid_roundtrip():
    # Same finite-lower-bound wrapping as Lognormal above: Gamma's
    # generic_params() has rel_min == loc (finite), so the result is wrapped
    # in a Truncate(lower=0.0).
    original = stats.Gamma(2.0, 3.0)
    p = original.generic_params()
    result = stats.makeDistribution(p)
    assert isinstance(result, stats.Truncate)
    base = result.base
    assert isinstance(base, stats.Gamma)
    assert base.alpha == pytest.approx(2.0, rel=1e-6)
    assert base.lam == pytest.approx(3.0, rel=1e-6)


def test_makedistribution_unknown_id_raises():
    p = stats.DistributionParameters(
        id=99, loc=0.0, scale=1.0, min_raw=0.0, max_raw=1.0, relative=False
    )
    with pytest.raises(ValueError, match="Unknown distribution id"):
        stats.makeDistribution(p)


def test_makedistribution_no_longer_accepts_tol_kwarg():
    # The formerly-dead `tol` kwarg (accepted but never referenced in the
    # body) has since been removed from makeDistribution's signature.
    p = stats.Normal(1.0, 2.0).generic_params()
    with pytest.raises(TypeError):
        stats.makeDistribution(p, tol=1e-3)


# ---------------------------------------------------------------------------
# asDistributionVector
# ---------------------------------------------------------------------------

def test_asdistributionvector_passthrough_same_length():
    v = stats.DistributionVector([0.0, 1.0])
    result = stats.asDistributionVector(v)
    assert isinstance(result, stats.DistributionVector)
    assert [d.value for d in result] == [0.0, 1.0]


def test_asdistributionvector_length_mismatch_raises():
    v = stats.DistributionVector([0.0, 1.0, 2.0])
    with pytest.raises(ValueError, match="Length mismatch"):
        stats.asDistributionVector(v, length=2)


def test_asdistributionvector_broadcasts_single_item_vector():
    v = stats.DistributionVector([0.0])
    result = stats.asDistributionVector(v, length=4)
    assert len(result) == 4
    assert all(d.value == 0.0 for d in result)


def test_asdistributionvector_generic_path_from_list():
    result = stats.asDistributionVector([0.0, stats.Normal(0.0, 1.0)])
    assert isinstance(result, stats.DistributionVector)
    assert isinstance(result[0], stats.Deterministic)
    assert isinstance(result[1], stats.Normal)


def test_asdistributionvector_generic_path_with_length():
    result = stats.asDistributionVector(5.0, length=3)
    assert len(result) == 3
    assert all(d.value == 5.0 for d in result)
