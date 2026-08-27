"""Tests for pyrockfall.stats.DistributionVector."""
import numpy as np
import pytest

import pyrockfall.stats as stats


def test_construct_from_scalar_list_coerces_to_deterministic():
    v = stats.DistributionVector([0.0, 2.0])
    assert len(v) == 2
    assert isinstance(v[0], stats.Deterministic)
    assert isinstance(v[1], stats.Deterministic)
    assert v[0].value == 0.0
    assert v[1].value == 2.0


def test_construct_mixed_distributions_and_scalars():
    v = stats.DistributionVector([stats.Normal(2.0, 0.1), 0.0])
    assert isinstance(v[0], stats.Normal)
    assert isinstance(v[1], stats.Deterministic)


def test_construct_broadcast_scalar_with_length():
    v = stats.DistributionVector(0.0, length=3)
    assert len(v) == 3
    assert all(isinstance(d, stats.Deterministic) and d.value == 0.0 for d in v)


def test_construct_length_mismatch_raises():
    with pytest.raises(ValueError, match="Length mismatch"):
        stats.DistributionVector([0.0, 1.0], length=3)


def test_construct_from_ndarray():
    v = stats.DistributionVector(np.array([1.0, 2.0, 3.0]))
    assert len(v) == 3
    assert [d.value for d in v] == [1.0, 2.0, 3.0]


def test_construct_unsupported_type_raises():
    with pytest.raises(TypeError):
        stats.DistributionVector({"a": 1})


def test_getitem_scalar_and_slice():
    v = stats.DistributionVector([0.0, 1.0, 2.0])
    assert isinstance(v[0], stats.Distribution)
    sliced = v[0:2]
    assert isinstance(sliced, stats.DistributionVector)
    assert len(sliced) == 2


def test_setitem_scalar_coerces():
    v = stats.DistributionVector([0.0, 1.0])
    v[0] = 5
    assert isinstance(v[0], stats.Deterministic)
    assert v[0].value == 5.0


def test_setitem_slice_broadcast_and_length_check():
    v = stats.DistributionVector([0.0, 1.0, 2.0])
    v[0:2] = 7.0
    assert v[0].value == 7.0 and v[1].value == 7.0
    v2 = stats.DistributionVector([0.0, 1.0])
    v2[0:2] = [1.0]
    assert v2[0].value == 1.0 and v2[1].value == 1.0
    with pytest.raises(ValueError, match="does not match"):
        v2[0:2] = [1.0, 2.0, 3.0]


def test_delitem_and_insert():
    v = stats.DistributionVector([0.0, 1.0, 2.0])
    del v[0]
    assert len(v) == 2
    v.insert(0, 9.0)
    assert len(v) == 3
    assert isinstance(v[0], stats.Deterministic)
    assert v[0].value == 9.0


def test_tolist_returns_plain_list_of_distributions():
    v = stats.DistributionVector([0.0, 1.0])
    result = v.tolist()
    assert isinstance(result, list)
    assert not isinstance(result, stats.DistributionVector)


def test_rvs_shape():
    v = stats.DistributionVector([stats.Normal(0.0, 1.0), stats.Uniform(2.0, 5.0)])
    samples = v.rvs(size=100, random_state=1)
    assert samples.shape == (100, 2)


def test_rvs_empty_vector_shape():
    v = stats.DistributionVector([0.0])
    del v[0]
    samples = v.rvs(size=5)
    assert samples.shape == (5, 0)


def test_mean_std_var_return_ndarray_of_correct_values():
    v = stats.DistributionVector([stats.Normal(0.0, 1.0), stats.Uniform(2.0, 5.0)])
    np.testing.assert_allclose(v.mean(), [0.0, 3.5])
    np.testing.assert_allclose(v.var(), [1.0, 0.75])
    np.testing.assert_allclose(v.std(), [1.0, np.sqrt(0.75)])


def test_add_scalar_elementwise():
    v = stats.DistributionVector([0.0, 1.0]) + 5.0
    assert isinstance(v, stats.DistributionVector)
    np.testing.assert_allclose(v.mean(), [5.0, 6.0])


def test_radd_symmetric():
    a = stats.DistributionVector([0.0, 1.0]) + 5.0
    b = 5.0 + stats.DistributionVector([0.0, 1.0])
    np.testing.assert_allclose(a.mean(), b.mean())


def test_sub_and_rsub_elementwise():
    sub = stats.DistributionVector([1.0, 2.0]) - 5.0
    np.testing.assert_allclose(sub.mean(), [-4.0, -3.0])
    rsub = 5.0 - stats.DistributionVector([1.0, 2.0])
    np.testing.assert_allclose(rsub.mean(), [4.0, 3.0])


def test_mul_scalar_and_by_zero():
    v = stats.DistributionVector([1.0, 2.0]) * 3.0
    np.testing.assert_allclose(v.mean(), [3.0, 6.0])
    zeroed = stats.DistributionVector([1.0, 2.0]) * 0.0
    assert all(isinstance(d, stats.Deterministic) and d.value == 0.0 for d in zeroed)


def test_truediv_scalar_and_by_zero_raises():
    v = stats.DistributionVector([2.0, 4.0]) / 2.0
    np.testing.assert_allclose(v.mean(), [1.0, 2.0])
    with pytest.raises(ValueError, match="Division by zero."):
        stats.DistributionVector([1.0, 2.0]) / 0.0


def test_arithmetic_with_non_numeric_raises_typeerror():
    with pytest.raises(TypeError):
        stats.DistributionVector([0.0]) + "x"


def test_repr_format():
    v = stats.DistributionVector([0.0])
    r = repr(v)
    assert r.startswith("DistributionVector([")
    assert repr(v[0]) in r
