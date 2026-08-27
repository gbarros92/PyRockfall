"""Tests for pyrockfall.Vegetation and pyrockfall.Drag.

Vegetation(geometry, drag, identities=None) mirrors Slope's design exactly
(geometry + a per-element assignment of "extra" objects -- Drag instead of
Material), supporting the same three constructor modes:
  1. A single Drag broadcast to every element.
  2. A list of Drag, one per element (deduplicated by identity into a
     dragTable via uniqueMaterialList).
  3. An explicit dragTable (list) + identities (indices into it).

For every mode, dragTable / dragIdentities / drag must stay mutually
consistent: drag[i] is dragTable[dragIdentities[i]].

Bug found and fixed while writing these tests (confirmed with the user
first): mode 2 (list of Drag) always raised TypeError, because Vegetation
reuses `_utils.uniqueMaterialList`, which hardcoded `isinstance(mobj,
Material)` -- Drag is not a Material subclass. Fixed by generalizing
uniqueMaterialList's identity-based dedup to accept any object type (it's
already shared between Slope and Vegetation and never actually needed the
Material-specific check).
"""
import numpy as np
import pytest

import pyrockfall as pr
from pyrockfall import stats


def make_geometry(n_nodes=4):
    # Default elements are sequential pairs -> E = n_nodes - 1.
    nodes = np.column_stack([np.arange(n_nodes, dtype=float), np.zeros(n_nodes)])
    return pr.Geometry(nodes=nodes)


def assert_drag_consistent(veg, expected_table_identity=None):
    table = veg.dragTable
    ids = veg.dragIdentities
    drags = veg.drag

    assert len(ids) == len(veg.elements)
    assert len(drags) == len(ids)
    assert all(0 <= i < len(table) for i in ids)

    for i, d in zip(ids, drags):
        assert d is table[i]

    if expected_table_identity is not None:
        assert [id(d) for d in table] == [id(d) for d in expected_table_identity]


# ---------------------------------------------------------------------------
# Drag: construction, validation, sampling
# ---------------------------------------------------------------------------

def test_drag_default_coefficient_is_deterministic_zero():
    d = pr.Drag()
    assert isinstance(d.coefficient, stats.Deterministic)
    assert d.coefficient.value == 0.0
    assert d.name.startswith("Drag ")


def test_drag_scalar_coefficient_is_coerced_to_deterministic():
    d = pr.Drag(name="D1", coefficient=2.5)
    assert isinstance(d.coefficient, stats.Deterministic)
    assert d.coefficient.value == 2.5


def test_drag_coefficient_setter_coerces_via_asdistribution():
    d = pr.Drag(coefficient=stats.Normal(1.0, 0.2))
    assert isinstance(d.coefficient, stats.Normal)
    d.coefficient = 3.0
    assert isinstance(d.coefficient, stats.Deterministic)
    assert d.coefficient.value == 3.0


def test_drag_num_random_variables_is_one():
    d = pr.Drag(coefficient=stats.Normal(1.0, 0.2))
    assert d.numRandomVariables == 1


def test_drag_ppf_rejects_wrong_length():
    d = pr.Drag(coefficient=stats.Normal(1.0, 0.2))
    with pytest.raises(ValueError, match="`q` must be a sequence of one quantile array."):
        d.ppf([np.array([0.5]), np.array([0.5])])
    with pytest.raises(ValueError, match="`q` must be a sequence of one quantile array."):
        d.ppf(0.5)


def test_drag_ppf_returns_expected_shape():
    d = pr.Drag(coefficient=stats.Normal(1.0, 0.2))
    result = d.ppf([np.array([0.1, 0.5, 0.9])])
    assert result.shape == (1, 3)


def test_drag_rvs_rejects_negative_num_samples():
    d = pr.Drag(coefficient=stats.Normal(1.0, 0.2))
    with pytest.raises(ValueError, match="`num_samples` must be non-negative."):
        d.rvs(-1)


def test_drag_rvs_returns_expected_shape():
    d = pr.Drag(coefficient=stats.Normal(1.0, 0.2))
    result = d.rvs(50)
    assert result.shape == (1, 50)


def test_drag_ppf_numerical_mean_and_variance_match_prescribed_distribution():
    d = pr.Drag(coefficient=stats.Normal(2.0, 0.5))
    n = 100_000
    grid = (np.arange(n) + 0.5) / n
    samples = d.ppf([grid])[0]
    assert samples.mean() == pytest.approx(2.0, rel=1e-3)
    assert samples.var() == pytest.approx(0.25, rel=1e-2)


def test_drag_rvs_sample_mean_and_variance_match_prescribed_distribution():
    d = pr.Drag(coefficient=stats.Normal(2.0, 0.5))
    np.random.seed(0)
    samples = d.rvs(100_000)[0]
    assert samples.mean() == pytest.approx(2.0, rel=5e-2)
    assert samples.var() == pytest.approx(0.25, rel=5e-2)


# ---------------------------------------------------------------------------
# Vegetation mode 1: a single Drag broadcast to all elements
# ---------------------------------------------------------------------------

def test_single_drag_broadcast_to_all_elements():
    geometry = make_geometry(n_nodes=4)  # E = 3
    d = pr.Drag(name="D1")
    veg = pr.Vegetation(geometry, drag=d)

    assert veg.dragTable == [d]
    np.testing.assert_array_equal(veg.dragIdentities, np.zeros(3, dtype=int))
    assert veg.drag == [d, d, d]
    assert_drag_consistent(veg, expected_table_identity=[d])


def test_single_drag_are_all_the_same_object():
    geometry = make_geometry(n_nodes=5)  # E = 4
    d = pr.Drag(name="D1")
    veg = pr.Vegetation(geometry, drag=d)
    assert all(x is d for x in veg.drag)


# ---------------------------------------------------------------------------
# Vegetation mode 2: a list of Drag, one per element (deduplicated by identity)
# ---------------------------------------------------------------------------

def test_drag_list_with_repeats_is_deduplicated_by_identity():
    geometry = make_geometry(n_nodes=4)  # E = 3
    d_a = pr.Drag(name="DA")
    d_b = pr.Drag(name="DB")
    per_element = [d_a, d_b, d_a]

    veg = pr.Vegetation(geometry, drag=per_element)

    assert veg.dragTable == [d_a, d_b]  # first-appearance order
    np.testing.assert_array_equal(veg.dragIdentities, [0, 1, 0])
    assert veg.drag == per_element
    assert_drag_consistent(veg, expected_table_identity=[d_a, d_b])


def test_drag_list_dedup_is_identity_based_not_name_based():
    geometry = make_geometry(n_nodes=3)  # E = 2
    d_x1 = pr.Drag(name="X")
    d_x2 = pr.Drag(name="X")  # same name, different instance
    veg = pr.Vegetation(geometry, drag=[d_x1, d_x2])

    assert len(veg.dragTable) == 2  # not deduplicated: distinct instances
    assert veg.dragTable[0] is d_x1
    assert veg.dragTable[1] is d_x2
    np.testing.assert_array_equal(veg.dragIdentities, [0, 1])


def test_drag_list_all_unique_no_dedup():
    geometry = make_geometry(n_nodes=4)  # E = 3
    drags = [pr.Drag(name=f"D{i}") for i in range(3)]
    veg = pr.Vegetation(geometry, drag=drags)

    assert veg.dragTable == drags
    np.testing.assert_array_equal(veg.dragIdentities, [0, 1, 2])
    assert veg.drag == drags
    assert_drag_consistent(veg, expected_table_identity=drags)


def test_drag_list_length_mismatch_raises():
    geometry = make_geometry(n_nodes=4)  # E = 3
    drags = [pr.Drag(name="A"), pr.Drag(name="B")]  # only 2, need 3
    with pytest.raises(ValueError, match="Number of elements and number of materials must match."):
        pr.Vegetation(geometry, drag=drags)


# ---------------------------------------------------------------------------
# Vegetation mode 3: explicit drag table + identities
# ---------------------------------------------------------------------------

def test_drag_table_and_identities_explicit():
    geometry = make_geometry(n_nodes=4)  # E = 3
    d_a = pr.Drag(name="DA")
    d_b = pr.Drag(name="DB")
    table = [d_a, d_b]
    ids = [1, 0, 1]

    veg = pr.Vegetation(geometry, drag=table, identities=ids)

    assert veg.dragTable == table
    np.testing.assert_array_equal(veg.dragIdentities, ids)
    assert veg.drag == [d_b, d_a, d_b]
    assert_drag_consistent(veg, expected_table_identity=table)


def test_drag_table_preserves_unused_entries():
    geometry = make_geometry(n_nodes=3)  # E = 2
    d_a = pr.Drag(name="A")
    d_b = pr.Drag(name="B")
    d_unused = pr.Drag(name="Unused")
    table = [d_a, d_b, d_unused]
    ids = [0, 1]

    veg = pr.Vegetation(geometry, drag=table, identities=ids)
    assert veg.dragTable == table
    assert d_unused in veg.dragTable
    assert d_unused not in veg.drag


def test_drag_table_identities_length_mismatch_raises():
    geometry = make_geometry(n_nodes=4)  # E = 3
    table = [pr.Drag(name="A"), pr.Drag(name="B")]
    with pytest.raises(ValueError, match="Number of elements and number of materials must match."):
        pr.Vegetation(geometry, drag=table, identities=[0, 1])  # only 2 ids, need 3


def test_identities_without_list_table_raises_typeerror():
    geometry = make_geometry(n_nodes=3)  # E = 2
    d = pr.Drag(name="A")
    with pytest.raises(TypeError, match="must be a list/array"):
        pr.Vegetation(geometry, drag=d, identities=[0, 0])


# ---------------------------------------------------------------------------
# Invalid constructor arguments
# ---------------------------------------------------------------------------

def test_drag_wrong_type_raises_typeerror():
    geometry = make_geometry(n_nodes=3)
    with pytest.raises(TypeError, match="materials.*must be a Material"):
        pr.Vegetation(geometry, drag="not a drag")


# ---------------------------------------------------------------------------
# Cross-mode consistency: same effective assignment via different constructors
# ---------------------------------------------------------------------------

def test_equivalent_assignment_via_all_three_constructor_modes_agree():
    d_a = pr.Drag(name="A")
    d_b = pr.Drag(name="B")

    veg_list = pr.Vegetation(make_geometry(4), drag=[d_a, d_b, d_a])
    veg_table = pr.Vegetation(make_geometry(4), drag=[d_a, d_b], identities=[0, 1, 0])

    assert veg_list.drag == veg_table.drag
    np.testing.assert_array_equal(veg_list.dragIdentities, veg_table.dragIdentities)
    assert veg_list.dragTable == veg_table.dragTable


# ---------------------------------------------------------------------------
# __getattr__ delegation to the underlying Geometry
# ---------------------------------------------------------------------------

def test_vegetation_delegates_unknown_attributes_to_geometry():
    geometry = make_geometry(n_nodes=4)
    d = pr.Drag(name="A")
    veg = pr.Vegetation(geometry, drag=d)

    np.testing.assert_array_equal(veg.nodes, geometry.nodes)
    np.testing.assert_array_equal(veg.elements, geometry.elements)
    assert veg.hasUncertainty == geometry.hasUncertainty
    assert veg.numRandomVariables == geometry.numRandomVariables
