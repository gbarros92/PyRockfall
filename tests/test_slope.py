"""Tests for pyrockfall.Slope: geometry + materials, across its three
constructor modes.

Slope(geometry, materials, materialIDs=None) supports:
  1. A single Material broadcast to every element.
  2. A list of Materials, one per element (deduplicated by identity into a
     materialTable via uniqueMaterialList).
  3. An explicit materialTable (list) + materialIDs (indices into it).

For every mode, the three related properties must stay mutually consistent:
  * materialTable: the unique list of Material objects.
  * materialIDs: per-element indices into materialTable.
  * materials: per-element Material objects, i.e. materialTable[materialIDs[i]].

Slope also delegates unknown attributes to its underlying Geometry via
__getattr__ (nodes, elements, hasUncertainty, etc.), which is checked too.
"""
import numpy as np
import pytest

import pyrockfall as pr


def make_geometry(n_nodes=4):
    # Default elements are sequential pairs -> E = n_nodes - 1.
    nodes = np.column_stack([np.arange(n_nodes, dtype=float), np.zeros(n_nodes)])
    return pr.Geometry(nodes=nodes)


def assert_materials_consistent(slope, expected_table_identity=None):
    """Shared invariant across all three constructor modes."""
    table = slope.materialTable
    ids = slope.materialIDs
    materials = slope.materials

    assert len(ids) == len(slope.elements)
    assert len(materials) == len(ids)
    assert all(0 <= i < len(table) for i in ids)

    for i, mat in zip(ids, materials):
        assert mat is table[i]

    if expected_table_identity is not None:
        assert [id(m) for m in table] == [id(m) for m in expected_table_identity]


# ---------------------------------------------------------------------------
# Mode 1: a single Material broadcast to all elements
# ---------------------------------------------------------------------------

def test_single_material_broadcast_to_all_elements():
    geometry = make_geometry(n_nodes=4)  # E = 3
    mat = pr.Material(name="Basalt")
    slope = pr.Slope(geometry, materials=mat)

    assert slope.materialTable == [mat]
    np.testing.assert_array_equal(slope.materialIDs, np.zeros(3, dtype=int))
    assert slope.materials == [mat, mat, mat]
    assert_materials_consistent(slope, expected_table_identity=[mat])


def test_single_material_materials_are_all_the_same_object():
    geometry = make_geometry(n_nodes=5)  # E = 4
    mat = pr.Material(name="Basalt")
    slope = pr.Slope(geometry, materials=mat)
    assert all(m is mat for m in slope.materials)


# ---------------------------------------------------------------------------
# Mode 2: a list of Materials, one per element (deduplicated by identity)
# ---------------------------------------------------------------------------

def test_material_list_with_repeats_is_deduplicated_by_identity():
    geometry = make_geometry(n_nodes=4)  # E = 3
    mat_a = pr.Material(name="Basalt")
    mat_b = pr.Material(name="Granite")
    per_element = [mat_a, mat_b, mat_a]

    slope = pr.Slope(geometry, materials=per_element)

    assert slope.materialTable == [mat_a, mat_b]  # first-appearance order
    np.testing.assert_array_equal(slope.materialIDs, [0, 1, 0])
    assert slope.materials == per_element
    assert_materials_consistent(slope, expected_table_identity=[mat_a, mat_b])


def test_material_list_dedup_is_identity_based_not_name_based():
    geometry = make_geometry(n_nodes=3)  # E = 2
    mat_x1 = pr.Material(name="X")
    mat_x2 = pr.Material(name="X")  # same name, different instance
    slope = pr.Slope(geometry, materials=[mat_x1, mat_x2])

    assert len(slope.materialTable) == 2  # not deduplicated: distinct instances
    assert slope.materialTable[0] is mat_x1
    assert slope.materialTable[1] is mat_x2
    np.testing.assert_array_equal(slope.materialIDs, [0, 1])


def test_material_list_all_unique_materials_no_dedup():
    geometry = make_geometry(n_nodes=4)  # E = 3
    mats = [pr.Material(name=f"M{i}") for i in range(3)]
    slope = pr.Slope(geometry, materials=mats)

    assert slope.materialTable == mats
    np.testing.assert_array_equal(slope.materialIDs, [0, 1, 2])
    assert slope.materials == mats
    assert_materials_consistent(slope, expected_table_identity=mats)


def test_material_list_length_mismatch_raises():
    geometry = make_geometry(n_nodes=4)  # E = 3
    mats = [pr.Material(name="A"), pr.Material(name="B")]  # only 2, need 3
    with pytest.raises(ValueError, match="Number of elements and number of materials must match."):
        pr.Slope(geometry, materials=mats)


# ---------------------------------------------------------------------------
# Mode 3: explicit material table + materialIDs
# ---------------------------------------------------------------------------

def test_material_table_and_ids_explicit():
    geometry = make_geometry(n_nodes=4)  # E = 3
    mat_a = pr.Material(name="Basalt")
    mat_b = pr.Material(name="Granite")
    table = [mat_a, mat_b]
    ids = [1, 0, 1]

    slope = pr.Slope(geometry, materials=table, materialIDs=ids)

    assert slope.materialTable == table
    np.testing.assert_array_equal(slope.materialIDs, ids)
    assert slope.materials == [mat_b, mat_a, mat_b]
    assert_materials_consistent(slope, expected_table_identity=table)


def test_material_table_preserves_unused_entries():
    # A table entry that no element references must still appear in
    # materialTable (unlike mode 2, this path does not deduplicate/prune).
    geometry = make_geometry(n_nodes=3)  # E = 2
    mat_a = pr.Material(name="A")
    mat_b = pr.Material(name="B")
    mat_unused = pr.Material(name="Unused")
    table = [mat_a, mat_b, mat_unused]
    ids = [0, 1]

    slope = pr.Slope(geometry, materials=table, materialIDs=ids)
    assert slope.materialTable == table
    assert mat_unused in slope.materialTable
    assert mat_unused not in slope.materials


def test_material_table_ids_length_mismatch_raises():
    geometry = make_geometry(n_nodes=4)  # E = 3
    table = [pr.Material(name="A"), pr.Material(name="B")]
    with pytest.raises(ValueError, match="Number of elements and number of materials must match."):
        pr.Slope(geometry, materials=table, materialIDs=[0, 1])  # only 2 ids, need 3


def test_materialids_without_list_table_raises_typeerror():
    geometry = make_geometry(n_nodes=3)  # E = 2
    mat = pr.Material(name="A")
    with pytest.raises(TypeError, match="materials.*must be a list/array"):
        pr.Slope(geometry, materials=mat, materialIDs=[0, 0])


# ---------------------------------------------------------------------------
# Invalid constructor arguments
# ---------------------------------------------------------------------------

def test_materials_wrong_type_raises_typeerror():
    geometry = make_geometry(n_nodes=3)
    with pytest.raises(TypeError, match="materials.*must be a Material"):
        pr.Slope(geometry, materials="not a material")


# ---------------------------------------------------------------------------
# Cross-mode consistency: same effective assignment via different constructors
# ---------------------------------------------------------------------------

def test_equivalent_assignment_via_all_three_constructor_modes_agree():
    mat_a = pr.Material(name="A")
    mat_b = pr.Material(name="B")

    # Same effective per-element assignment: [A, B, A]
    slope_list = pr.Slope(make_geometry(4), materials=[mat_a, mat_b, mat_a])
    slope_table = pr.Slope(make_geometry(4), materials=[mat_a, mat_b], materialIDs=[0, 1, 0])

    assert slope_list.materials == slope_table.materials
    np.testing.assert_array_equal(slope_list.materialIDs, slope_table.materialIDs)
    assert slope_list.materialTable == slope_table.materialTable


# ---------------------------------------------------------------------------
# __getattr__ delegation to the underlying Geometry
# ---------------------------------------------------------------------------

def test_slope_delegates_unknown_attributes_to_geometry():
    geometry = make_geometry(n_nodes=4)
    mat = pr.Material(name="A")
    slope = pr.Slope(geometry, materials=mat)

    np.testing.assert_array_equal(slope.nodes, geometry.nodes)
    np.testing.assert_array_equal(slope.elements, geometry.elements)
    assert slope.hasUncertainty == geometry.hasUncertainty
    assert slope.numRandomVariables == geometry.numRandomVariables


def test_slope_floor_matches_minimum_vertical_coordinate():
    nodes = np.array([[0.0, 5.0], [1.0, -2.0], [2.0, 3.0]])
    geometry = pr.Geometry(nodes=nodes)
    mat = pr.Material(name="A")
    slope = pr.Slope(geometry, materials=mat)
    assert slope.floor == pytest.approx(-2.0)
