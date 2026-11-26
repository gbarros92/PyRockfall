"""Tests for pyrockfall._filesystem: save-to-tmp-file / load-back round trips.

Every public export/import (or write/read) pair is exercised the same way:
write to a pytest `tmp_path` file, read it back, and compare the loaded data
against what was originally saved.

Two real bugs were found and fixed while writing these tests (confirmed with
the user before changing production code):

1. writeMaterials/readMaterials (CSV): the writer stored
   ``generic_params().rel_min``/``rel_max`` (offsets relative to the mean)
   but the reader interpreted the same columns as absolute bounds via
   ``DistributionParameters.absolute(...)``. For Beta(2,3) this produced
   abs_min=0.4, abs_max=0.6 instead of the correct [0,1] and made
   ``makeDistribution`` raise. Fixed by writing ``abs_min``/``abs_max``
   instead, matching what the reader already expected.

2. exportSeeders/importSeeders (plain text): the parser distinguished a
   points row from the trailing rock-ids row using an "all tokens are
   integers" heuristic. Since ``np.savetxt`` writes whole-valued floats like
   10.0 as "10", any point row with all-integer coordinates was misread as
   the rock-ids row, corrupting the parse. Fixed by writing an explicit
   point count instead of relying on that ambiguous heuristic.

A third, pre-existing limitation is documented (not fixed, since it matches
Material's own design where `roughness` is excluded from
`numRandomVariables`): exportMaterials/importMaterials (the RocFall "fal8"
text format) never actually serializes a material's `roughness` -- it always
writes a hardcoded placeholder, so `roughness` never round-trips through
that format.
"""
import numpy as np
import pytest

import pyrockfall as pr
from pyrockfall import stats


# ---------------------------------------------------------------------------
# writeMaterials / readMaterials (CSV)
# ---------------------------------------------------------------------------

def test_write_materials_read_materials_roundtrip(tmp_path):
    mat0 = pr.Material(
        name="Basalt",
        normalRestitution=stats.Beta(2.0, 3.0),
        tangentialRestitution=stats.Normal(0.5, 0.1),
        frictionAngle=stats.Uniform(20.0, 40.0),
    )
    mat1 = pr.Material(
        name="Granite",
        normalRestitution=0.4,
        tangentialRestitution=0.6,
        frictionAngle=35.0,
    )
    materials = {0: mat0, 1: mat1}

    path = tmp_path / "materials.csv"
    pr.writeMaterials(str(path), materials)
    assert path.exists()

    loaded = pr.readMaterials(str(path))
    assert set(loaded.keys()) == {0, 1}

    for mat_id, original in materials.items():
        result = loaded[mat_id]
        assert result.name == original.name
        assert isinstance(result.normalRestitution, stats.Distribution)
        assert result.normalRestitution.mean() == pytest.approx(original.normalRestitution.mean(), rel=1e-6)
        assert result.normalRestitution.std() == pytest.approx(original.normalRestitution.std(), rel=1e-6)
        assert result.tangentialRestitution.native_params() == pytest.approx(
            original.tangentialRestitution.native_params()
        )
        assert result.frictionAngle.native_params() == pytest.approx(original.frictionAngle.native_params())


def test_write_materials_read_materials_roundtrip_deterministic_values(tmp_path):
    # Regression case: this is exactly the scenario that used to raise
    # ValueError before the abs_min/abs_max fix (Beta specifically, but any
    # bounded family was affected).
    mat = pr.Material(name="M", normalRestitution=stats.Beta(2.0, 3.0))
    path = tmp_path / "materials_beta.csv"
    pr.writeMaterials(str(path), {0: mat})
    loaded = pr.readMaterials(str(path))
    a, b = loaded[0].normalRestitution.generic_params().a, loaded[0].normalRestitution.generic_params().b
    assert a == pytest.approx(0.0, abs=1e-6)
    assert b == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# exportSlope / importSlope
# ---------------------------------------------------------------------------

def test_export_slope_import_slope_roundtrip(tmp_path):
    nodes = np.array([[0.0, 0.0], [1.0, 0.5], [2.0, 0.0]])
    nodes_std = np.array([[0.1, 0.0], [0.0, 0.05], [0.0, 0.0]])
    geometry = pr.Geometry(nodes=nodes, nodes_std=nodes_std)
    mat0 = pr.Material(name="A")
    mat1 = pr.Material(name="B")
    slope = pr.Slope(geometry, materials=[mat0, mat1], materialIDs=[0, 1])

    path = tmp_path / "slope.csv"
    pr.exportSlope(str(path), slope)
    assert path.exists()

    loaded = pr.importSlope(str(path), materials=[mat0, mat1])
    np.testing.assert_allclose(loaded.nodes, slope.nodes)
    np.testing.assert_allclose(loaded.nodes_std, slope.nodes_std)
    np.testing.assert_array_equal(loaded.materialIDs, slope.materialIDs)


def test_export_slope_import_slope_roundtrip_no_uncertainty(tmp_path):
    nodes = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
    geometry = pr.Geometry(nodes=nodes)
    mat = pr.Material(name="A")
    slope = pr.Slope(geometry, materials=mat)

    path = tmp_path / "slope_det.csv"
    pr.exportSlope(str(path), slope)
    loaded = pr.importSlope(str(path), materials=[mat])
    np.testing.assert_allclose(loaded.nodes, slope.nodes)
    np.testing.assert_allclose(loaded.nodes_std, np.zeros_like(slope.nodes))


# ---------------------------------------------------------------------------
# exportSeeders / importSeeders
# ---------------------------------------------------------------------------

def test_export_seeders_import_seeders_roundtrip_with_integer_valued_points(tmp_path):
    # Regression test: (0,0) and (10,0) are exactly the values that used to
    # be misread as the rock-ids line before the explicit point-count fix,
    # because both rows tokenize as "all integers".
    rock = pr.Rock(name="R1", mass=stats.Normal(50.0, 5.0), density=stats.Uniform(2500.0, 2700.0))
    rock_types = [rock]
    points = np.array([[0.0, 10.0], [0.0, 0.0]])
    seeder = pr.Seeder(points, rocks=[rock], name="S1")
    seeder.numberOfRocks = 42
    seeder.rockThrowMode = pr.SeederRocksThrown.PerRockType
    seeder.translationalVelocity = [stats.Normal(1.0, 0.5), stats.Normal(-2.0, 0.3)]
    seeder.angularVelocity = [stats.Normal(0.0, 0.1)]

    path = tmp_path / "seeders.txt"
    pr.exportSeeders(str(path), [seeder], rock_types)
    assert path.exists()

    loaded = pr.importSeeders(str(path), rock_types)
    assert len(loaded) == 1
    result = loaded[0]

    np.testing.assert_allclose(result.points, seeder.points)
    assert result.numberOfRocks == seeder.numberOfRocks
    assert result.rockThrowMode == seeder.rockThrowMode
    assert result.rocks == seeder.rocks
    for orig_d, loaded_d in zip(seeder.translationalVelocity, result.translationalVelocity):
        assert loaded_d.native_params() == pytest.approx(orig_d.native_params())
    for orig_d, loaded_d in zip(seeder.angularVelocity, result.angularVelocity):
        assert loaded_d.native_params() == pytest.approx(orig_d.native_params())


def test_export_seeders_import_seeders_roundtrip_non_integer_points(tmp_path):
    rock = pr.Rock(name="R1", mass=1.0, density=2500.0)
    rock_types = [rock]
    points = np.array([[-3.5, 7.25, 12.1], [0.0, 4.4, -2.2]])
    seeder = pr.Seeder(points, rocks=[rock], name="Line")
    seeder.numberOfRocks = 5

    path = tmp_path / "seeders_multi.txt"
    pr.exportSeeders(str(path), [seeder], rock_types)
    loaded = pr.importSeeders(str(path), rock_types)[0]
    np.testing.assert_allclose(loaded.points, seeder.points)


def test_export_seeders_import_seeders_roundtrip_multiple_seeders(tmp_path):
    rock_a = pr.Rock(name="A", mass=1.0, density=2500.0)
    rock_b = pr.Rock(name="B", mass=2.0, density=2600.0)
    rock_types = [rock_a, rock_b]

    seeder1 = pr.Seeder(np.array([5.0, 5.0]), rocks=[rock_a])
    seeder1.numberOfRocks = 10
    seeder2 = pr.Seeder(np.array([[0.0, 1.0], [0.0, 1.0]]), rocks=[rock_a, rock_b])
    seeder2.numberOfRocks = 20

    path = tmp_path / "seeders_two.txt"
    pr.exportSeeders(str(path), [seeder1, seeder2], rock_types)
    loaded = pr.importSeeders(str(path), rock_types)

    assert len(loaded) == 2
    np.testing.assert_allclose(loaded[0].points, seeder1.points)
    np.testing.assert_allclose(loaded[1].points, seeder2.points)
    assert loaded[0].rocks == seeder1.rocks
    assert loaded[1].rocks == seeder2.rocks


# ---------------------------------------------------------------------------
# exportMaterials / importMaterials (RocFall "fal8" text format)
# ---------------------------------------------------------------------------

def test_export_materials_import_materials_roundtrip(tmp_path):
    mat0 = pr.Material(
        name="Basalt",
        normalRestitution=stats.Beta(2.0, 3.0),
        tangentialRestitution=stats.Normal(0.5, 0.1),
        frictionAngle=stats.Uniform(20.0, 40.0),
    )
    mat1 = pr.Material(name="Granite", normalRestitution=0.4, tangentialRestitution=0.6, frictionAngle=35.0)
    mat0.color = (255, 0, 0)
    mat1.color = (0, 255, 0)

    path = tmp_path / "materials.fal8"
    pr.exportMaterials(str(path), [mat0, mat1])
    assert path.exists()

    loaded = pr.importMaterials(str(path))
    assert [m.color for m in loaded] == [mat0.color, mat1.color]
    assert [m.name for m in loaded] == ["Basalt", "Granite"]

    assert loaded[0].normalRestitution.mean() == pytest.approx(mat0.normalRestitution.mean(), rel=1e-5)
    assert loaded[0].normalRestitution.std() == pytest.approx(mat0.normalRestitution.std(), rel=1e-5)
    assert loaded[0].tangentialRestitution.native_params() == pytest.approx(
        mat0.tangentialRestitution.native_params(), rel=1e-5
    )
    assert loaded[0].frictionAngle.native_params() == pytest.approx(mat0.frictionAngle.native_params(), rel=1e-5)

    assert loaded[1].normalRestitution.native_params() == pytest.approx(mat1.normalRestitution.native_params())
    assert loaded[1].tangentialRestitution.native_params() == pytest.approx(mat1.tangentialRestitution.native_params())
    assert loaded[1].frictionAngle.native_params() == pytest.approx(mat1.frictionAngle.native_params())


def test_export_materials_import_materials_default_colors_are_generated(tmp_path):
    mat = pr.Material(name="M")
    path = tmp_path / "materials_nocolor.fal8"
    pr.exportMaterials(str(path), [mat])  # color unset -> random color generated
    loaded = pr.importMaterials(str(path))
    assert len(loaded) == 1
    r, g, b = loaded[0].color
    assert 0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255


def test_export_materials_import_materials_does_not_preserve_roughness(tmp_path):
    # Known limitation, not fixed: _materialProperties always writes a
    # hardcoded placeholder for slope_roughness instead of the material's
    # actual `roughness` distribution, so it never round-trips through this
    # format (consistent with Material.numRandomVariables also excluding
    # roughness).
    mat = pr.Material(name="M", roughness=stats.Normal(0.0, 0.05))
    path = tmp_path / "materials_roughness.fal8"
    pr.exportMaterials(str(path), [mat])
    loaded = pr.importMaterials(str(path))
    assert isinstance(loaded[0].roughness, stats.Deterministic)
    assert loaded[0].roughness.value == 0.0


# ---------------------------------------------------------------------------
# exportRocks / importRocks (RocFall "fal8" text format)
# ---------------------------------------------------------------------------

def test_export_rocks_import_rocks_roundtrip(tmp_path):
    rock0 = pr.Rock(name="Small", mass=stats.Normal(50.0, 5.0), density=stats.Uniform(2500.0, 2700.0))
    rock1 = pr.Rock(name="Big", mass=500.0, density=2600.0)
    rock0.color = (1, 2, 3)
    rock1.color = (4, 5, 6)

    path = tmp_path / "rocks.fal8"
    pr.exportRocks(str(path), [rock0, rock1])
    assert path.exists()

    loaded = pr.importRocks(str(path))
    assert [r.color for r in loaded] == [rock0.color, rock1.color]
    assert [r.name for r in loaded] == ["Small", "Big"]

    assert loaded[0].mass.native_params() == pytest.approx(rock0.mass.native_params(), rel=1e-5)
    assert loaded[0].density.native_params() == pytest.approx(rock0.density.native_params(), rel=1e-5)
    assert loaded[1].mass.native_params() == pytest.approx(rock1.mass.native_params())
    assert loaded[1].density.native_params() == pytest.approx(rock1.density.native_params())


# ---------------------------------------------------------------------------
# exportSettings / importSettings
# ---------------------------------------------------------------------------

def test_export_settings_import_settings_roundtrip(tmp_path):
    analysis = pr.Analysis()
    analysis.rockThrowMode = pr.AnalysisRocksThrown.DistributedFromNumberOfRocks
    analysis.numberOfRocks = 500
    analysis.samplingMethod = pr.Sampling.MonteCarlo
    analysis.useSpecificSeed = True
    analysis.specificSeed = 777
    analysis.maxIter = 5000
    analysis.normalVelocityThreshold = 0.2
    analysis.stoppedVelocity = 0.01
    analysis.timeStep = 0.005
    analysis.scaleByVelocity = True
    analysis.K = 12.5
    analysis.scaleByMass = True
    analysis.C = 2000.0
    analysis.considerRotationalVelocity = True

    path = tmp_path / "settings.fal8"
    pr.exportSettings(str(path), analysis)
    assert path.exists()

    loaded = pr.importSettings(str(path))
    assert loaded.rockThrowMode == analysis.rockThrowMode
    assert loaded.numberOfRocks == analysis.numberOfRocks
    assert loaded.samplingMethod == analysis.samplingMethod
    assert loaded.useSpecificSeed == analysis.useSpecificSeed
    assert loaded.specificSeed == analysis.specificSeed
    assert loaded.maxIter == analysis.maxIter
    assert loaded.normalVelocityThreshold == pytest.approx(analysis.normalVelocityThreshold, rel=1e-5)
    assert loaded.stoppedVelocity == pytest.approx(analysis.stoppedVelocity, rel=1e-5)
    assert loaded.timeStep == pytest.approx(analysis.timeStep, rel=1e-5)
    assert loaded.scaleByVelocity == analysis.scaleByVelocity
    assert loaded.K == pytest.approx(analysis.K, rel=1e-5)
    assert loaded.scaleByMass == analysis.scaleByMass
    assert loaded.C == pytest.approx(analysis.C, rel=1e-5)
    assert loaded.considerRotationalVelocity == analysis.considerRotationalVelocity


def test_export_settings_import_settings_roundtrip_defaults(tmp_path):
    analysis = pr.Analysis()  # all defaults
    path = tmp_path / "settings_default.fal8"
    pr.exportSettings(str(path), analysis)
    loaded = pr.importSettings(str(path))
    assert loaded.rockThrowMode == analysis.rockThrowMode
    assert loaded.samplingMethod == analysis.samplingMethod
    assert loaded.useSpecificSeed == analysis.useSpecificSeed


# ---------------------------------------------------------------------------
# write_npz_full_fidelity / read_properties, read_property
# ---------------------------------------------------------------------------

def test_write_npz_read_properties_roundtrip(tmp_path):
    from pyrockfall._filesystem import write_npz_full_fidelity, read_properties, read_property

    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    normals = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    scalar = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    path = tmp_path / "cloud.npz"
    write_npz_full_fidelity(str(path), core={"points": points}, attrs={"normals": normals, "myscalar": scalar})
    assert path.exists()

    out = read_properties(str(path), required=["points"])
    np.testing.assert_allclose(out["points"], points)

    scalar_val, scalar_meta = read_property(str(path), "myscalar")
    np.testing.assert_allclose(scalar_val, scalar)
    assert scalar_meta["is_vector"] is False

    normal_val, normal_meta = read_property(str(path), "normals")
    np.testing.assert_allclose(normal_val, normals)
    assert normal_meta["is_vector"] is True


def test_read_properties_missing_file_raises(tmp_path):
    from pyrockfall._filesystem import read_properties

    with pytest.raises(FileNotFoundError):
        read_properties(str(tmp_path / "does_not_exist.npz"), required=["points"])


# ---------------------------------------------------------------------------
# write_ply_with_attrs / read_properties, read_property
# ---------------------------------------------------------------------------

def test_write_ply_read_properties_roundtrip(tmp_path):
    from pyrockfall._filesystem import write_ply_with_attrs, read_properties, read_property

    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    normals = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    scalar = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    triangles = np.array([[0, 1, 2]], dtype=np.int32)

    path = tmp_path / "mesh.ply"
    write_ply_with_attrs(
        str(path),
        core={"points": points, "triangles": triangles},
        attrs={"normals": normals, "myscalar": scalar},
        comments={"source": "test"},
    )
    assert path.exists()

    out = read_properties(str(path), required=["xyz"])
    np.testing.assert_allclose(out["xyz"], points, atol=1e-5)

    normal_val, normal_meta = read_property(str(path), "normal")
    np.testing.assert_allclose(normal_val, normals, atol=1e-5)
    assert normal_meta["components"] == ["nx", "ny", "nz"]

    scalar_val, scalar_meta = read_property(str(path), "myscalar")
    np.testing.assert_allclose(scalar_val, scalar, atol=1e-5)
    assert scalar_meta["is_vector"] is False


def test_write_ply_with_colors_roundtrip(tmp_path):
    from pyrockfall._filesystem import write_ply_with_attrs, read_property

    points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)
    colors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # 0..1 floats -> quantized to uint8

    path = tmp_path / "mesh_colors.ply"
    write_ply_with_attrs(str(path), core={"points": points}, attrs={"colors": colors}, comments=None)

    rgb, meta = read_property(str(path), "rgb")
    np.testing.assert_allclose(rgb, np.array([[255, 0, 0], [0, 255, 0]]), atol=1)


# ---------------------------------------------------------------------------
# openPly / plyFieldName and row subsets (native PLY reader)
# ---------------------------------------------------------------------------

def _write_raw_ply(path, fmt, columns, types, pre_element=None):
    """A vertex-only PLY (optionally after another fixed-size element) written by
    hand, so the reader is not tested against its own writer."""
    names = list(columns)
    n = len(columns[names[0]])
    header = ["ply", f"format {fmt} 1.0"]
    if pre_element is not None:
        header += [f"element camera {len(pre_element)}", "property float view"]
    header += [f"element vertex {n}"] + [f"property {types[k]} {k}" for k in names] + ["end_header"]
    np_types = {"float": "f4", "double": "f8", "uchar": "u1", "int": "i4"}
    with open(path, "wb") as f:
        f.write(("\n".join(header) + "\n").encode("ascii"))
        if fmt == "ascii":
            if pre_element is not None:
                for v in pre_element:
                    f.write(f"{v}\n".encode())
            for i in range(n):
                f.write((" ".join(str(columns[k][i]) for k in names) + "\n").encode())
        else:
            e = "<" if fmt == "binary_little_endian" else ">"
            if pre_element is not None:
                f.write(np.asarray(pre_element, dtype=e + "f4").tobytes())
            rec = np.empty(n, dtype=[(k, e + np_types[types[k]]) for k in names])
            for k in names:
                rec[k] = columns[k]
            f.write(rec.tobytes())


_COLS = {
    "x": [0.0, 1.0, 2.0, 3.0], "y": [10.0, 11.0, 12.0, 13.0], "z": [-1.0, -2.0, -3.0, -4.0],
    "nx": [0.0, 0.0, 1.0, 0.0], "ny": [0.0, 1.0, 0.0, 0.0], "nz": [1.0, 0.0, 0.0, 1.0],
    "red": [255, 0, 0, 51], "green": [0, 255, 0, 51], "blue": [0, 0, 255, 51],
    "scalar_Material": [3.0, 7.0, 7.0, 0.0],
}
_TYPES = {"x": "double", "y": "double", "z": "double", "nx": "float", "ny": "float", "nz": "float",
          "red": "uchar", "green": "uchar", "blue": "uchar", "scalar_Material": "float"}


@pytest.mark.parametrize("fmt", ["binary_little_endian", "binary_big_endian", "ascii"])
@pytest.mark.parametrize("pre", [None, [1.5, 2.5]])
def test_openPly_reads_vertices(tmp_path, fmt, pre):
    path = tmp_path / "cloud.ply"
    _write_raw_ply(path, fmt, _COLS, _TYPES, pre_element=pre)
    v = pr.openPly(str(path))
    assert v.shape == (4,)
    assert v.dtype.names == tuple(_COLS)
    for k, vals in _COLS.items():
        np.testing.assert_allclose(v[k], vals)
    if fmt != "ascii":
        assert isinstance(v, np.memmap)


def test_plyFieldName_exact_then_cloudcompare_prefix(tmp_path):
    path = tmp_path / "cloud.ply"
    _write_raw_ply(path, "binary_little_endian", _COLS, _TYPES)
    v = pr.openPly(str(path))
    assert pr.plyFieldName(v, "Material") == "scalar_Material"
    assert pr.plyFieldName(v, "x") == "x"
    assert pr.plyFieldName(v.dtype.names, "red") == "red"
    with pytest.raises(KeyError):
        pr.plyFieldName(v, "Profile")


def test_openPly_rejects_truncated_and_non_ply(tmp_path):
    path = tmp_path / "cloud.ply"
    _write_raw_ply(path, "binary_little_endian", _COLS, _TYPES)
    data = path.read_bytes()
    (tmp_path / "short.ply").write_bytes(data[:-3])
    with pytest.raises(ValueError, match="truncated"):
        pr.openPly(str(tmp_path / "short.ply"))
    (tmp_path / "bad.ply").write_bytes(b"not a ply\n")
    with pytest.raises(ValueError):
        pr.openPly(str(tmp_path / "bad.ply"))


def test_read_property_prefers_exact_name(tmp_path):
    """'Material' must not resolve to 'Material_P1', whichever comes first."""
    from pyrockfall._filesystem import read_property, read_properties
    cols = {"x": [0.0, 1.0], "y": [0.0, 1.0], "z": [0.0, 1.0],
            "Material_P1": [0.25, 0.5], "Material": [4.0, 5.0]}
    types = {k: "float" for k in cols}
    path = tmp_path / "filled.ply"
    _write_raw_ply(path, "binary_little_endian", cols, types)
    val, _ = read_property(str(path), "Material")
    np.testing.assert_allclose(val, [4.0, 5.0])
    out = read_properties(str(path), ["Material", "Material_P1"])
    np.testing.assert_allclose(out["Material_P1"], [0.25, 0.5])
    assert out["Material"].dtype == np.float64


def test_read_properties_subset_of_rows(tmp_path):
    from pyrockfall._filesystem import read_properties
    path = tmp_path / "cloud.ply"
    _write_raw_ply(path, "binary_little_endian", _COLS, _TYPES)
    out = read_properties(str(path), ["Material", "xyz"], indices=[3, 1])
    np.testing.assert_allclose(out["Material"], [0.0, 7.0])
    np.testing.assert_allclose(out["xyz"], [[3.0, 13.0, -4.0], [1.0, 11.0, -2.0]])
    mask = np.array([False, True, True, False])
    np.testing.assert_allclose(read_properties(str(path), ["Material"], indices=mask)["Material"], [7.0, 7.0])


def test_pointcloud_load_ply_native(tmp_path):
    path = tmp_path / "cloud.ply"
    _write_raw_ply(path, "binary_little_endian", _COLS, _TYPES)
    pc = pr.PointCloud.load(str(path), attributes=["Material"])
    np.testing.assert_allclose(pc.points, np.column_stack([_COLS["x"], _COLS["y"], _COLS["z"]]))
    np.testing.assert_allclose(pc.get_attr("normals"), np.column_stack([_COLS["nx"], _COLS["ny"], _COLS["nz"]]))
    np.testing.assert_allclose(pc.get_attr("colors")[3], [0.2, 0.2, 0.2])      # uchar / 255
    np.testing.assert_allclose(pc.get_attr("Material"), _COLS["scalar_Material"])

    sub = pr.PointCloud.load(str(path), attributes=["Material"], indices=[0, 2], normals=False, colors=False)
    np.testing.assert_allclose(sub.points, [[0.0, 10.0, -1.0], [2.0, 12.0, -3.0]])
    np.testing.assert_allclose(sub.get_attr("Material"), [3.0, 7.0])
    assert not sub.has_attr("normals") and not sub.has_attr("colors")


def test_mesh_load_rejects_indices(tmp_path):
    path = tmp_path / "cloud.ply"
    _write_raw_ply(path, "binary_little_endian", _COLS, _TYPES)
    with pytest.raises(ValueError, match="indices"):
        pr.Mesh.load(str(path), indices=[0, 1])


# ---------------------------------------------------------------------------
# write_xyz_ascii / write_pcd_minimal (via Open3D-backed read_property)
# ---------------------------------------------------------------------------

def test_write_xyz_ascii_roundtrip(tmp_path):
    pytest.importorskip("open3d")
    from pyrockfall._filesystem import write_xyz_ascii, read_properties

    points = np.array([[0.0, 0.0, 0.0], [1.5, 2.5, 3.5], [4.0, 5.0, 6.0]])
    path = tmp_path / "cloud.xyz"
    write_xyz_ascii(str(path), core={"points": points}, attrs={}, allow_lossy=False)
    assert path.exists()

    out = read_properties(str(path), required=["xyz"])
    np.testing.assert_allclose(out["xyz"], points, atol=1e-5)


def test_write_xyz_ascii_refuses_lossy_attrs_without_consent(tmp_path):
    from pyrockfall._filesystem import write_xyz_ascii

    points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    scalar = np.array([1.0, 2.0])
    path = tmp_path / "cloud_lossy.xyz"
    with pytest.raises(ValueError, match="Refusing to write .xyz"):
        write_xyz_ascii(str(path), core={"points": points}, attrs={"scalar": scalar}, allow_lossy=False)


def test_write_pcd_minimal_roundtrip(tmp_path):
    pytest.importorskip("open3d")
    from pyrockfall._filesystem import write_pcd_minimal, read_properties

    points = np.array([[0.0, 0.0, 0.0], [1.5, 2.5, 3.5], [4.0, 5.0, 6.0]])
    path = tmp_path / "cloud.pcd"
    write_pcd_minimal(str(path), core={"points": points}, attrs={}, allow_lossy=False)
    assert path.exists()

    out = read_properties(str(path), required=["xyz"])
    np.testing.assert_allclose(out["xyz"], points, atol=1e-5)
