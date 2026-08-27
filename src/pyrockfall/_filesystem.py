"""
Filesystem utilities for pyrockfall
===================================

Provides functions for reading and writing material and configuration files
used by the pyrockfall package.

This module is responsible for handling CSV and other file formats that define
material properties, distribution parameters, and simulation inputs.

Key features:
    - Reading material parameter files and constructing random variable objects.
    - Writing/exporting parameter data for reproducibility and sharing.
    - Designed to support core pyrockfall workflows.

This module should remain general to facilitate easy extension and maintenance
as the pyrockfall project evolves.
"""
from __future__ import annotations

from typing import Mapping, List, Iterable, Any, Tuple

import os
import uuid
import struct
import numpy as np
import pandas as pd
from plyfile import PlyData, PlyElement

from ._geometry import Geometry
from ._material import Material
from ._slope import Slope
from ._seeder import Seeder, SeederRocksThrown
from ._rock import Rock
from ._analysis import Analysis, AnalysisRocksThrown, Sampling
from . import stats

def readMaterials(filepath):
    """
    Reads a CSV file defining material parameters and constructs random variables for
    normal restitution, tangential restitution, and friction angle for each material.

    The function expects the CSV to have, for each material:
        - name: Material name.
        - For each property (Rn, Rt, phi):
            - <Property>_id: Distribution type as integer (see stats.DistributionID enum).
            - <Property>_mean: Target mean of the truncated distribution.
            - <Property>_std: Target standard deviation of the truncated distribution.
            - <Property>_min: Lower bound of support.
            - <Property>_max: Upper bound of support.

    Args:
        filepath (str): Path to the CSV file.
    Returns:
        dict: Material instances, with the following attributes set:
            - normalRestitution: Random variable for normal restitution (rn)
            - tangentialRestitution: Random variable for tangential restitution (rt)
            - frictionAngle: Random variable for friction angle (phi)
    """
    df = pd.read_csv(filepath)
    result = {}
    for _, row in df.iterrows():
        mat = Material(name=row["name"])
        mat.normalRestitution = stats.makeDistribution(
            stats.DistributionParameters.absolute(
                id=stats.DistributionID(row["Rn_id"]),
                loc=row["Rn_mean"],
                scale=row["Rn_std"],
                abs_min=row["Rn_min"],
                abs_max=row["Rn_max"]
                )
            )
        mat.tangentialRestitution = stats.makeDistribution(
            stats.DistributionParameters.absolute(
                id=stats.DistributionID(row["Rt_id"]),
                loc=row["Rt_mean"],
                scale=row["Rt_std"],
                abs_min=row["Rt_min"],
                abs_max=row["Rt_max"]
            )
        )
        mat.frictionAngle = stats.makeDistribution(
            stats.DistributionParameters.absolute(
                id=stats.DistributionID(row["phi_id"]),
                loc=row["phi_mean"],
                scale=row["phi_std"],
                abs_min=row["phi_min"],
                abs_max=row["phi_max"]
            )
        )
        result[row["material_id"]] = mat
    return result

def writeMaterials(filepath: str, materials: dict[int, Material]):
    """
    Writes material parameters to a CSV file.

    Args:
        filepath (str): Path to the output CSV file.
        materials (dict): Dictionary mapping material_id to Material instances.
    """
    data = []
    for material_id, material in materials.items():
        rn_params = material.normalRestitution.generic_params()
        rt_params = material.tangentialRestitution.generic_params()
        phi_params = material.frictionAngle.generic_params()
        data.append({
            "material_id": material_id,
            "name": material.name,
            "Rn_id": rn_params.id,
            "Rn_mean": rn_params.loc,
            "Rn_std": rn_params.scale,
            "Rn_min": rn_params.abs_min,
            "Rn_max": rn_params.abs_max,
            "Rt_id": rt_params.id,
            "Rt_mean": rt_params.loc,
            "Rt_std": rt_params.scale,
            "Rt_min": rt_params.abs_min,
            "Rt_max": rt_params.abs_max,
            "phi_id": phi_params.id,
            "phi_mean": phi_params.loc,
            "phi_std": phi_params.scale,
            "phi_min": phi_params.abs_min,
            "phi_max": phi_params.abs_max
        })
    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)


_PLY_TYPEMAP = {
    "char":   ("b", 1), "uchar":  ("B", 1),
    "int8":   ("b", 1), "uint8":  ("B", 1),
    "short":  ("h", 2), "ushort": ("H", 2),
    "int16":  ("h", 2), "uint16": ("H", 2),
    "int":    ("i", 4), "uint":   ("I", 4),
    "int32":  ("i", 4), "uint32": ("I", 4),
    "float":  ("f", 4), "float32":("f", 4),
    "double": ("d", 8), "float64":("d", 8),
}

# Heuristics for grouping vectors from scalar components
_VECTOR_SETS = [
    # name -> list of component name patterns to search (ordered)
    ("xyz",    [["x","y","z"], ["position_x","position_y","position_z"]]),
    ("normal", [["nx","ny","nz"], ["normal_x","normal_y","normal_z"]]),
    ("rgb",    [["red","green","blue"], ["r","g","b"]]),
    ("rgba",   [["red","green","blue","alpha"], ["r","g","b","a"]]),
    ("uv",     [["u","v"], ["tex_u","tex_v"]]),
]

def _group_components(props, base):
    """
    Try to assemble a vector from per-component properties using heuristics.
    props: dict name->array
    base: requested base name, e.g., 'normal', 'xyz', 'rgb'
    Returns (array, component_names) or (None, None)
    """
    # direct canonical groups first
    for key, candidates in _VECTOR_SETS:
        if base == key:
            for pat in candidates:
                if all(p in props for p in pat):
                    return np.column_stack([props[p] for p in pat]), pat
    # generic patterns: base_[x,y,z] or base[x,y,z]
    for suffixes in (["x","y","z"], ["r","g","b"], ["u","v"]):
        patterns = [f"{base}_{s}" for s in suffixes]
        if all(p in props for p in patterns):
            return np.column_stack([props[p] for p in patterns]), patterns
        patterns2 = [f"{base}{s}" for s in suffixes]  # e.g. normalx, normaly
        if all(p in props for p in patterns2):
            return np.column_stack([props[p] for p in patterns2]), patterns2
    return None, None

def _parse_ply_header(f):
    """
    Parse PLY header. Returns:
      fmt ('ascii'|'binary_little_endian'|'binary_big_endian'),
      vcount (int),
      vprops (list[(name, type)]),  # only scalar properties for vertex
      header_end_offset (int)
    """
    fmt = None
    vcount = 0
    vprops = []
    in_vertex = False
    header_lines = []
    while True:
        line = f.readline()
        if not line:
            raise ValueError("Invalid PLY: unexpected EOF in header")
        try:
            s = line.decode('utf-8').strip()
        except UnicodeDecodeError:
            # Some headers are pure ASCII but decoded already; fallback
            s = line.decode(errors='ignore').strip()
        header_lines.append(s)
        if s.startswith("format "):
            # e.g. 'format binary_little_endian 1.0'
            fmt = s.split()[1]
        elif s.startswith("element vertex"):
            vcount = int(s.split()[-1])
            in_vertex = True
        elif s.startswith("element ") and not s.startswith("element vertex"):
            in_vertex = False  # we only care about vertex properties
        elif s.startswith("property") and in_vertex:
            parts = s.split()
            if parts[1] == "list":
                # Vertex properties should not be lists; skip (faces will be lists)
                # parts: property list <count_type> <data_type> <name>
                continue
            # parts: property <type> <name>
            ptype, pname = parts[1], parts[2]
            if ptype not in _PLY_TYPEMAP:
                raise ValueError(f"Unsupported PLY type: {ptype}")
            vprops.append((pname, ptype))
        elif s == "end_header":
            break
    header_end_offset = f.tell()
    if fmt is None:
        raise ValueError("Invalid PLY header: missing format")
    return fmt, vcount, vprops, header_end_offset

def _read_ply_ascii(f, vcount, vprops):
    # Read vcount lines; each line has len(vprops) tokens
    ncols = len(vprops)
    data = []
    for _ in range(vcount):
        line = f.readline()
        if not line:
            raise ValueError("PLY ASCII truncated")
        vals = line.decode('utf-8').strip().split()
        if len(vals) < ncols:
            raise ValueError("PLY ASCII row with fewer columns than header")
        data.append(vals[:ncols])
    arr = np.array(data, dtype=float)
    return arr  # shape (N, P)

def _read_ply_binary(f, fmt, vcount, vprops):
    endian = "<" if "little" in fmt else ">"
    fmts = "".join(_PLY_TYPEMAP[t][0] for _, t in vprops)
    row_size = sum(_PLY_TYPEMAP[t][1] for _, t in vprops)
    struct_fmt = endian + fmts
    # Read all vertex bytes at once
    buf = f.read(row_size * vcount)
    if len(buf) != row_size * vcount:
        raise ValueError("PLY binary truncated")
    it = struct.iter_unpack(struct_fmt, buf)
    arr = np.fromiter((x for row in it for x in row), dtype=float, count=vcount*len(vprops))
    arr = arr.reshape(vcount, len(vprops))
    return arr

def _read_ply_all_properties(filename):
    with open(filename, "rb") as f:
        fmt, vcount, vprops, _ = _parse_ply_header(f)
        # After header, file pointer is at first vertex row
        if fmt == "ascii":
            arr = _read_ply_ascii(f, vcount, vprops)
        elif fmt in ("binary_little_endian", "binary_big_endian"):
            arr = _read_ply_binary(f, fmt, vcount, vprops)
        else:
            raise ValueError(f"Unsupported PLY format: {fmt}")
    names = [n for n, _ in vprops]
    props = {n: arr[:, i] for i, n in enumerate(names)}
    return props, vcount

def _from_open3d(filename, property_name):
    try:
        import open3d as o3d
    except Exception as e:
        raise ImportError("open3d is required to read this file type") from e

    pcd = o3d.io.read_point_cloud(filename)
    if pcd.is_empty():
        raise ValueError("Empty point cloud")

    # Collect what we can
    props = {}
    P = np.asarray(pcd.points)
    if P.size:
        props["x"], props["y"], props["z"] = P[:,0], P[:,1], P[:,2]

    # normals
    if pcd.has_normals():
        N = np.asarray(pcd.normals)
        props["nx"], props["ny"], props["nz"] = N[:,0], N[:,1], N[:,2]
        props["normal_x"], props["normal_y"], props["normal_z"] = N[:,0], N[:,1], N[:,2]

    # colors
    if pcd.has_colors():
        C = np.asarray(pcd.colors)
        # Open3D colors are usually float in [0,1]
        props["r"], props["g"], props["b"] = C[:,0], C[:,1], C[:,2]
        props["red"], props["green"], props["blue"] = C[:,0], C[:,1], C[:,2]

    # If Open3D exposes additional attributes (Tensor/legacy differences), try to include them
    # Legacy PointCloud doesn't support arbitrary attributes; Tensor does, but we keep it simple here.

    return _resolve_property(props, property_name)

def _from_npz(filename, property_name):
    data = np.load(filename)
    props = {k: np.asarray(v) for k, v in data.items()}
    return _resolve_property(props, property_name)

def _resolve_property(props, property_name):
    # Exact match => scalar (return 1D array)
    for prop in props.keys():
        if property_name in prop:
            arr = np.asarray(props[prop])
            if arr.ndim == 2 and arr.shape[1] in (2,3,4):  # already vector?
                return arr, {"is_vector": True, "components": list(range(arr.shape[1]))}
            return arr.ravel(), {"is_vector": False, "components": [property_name]}

    # Try to assemble a vector from components
    vec, comps = _group_components(props, property_name)
    if vec is not None:
        return vec, {"is_vector": True, "components": comps}

    # As a convenience: if user asked "xyz" and we only have x,y,z
    if property_name == "xyz" and all(k in props for k in ("x","y","z")):
        vec = np.column_stack([props["x"], props["y"], props["z"]])
        return vec, {"is_vector": True, "components": ["x","y","z"]}

    raise KeyError(f"Property '{property_name}' not found (and no matching components). "
                   f"Available: {sorted(props.keys())[:20]}{'...' if len(props)>20 else ''}")

def read_property(
    filename: str,
    property_name: str,
):
    """
    Read a scalar or vector property from a point set file.

    Supported:
      - .ply (ASCII, binary little-endian, binary big-endian) [native]
      - .npz (keys become properties) [native]
      - .pcd, .xyz (and .ply fallback) via lazy Open3D if available

    Vector auto-detection:
      - exact groups: xyz, normal(n: nx,ny,nz or normal_x,...) ; rgb/rgba ; uv
      - generic: <name>_{x,y,z} or <name>{x,y,z}

    Returns
    -------
    data : np.ndarray
        (N,) for scalar or (N, k) for vectors.
    meta : dict
        {"is_vector": bool, "components": [names...]}
    """
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".ply":
        props, _ = _read_ply_all_properties(filename)
        return _resolve_property(props, property_name)
    elif ext == ".npz":
        return _from_npz(filename, property_name)
    elif ext in (".pcd", ".xyz"):
        return _from_open3d(filename, property_name)
    else:
        # Try Open3D for other common point formats; will still allow xyz/normals/colors if present
        try:
            return _from_open3d(filename, property_name)
        except Exception as e:
            raise ValueError(f"Unsupported extension '{ext}' and Open3D fallback failed: {e}") from e

def read_properties(filename: str, required: list[str]) -> dict[str, np.ndarray]:
    """
    Read properties from a geometry file and return them.

    Args:
        filename: Path to the file (.npz, .ply, .pcd, .xyz, .obj, .stl, .off).
        required: List of attributes to read (e.g. ["points", "normals"]).

    Returns:
        dict[str, np.ndarray]: Mapping from attribute name to array.

    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If a required attribute is missing or unsupported.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File '{filename}' not found.")
    
    attrs: dict[str, np.ndarray] = {}        
    for key in required:
        data, meta = read_property(filename, key)
        attrs[key] = np.asarray(data)

    return attrs

def write_ply_with_attrs(
    filename: str,
    core: Mapping[str, np.ndarray],
    attrs: Mapping[str, np.ndarray],
    *,
    comments: dict[str, str] | None,
):
    """PLY writer using plyfile, supporting arbitrary per-vertex attributes."""
    pts = np.asarray(core["points"], dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must be (N,3)")
    N = pts.shape[0]

    # Gather per-vertex arrays ready for PLY
    #   - map 'normals' -> nx,ny,nz (float32)
    #   - map 'colors'  -> red,green,blue (uint8) if provided
    #   - flatten any (N,k) attr to multiple columns (suffixes)
    columns: list[tuple[str, str]] = [("x","f8"), ("y","f8"), ("z","f8")]
    data = {"x": pts[:,0], "y": pts[:,1], "z": pts[:,2]}

    def add_vec3(name: str, arr: np.ndarray, as_color=False):
        a = np.asarray(arr)
        if a.shape != (N,3):
            raise ValueError(f"{name} must be (N,3)")
        if as_color:
            if a.dtype.kind == "f":  # 0..1 floats
                a = np.clip(np.round(a * 255.0), 0, 255).astype(np.uint8)
            else:
                a = np.clip(a, 0, 255).astype(np.uint8)
            cols = ("red","green","blue")
            dts  = ("u1","u1","u1")
        else:
            a = a.astype(np.float32)
            cols = (f"{name[0]}x", f"{name[0]}y", f"{name[0]}z") if name not in ("normals","colors") else ("nx","ny","nz")
            dts  = ("f8","f8","f8")
        for j, (c, dt) in enumerate(zip(cols, dts)):
            columns.append((c, dt))
            data[c] = a[:, j].astype(np.float32)

    # Standard fields if present in attrs
    if "normals" in attrs:
        add_vec3("normals", attrs["normals"], as_color=False)
    if "colors" in attrs:
        add_vec3("colors", attrs["colors"], as_color=True)

    # Add remaining per-vertex attrs
    for name, arr in attrs.items():
        if name in ("normals","colors"):
            continue  # already handled
        a = np.asarray(arr)
        if a.shape[0] != N:
            continue  # skip mismatched shapes silently or raise
        if a.ndim == 1:
            # scalar
            a = a
            dt = _best_dtype_for_scalar(a)
            columns.append((name, dt))
            data[name] = a.astype(_numpy_dtype_from_ply(dt))
        elif a.ndim == 2:
            # vector -> flatten columns with suffixes
            k = a.shape[1]
            base = name
            suffixes = _suffixes_for_k(k)  # ['x','y','z'] if k==3 else ['0','1',...]
            for j in range(k):
                cname = f"{base}_{suffixes[j]}"
                dt = _best_dtype_for_scalar(a[:, j])
                columns.append((cname, dt))
                data[cname] = a[:, j].astype(_numpy_dtype_from_ply(dt))
        else:
            # skip higher-dim attrs
            pass

    # Build structured array
    dt = [(n, t) for (n, t) in columns]
    vertex = np.empty(N, dtype=dt)
    for n, _t in columns:
        vertex[n] = data[n]
    vert_el = PlyElement.describe(vertex, "vertex")

    elements = [vert_el]
    if "triangles" in core and core["triangles"].size:
        tri = np.asarray(core["triangles"], dtype=np.int32)
        if tri.ndim != 2 or tri.shape[1] != 3:
            raise ValueError("triangles must be (E,3) int indices")
        if tri.min() < 0 or tri.max() >= N:
            raise ValueError("triangle indices out of range of vertex array")

        # plyfile expects a list/array property named 'vertex_indices'
        # Using a fixed-length (3,) field writes a standard face list:
        #   property list uchar int vertex_indices
        faces = np.empty(tri.shape[0], dtype=[("vertex_indices", "i4", (3,))])
        faces["vertex_indices"] = tri
        face_el = PlyElement.describe(faces, "face")
        elements.append(face_el)

    # Write with plyfile
    ply = PlyData(elements, text=False)  # binary
    if comments:
        for k, v in comments.items():
            ply.comments.append(f"{k}: {v}")
    ply.write(filename)


def write_npz_full_fidelity(filename: str, core: Mapping[str, np.ndarray], attrs: Mapping[str, np.ndarray]):
    """Store everything exactly: core dict + all attributes (including per-triangle)."""
    # Keep dtypes as-is
    np.savez_compressed(filename, **{**core, **{f"attr::{k}": v for k,v in attrs.items()}})


def write_xyz_ascii(
    filename: str,
    core: Mapping[str, np.ndarray],
    attrs: Mapping[str, np.ndarray],
    *,
    allow_lossy: bool,
):
    """
    Very simple ASCII XYZ(+ extras). CloudCompare can import and you can map columns to SFs,
    but there is no header to name fields. Prefer PLY.
    """
    pts = np.asarray(core["points"], dtype=float)
    N = pts.shape[0]

    # Prepare extra per-vertex columns (flattened)
    extras = []

    # Count how many columns we’ll add
    col_count = 0
    for name, arr in attrs.items():
        a = np.asarray(arr)
        if a.shape[0] != N:
            continue
        if a.ndim == 1:
            col_count += 1
        elif a.ndim == 2:
            col_count += a.shape[1]
        else:
            continue

    if col_count > 0 and not allow_lossy:
        # XYZ has no names; you’ll lose semantics. Force explicit consent.
        raise ValueError("Refusing to write .xyz with attributes without allow_lossy=True (no headers).")

    # Build matrix
    rows = [pts]
    for name, arr in attrs.items():
        a = np.asarray(arr)
        if a.shape[0] != N: continue
        if a.ndim == 1:
            rows.append(a.reshape(-1,1))
        elif a.ndim == 2:
            rows.append(a)
    M = np.hstack(rows)
    np.savetxt(filename, M, fmt="%.6f")


def write_pcd_minimal(filename: str, core: Mapping[str, np.ndarray], attrs: Mapping[str, np.ndarray], *, allow_lossy: bool):
    """
    Minimal PCD writer (xyz + optional rgb). For anything richer, prefer PLY.
    """
    pts = np.asarray(core["points"], dtype=np.float32)
    N = pts.shape[0]

    rgb = None
    if "colors" in attrs:
        c = np.asarray(attrs["colors"])
        if c.shape == (N,3):
            if c.dtype.kind == "f":
                c = np.clip(np.round(c*255), 0, 255).astype(np.uint8)
            else:
                c = np.clip(c, 0, 255).astype(np.uint8)
            # pack to single float RGB as PCL expects
            packed = (c[:,0].astype(np.uint32) << 16) | (c[:,1].astype(np.uint32) << 8) | c[:,2].astype(np.uint32)
            rgb = packed.view(np.float32)

    if (len(attrs) > (1 if rgb is not None else 0)) and not allow_lossy:
        raise ValueError("PCD writer is minimal (xyz+rgb). Set allow_lossy=True to drop other attributes.")

    # Write simple ASCII PCD
    fields = ["x","y","z"] + (["rgb"] if rgb is not None else [])
    sizes  = [4,4,4] + ([4] if rgb is not None else [])
    types  = ["F","F","F"] + (["F"] if rgb is not None else [])
    counts = [1,1,1] + ([1] if rgb is not None else [])
    header = [
        "# .PCD v0.7 - Point Cloud Data file format",
        "VERSION 0.7",
        f"FIELDS {' '.join(fields)}",
        f"SIZE {' '.join(map(str, sizes))}",
        f"TYPE {' '.join(types)}",
        f"COUNT {' '.join(map(str, counts))}",
        f"WIDTH {N}",
        "HEIGHT 1",
        "VIEWPOINT 0 0 0 1 0 0 0",
        f"POINTS {N}",
        "DATA ascii"
    ]
    with open(filename, "w") as f:
        f.write("\n".join(header) + "\n")
        if rgb is None:
            np.savetxt(f, pts, fmt="%.6f %.6f %.6f")
        else:
            M = np.c_[pts, rgb]
            np.savetxt(f, M, fmt="%.6f %.6f %.6f %.8e")

def _best_dtype_for_scalar(a: np.ndarray) -> str:
    """Choose a compact PLY dtype for a 1-D array."""
    if a.dtype.kind in "f":
        return "f8" if a.dtype.itemsize > 4 else "f4"
    if a.dtype.kind == "u":
        # choose smallest that fits
        mx = a.max(initial=0)
        if mx <= 255: return "u1"
        if mx <= 65535: return "u2"
        return "u4"
    if a.dtype.kind == "i":
        mn, mx = a.min(initial=0), a.max(initial=0)
        if -128 <= mn and mx <= 127: return "i1"
        if -32768 <= mn and mx <= 32767: return "i2"
        return "i4"
    # fallback to float
    return "f4"

def _numpy_dtype_from_ply(ply_dt: str):
    return {
        "f4": np.float32, "f8": np.float64,
        "u1": np.uint8, "u2": np.uint16, "u4": np.uint32,
        "i1": np.int8,  "i2": np.int16,  "i4": np.int32,
    }[ply_dt]

def _suffixes_for_k(k: int) -> list[str]:
    if k == 2: return ["x","y"]
    if k == 3: return ["x","y","z"]
    return [str(i) for i in range(k)]


def exportDistribution(var):
    params = var.generic_params()
    return [int(params.id), params.loc, params.scale, params.rel_min, params.rel_max]


def exportSlope(filepath: str, slope: Slope):
    nodes = slope.nodes
    if slope.hasUncertainty:
        nodes_std = slope.nodes_std
    else:
        nodes_std = np.zeros_like(nodes)
    out = np.empty((nodes.shape[0], 5), dtype=nodes.dtype)
    out[:, 0:4:2] = nodes  # columns 0,2
    out[:, 1:4:2] = nodes_std  # columns 1,3
    out[:-1, 4] = slope.materialIDs; out[-1, 4] = -1
    np.savetxt(filepath, out, fmt=["%.6f"]*4+["%d"], delimiter=",")


def importSlope(filepath: str, materials: List[Material]) -> Slope:
    data = np.loadtxt(filepath, delimiter=",")
    nodes = data[:, 0:4:2]
    nodes_std = data[:, 1:4:2]
    materialIDs = data[:-1, 4].astype(int)
    return Slope(Geometry(nodes=nodes, nodes_std=nodes_std), materials=materials, materialIDs=materialIDs)


def exportSeeders(filepath: str, seeders: List[Seeder], rockTypes: List[Rock]):
    with open(filepath, "w", encoding="utf-8") as f:
        for i, s in enumerate(seeders):
            m = s.rockThrowMode.value
            n = s.numberOfRocks
            v = s.translationalVelocity
            w = s.angularVelocity
            p = np.asarray(s.points, dtype=float)
            # Compute rock ids (indices into rockTypes)
            rock_ids: List[int] = []
            for r in s.rocks:
                try:
                    rock_ids.append(rockTypes.index(r))
                except ValueError as e:
                    raise ValueError(f"Seeder[{i}] contains a rock not present in rockTypes: {r!r}") from e
            f.write(f"{m} {n}\n")
            f.write(" ".join(map(str, exportDistribution(v[0]))) + "\n")
            f.write(" ".join(map(str, exportDistribution(v[1]))) + "\n")
            f.write(" ".join(map(str, exportDistribution(w[0]))) + "\n")
            f.write(f"{p.shape[1]}\n")
            np.savetxt(f, p.T, fmt="%.16g", delimiter=" ")
            f.write(" ".join(map(str, rock_ids)) + "\n")


def importSeeders(filepath: str, rockTypes: List[Rock]) -> List[Seeder]:
    mkdist  = lambda s: (p:=s.split()) and stats.makeDistribution(stats.DistributionParameters.relative(
                    id=int(float(p[0])), loc=float(p[1]), scale=float(p[2]), rel_min=float(p[3]), rel_max=float(p[4])
                )) or (_ for _ in ()).throw(ValueError(f"Bad dist line: {s!r}"))
    out, ln = [], [l.strip() for l in open(filepath, "r", encoding="utf-8") if l.strip()]
    i = 0
    while i < len(ln):
        m, n = ln[i].split()[:2]; i += 1
        m, n = SeederRocksThrown(int(m)), int(n)
        vh, vv, w = mkdist(ln[i]), mkdist(ln[i+1]), mkdist(ln[i+2]); i += 3
        num_points = int(ln[i]); i += 1
        pts = []
        for _ in range(num_points):
            pts.append([float(x) for x in ln[i].split()]); i += 1
        if i >= len(ln): raise ValueError("Missing rock_ids line.")
        rock_ids = [int(x) for x in ln[i].split()]; i += 1
        p = np.asarray(pts, float).T
        rocks = [rockTypes[j] for j in rock_ids]
        s = Seeder(p, rocks)
        s.rockThrowMode = m
        s.numberOfRocks = n
        s.translationalVelocity = [vh,vv]
        s.angularVelocity = [w]
        out.append(s)
    return out


def _indentation(level: int) -> str:
    return '  '*level


def _write(file_id, text, indentation=0):
    file_id.write(_indentation(indentation) + text + '\n')


def _valueToStr(value: Any) -> str:
    if isinstance(value, bool):
        txt = 'yes' if value else 'no'
    elif isinstance(value, float):
        txt = f'{value:.6e}'
    elif isinstance(value, int):
        txt = f'{value}'
    elif isinstance(value, str):
        txt = f'"{value}"'
    elif isinstance(value, Iterable):
        txt = ', '.join([_valueToStr(v) for v in value])
    else:
        txt = f'"{value}"'
    return txt


def _materialProperties(materials: List[Material], colors: List[tuple[int, int, int]] | None = None) -> List[Mapping[str, Any]]:
    materials_props = []
    for id, mat in enumerate(materials):
        name  = mat.name
        if colors is None:
            rgb = (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))
        else:
            rgb = colors[id]
        mat_prop = {
            'name': name,
            'id': id,
            'uid': f'{{{uuid.uuid4()}}}',
            'color': rgb,
            'hatch_color': (0, 0, 0),
            'hatch_type': 'none',
            'normal_rest': exportDistribution(mat.normalRestitution),
            'tangential_rest': exportDistribution(mat.tangentialRestitution),
            'friction_angle': exportDistribution(mat.frictionAngle),
            'dynamic_friction': [2, 0.5, 0.04, 0.12, 0.12],
            'rolling_friction': [2, 0.15, 0.02, 0.06, 0.06],
            'slope_roughness': [2, 0.0, 0.0, 0.0, 0.0],
            'rb_slope_roughness': False,
            'rb_sr_spacing': [2, 1.0, 0.2, 0.6, 0.6],
            'rb_sr_amplitude': [2, 0.0, 0.2, 0.6, 0.6],
            'max_dynamic_friction': 2.0,
            'dynamic_beta': 185.0,
            'dynamic_kappa': 3.0,
            'dynamic_ground_drag': 0.4,
            'use_viscoplastic_damping': False,
            'use_advanced_friction': False,
            'material_type': 5,
            'use_forest': False,
            'effective_forest_height': 5.0,
            'forest_drag_coefficient': 500.0,
            'forest_type': 1
        }
        materials_props.append(mat_prop)
    return materials_props


def _properties_to_materials(materials: List[Mapping[str, Any]]) -> Tuple[List[Material], List[Tuple[int, int, int]]]:
    material_objects = []
    colors = []
    for mat in materials:
        name  = mat['name']
        normal_rest = stats.makeDistribution(
            stats.DistributionParameters.relative(
                id=mat['normal_rest'][0],
                loc=mat['normal_rest'][1],
                scale=mat['normal_rest'][2],
                rel_min=mat['normal_rest'][3],
                rel_max=mat['normal_rest'][4],
            )
        )
        tangential_rest = stats.makeDistribution(
            stats.DistributionParameters.relative(
                id=mat['tangential_rest'][0],
                loc=mat['tangential_rest'][1],
                scale=mat['tangential_rest'][2],
                rel_min=mat['tangential_rest'][3],
                rel_max=mat['tangential_rest'][4],
            )
        )
        friction_angle = stats.makeDistribution(
            stats.DistributionParameters.relative(
                id=mat['friction_angle'][0],
                loc=mat['friction_angle'][1],
                scale=mat['friction_angle'][2],
                rel_min=mat['friction_angle'][3],
                rel_max=mat['friction_angle'][4],
            )
        )
        slope_roughness = stats.makeDistribution(
            stats.DistributionParameters.relative(
                id=mat['slope_roughness'][0],
                loc=mat['slope_roughness'][1],
                scale=mat['slope_roughness'][2],
                rel_min=mat['slope_roughness'][3],
                rel_max=mat['slope_roughness'][4],
            )
        )
        colour = mat['color']
        colors.append(colour)
        material_objects.append(
            Material(
                name=name,
                normalRestitution=normal_rest,
                tangentialRestitution=tangential_rest,
                frictionAngle=friction_angle,
                roughness=slope_roughness,
            )
        )
    return material_objects, colors


def _writeMaterials_fal8(filepath: str, materials: List[Mapping[str, Any]]):
    with open(filepath, 'a') as f:
        _write(f, 'material definitions start:', indentation=1)
        _write(f, f'num of material defs: {len(materials)}', indentation=2)
        _write(f, 'start user id: 0', indentation=2)
        for id, material in enumerate(materials):
            _write(f, 'material definition start:', indentation=2)
            for attrib, value in material.items():
                attrib = attrib.replace('_', ' ')
                _write(f, f'{attrib}: {_valueToStr(value)}', indentation=3)
            _write(f, 'material definition end:', indentation=2)
        _write(f, 'material definitions end:', indentation=1)
        _write(f, 'barrier definitions start:', indentation=1)
        _write(f, 'num of barrier defs: 0', indentation=2)
        _write(f, 'start user id: 0', indentation=2)
        _write(f, 'barrier definitions end:', indentation=1)
        _write(f, 'berm definitions start:', indentation=1)
        _write(f, 'num of berm defs: 0', indentation=2)
        _write(f, 'start user id: 0', indentation=2)
        _write(f, 'berm definitions end:', indentation=1)
        _write(f, 'barrier properties definitions start:', indentation=1)
        _write(f, 'num of barrier properties defs: 0', indentation=2)
        _write(f, 'start user id: 0', indentation=2)
        _write(f, 'barrier properties definitions end:', indentation=1)
        _write(f, 'custom_shapes start:', indentation=1)
        _write(f, 'custom_shapes end:', indentation=1)


def _readMaterials_fal8(filepath: str) -> List[Mapping[str, Any]]:
    with open(filepath, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                raise ValueError('Material definitions not found in file!')
            if 'material definitions start:' in line:
                break
        
        num_materials = int(f.readline().strip()[22:])
        materials = []
        assert 'start user id:' in f.readline()
        for m in range(num_materials):
            assert 'material definition start:' in f.readline()
            name = f.readline().strip()[7:-1]
            id = int(f.readline().strip()[4:])
            material = {}
            material['name'] = name
            material['uid'] = f.readline().strip()[6:-1]
            color = f.readline().strip()[7:].split(',')
            material['color'] = (int(color[0]), int(color[1]), int(color[2]))
            hatch_color = f.readline().strip()[13:].split(',')
            material['hatch_color'] = (int(hatch_color[0]), int(hatch_color[1]), int(hatch_color[2]))
            material['hatch_type'] = f.readline().strip()[13:-1]
            for _ in range(6):
                attrib, values = f.readline().strip().split(':')
                attrib = attrib.replace(' ', '_')
                values = values.split(',')
                material[attrib] = [int(values[0])] + [float(v) for v in values[1:]]
            attrib, values = f.readline().strip().split(':')
            attrib = attrib.replace(' ', '_')
            material[attrib] = True if 'yes' in values else False
            for _ in range(2):
                attrib, values = f.readline().strip().split(':')
                attrib = attrib.replace(' ', '_')
                values = values.split(',')
                material[attrib] = [int(values[0])] + [float(v) for v in values[1:]]
            for _ in range(11):
                attrib, value = f.readline().strip().split(':')
                is_int = True if 'type' in attrib else False
                attrib = attrib.replace(' ', '_')
                if 'no' in value or 'yes' in value:
                    value = True if 'yes' in value else False
                else:
                    if is_int:
                        value = int(value)
                    else:
                        value = float(value)
                material[attrib] = value
            assert 'material definition end:' in f.readline()
            materials.append(material)
        assert 'material definitions end:' in f.readline()
    return materials


def exportMaterials(filepath: str, materials: List[Material], colors: List[Tuple[int, int, int]] | None = None):
    materials_props = _materialProperties(materials, colors)
    with open(filepath, 'w') as f:
        _write(f, 'RocFall materials:', indentation=0)
    _writeMaterials_fal8(filepath, materials_props)


def importMaterials(filepath: str) -> Tuple[List[Material], List[Tuple[int, int, int]]]:
    return _properties_to_materials(_readMaterials_fal8(filepath))


def _rocksProperties(rock_types: List[Rock], colors: List[tuple[int, int, int]] | None = None) -> List[Mapping[str, Any]]:
    rocks_props = []
    for id, rock in enumerate(rock_types):
        density = exportDistribution(rock.density)
        mass = exportDistribution(rock.mass)
        if colors is None:
            rgb = (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))
        else:
            rgb = colors[id]
        rocks_props.append({
            'name': rock.name,
            'id': id,
            'color': rgb,
            'mass': mass,
            'density': density,
            'number_of_custom_polygons': 0,
            'shape_types': [True] + [False]*29
        })
    return rocks_props


def _properties_to_rocks(rocks: List[Mapping[str, Any]]) -> Tuple[List[Rock], List[Tuple[int, int, int]]]:
    rocks_objects = []
    colors = []
    for rock in rocks:
        name = rock['name']
        density = stats.makeDistribution(
            stats.DistributionParameters.relative(
                id=rock['density'][0],
                loc=rock['density'][1],
                scale=rock['density'][2],
                rel_min=rock['density'][3],
                rel_max=rock['density'][4]
            )
        )
        mass = stats.makeDistribution(
            stats.DistributionParameters.relative(
                id=rock['mass'][0],
                loc=rock['mass'][1],
                scale=rock['mass'][2],
                rel_min=rock['mass'][3],
                rel_max=rock['mass'][4]
            )
        )
        rocks_objects.append(
            Rock(
                name=name,
                density=density,
                mass=mass
            )
        )
        colors.append(rock['color'])
    return rocks_objects, colors


def _writeRocks_fal8(filepath: str, rocks: List[Mapping[str, Any]]):
    with open(filepath, 'a') as f:
        _write(f, 'rock definitions start:', indentation=1)
        _write(f, f'num of rock defs: {len(rocks)}', indentation=2)
        _write(f, 'start user id: 0', indentation=2)
        for id, rock in enumerate(rocks):
            _write(f, 'rock definition start:', indentation=2)
            for attrib, value in rock.items():
                attrib = attrib.replace('_', ' ')
                if attrib == 'shape types':
                    _write(f, 'shape type start:', indentation=3)
                    for s_id, shape in enumerate(value):
                        _write(f, f'shape {s_id}: {_valueToStr(shape)}', indentation=4)
                    _write(f, 'shape type end:', indentation=3)
                else:
                    _write(f, f'{attrib}: {_valueToStr(value)}', indentation=3)
            _write(f, 'rock definition end:', indentation=2)
        _write(f, 'rock definitions end:', indentation=1)


def _readRocks_fal8(filepath: str) -> List[Mapping[str, Any]]:
    with open(filepath, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                raise ValueError('Rock definitions not found in file!')
            if 'rock definitions start:' in line:
                break

        num_rocks = int(f.readline().strip()[18:])
        rocks = []
        assert 'start user id:' in f.readline()
        for r in range(num_rocks):
            assert 'rock definition start:' in f.readline()
            name = f.readline().strip()[7:-1]
            id = int(f.readline().strip()[4:])
            rock = {}
            rock['name'] = name
            color = f.readline().strip()[7:].split(',')
            rock['color'] = (int(color[0]), int(color[1]), int(color[2]))
            for _ in range(2):
                attrib, values = f.readline().strip().split(':')
                attrib = attrib.replace(' ', '_')
                values = values.split(',')
                rock[attrib] = [int(values[0])] + [float(v) for v in values[1:]]

            num_polygons = int(f.readline().strip()[27:])
            assert 'shape type start:' in f.readline()
            shape_types = []
            while True:
                line = f.readline()
                if 'shape type end:' in line:
                    break
                value = True if 'yes' in line else False
                shape_types.append(value)
            rock['shape_types'] = shape_types
            assert 'rock definition end:' in f.readline()
            rocks.append(rock)
        assert 'rock definitions end:' in f.readline()
    return rocks


def exportRocks(filepath: str, rocks: List[Rock], colors: List[tuple[int, int, int]] | None = None):
    rocks_props = _rocksProperties(rocks, colors)
    with open(filepath, 'w') as f:
        _write(f, 'RocFall rock types:', indentation=0)
    _writeRocks_fal8(filepath, rocks_props)


def importRocks(filepath: str) -> Tuple[List[Rock], List[Tuple[int, int, int]]]:
    return _properties_to_rocks(_readRocks_fal8(filepath))


def _settings(analysis: Analysis) -> Mapping[str, Any]:
    settings = {
        'rock_throw_mode': 'CONTROL SEEDER',
        'rocks_throw': 100,
        'slope_material_sample_mode': 'SAMPLE PER SIMULATION',
        'save_and_load_results': False,
        'first_collector': False,
        'seed_mode': 'Pseudo Random Seed',
        'sample_method': 'Latin Hypercube',
        'use_specific_seed': False,
        'specific_seed': 12345234,
        'seeder_probably_length': False,
        'engine': 'Lump Mass',
        'units': 'Metric',
        'engine_max_steps': 20000,
        'engine_normal_velocity': 0.1,
        'engine_stopped_velocity': 0.001,
        'engine_timestep': 0.01,
        'engine_use_scale_rn_velocity': True,
        'engine_scale_rn_velocity_k': 9.144,
        'engine_use_scale_rn_mass': False,
        'engine_use_scan_rn_mass_c': 1000,
        'engine_use_angular_velocity': True,
        'engine_friction_angle_mode': 'from dialog',
        'engine_use_kyle_fudge': True,
        'engine_switch_velocity': -1e-09
    }
    if analysis.rockThrowMode == AnalysisRocksThrown.IndividuallyPerSeeder:
        settings['rock_throw_mode'] = 'CONTROL SEEDER'
    elif analysis.rockThrowMode == AnalysisRocksThrown.DistributedFromNumberOfRocks:
        settings['rock_throw_mode'] = 'CONTROL TOTAL'
    settings['rocks_throw'] = min(max(analysis.numberOfRocks, 1), 10_000)
    if analysis.samplingMethod == Sampling.LatinHypercube:
        settings['sample_method'] = 'Latin Hypercube'
    elif analysis.samplingMethod == Sampling.MonteCarlo:
        settings['sample_method'] = 'Monte Carlo'
    settings['use_specific_seed'] = analysis.useSpecificSeed
    settings['specific_seed'] = analysis.specificSeed
    settings['engine_max_steps'] = analysis.maxIter
    settings['engine_normal_velocity'] = analysis.normalVelocityThreshold
    settings['engine_stopped_velocity'] = min(max(analysis.stoppedVelocity, 0.001), 10)
    settings['engine_timestep'] = analysis.timeStep
    settings['engine_use_scale_rn_velocity'] = analysis.scaleByVelocity
    settings['engine_scale_rn_velocity_k'] = analysis.K
    settings['engine_use_scale_rn_mass'] = analysis.scaleByMass
    settings['engine_use_scan_rn_mass_c'] = analysis.C
    settings['engine_use_angular_velocity'] = analysis.considerRotationalVelocity
    return settings


def _analysis(settings: Mapping[str, Any]) -> Analysis:
    analysis = Analysis()
    analysis.rockThrowMode = AnalysisRocksThrown.IndividuallyPerSeeder if settings['rock_throw_mode'] == 'CONTROL SEEDER' else AnalysisRocksThrown.DistributedFromNumberOfRocks
    analysis.numberOfRocks = settings['rocks_throw']
    analysis.samplingMethod = Sampling.LatinHypercube if settings['sample_method'] == 'Latin Hypercube' else Sampling.MonteCarlo
    analysis.useSpecificSeed = settings['use_specific_seed']
    analysis.specificSeed = settings['specific_seed']
    analysis.maxIter = settings['engine_max_steps']
    analysis.normalVelocityThreshold = settings['engine_normal_velocity']
    analysis.stoppedVelocity = settings['engine_stopped_velocity']
    analysis.timeStep = settings['engine_timestep']
    analysis.scaleByVelocity = settings['engine_use_scale_rn_velocity']
    analysis.K = settings['engine_scale_rn_velocity_k']
    analysis.scaleByMass = settings['engine_use_scale_rn_mass']
    analysis.C = settings['engine_use_scan_rn_mass_c']
    analysis.considerRotationalVelocity = settings['engine_use_angular_velocity']
    return analysis


def _writeSettings_fal8(filepath: str, settings: Mapping[str, Any]):
    with open(filepath, 'a') as f:
        for attrib, value in settings.items():
            if attrib.startswith('engine_'):
                continue
            attrib = attrib.replace('_', ' ')  # Replace underscores with spaces
            _write(f, f'{attrib}: {_valueToStr(value)}', indentation=1)
        _write(f, 'engine settings start:', indentation=1)
        for attrib, value in settings.items():
            if attrib.startswith('engine_'):
                attrib = attrib.replace('_', ' ')  # Replace underscores with spaces
                attrib = attrib.replace('engine ', '')  # Remove "engine" prefix
                _write(f, f'{attrib}: {_valueToStr(value)}', indentation=2)
        _write(f, 'engine settings end:', indentation=1)


def _readSettings_fal8(filepath: str) -> Mapping[str, Any]:
    with open(filepath, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                raise ValueError('Project settings not found in file!')
            if 'project settings:' in line:
                break
        settings = {}
        while True:
            line = f.readline()
            if 'engine settings' in line:
                break
            if not line:
                break
            attrib, value = line.strip().split(':')
            attrib = attrib.replace(' ', '_')  # Replace spaces with underscores
            value = value[1:]  # Remove leading space
            value = value.replace('"', '')  # Remove quotes
            try:
                value_i = int(value)
                value_f = float(value)
                value = value_i if value_i - value_f == 0.0 else value_f
            except ValueError:
                if value == 'yes' or value == 'no':
                    value = True if value == 'yes' else False
            settings[attrib] = value
        while True:
            line = f.readline()
            if 'engine settings end:' in line:
                break
            attrib, value = line.strip().split(':')
            attrib = attrib.replace(' ', '_')  # Replace spaces with underscores
            attrib = 'engine_' + attrib
            value = value[1:]  # Remove leading space
            value = value.replace('"', '')  # Remove quotes
            try:
                value_f = float(value)
                value_i = int(value_f)
                value = value_i if value_i - value_f == 0.0 else value_f
            except ValueError:
                if value == 'yes' or value == 'no':
                    value = True if value == 'yes' else False
            settings[attrib] = value
    return settings


def exportSettings(filepath: str, analysis: Analysis):
    materials_props = _settings(analysis)
    with open(filepath, 'w') as f:
        _write(f, 'RocFall project settings:', indentation=0)
    _writeSettings_fal8(filepath, materials_props)

def importSettings(filepath: str) -> Analysis:
    return _analysis(_readSettings_fal8(filepath))
