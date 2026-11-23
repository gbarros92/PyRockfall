"""
Prepare a wall for 2D rockfall simulations: rock profiles with their uncertainty
under the vegetation, and the canopy above them.

Slicing the original cloud gives the exposed rock and, where it is covered, the
vegetation. This script:

1. Loads the original cloud (binary PLY is memory-mapped, so a cloud of 100M+
   points is fine), optionally thinned on a voxel grid for the kriging.
2. Fits the curvilinear frame on it (or loads one with --frame). Every cloud
   below is cut in this one frame.
3. Removes the vegetation and fills the holes by kriging (as fill_holes.py):
   the mean surface, the std of its height (kriging_std) and the material
   probabilities. The vegetation of the full-resolution cloud is the cover: the
   rock must lie behind its deepest returns (fillHoles(cover=...)), so the
   kriging cannot bridge across recesses that the vegetation fills. Cached in
   <out>/filled.ply, filled_info.json.
4. Chooses the stations: s positions every --spacing along the strike.
5. Mean profiles: slices the filled cloud at the stations. At every node, the
   kriging std is interpolated and turned into nodes_std in the profile plane.
6. Realisations: draws N realisations of the filled cloud (pr.sampleRealisations)
   and moves every node of the mean profile along the frame's w axis by the
   realisation's displacement there, then back onto the plane of its station.
   Every realisation profile has the same nodes and elements as the mean one, so
   they can be the samples of one Geometry (Geometry(..., realisations=...)).
   Nodes on exposed rock do not move. The sampled material of every element is
   saved too.
7. Canopy: from the vegetation points of the original cloud (full resolution),
   at the same stations and in the same frame. Per bin up the face, the
   outermost vegetation point (a high quantile, to skip stray points) gives the
   canopy, kept at least --canopy-gap outside the most outward realisation of the
   rock. Each run of vegetation is one polyline, closed down to the mean rock at
   both ends unless --open-canopy.
8. Saves one file per station, all in the same 2D coordinates: (d, e) of the
   frame minus the toe of the mean profile. Also a summary, a 3D point cloud of
   all profiles (to check in CloudCompare that they coincide) and plots.

Outputs (in <input>_simulation/ unless --out)
---------------------------------------------
    frame.json                  the frame everything is cut in
    filled.ply, filled_info.json   the kriged cloud and its variograms (cache)
    stations/station_####.npz   per station (see save_station)
    summary.json                per-station statistics and settings
    sections_qa.ply             every profile node in 3D: source (0 mean,
                                1 realisation, 2 canopy), Profile, realisation
    plots/station_####.png      with --plot-every

Use the stations with build_simulation() (bottom of this file).

Usage
-----
    python prepare_simulation.py model_3holes.ply -N 50
    python prepare_simulation.py susan.ply --voxel 0.15 -N 100 --plot-every 20
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial import Delaunay, cKDTree

import pyrockfall as pr

MATERIAL = "Material"


# =============================================================================
# PLY input
# =============================================================================

_PLY_TYPES = {"float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
              "uchar": "u1", "uint8": "u1", "char": "i1", "int8": "i1",
              "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
              "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4"}


def open_ply(path) -> np.memmap:
    """Memory-map the vertices of a binary little-endian PLY (vertex element only)."""
    with open(path, "rb") as f:
        if f.readline().strip() != b"ply":
            raise ValueError(f"{path} is not a PLY file")
        fields, n = [], None
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: no end_header")
            words = line.decode("ascii", "replace").split()
            if not words:
                continue
            if words[0] == "format" and words[1] != "binary_little_endian":
                raise ValueError(f"{path}: only binary little-endian PLY can be memory-mapped")
            if words[0] == "element":
                if words[1] != "vertex":
                    raise ValueError(f"{path}: only a vertex element is supported")
                n = int(words[2])
            if words[0] == "property":
                if words[1] == "list":
                    raise ValueError(f"{path}: list properties are not supported")
                fields.append((words[2], "<" + _PLY_TYPES[words[1]]))
            if words[0] == "end_header":
                offset = f.tell()
                break
    return np.memmap(path, dtype=np.dtype(fields), mode="r", offset=offset, shape=(n,))


def field_name(names, label: str) -> str:
    """`label`, or CloudCompare's ``scalar_<label>``."""
    for cand in (label, f"scalar_{label}"):
        if cand in names:
            return cand
    raise ValueError(f"No field '{label}' (or 'scalar_{label}') in {names}")


def xyz_of(ply, idx=slice(None)) -> np.ndarray:
    return np.column_stack([np.asarray(ply[k][idx], dtype=np.float64) for k in ("x", "y", "z")])


def voxel_thin(xyz: np.ndarray, voxel: float) -> np.ndarray:
    """Indices of one point per voxel (the first one)."""
    keys = np.floor((xyz - xyz.min(axis=0)) / voxel).astype(np.int64)
    dims = keys.max(axis=0) + 1
    flat = (keys[:, 0] * dims[1] + keys[:, 1]) * dims[2] + keys[:, 2]
    _, first = np.unique(flat, return_index=True)
    return np.sort(first)


def load_input(path: Path, label: str, voxel: float | None, normals: bool = False) -> pr.PointCloud:
    """The cloud to fill: xyz, the material and, if asked and present, the normals."""
    ply = open_ply(path)
    names = ply.dtype.names
    mat_name = field_name(names, label)
    idx = slice(None)
    if voxel:
        idx = voxel_thin(xyz_of(ply), voxel)
    pc = pr.PointCloud(points=xyz_of(ply, idx))
    pc.set_attr(label, np.asarray(ply[mat_name][idx], dtype=np.float64))
    if normals and all(k in names for k in ("nx", "ny", "nz")):
        pc.set_attr("normals", np.column_stack([np.asarray(ply[k][idx], dtype=np.float64) for k in ("nx", "ny", "nz")]))
    return pc


def load_filled(path: Path, label: str) -> pr.PointCloud:
    """A cloud written by stage 3, with the attributes the realisations need."""
    ply = open_ply(path)
    pc = pr.PointCloud(points=xyz_of(ply))
    keep = [n for n in ply.dtype.names
            if n in ("filled", "hole_id", "kriging_std", "cover_bound", "cover_free_w", label) or n.startswith(f"{label}_P")]
    for n in keep:
        pc.set_attr(n, np.asarray(ply[n], dtype=np.float64))
    return pc


def _to_json(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(type(o).__name__)


# =============================================================================
# Geometry helpers in the frame
# =============================================================================

def outward_sign(frame) -> float:
    """+1 if +w points out of the wall (into the air), else -1."""
    if abs(frame.pca_axes[2, 2]) < 0.2:
        raise ValueError("The face is nearly vertical: the outward side of w cannot be read from the frame.")
    return frame.outward_sign()


def to_uvw_chunked(frame, ply, idx, chunk: int = 2_000_000) -> np.ndarray:
    """(u, v, w) of the points `idx` of a memory-mapped PLY, a chunk at a time."""
    out = np.empty((idx.size, 3))
    for a in range(0, idx.size, chunk):
        out[a:a + chunk] = frame.xyz_to_uvw(xyz_of(ply, idx[a:a + chunk]))
    return out


def deepest_cover(path: Path, label: str, veg_ids, frame, cell: float) -> np.ndarray:
    """(x, y, z) of the deepest vegetation return in each (u, v) cell of size `cell`:
    the tightest bounds on the rock behind the vegetation."""
    ply = open_ply(path)
    mat = np.asarray(ply[field_name(ply.dtype.names, label)])
    veg = np.flatnonzero(np.isin(mat, veg_ids))
    del mat
    uvw = to_uvw_chunked(frame, ply, veg)
    uvw = uvw[np.isfinite(uvw).all(axis=1)]
    depth = outward_sign(frame) * uvw[:, 2]                      # smaller = deeper
    key = np.floor(uvw[:, :2] / cell).astype(np.int64)
    key = (key[:, 0] - key[:, 0].min()) * (key[:, 1].max() - key[:, 1].min() + 1) + (key[:, 1] - key[:, 1].min())
    order = np.lexsort((depth, key))
    _, first = np.unique(key[order], return_index=True)
    return frame.uvw_to_xyz(uvw[order[first]])


def vw_to_de(frame, s_k: float, v: np.ndarray, w: np.ndarray) -> np.ndarray:
    """(d, e) of the points (v, w) on the plane s = s_k: u is solved so that s = s_k."""
    A, m = frame.pca_axes, frame.pca_mean
    u = (s_k - m[0] - v * A[1, 0] - w * A[2, 0]) / A[0, 0]
    return frame.uvw_to_sde(np.column_stack([u, v, w]))[:, 1:]


class SurfaceInterpolator:
    """Linear interpolation, in (u, v), of values given at the filled points.

    Built on the filled (uncertain) points plus a ring of certain points around
    them, where the values are 0, so interpolated values fade to 0 at the edge of
    each hole. Queries farther than `reach` from any filled point get 0.
    """

    def __init__(self, uv_unc: np.ndarray, uv_certain: np.ndarray, spacing: float):
        self.n_unc = uv_unc.shape[0]
        self.reach = 1.5 * spacing
        self.tree_unc = cKDTree(uv_unc)
        dist, _ = self.tree_unc.query(uv_certain, k=1, distance_upper_bound=3.0 * spacing)
        ring = uv_certain[np.isfinite(dist)]
        self.n_ring = ring.shape[0]
        self.tri = Delaunay(np.vstack([uv_unc, ring]))

    def weights(self, uv: np.ndarray):
        """Vertices (n, 3) and barycentric weights (n, 3) of each query; weights
        are 0 where the query is off the filled region."""
        simp = self.tri.find_simplex(uv)
        near = self.tree_unc.query(uv, k=1, distance_upper_bound=self.reach)[0]
        ok = (simp >= 0) & np.isfinite(near)
        T = self.tri.transform[np.maximum(simp, 0)]
        b = np.einsum("nij,nj->ni", T[:, :2], uv - T[:, 2])
        bary = np.column_stack([b, 1.0 - b.sum(axis=1)])
        bary[~ok] = 0.0
        return self.tri.simplices[np.maximum(simp, 0)], bary

    def __call__(self, values_unc: np.ndarray, verts, bary) -> np.ndarray:
        """Interpolate `values_unc` (..., n_unc) given at the filled points."""
        vals = np.concatenate([values_unc, np.zeros(values_unc.shape[:-1] + (self.n_ring,))], axis=-1)
        return (vals[..., verts] * bary).sum(axis=-1)


# =============================================================================
# Stages
# =============================================================================

def stage_fill(args, out: Path):
    """Stages 1-3: the frame and the filled cloud (cached)."""
    filled_path, info_path, frame_path = out / "filled.ply", out / "filled_info.json", out / "frame.json"
    if filled_path.exists() and info_path.exists() and frame_path.exists() and not args.refill:
        t0 = time.perf_counter()
        frame = pr.CurvilinearFrame.load(frame_path)
        filled = load_filled(filled_path, args.material)
        filled.frame = frame
        info = json.loads(info_path.read_text())
        print(f"[fill] reused {filled_path.name} ({filled.points.shape[0]} points) and {frame_path.name} "
              f"({time.perf_counter() - t0:.1f} s); --refill to redo")
        return frame, filled, info

    t0 = time.perf_counter()
    pc = load_input(args.input, args.material, args.voxel, normals=args.frame_normals)
    print(f"[fill] loaded {args.input.name}: {pc.points.shape[0]} points"
          f"{f' (one per {args.voxel:g} m voxel)' if args.voxel else ''} ({time.perf_counter() - t0:.1f} s)")

    t0 = time.perf_counter()
    if args.frame:
        frame = pr.CurvilinearFrame.load(args.frame)
        pc.frame = frame
        if frame.pca_axes is None:
            pc.uvw()
        print(f"[fill] frame from {args.frame}")
    else:
        frame = pc.fitFrame()
        pc.uvw()                                  # principal axes of (s, d, e)
        print(f"[fill] frame fitted: {frame.length:.1f} m, {frame.n_segments} segments "
              f"({time.perf_counter() - t0:.1f} s)")
    diag = frame.diagnostics(pc.points)
    print(f"[fill] {diag['n_nan']} points outside the frame, {diag['n_ambiguous']} ambiguous")
    frame.save(frame_path)

    n_removed = pc.removePoints(args.material, args.veg_ids)
    print(f"[fill] removed {n_removed} vegetation points ({args.material} in {args.veg_ids})")
    cover_kw = {}
    if not args.no_cover:
        t0 = time.perf_counter()
        cover = deepest_cover(args.input, args.material, args.veg_ids, frame, args.cover_cell)
        cover_kw = dict(cover=cover, cover_side=outward_sign(frame), cover_margin=args.cover_margin)
        print(f"[fill] cover: deepest vegetation return in each {args.cover_cell:g} m cell, "
              f"{cover.shape[0]} points ({time.perf_counter() - t0:.1f} s)")
    t0 = time.perf_counter()
    filled, info = pc.fillHoles(attributes=[args.material], return_info=True, **cover_kw)
    print(f"[fill] {len(info['holes'])} holes, {info['n_new']} new points ({time.perf_counter() - t0:.1f} s)")
    for h in info["holes"]:
        if "skipped" in h:
            print(f"       hole {h['hole']}: skipped ({h['skipped']})")
    if not args.no_cover:
        n_b = sum(h.get("n_cover_clipped", 0) for h in info["holes"])
        n_a = sum(h.get("n_cover_active", 0) for h in info["holes"])
        print(f"[fill] cover: {n_b} of {info['n_new']} new points bounded, {n_a} bounds active in the kriging")
    filled.frame = frame
    if filled.has_attr("normals"):
        filled.del_attr("normals")
    filled.save(str(filled_path), overwrite=True)
    info_path.write_text(json.dumps(info, default=_to_json))
    # back from disk, so a rerun from the cache gives the same result
    filled = load_filled(filled_path, args.material)
    filled.frame = frame
    info = json.loads(info_path.read_text())
    return frame, filled, info


def stage_stations(args, frame, filled) -> np.ndarray:
    """Stage 4: the s positions of the profiles."""
    s = frame.xyz_to_sde(filled.points)[:, 0]
    s = s[np.isfinite(s)]
    lo, hi = (args.s_range if args.s_range else (s.min(), s.max()))
    stations = np.arange(lo + 0.5 * args.spacing, hi, args.spacing)
    print(f"[stations] {stations.size} stations every {args.spacing:g} m, s = {stations[0]:.2f} .. {stations[-1]:.2f}")
    return stations


def stage_mean_profiles(args, frame, filled, stations):
    """Stage 5: mean profiles of the filled cloud, with nodes_std."""
    t0 = time.perf_counter()
    profiles, s_ok = filled.slice(None, args.slab, args.material, frame=frame, positions=stations,
                                  slice_height_increment=args.node_step, method="interp")
    print(f"[mean] {len(profiles)} of {stations.size} stations have a profile ({time.perf_counter() - t0:.1f} s)")
    out = []
    for prof, s_k in zip(profiles, s_ok):
        de = prof.nodes.copy()
        if de[0, 1] < de[-1, 1]:                  # crest first
            de = de[::-1]
            mats = np.asarray(prof.attributes)[::-1]
        else:
            mats = np.asarray(prof.attributes)
        uvw = frame.sde_to_uvw(np.column_stack([np.full(len(de), s_k), de]))
        out.append(dict(s=float(s_k), de=de, uvw=uvw, materials=mats.astype(np.int64)))
    return out


def stage_realisations(args, frame, filled, info, stations_data):
    """Stage 6: realisation profiles with the nodes of the mean profiles."""
    sigma = np.asarray(filled.get_attr("kriging_std"), dtype=float)
    p_names = [n for n in filled.list_attrs() if n.startswith(f"{args.material}_P") and n != f"{args.material}_Pmax"]
    P = np.column_stack([np.asarray(filled.get_attr(n), dtype=float) for n in p_names])
    uncertain = (sigma > 0) | (P.max(axis=1) < 1.0 - 1e-9)       # as in sampleRealisations
    idx = np.flatnonzero(uncertain)
    uvw_all = frame.xyz_to_uvw(filled.points)
    uv_unc = uvw_all[idx, :2]
    cert = ~uncertain & np.isfinite(uvw_all).all(axis=1)
    spacing = float(info.get("spacing") or np.median(cKDTree(uv_unc).query(uv_unc[:5000], k=2)[0][:, 1]))
    interp = SurfaceInterpolator(uv_unc, uvw_all[cert, :2], spacing)
    tree_unc = interp.tree_unc

    # All nodes and element midpoints of all stations, interpolated at once
    sizes = [st["de"].shape[0] for st in stations_data]
    node_uvw = np.vstack([st["uvw"] for st in stations_data])
    verts, bary = interp.weights(node_uvw[:, :2])
    node_sigma = interp(sigma[idx], verts, bary)
    mid_uv = np.vstack([0.5 * (st["uvw"][:-1, :2] + st["uvw"][1:, :2]) for st in stations_data])
    d_mid, nn_mid = tree_unc.query(mid_uv, k=1, distance_upper_bound=interp.reach)
    mid_unc = np.isfinite(d_mid)
    nn_mid = np.where(mid_unc, nn_mid, 0)

    # nodes_std: the std of w, projected on (d, e)
    w_de = np.abs(frame.pca_axes[2, 1:])
    offsets = np.cumsum([0] + sizes)
    e_offsets = np.cumsum([0] + [n - 1 for n in sizes])
    for k, st in enumerate(stations_data):
        sig = node_sigma[offsets[k]:offsets[k + 1]]
        st["sigma_w"] = sig
        st["nodes_std"] = sig[:, None] * w_de[None, :]
        st["realisations"] = np.empty((args.n_realisations,) + st["de"].shape)
        st["realisation_materials"] = np.empty((args.n_realisations, sizes[k] - 1), dtype=np.int64)

    n_moving = int(np.count_nonzero(np.abs(bary).sum(axis=1) > 0))
    print(f"[realisations] {idx.size} uncertain points; {n_moving} of {node_uvw.shape[0]} profile nodes "
          f"on filled rock; drawing {args.n_realisations} realisations")

    seeds = np.random.SeedSequence(args.seed).generate_state(-(-args.n_realisations // args.batch))
    t0 = time.perf_counter()
    for b, start in enumerate(range(0, args.n_realisations, args.batch)):
        n_b = min(args.batch, args.n_realisations - start)
        reals = pr.sampleRealisations(filled, n_b, args.material, info=info, frame=frame,
                                      uncertain_only=True, seed=int(seeds[b]))
        DW = np.stack([np.asarray(r.get_attr("dw"), dtype=float) for r in reals])       # (n_b, n_unc)
        MAT = np.stack([np.asarray(r.get_attr(args.material)) for r in reals])           # (n_b, n_unc)
        dw_nodes = interp(DW, verts, bary)                                               # (n_b, n_nodes)
        mat_mid = MAT[:, nn_mid].astype(np.int64)                                        # (n_b, n_elems)
        for k, st in enumerate(stations_data):
            sl = slice(offsets[k], offsets[k + 1])
            moving = np.abs(bary[sl]).sum(axis=1) > 0
            el = slice(e_offsets[k], e_offsets[k + 1])
            for j in range(n_b):
                de = st["de"].copy()
                if moving.any():
                    v = st["uvw"][moving, 1]
                    w = st["uvw"][moving, 2] + dw_nodes[j, sl][moving]
                    de[moving] = vw_to_de(frame, st["s"], v, w)
                st["realisations"][start + j] = de
                st["realisation_materials"][start + j] = np.where(mid_unc[el], mat_mid[j, el], st["materials"])
        print(f"  {start + n_b}/{args.n_realisations} ({time.perf_counter() - t0:.1f} s)")
        del reals, DW, MAT

    # w of every realisation node (back from (d, e)), and its spread over the realisations
    for st in stations_data:
        R, N = st["realisations"].shape[:2]
        sde = np.column_stack([np.full(R * N, st["s"]), st["realisations"].reshape(-1, 2)])
        st["real_w"] = frame.sde_to_uvw(sde)[:, 2].reshape(R, N)
        st["spread_w"] = st["real_w"].std(axis=0) if R > 1 else np.zeros(N)


def stage_canopy(args, frame, stations_data):
    """Stage 7: canopy polylines from the vegetation of the original cloud."""
    t0 = time.perf_counter()
    ply = open_ply(args.canopy_source)
    mat = np.asarray(ply[field_name(ply.dtype.names, args.material)])
    veg = np.flatnonzero(np.isin(mat, args.veg_ids))
    if args.canopy_stride > 1:
        veg = veg[::args.canopy_stride]
    veg_class = mat[veg].astype(np.int64)
    del mat
    sde = frame.uvw_to_sde(to_uvw_chunked(frame, ply, veg))
    ok = np.isfinite(sde).all(axis=1)
    sde, veg_class = sde[ok], veg_class[ok]
    s_st = np.array([st["s"] for st in stations_data])
    order = np.argsort(sde[:, 0])
    s_sorted = sde[order, 0]
    print(f"[canopy] {veg.size} vegetation points from {Path(args.canopy_source).name} ({time.perf_counter() - t0:.1f} s)")

    out_sign = outward_sign(frame)
    h = args.node_step
    for st in stations_data:
        lo, hi = np.searchsorted(s_sorted, [st["s"] - 0.5 * args.slab, st["s"] + 0.5 * args.slab])
        rows = order[lo:hi]
        st["canopy_de"] = np.empty((0, 2))
        st["canopy_elements"] = np.empty((0, 2), dtype=np.int64)
        st["canopy_ids"] = np.empty(0, dtype=np.int64)
        st["n_veg"] = int(rows.size)
        if rows.size < args.canopy_min_points:
            continue
        uvw = frame.sde_to_uvw(sde[rows])
        v_nodes = st["uvw"][:, 1]
        v_lo, v_hi = v_nodes.min(), v_nodes.max()
        inside = (uvw[:, 1] >= v_lo) & (uvw[:, 1] <= v_hi)
        if inside.sum() < args.canopy_min_points:
            continue
        uvw, cls = uvw[inside], veg_class[rows][inside]
        bins = np.floor((uvw[:, 1] - v_lo) / h).astype(np.int64)
        n_bins = int(bins.max()) + 1
        o = out_sign * uvw[:, 2]                                   # distance outward along w
        # rock, outward, at the bin centres: mean and most outward realisation
        vc = v_lo + (np.arange(n_bins) + 0.5) * h
        asc = np.argsort(v_nodes)
        rock_mean = out_sign * np.interp(vc, v_nodes[asc], st["uvw"][asc, 2])
        rock_out = np.max([np.interp(vc, v_nodes[asc], out_sign * rw[asc]) for rw in st["real_w"]], axis=0)

        top = np.full(n_bins, np.nan)
        cls_bin = np.zeros(n_bins, dtype=np.int64)
        srt = np.argsort(bins, kind="stable")
        starts = np.searchsorted(bins[srt], np.arange(n_bins))
        ends = np.searchsorted(bins[srt], np.arange(n_bins), side="right")
        for i in range(n_bins):
            sel = srt[starts[i]:ends[i]]
            if sel.size < 3:
                continue
            top[i] = np.quantile(o[sel], args.canopy_quantile)
            vals, cnt = np.unique(cls[sel], return_counts=True)
            cls_bin[i] = vals[np.argmax(cnt)]
        has = np.isfinite(top) & (top - rock_mean > args.canopy_min_thickness)
        top = np.where(has, np.maximum(top, rock_out + args.canopy_gap), np.nan)

        # runs of bins with vegetation (bridging up to canopy_max_gap empty bins), crest (high v) first
        nodes, elems, ids = [], [], []
        filled_bins = np.flatnonzero(has)
        runs = np.split(filled_bins, np.flatnonzero(np.diff(filled_bins) > args.canopy_max_gap + 1) + 1)
        for run in runs[::-1]:
            run = run[::-1]
            if run.size < 2:
                continue
            v_run, o_run = vc[run], top[run]
            if not args.open_canopy:                               # down to the mean rock at both ends
                v_run = np.r_[v_run[0], v_run, v_run[-1]]
                o_run = np.r_[rock_mean[run[0]], o_run, rock_mean[run[-1]]]
                c_run = np.r_[cls_bin[run[0]], cls_bin[run]]
            else:
                c_run = cls_bin[run[:-1]]
            base = sum(len(n) for n in nodes)
            de = vw_to_de(frame, st["s"], v_run, out_sign * o_run)
            nodes.append(de)
            m = de.shape[0]
            elems.append(np.column_stack([np.arange(base, base + m - 1), np.arange(base + 1, base + m)]))
            ids.append(c_run[:m - 1])
        if nodes:
            st["canopy_de"] = np.vstack(nodes)
            st["canopy_elements"] = np.vstack(elems).astype(np.int64)
            st["canopy_ids"] = np.concatenate(ids).astype(np.int64)
    n_with = sum(1 for st in stations_data if st["canopy_elements"].size)
    print(f"[canopy] {n_with} of {len(stations_data)} stations have a canopy ({time.perf_counter() - t0:.1f} s)")


def rock_misfit(args, frame, stations_data, stride: int):
    """Median distance (along w) between the rock of the original cloud and the mean
    profile, at the nodes off the filled region: a check that both coincide."""
    ply = open_ply(args.canopy_source)
    mat = np.asarray(ply[field_name(ply.dtype.names, args.material)][::stride])
    rock = np.flatnonzero(~np.isin(mat, args.veg_ids)) * stride
    sde = frame.uvw_to_sde(to_uvw_chunked(frame, ply, rock))
    ok = np.isfinite(sde).all(axis=1)
    sde = sde[ok]
    order = np.argsort(sde[:, 0])
    s_sorted = sde[order, 0]
    for st in stations_data:
        lo, hi = np.searchsorted(s_sorted, [st["s"] - 0.25 * args.slab, st["s"] + 0.25 * args.slab])
        st["rock_misfit"] = np.nan
        if hi - lo < 10:
            continue
        uvw = frame.sde_to_uvw(sde[order[lo:hi]])
        fixed = st["sigma_w"] == 0
        if fixed.sum() < 2:
            continue
        v, w = st["uvw"][fixed, 1], st["uvw"][fixed, 2]
        asc = np.argsort(v)
        inside = (uvw[:, 1] > v.min()) & (uvw[:, 1] < v.max())
        if inside.sum() < 10:
            continue
        st["rock_misfit"] = float(np.median(np.abs(uvw[inside, 2] - np.interp(uvw[inside, 1], v[asc], w[asc]))))


# =============================================================================
# Output
# =============================================================================

def save_station(path: Path, st: dict, frame) -> None:
    """One station, in 2D coordinates relative to the toe of the mean profile:

    s, origin_de, origin_xyz   position of the station and of its origin
    nodes, nodes_std           mean profile (N, 2) and the std of its nodes (N, 2)
    materials                  material ID of each element (E,), most probable
    realisations               (R, N, 2), same nodes and elements as `nodes`
    realisation_materials      (R, E) sampled material of each element
    canopy_nodes, canopy_elements, canopy_ids   canopy polylines (M, 2), (K, 2), (K,)
    """
    origin = st["de"][-1].copy()
    np.savez_compressed(
        path,
        s=st["s"], origin_de=origin,
        origin_xyz=frame.sde_to_xyz(np.r_[st["s"], origin][None])[0],
        nodes=st["de"] - origin, nodes_std=st["nodes_std"], materials=st["materials"],
        realisations=st["realisations"] - origin, realisation_materials=st["realisation_materials"],
        canopy_nodes=st["canopy_de"] - origin if st["canopy_de"].size else st["canopy_de"],
        canopy_elements=st["canopy_elements"], canopy_ids=st["canopy_ids"],
    )


def save_qa_cloud(path: Path, stations_data, frame, n_reals: int) -> None:
    """Every profile node in world coordinates, to overlay on the original cloud."""
    pts, src, prof, real, mat = [], [], [], [], []
    for k, st in enumerate(stations_data):
        s = st["s"]
        def add(de, source, r, m):
            pts.append(frame.sde_to_xyz(np.column_stack([np.full(len(de), s), de])))
            src.append(np.full(len(de), source)); prof.append(np.full(len(de), k))
            real.append(np.full(len(de), r)); mat.append(m)
        node_mat = np.r_[st["materials"], st["materials"][-1]]
        add(st["de"], 0, -1, node_mat)
        for r in range(min(n_reals, st["realisations"].shape[0])):
            rm = st["realisation_materials"][r]
            add(st["realisations"][r], 1, r, np.r_[rm, rm[-1]])
        if st["canopy_de"].size:
            add(st["canopy_de"], 2, -1, np.full(len(st["canopy_de"]), -1))
    pc = pr.PointCloud(points=np.vstack(pts))
    for name, arr in (("source", src), ("Profile", prof), ("realisation", real), (MATERIAL, mat)):
        pc.set_attr(name, np.concatenate(arr).astype(np.float64))
    pc.save(str(path), overwrite=True)


def plot_station(path: Path, st: dict, frame, args) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    origin = st["de"][-1]
    fig, ax = plt.subplots(figsize=(7, 7))
    for r in st["realisations"]:
        ax.plot(*(r - origin).T, color="0.6", lw=0.4, alpha=0.5)
    m = st["de"] - origin
    ax.plot(*m.T, "k-", lw=1.2, label="mean (filled)")
    unc = st["sigma_w"] > 0
    if unc.any():
        ax.plot(*m[unc].T, "o", ms=2, color="tab:red", label="uncertain nodes")
    if st["canopy_de"].size:
        c = st["canopy_de"] - origin
        for i, (a, b) in enumerate(st["canopy_elements"]):
            ax.plot(*c[[a, b]].T, color="tab:green", lw=1.5, label="canopy" if i == 0 else None)
    ax.set_aspect("equal")
    ax.set_xlabel("d - d_toe [m]")
    ax.set_ylabel("e - e_toe [m]")
    ax.set_title(f"s = {st['s']:.2f} m, {st['realisations'].shape[0]} realisations")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# =============================================================================
# Using a station
# =============================================================================

def build_simulation(station_path, materials: dict, drags: dict | None = None):
    """
    Slope (and Vegetation) of one station, ready for an Analysis.

    Parameters
    ----------
    station_path : path of a station_####.npz
    materials : {material ID: pr.Material}
    drags : {vegetation class ID: pr.Drag}, optional. Without it there is no vegetation.

    Returns
    -------
    slope : pr.Slope whose geometry samples the realisations (one per block, Latin
        hypercube spreads them evenly). Materials are the most probable ones; the
        sampled materials of each realisation are in the file
        (``realisation_materials``) but not used yet.
    vegetation : pr.Vegetation or None
    """
    z = np.load(station_path)
    ids = z["materials"]
    table_ids = sorted(set(ids.tolist()))
    geom = pr.Geometry(nodes=z["nodes"], nodes_std=z["nodes_std"],
                       realisations=z["realisations"] if z["realisations"].size else None)
    slope = pr.Slope(geom, materials=[materials[i] for i in table_ids],
                     materialIDs=np.searchsorted(table_ids, ids))
    vegetation = None
    if drags is not None and z["canopy_elements"].size:
        cids = z["canopy_ids"]
        d_ids = sorted(set(cids.tolist()))
        canopy = pr.Geometry(nodes=z["canopy_nodes"], elements=z["canopy_elements"])
        vegetation = pr.Vegetation(canopy, drag=[drags[i] for i in d_ids], identities=np.searchsorted(d_ids, cids))
    return slope, vegetation


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path, help="Original cloud (binary PLY with the material field).")
    ap.add_argument("--out", type=Path, help="Output folder. Default: <input>_simulation/")
    ap.add_argument("--material", default=MATERIAL, help=f"Material field (or scalar_<name>). Default: {MATERIAL}")
    ap.add_argument("--veg-ids", type=float, nargs="+", default=[0.0],
                    help="Material IDs of the vegetation. Default: 0")
    ap.add_argument("--frame", type=Path, help="Frame JSON to use instead of fitting one.")
    ap.add_argument("--frame-normals", action="store_true",
                    help="Let the frame fit use the point normals (to pick the steep points). On susan.ply "
                         "they give a wrong frame, so they are not used by default.")
    ap.add_argument("--voxel", type=float, help="Thin the cloud to one point per voxel of this size before filling.")
    ap.add_argument("--refill", action="store_true", help="Fill again even if <out>/filled.ply exists.")
    ap.add_argument("--no-cover", action="store_true", help="Do not keep the filled rock behind the vegetation.")
    ap.add_argument("--cover-cell", type=float, default=0.1,
                    help="Cell (m) in which the deepest vegetation return is a bound. Default: 0.1")
    ap.add_argument("--cover-margin", type=float, default=0.0,
                    help="Least distance (m) of the rock behind the vegetation. Default: 0")
    g = ap.add_argument_group("profiles")
    g.add_argument("--spacing", type=float, default=1.0, help="Station spacing along the strike [m]. Default: 1")
    g.add_argument("--slab", type=float, help="Slab thickness around each station [m]. Default: --spacing")
    g.add_argument("--node-step", type=float, default=0.5, help="Node spacing of the profiles [m]. Default: 0.5")
    g.add_argument("--s-range", type=float, nargs=2, metavar=("S_MIN", "S_MAX"), help="Only stations in this s range.")
    g = ap.add_argument_group("realisations")
    g.add_argument("-N", "--n-realisations", type=int, default=50, help="Default: 50")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--batch", type=int, default=10, help="Realisations in memory at once. Default: 10")
    g = ap.add_argument_group("canopy")
    g.add_argument("--canopy-source", type=Path, help="Cloud the vegetation is read from. Default: input")
    g.add_argument("--canopy-stride", type=int, default=1, help="Use every n-th vegetation point. Default: 1")
    g.add_argument("--canopy-quantile", type=float, default=0.95,
                   help="Quantile of the outward distance in each bin taken as the canopy. Default: 0.95")
    g.add_argument("--canopy-min-thickness", type=float, default=0.2,
                   help="Vegetation thinner than this over the mean rock is ignored [m]. Default: 0.2")
    g.add_argument("--canopy-gap", type=float, default=0.05,
                   help="Least distance of the canopy outside every realisation of the rock [m]. Default: 0.05")
    g.add_argument("--canopy-max-gap", type=int, default=1, help="Empty bins bridged within one canopy run. Default: 1")
    g.add_argument("--canopy-min-points", type=int, default=20)
    g.add_argument("--open-canopy", action="store_true", help="Do not close the canopy down to the rock.")
    g = ap.add_argument_group("checks")
    g.add_argument("--plot-every", type=int, default=0, help="Plot every n-th station. Default: 0 (none)")
    g.add_argument("--qa-realisations", type=int, default=5, help="Realisations written to sections_qa.ply. Default: 5")
    g.add_argument("--misfit-stride", type=int,
                   help="Use every n-th original point for the rock misfit check. Default: about 5M points")
    args = ap.parse_args()
    args.slab = args.slab or args.spacing
    args.canopy_source = args.canopy_source or args.input
    if args.n_realisations < 1:
        ap.error("-N must be >= 1")
    out = args.out or args.input.with_name(f"{args.input.stem}_simulation")
    (out / "stations").mkdir(parents=True, exist_ok=True)
    t_all = time.perf_counter()

    frame, filled, info = stage_fill(args, out)
    stations = stage_stations(args, frame, filled)
    data = stage_mean_profiles(args, frame, filled, stations)
    stage_realisations(args, frame, filled, info, data)
    stage_canopy(args, frame, data)
    t0 = time.perf_counter()
    stride = args.misfit_stride or max(1, open_ply(args.canopy_source).shape[0] // 5_000_000)
    rock_misfit(args, frame, data, stride)
    print(f"[check] rock misfit ({time.perf_counter() - t0:.1f} s)")

    for old in (out / "stations").glob("station_*.npz"):
        old.unlink()
    summary = []
    for k, st in enumerate(data):
        save_station(out / "stations" / f"station_{k:04d}.npz", st, frame)
        summary.append(dict(
            station=k, s=st["s"], n_nodes=int(st["de"].shape[0]),
            n_uncertain_nodes=int(np.count_nonzero(st["sigma_w"] > 0)),
            # kriging std of w at the nodes vs the spread of the realisations (should be close)
            max_sigma_w=float(st["sigma_w"].max()),
            max_spread_w=float(st["spread_w"].max()),
            n_canopy_segments=int(st["canopy_elements"].shape[0]), n_veg_points=st["n_veg"],
            rock_misfit=st["rock_misfit"],
        ))
    meta = dict(input=str(args.input), frame=str(out / "frame.json"), n_realisations=args.n_realisations,
                spacing=args.spacing, slab=args.slab, node_step=args.node_step, veg_ids=args.veg_ids,
                outward_sign=outward_sign(frame), stations=summary)
    (out / "summary.json").write_text(json.dumps(meta, indent=1, default=_to_json))
    save_qa_cloud(out / "sections_qa.ply", data, frame, args.qa_realisations)
    if args.plot_every:
        (out / "plots").mkdir(exist_ok=True)
        for k in range(0, len(data), args.plot_every):
            plot_station(out / "plots" / f"station_{k:04d}.png", data[k], frame, args)

    mis = np.array([s["rock_misfit"] for s in summary], dtype=float)
    print(f"Saved {len(data)} stations to {out}/ ({time.perf_counter() - t_all:.1f} s). "
          f"{sum(s['n_uncertain_nodes'] > 0 for s in summary)} cross filled rock, "
          f"{sum(s['n_canopy_segments'] > 0 for s in summary)} have a canopy; "
          f"rock misfit median {np.nanmedian(mis):.3f} m, max {np.nanmax(mis):.3f} m")


if __name__ == "__main__":
    main()
