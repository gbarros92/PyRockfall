"""
Prepare a wall for 2D rockfall simulations: profiles every --spacing along the
strike, one line seeder covering each whole profile, and the material, rock and
settings files, in the RocFall-style formats that pr.import*() read back.

If the cloud has vegetation (material in --veg-ids), the rock behind it is
uncertain. The vegetation is removed, the holes are filled by kriging, and
realisations of every profile are drawn (as in fill_holes.py and
pr.sampleRealisations); they are saved next to each profile. Without vegetation
the profiles are cut straight from the cloud and are deterministic.

Steps
-----
1. Loads the cloud (pr.PointCloud.load reads binary PLY memory-mapped, so a
   cloud of 100M+ points is fine), optionally thinned on a voxel grid, and
   fits the curvilinear frame (or loads one with --frame). Every profile is cut
   in this one frame.
2. With vegetation: removes it and fills the holes by kriging: the mean surface,
   the std of its height (kriging_std) and the material probabilities. The
   vegetation of the full-resolution cloud is the cover: the rock must lie
   behind its deepest returns (fillHoles(cover=...)). Cached in <out>/cache/.
3. Stations: s positions every --spacing along the strike.
4. Mean profiles: slices of the (filled) cloud at the stations.
5. With vegetation: N realisations of the filled cloud (pr.sampleRealisations);
   every node of the mean profile moves along the frame's w axis by the
   realisation's displacement there, so every realisation has the same nodes
   and elements as the mean profile. Nodes on exposed rock do not move.
   Elements on filled rock take the realisation's material there, or, with
   --use-vegetation-cor, the vegetation material of the library (a topsoil
   layer under the vegetation, say, with a lower restitution).
6. With vegetation: the canopy, from the vegetation points of the cloud at the
   same stations. Per bin up the face, the outermost vegetation (a high
   quantile, to skip stray points) gives the canopy, kept at least --canopy-gap
   outside every realisation of the rock, closed down to the mean rock at both
   ends unless --open-canopy. Blocks crossing it are slowed by its drag (a
   linear damping rate, from --drags or --drag-coefficient).
7. Writes the simulation inputs. Profiles are in 2D (x, y) relative to the toe
   of the mean profile, x increasing down the slope. A horizontal floor of
   --floor-length is added past the toe. One line seeder per profile runs along
   the whole profile (not the floor), so blocks can start anywhere on it, with a
   downward velocity equal to a free fall of --drop. Its points (every 0.1 m)
   sit --seeder-offset outside the rock along the normal, pushed further out
   where needed so they are outside every realisation too.

Outputs (in <input>_profiles/ unless --out)
-------------------------------------------
    materials.ini               the material library (--materials), in its order
    rocks.ini                   the block
    settings.ini                analysis settings (number of blocks, ...)
    profile_####.txt            nodes, nodes_std and material of each element
    seeders_####.txt            the line seeder of the profile
    realisations_####.npz       with vegetation only (see save_realisations)
    canopy_####.txt             with vegetation only: canopy polylines (see export_canopy)
    drags.csv                   with vegetation only: drag of each vegetation ID
    summary.json                position of every profile and the settings
    cache/                      frame.json and, with vegetation, the kriged cloud

Run them with run_simulations.py.

Usage
-----
    python prepare_simulation.py model.ply --spacing 2
    python prepare_simulation.py susan.ply --spacing 1 -N 50 -m mock_materials.csv
    python prepare_simulation.py susan.ply -m mock_materials.csv --use-vegetation-cor
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
# Input (read with pr.openPly / pr.PointCloud.load)
# =============================================================================

def voxel_thin(xyz: np.ndarray, voxel: float) -> np.ndarray:
    """Indices of one point per voxel (the first one)."""
    keys = np.floor((xyz - xyz.min(axis=0)) / voxel).astype(np.int64)
    dims = keys.max(axis=0) + 1
    flat = (keys[:, 0] * dims[1] + keys[:, 1]) * dims[2] + keys[:, 2]
    _, first = np.unique(flat, return_index=True)
    return np.sort(first)


def load_input(path: Path, label: str, voxel: float | None, normals: bool = False) -> pr.PointCloud:
    """The cloud to fill: xyz, the material and, if asked and present, the normals."""
    idx = None
    if voxel:
        idx = voxel_thin(pr.PointCloud.load(str(path), normals=False, colors=False).points, voxel)
    return pr.PointCloud.load(str(path), attributes=[label], indices=idx, normals=normals, colors=False)


def load_filled(path: Path, label: str) -> pr.PointCloud:
    """A cloud written by stage_fill, with the attributes the realisations need."""
    keep = [n for n in pr.openPly(str(path)).dtype.names
            if n in ("filled", "hole_id", "kriging_std", "cover_bound", "cover_free_w", label) or n.startswith(f"{label}_P")]
    return pr.PointCloud.load(str(path), attributes=keep, normals=False, colors=False)


def vegetation_points(path: Path, label: str, veg_ids) -> tuple[np.ndarray, np.ndarray]:
    """Indices and material of the vegetation points of a PLY (only its material
    column is read)."""
    v = pr.openPly(str(path))
    mat = np.asarray(v[pr.plyFieldName(v, label)])
    veg = np.flatnonzero(np.isin(mat, veg_ids))
    return veg, mat[veg].astype(np.int64)


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


def to_uvw_chunked(frame, path: Path, idx, chunk: int = 2_000_000) -> np.ndarray:
    """(u, v, w) of the points `idx` of a PLY, loaded a chunk at a time."""
    out = np.empty((idx.size, 3))
    for a in range(0, idx.size, chunk):
        pc = pr.PointCloud.load(str(path), indices=idx[a:a + chunk], normals=False, colors=False)
        out[a:a + chunk] = frame.xyz_to_uvw(pc.points)
    return out


def deepest_cover(path: Path, label: str, veg_ids, frame, cell: float) -> np.ndarray:
    """(x, y, z) of the deepest vegetation return in each (u, v) cell of size `cell`:
    the tightest bounds on the rock behind the vegetation."""
    veg, _ = vegetation_points(path, label, veg_ids)
    uvw = to_uvw_chunked(frame, path, veg)
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
    if args.s_range:
        lo, hi = args.s_range
    else:
        # Past the ends the frame only extrapolates; a few metres is fine, far beyond is a bad mapping.
        m = args.end_margin
        inside = (s >= -m) & (s <= frame.length + m)
        if not inside.all():
            print(f"[stations] {np.count_nonzero(~inside)} points map more than {m:g} m past the frame ends "
                  f"(s = {s.min():.1f} .. {s.max():.1f}, frame {frame.length:.1f} m); no stations there")
        lo, hi = s[inside].min(), s[inside].max()
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
        st["under_vegetation"] = mid_unc[e_offsets[k]:e_offsets[k + 1]]   # elements on filled rock

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
                st["realisation_materials"][start + j] = np.where(st["under_vegetation"], mat_mid[j, el],
                                                                  st["materials"])
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
    veg, veg_class = vegetation_points(args.canopy_source, args.material, args.veg_ids)
    if args.canopy_stride > 1:
        veg, veg_class = veg[::args.canopy_stride], veg_class[::args.canopy_stride]
    sde = frame.uvw_to_sde(to_uvw_chunked(frame, args.canopy_source, veg))
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


def count_vegetation(path: Path, label: str, veg_ids) -> int:
    """Number of vegetation points in a PLY."""
    return int(vegetation_points(path, label, veg_ids)[0].size)


def stage_frame_bare(args, cache: Path):
    """Without vegetation: the cloud itself and its frame."""
    t0 = time.perf_counter()
    pc = load_input(args.input, args.material, args.voxel, normals=args.frame_normals)
    print(f"[load] {args.input.name}: {pc.points.shape[0]} points"
          f"{f' (one per {args.voxel:g} m voxel)' if args.voxel else ''} ({time.perf_counter() - t0:.1f} s)")
    t0 = time.perf_counter()
    frame_path = cache / "frame.json"
    if args.frame:
        frame = pr.CurvilinearFrame.load(args.frame)
        pc.frame = frame
        if frame.pca_axes is None:
            pc.uvw()
        print(f"[frame] from {args.frame}")
    else:
        frame = pc.fitFrame()
        pc.uvw()                                  # principal axes of (s, d, e)
        print(f"[frame] fitted: {frame.length:.1f} m, {frame.n_segments} segments "
              f"({time.perf_counter() - t0:.1f} s)")
    diag = frame.diagnostics(pc.points)
    print(f"[frame] {diag['n_nan']} points outside the frame, {diag['n_ambiguous']} ambiguous")
    frame.save(frame_path)
    if pc.has_attr("normals"):
        pc.del_attr("normals")
    return frame, pc


# =============================================================================
# Simulation inputs
# =============================================================================

def load_material_table(args):
    """Material IDs (CSV order) and the pr.Material of each, coloured by name."""
    from pyrockfall.scripts.create import colorMaterials, loadMaterials
    from pyrockfall.scripts._style import get_material_colors
    materials = loadMaterials(str(args.materials) if args.materials else None)
    colorMaterials(materials)
    known = get_material_colors(latex_escape=False)
    import matplotlib
    cycle = matplotlib.colormaps["tab20"].colors
    for j, m in enumerate(m for m in materials.values() if m.name not in known):
        m.color = tuple(int(round(255 * c)) for c in cycle[j % len(cycle)])
    ids = [int(k) for k in materials]
    return ids, [materials[k] for k in materials]


DRAG_COLUMNS = ["veg_id", "name", "drag_id", "drag_mean", "drag_std", "drag_min", "drag_max"]


def load_drag_table(args):
    """Vegetation IDs and their drag rows (as in drags.csv): --drags, else one
    deterministic --drag-coefficient for every --veg-ids."""
    import pandas as pd
    if args.drags:
        df = pd.read_csv(args.drags)
        missing = [c for c in DRAG_COLUMNS if c not in df.columns]
        if missing:
            raise SystemExit(f"{args.drags}: missing columns {missing}; expected {DRAG_COLUMNS}")
    else:
        c = args.drag_coefficient
        df = pd.DataFrame([dict(veg_id=int(v), name=f"Vegetation {int(v)}", drag_id=1, drag_mean=c,
                                drag_std=0.0, drag_min=c, drag_max=c) for v in args.veg_ids])
    return [int(v) for v in df["veg_id"]], df[DRAG_COLUMNS]


def export_canopy(path: Path, nodes: np.ndarray, elements: np.ndarray, ids: np.ndarray) -> None:
    """Canopy polylines in the format of profile_####.txt: x, x_std, y, y_std and the
    drag (index in drags.csv) of the element from each node to the next; -1 ends a
    polyline, so several polylines follow each other in one file."""
    if not np.array_equal(elements[:, 1], elements[:, 0] + 1):
        raise ValueError("canopy elements must join consecutive nodes")
    drag = np.full(nodes.shape[0], -1, dtype=np.int64)
    drag[elements[:, 0]] = ids
    out = np.column_stack([nodes[:, 0], np.zeros(len(nodes)), nodes[:, 1], np.zeros(len(nodes)), drag])
    np.savetxt(path, out, fmt=["%.6f"] * 4 + ["%d"], delimiter=",")


def floor_index(args, ids, table):
    """Index in the table of the floor material: --floor-material, else the one
    named 'Floor', else None (the toe material is used)."""
    if args.floor_material is not None:
        if args.floor_material not in ids:
            raise SystemExit(f"--floor-material {args.floor_material} is not in the material library")
        return ids.index(args.floor_material)
    for i, m in enumerate(table):
        if m.name.strip().lower() == "floor":
            return i
    return None


def vegetation_material(args, ids, table) -> int:
    """Material ID of the ground under the vegetation (--use-vegetation-cor):
    --vegetation-material, else the one named 'Vegetation'."""
    if args.vegetation_material is not None:
        if args.vegetation_material not in ids:
            raise SystemExit(f"--vegetation-material {args.vegetation_material} is not in the material library")
        return args.vegetation_material
    for m_id, m in zip(ids, table):
        if m.name.strip().lower() == "vegetation":
            return m_id
    raise SystemExit("--use-vegetation-cor needs a material named 'Vegetation' in the material library "
                     f"({args.materials or 'bundled material_lib.csv'}), or its ID with --vegetation-material")


def apply_vegetation_material(stations_data, veg_mat: int) -> int:
    """Every element under the vegetation (on filled rock) gets `veg_mat`, in the
    mean profile and in every realisation, instead of the predicted material.
    Returns the number of elements changed."""
    n = 0
    for st in stations_data:
        under = st.get("under_vegetation")
        if under is None or not under.any():
            continue
        st["materials"] = np.where(under, veg_mat, st["materials"])
        st["realisation_materials"] = np.where(under[None, :], veg_mat, st["realisation_materials"])
        n += int(np.count_nonzero(under))
    return n


def to_table(material_ids: np.ndarray, id_to_idx: dict) -> np.ndarray:
    """Material IDs of the cloud -> indices in the material table."""
    lut = np.vectorize(lambda m: id_to_idx.get(int(m), -1), otypes=[np.int64])
    return lut(material_ids) if material_ids.size else material_ids.astype(np.int64)


def node_normals(nodes: np.ndarray) -> np.ndarray:
    """Unit normals at the nodes of a polyline ordered crest to toe (x increasing
    down the slope): the average of the adjacent element normals, pointing out of
    the rock (as Analysis computes them)."""
    edge = np.diff(nodes, axis=0)
    n_el = np.column_stack([-edge[:, 1], edge[:, 0]])
    n_el /= np.maximum(np.linalg.norm(n_el, axis=1, keepdims=True), 1e-12)
    n = np.zeros_like(nodes)
    n[:-1] += n_el
    n[1:] += n_el
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


def rock_path(nodes: np.ndarray):
    """The rock below/behind a profile ordered crest to toe (x increasing down the
    slope), closed far below and behind it, as a matplotlib Path."""
    from matplotlib.path import Path as MplPath
    far = 10.0 * (np.ptp(nodes[:, 0]) + np.ptp(nodes[:, 1]) + 1.0)
    (x0, y0), (x1, _) = nodes[0], nodes[-1]
    low = nodes[:, 1].min() - far
    ring = np.vstack([nodes, [[x1, low], [x0 - far, low], [x0 - far, y0]]])
    return MplPath(ring, closed=False)


def seeder_points(nodes: np.ndarray, reals: np.ndarray | None, offset: float, spacing: float = 0.1,
                  step: float = 0.02, max_push: float = 3.0) -> tuple[np.ndarray, int]:
    """(M, 2) line along the whole profile, just outside the rock, and the number of
    points dropped.

    Points every `spacing` along the profile each move out along their normal (the
    element's, or the average at a node): `offset`, and further in steps of `step`
    until they are outside the rock of the profile and of every realisation, so no
    block starts inside the rock it runs on. Points still inside after `max_push`
    (e.g. a crest that starts with a vertical step, or a profile folding over
    itself) are dropped."""
    n_node = node_normals(nodes)
    edge = np.diff(nodes, axis=0)
    length = np.linalg.norm(edge, axis=1)
    n_el = np.column_stack([-edge[:, 1], edge[:, 0]]) / np.maximum(length, 1e-12)[:, None]
    pts, nrm, env = [], [], []
    env_node = np.zeros(nodes.shape[0])
    if reals is not None and reals.size:      # how far out the realisations reach, as a first guess
        env_node = np.maximum(0.0, np.einsum("rnd,nd->rn", reals - nodes[None], n_node).max(axis=0))
    for e in range(edge.shape[0]):
        f = np.arange(max(1, int(np.ceil(length[e] / spacing)))) / max(1, int(np.ceil(length[e] / spacing)))
        pts.append(nodes[e] + f[:, None] * edge[e])
        nn = np.repeat(n_el[e][None], f.size, axis=0)
        nn[0] = n_node[e]
        nrm.append(nn)
        env.append((1 - f) * env_node[e] + f * env_node[e + 1])
    pts.append(nodes[-1:]); nrm.append(n_node[-1:]); env.append(env_node[-1:])
    base, nrm, out = np.vstack(pts), np.vstack(nrm), np.concatenate(env) + offset

    paths = [rock_path(nodes)] + ([rock_path(r) for r in reals] if reals is not None else [])

    def inside(q):
        res = np.zeros(q.shape[0], dtype=bool)
        for p in paths:
            res |= p.contains_points(q)
        return res

    todo = np.arange(base.shape[0])
    while todo.size:
        bad = inside(base[todo] + nrm[todo] * out[todo, None])
        todo = todo[bad]
        out[todo] += step
        todo = todo[out[todo] <= max_push + offset]
    seeds = base + nrm * out[:, None]
    keep = ~inside(seeds)
    seeds = seeds[keep]
    seeds = seeds[np.r_[True, np.linalg.norm(np.diff(seeds, axis=0), axis=1) > 1e-9]]
    return seeds, int(np.count_nonzero(~keep))


def station_arrays(st: dict, id_to_idx: dict, floor_idx, floor_length: float) -> dict:
    """The 2D arrays of one station for the simulation, relative to the toe of the
    mean profile, x increasing down the slope, with the floor appended."""
    origin = st["de"][-1].copy()
    sign = 1.0 if st["de"][0, 0] <= st["de"][-1, 0] else -1.0      # crest left, toe right
    flip = np.array([sign, 1.0])
    nodes = (st["de"] - origin) * flip
    std = st.get("nodes_std", np.zeros_like(nodes))
    mats = to_table(st["materials"], id_to_idx)
    reals = rmats = None
    if "realisations" in st:
        reals = (st["realisations"] - origin) * flip
        rmats = to_table(st["realisation_materials"], id_to_idx)
    seeder, n_dropped = seeder_points(nodes, reals, st["seeder_offset"])
    if floor_length > 0:
        f_node = np.array([[nodes[-1, 0] + floor_length, nodes[-1, 1]]])
        f_mat = floor_idx if floor_idx is not None else mats[-1]
        nodes = np.vstack([nodes, f_node])
        std = np.vstack([std, np.zeros((1, 2))])
        mats = np.r_[mats, f_mat]
        if reals is not None:
            reals = np.concatenate([reals, np.broadcast_to(f_node, (reals.shape[0], 1, 2))], axis=1)
            rmats = np.column_stack([rmats, np.full(rmats.shape[0], f_mat)])
    canopy = None
    if st.get("canopy_elements") is not None and st["canopy_elements"].size:
        canopy = dict(nodes=(st["canopy_de"] - origin) * flip, elements=st["canopy_elements"],
                      ids=st["canopy_ids"])
    return dict(nodes=nodes, nodes_std=std, materials=mats, realisations=reals, canopy=canopy,
                realisation_materials=rmats, seeder=seeder, n_seeder_dropped=n_dropped,
                origin=origin, x_sign=sign)


def save_realisations(path: Path, arr: dict, st: dict, frame) -> None:
    """The realisations of one profile, in the coordinates of profile_####.txt:

    realisations           (R, N, 2) same nodes and elements as the profile
    realisation_materials  (R, E) sampled material of each element (index in materials.ini)
    nodes, materials       the mean profile (as in profile_####.txt)
    s, origin_xyz          position of the profile along the strike and of its origin
    """
    np.savez_compressed(
        path,
        realisations=arr["realisations"], realisation_materials=arr["realisation_materials"],
        nodes=arr["nodes"], materials=arr["materials"],
        s=st["s"], origin_xyz=frame.sde_to_xyz(np.r_[st["s"], arr["origin"]][None])[0],
    )


def write_inputs(args, out: Path, frame, stations_data, uncertain: bool) -> list:
    """materials.ini, rocks.ini, settings.ini and the files of every profile."""
    ids, table = load_material_table(args)
    id_to_idx = {m: i for i, m in enumerate(ids)}
    if args.use_vegetation_cor:
        if uncertain:
            veg_mat = vegetation_material(args, ids, table)
            n = apply_vegetation_material(stations_data, veg_mat)
            print(f"[materials] {n} elements under the vegetation use material {veg_mat} "
                  f"({table[ids.index(veg_mat)].name}) instead of the predicted one")
        else:
            print("[materials] --use-vegetation-cor: no vegetation in the cloud, nothing to change")
    used = set()
    for st in stations_data:
        used.update(np.unique(st["materials"]).astype(int).tolist())
        if "realisation_materials" in st:
            used.update(np.unique(st["realisation_materials"]).astype(int).tolist())
    missing = sorted(used - set(ids))
    if missing:
        raise SystemExit(f"Material IDs {missing} of the cloud are not in the material library "
                         f"({args.materials or 'bundled material_lib.csv'}); pass one that has them with -m")
    floor_idx = floor_index(args, ids, table)
    for pattern in ("profile_*.txt", "seeders_*.txt", "realisations_*.npz", "canopy_*.txt", "drags.csv"):
        for old in out.glob(pattern):                 # outputs of a previous run
            old.unlink()

    pr.exportMaterials(str(out / "materials.ini"), table)
    rock = pr.Rock("Block", mass=args.rock_mass, density=args.rock_density, color=(0, 0, 255))
    pr.exportRocks(str(out / "rocks.ini"), [rock])
    analysis = pr.Analysis()
    analysis.rockThrowMode = pr.AnalysisRocksThrown.IndividuallyPerSeeder
    analysis.numberOfRocks = args.n_rocks
    analysis.scaleByVelocity = True
    analysis.considerRotationalVelocity = True
    pr.exportSettings(str(out / "settings.ini"), analysis)

    drag_lut = None
    if uncertain and not args.no_canopy:
        veg_ids, drag_df = load_drag_table(args)
        drag_lut = {v: i for i, v in enumerate(veg_ids)}
        used = {int(c) for st in stations_data for c in np.unique(st.get("canopy_ids", []))}
        if used - set(veg_ids):
            raise SystemExit(f"Vegetation IDs {sorted(used - set(veg_ids))} are not in the drag table "
                             f"({args.drags or '--veg-ids'})")
        drag_df.to_csv(out / "drags.csv", index=False)

    v_drop = -np.sqrt(2.0 * 9.80665 * args.drop)
    summary = []
    for k, st in enumerate(stations_data):
        st["seeder_offset"] = args.seeder_offset
        arr = station_arrays(st, id_to_idx, floor_idx, args.floor_length)
        slope = pr.Slope(pr.Geometry(nodes=arr["nodes"], nodes_std=arr["nodes_std"]),
                         materials=table, materialIDs=arr["materials"])
        pr.exportSlope(str(out / f"profile_{k:04d}.txt"), slope)
        seeder = pr.LineSeeder(arr["seeder"].T, [rock], name=f"Profile {k}")
        seeder.rockThrowMode = pr.SeederRocksThrown.Overall
        seeder.numberOfRocks = args.n_rocks
        seeder.translationalVelocity = [0.0, v_drop]
        seeder.angularVelocity = [0.0]
        pr.exportSeeders(str(out / f"seeders_{k:04d}.txt"), [seeder], [rock])
        if uncertain:
            save_realisations(out / f"realisations_{k:04d}.npz", arr, st, frame)
        if drag_lut is not None and arr["canopy"] is not None:
            c = arr["canopy"]
            export_canopy(out / f"canopy_{k:04d}.txt", c["nodes"], c["elements"],
                          np.array([drag_lut[int(i)] for i in c["ids"]], dtype=np.int64))
        summary.append(dict(
            profile=k, s=st["s"],
            origin_xyz=frame.sde_to_xyz(np.r_[st["s"], arr["origin"]][None])[0],
            x_sign=arr["x_sign"], n_nodes=int(arr["nodes"].shape[0]),
            n_seeder_points=int(arr["seeder"].shape[0]), n_seeder_dropped=arr["n_seeder_dropped"],
            height=float(arr["nodes"][:, 1].max()),
            n_realisations=0 if arr["realisations"] is None else int(arr["realisations"].shape[0]),
            n_uncertain_nodes=int(np.count_nonzero(st["sigma_w"] > 0)) if "sigma_w" in st else 0,
            n_elements_under_vegetation=int(np.count_nonzero(st.get("under_vegetation", []))),
            max_sigma_w=float(st["sigma_w"].max()) if "sigma_w" in st else 0.0,
            n_canopy_segments=0 if arr["canopy"] is None else int(arr["canopy"]["elements"].shape[0]),
        ))
    return summary


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path, help="The cloud (binary PLY with the material field).")
    ap.add_argument("--out", type=Path, help="Output folder. Default: <input>_profiles/")
    ap.add_argument("--material", default=MATERIAL, help=f"Material field (or scalar_<name>). Default: {MATERIAL}")
    ap.add_argument("--veg-ids", type=float, nargs="+", default=[0.0],
                    help="Material IDs of the vegetation. Default: 0")
    ap.add_argument("--frame", type=Path, help="Frame JSON to use instead of fitting one.")
    ap.add_argument("--frame-normals", action=argparse.BooleanOptionalAction, default=True,
                    help="Let the frame fit use the point normals, when the cloud has them, to pick the "
                         "steep points. Default: on (--no-frame-normals to fit on xyz only).")
    ap.add_argument("--voxel", type=float, help="Thin the cloud to one point per voxel of this size first.")
    g = ap.add_argument_group("vegetation (hole filling and realisations)")
    g.add_argument("--refill", action="store_true", help="Fill again even if <out>/cache/filled.ply exists.")
    g.add_argument("--no-cover", action="store_true", help="Do not keep the filled rock behind the vegetation.")
    g.add_argument("--cover-cell", type=float, default=0.1,
                   help="Cell (m) in which the deepest vegetation return is a bound. Default: 0.1")
    g.add_argument("--cover-margin", type=float, default=0.0,
                   help="Least distance (m) of the rock behind the vegetation. Default: 0")
    g.add_argument("-N", "--n-realisations", type=int, default=50, help="Default: 50")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--batch", type=int, default=10, help="Realisations in memory at once. Default: 10")
    g.add_argument("--use-vegetation-cor", action="store_true",
                   help="Give every element under the vegetation (on filled rock) the vegetation material, "
                        "in the mean profile and every realisation, instead of the predicted rock material "
                        "(e.g. a topsoil layer with a lower restitution). The material library must have "
                        "a material named 'Vegetation', or pass its ID with --vegetation-material.")
    g.add_argument("--vegetation-material", type=int,
                   help="Material ID used under the vegetation with --use-vegetation-cor. "
                        "Default: the material named 'Vegetation'")
    g = ap.add_argument_group("canopy (with vegetation)")
    g.add_argument("--no-canopy", action="store_true", help="Do not write the canopy (vegetation) files.")
    g.add_argument("--drags", type=Path,
                   help="Drag table CSV, columns " + ",".join(DRAG_COLUMNS) + " (drag_id as the material "
                        "library's *_id, the coefficient is a linear damping rate [1/s]). Default: one "
                        "deterministic --drag-coefficient for every --veg-ids")
    g.add_argument("--drag-coefficient", type=float, default=1.0,
                   help="Drag (linear damping rate) of the vegetation without --drags [1/s]. Default: 1")
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
    g = ap.add_argument_group("profiles")
    g.add_argument("-S", "--spacing", type=float, default=1.0, help="Profile spacing along the strike [m]. Default: 1")
    g.add_argument("--slab", type=float, help="Slab thickness around each profile [m]. Default: --spacing")
    g.add_argument("--node-step", type=float, default=0.5, help="Node spacing of the profiles [m]. Default: 0.5")
    g.add_argument("--s-range", type=float, nargs=2, metavar=("S_MIN", "S_MAX"), help="Only profiles in this s range.")
    g.add_argument("--end-margin", type=float, default=10.0,
                   help="Without --s-range, profiles reach at most this far (m) past the frame ends. Default: 10")
    g.add_argument("--floor-length", type=float, default=50.0,
                   help="Horizontal floor added past the toe [m]; 0 for none. Default: 50")
    g.add_argument("--floor-material", type=int,
                   help="Material ID of the floor. Default: the material named 'Floor', else the toe's")
    g = ap.add_argument_group("simulation")
    g.add_argument("-m", "--materials", type=Path,
                   help="Material library CSV (as material_lib.csv). Default: materials.csv next to the "
                        "input if there is one, else the bundled material_lib.csv")
    g.add_argument("-n", "--n-rocks", type=int, default=1000, help="Blocks thrown per profile. Default: 1000")
    g.add_argument("--drop", type=float, default=0.5,
                   help="Initial downward velocity of the blocks, as the height of a free fall [m]. Default: 0.5")
    g.add_argument("--seeder-offset", type=float, default=0.01,
                   help="Distance of the seeder outside the rock, along the normal [m]. Default: 0.01")
    g.add_argument("--rock-mass", type=float, default=1000.0, help="Block mass [kg]. Default: 1000")
    g.add_argument("--rock-density", type=float, default=2700.0, help="Block density [kg/m3]. Default: 2700")
    args = ap.parse_args()
    args.slab = args.slab or args.spacing
    args.canopy_source = args.canopy_source or args.input
    if args.n_realisations < 1:
        ap.error("-N must be >= 1")
    if args.materials is None and args.input.with_name("materials.csv").exists():
        args.materials = args.input.with_name("materials.csv")
    out = args.out or args.input.with_name(f"{args.input.stem}_profiles")
    cache = out / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    t_all = time.perf_counter()

    n_veg = count_vegetation(args.input, args.material, args.veg_ids)
    uncertain = n_veg > 0
    print(f"[load] {n_veg} vegetation points ({args.material} in {args.veg_ids}): "
          + ("filling the holes and drawing realisations" if uncertain else "deterministic profiles"))
    if uncertain:
        frame, cloud, info = stage_fill(args, cache)
    else:
        frame, cloud = stage_frame_bare(args, cache)
    stations = stage_stations(args, frame, cloud)
    data = stage_mean_profiles(args, frame, cloud, stations)
    if uncertain:
        stage_realisations(args, frame, cloud, info, data)
        if not args.no_canopy:
            stage_canopy(args, frame, data)

    summary = write_inputs(args, out, frame, data, uncertain)
    meta = dict(input=str(args.input), frame=str(cache / "frame.json"),
                materials=str(args.materials or "bundled material_lib.csv"),
                uncertain=uncertain, n_realisations=args.n_realisations if uncertain else 0,
                n_rocks=args.n_rocks, drop=args.drop, seeder_offset=args.seeder_offset,
                floor_length=args.floor_length, spacing=args.spacing, slab=args.slab,
                canopy=uncertain and not args.no_canopy,
                use_vegetation_cor=bool(args.use_vegetation_cor and uncertain),
                node_step=args.node_step, veg_ids=args.veg_ids, profiles=summary)
    (out / "summary.json").write_text(json.dumps(meta, indent=1, default=_to_json))
    print(f"Saved {len(summary)} profiles to {out}/ ({time.perf_counter() - t_all:.1f} s)"
          + (f"; {sum(s['n_uncertain_nodes'] > 0 for s in summary)} cross filled rock" if uncertain else ""))
    n_drop = [s["n_seeder_dropped"] for s in summary]
    if any(n_drop):
        print(f"[seeders] {sum(n_drop)} seeder points dropped (could not be put outside the rock) "
              f"in {sum(d > 0 for d in n_drop)} profiles; see n_seeder_dropped in summary.json")


if __name__ == "__main__":
    main()
