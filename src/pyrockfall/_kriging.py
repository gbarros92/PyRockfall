"""
_kriging.py
===========

Ordinary kriging of scattered 2D data and hole filling in the ``(u, v)`` plane
of a :class:`~._frame.CurvilinearFrame`.

- :class:`Variogram`, :func:`fit_variogram` and :func:`ordinary_kriging` are
  general-purpose and work on plain arrays.
- :func:`find_holes` locates the missing parts of a point set on an occupancy
  grid and :func:`fill_holes_uv` creates new points there, kriging their height
  ``w`` above the ``(u, v)`` plane and any number of attributes.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage
from scipy.linalg import lu_factor, lu_solve
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist, pdist

__all__ = [
    "VARIOGRAM_MODELS",
    "Variogram",
    "fit_variogram",
    "ordinary_kriging",
    "poly_design",
    "find_holes",
    "fill_holes_uv",
]

VARIOGRAM_MODELS = ("spherical", "exponential", "gaussian", "linear")


# ---------------------------------------------------------------------------
# Variogram
# ---------------------------------------------------------------------------
@dataclass
class Variogram:
    """Variogram model with geometric anisotropy.

    Attributes
    ----------
    model : str
        One of :data:`VARIOGRAM_MODELS`.
    psill : float
        Partial sill (sill minus nugget). For ``"linear"`` the value reached at
        ``range``, i.e. the slope is ``psill / range``.
    range : float
        Range along the major axis (practical range for ``"exponential"`` and
        ``"gaussian"``).
    nugget : float
        Nugget: discontinuity at the origin. The model is zero at zero distance,
        so kriging honours the data exactly.
    aniso : float
        Range along the major axis divided by the range across it (1 = isotropic).
    angle : float
        Direction of the major axis, in degrees counter-clockwise from the first
        coordinate.
    """
    model: str = "spherical"
    psill: float = 1.0
    range: float = 1.0
    nugget: float = 0.0
    aniso: float = 1.0
    angle: float = 0.0

    def __post_init__(self):
        if self.model not in VARIOGRAM_MODELS:
            raise ValueError(f"Unknown variogram model '{self.model}'. Use one of {VARIOGRAM_MODELS}.")

    def gamma(self, h) -> np.ndarray:
        """Semivariance at (isotropic) distance ``h``."""
        h = np.asarray(h, dtype=float)
        a = max(float(self.range), 1e-12)
        if self.model == "spherical":
            r = np.minimum(h / a, 1.0)
            g = self.psill * (1.5 * r - 0.5 * r ** 3)
        elif self.model == "exponential":
            g = self.psill * (1.0 - np.exp(-3.0 * h / a))
        elif self.model == "gaussian":
            g = self.psill * (1.0 - np.exp(-(h / (a * 4.0 / 7.0)) ** 2))
        else:  # linear
            g = self.psill * h / a
        return np.where(h > 0, g + self.nugget, 0.0)

    def transform(self, xy) -> np.ndarray:
        """Rotate and scale coordinates so that the model is isotropic in them."""
        xy = np.asarray(xy, dtype=float)
        if self.aniso == 1.0 and self.angle == 0.0:
            return xy
        t = np.radians(self.angle)
        c, s = np.cos(t), np.sin(t)
        major = xy[:, 0] * c + xy[:, 1] * s
        minor = -xy[:, 0] * s + xy[:, 1] * c
        return np.column_stack([major, minor * self.aniso])

    def to_dict(self) -> dict:
        return asdict(self)


def fit_variogram(
    xy,
    values,
    model: str = "spherical",
    *,
    nlags: int = 20,
    aniso: float = 1.0,
    angle: float = 0.0,
    max_points: int = 2000,
    max_lag: Optional[float] = None,
    seed: int = 0,
) -> Variogram:
    """Fit a variogram model to scattered data.

    The experimental variogram is computed from all pairs of (at most
    `max_points` randomly chosen) data points, binned into `nlags` lags up to
    `max_lag`, and the model is fitted by weighted least squares with more
    weight on short, well-populated lags.

    Parameters
    ----------
    xy : (n, 2) array
        Data locations.
    values : (n,) array
        Data values.
    model : str
        One of :data:`VARIOGRAM_MODELS`.
    nlags : int
        Number of lag bins.
    aniso, angle : float
        Geometric anisotropy (see :class:`Variogram`). They are imposed, not fitted.
    max_points : int
        Largest number of points used for the experimental variogram.
    max_lag : float, optional
        Largest lag. Defaults to half the largest distance between points.

    Returns
    -------
    Variogram
    """
    out = Variogram(model=model, aniso=float(aniso), angle=float(angle))
    xy = out.transform(np.asarray(xy, dtype=float))
    values = np.asarray(values, dtype=float).ravel()
    n = xy.shape[0]
    if n > max_points:
        pick = np.random.default_rng(seed).choice(n, size=max_points, replace=False)
        xy, values = xy[pick], values[pick]
        n = max_points

    var = float(values.var()) if n > 0 else 0.0
    if n < 3 or var <= 0:
        out.psill, out.range, out.nugget = max(var, 1e-12), 1.0, 0.0
        return out

    d = pdist(xy)
    g = 0.5 * pdist(values[:, None], "sqeuclidean")
    d_max = float(d.max())
    if max_lag is None:
        max_lag = 0.5 * d_max
    if not np.isfinite(max_lag) or max_lag <= 0:
        out.psill, out.range, out.nugget = var, 1.0, 0.0
        return out

    keep = (d > 0) & (d <= max_lag)
    d, g = d[keep], g[keep]
    k = np.minimum((d / max_lag * nlags).astype(np.int64), nlags - 1)
    cnt = np.bincount(k, minlength=nlags).astype(float)
    ok = cnt > 0
    if ok.sum() < 3:
        out.psill, out.range, out.nugget = var, float(max_lag), 0.0
        return out
    lag = np.bincount(k, weights=d, minlength=nlags)[ok] / cnt[ok]
    sem = np.bincount(k, weights=g, minlength=nlags)[ok] / cnt[ok]
    wgt = np.sqrt(cnt[ok]) / lag
    wgt /= wgt.max()

    s_max = float(max(sem.max(), 1e-12))
    psill0 = max(float(sem.max() - sem.min()), 1e-3 * s_max)
    nugget0 = float(np.clip(sem.min(), 0.0, s_max))

    if model == "linear":
        out.range = float(max_lag)

        def resid(p):
            out.psill, out.nugget = p
            return wgt * (out.gamma(lag) - sem)

        sol = least_squares(resid, [psill0, nugget0], bounds=([0.0, 0.0], [10 * s_max, s_max]), loss="soft_l1")
        out.psill, out.nugget = (float(v) for v in sol.x)
        return out

    r_lo, r_hi = 0.5 * float(lag[0]), 5.0 * float(max_lag)
    r0 = float(np.clip(0.5 * max_lag, r_lo * 1.01, r_hi * 0.99))

    def resid(p):
        out.psill, out.range, out.nugget = p
        return wgt * (out.gamma(lag) - sem)

    sol = least_squares(
        resid, [psill0, r0, nugget0],
        bounds=([0.0, r_lo, 0.0], [10 * s_max, r_hi, s_max]),
        loss="soft_l1",
    )
    out.psill, out.range, out.nugget = (float(v) for v in sol.x)
    return out


# ---------------------------------------------------------------------------
# Ordinary kriging
# ---------------------------------------------------------------------------
def ordinary_kriging(
    xy,
    values,
    xy_new,
    variogram: Variogram,
    *,
    n_closest: Optional[int] = None,
    chunk_bytes: float = 1.5e8,
) -> Tuple[np.ndarray, np.ndarray]:
    """Ordinary kriging at scattered locations.

    Parameters
    ----------
    xy : (n, 2) array
        Data locations.
    values : (n,) or (n, k) array
        Data values. Several columns are kriged with the same weights (one
        factorisation), e.g. the indicators of all classes.
    xy_new : (m, 2) array
        Locations to estimate.
    variogram : Variogram
        Variogram model (with its anisotropy).
    n_closest : int, optional
        ``None`` (default) uses all data for every estimate: one system is
        factorised once, so the estimate is continuous. A number uses only the
        `n_closest` nearest data for each estimate (moving neighbourhood), which
        scales to very large data sets.

    Returns
    -------
    estimate : (m,) or (m, k) array
    variance : (m,) array
        Kriging variance, in the units of the variogram.
    """
    xy = variogram.transform(np.asarray(xy, dtype=float))
    x0 = variogram.transform(np.asarray(xy_new, dtype=float))
    vals = np.asarray(values, dtype=float)
    one_d = vals.ndim == 1
    if one_d:
        vals = vals[:, None]
    if xy.shape[0] != vals.shape[0]:
        raise ValueError("xy and values must have the same number of rows.")
    if xy.shape[0] == 0:
        raise ValueError("ordinary_kriging() needs at least one data point.")

    # Exact duplicates make the system singular: keep the first of each
    _, first = np.unique(xy, axis=0, return_index=True)
    if first.size < xy.shape[0]:
        first.sort()
        xy, vals = xy[first], vals[first]

    n, m, k = xy.shape[0], x0.shape[0], vals.shape[1]
    est = np.empty((m, k), dtype=float)
    var = np.empty(m, dtype=float)
    if m == 0:
        return (est[:, 0] if one_d else est), var

    if n_closest is not None and 0 < n_closest < n:
        kc = int(n_closest)
        tree = cKDTree(xy)
        step = max(1, int(chunk_bytes // (8 * 3 * (kc + 1) ** 2)))
        for a in range(0, m, step):
            b = min(a + step, m)
            dist, idx = tree.query(x0[a:b], k=kc)
            P = xy[idx]                                         # (mc, kc, 2)
            D = np.linalg.norm(P[:, :, None, :] - P[:, None, :, :], axis=-1)
            A = np.zeros((b - a, kc + 1, kc + 1))
            A[:, :kc, :kc] = variogram.gamma(D)
            A[:, :kc, kc] = 1.0
            A[:, kc, :kc] = 1.0
            rhs = np.ones((b - a, kc + 1))
            rhs[:, :kc] = variogram.gamma(dist)
            w = np.linalg.solve(A, rhs[..., None])[..., 0]
            est[a:b] = np.einsum("ij,ijk->ik", w[:, :kc], vals[idx])
            var[a:b] = np.einsum("ij,ij->i", w, rhs)
    else:
        A = np.empty((n + 1, n + 1))
        A[:n, :n] = variogram.gamma(cdist(xy, xy))
        A[:n, n] = 1.0
        A[n, :n] = 1.0
        A[n, n] = 0.0
        lu = lu_factor(A, check_finite=False)
        step = max(1, int(chunk_bytes // (8 * (n + 1))))
        for a in range(0, m, step):
            b = min(a + step, m)
            rhs = np.empty((n + 1, b - a))
            rhs[:n] = variogram.gamma(cdist(xy, x0[a:b]))
            rhs[n] = 1.0
            w = lu_solve(lu, rhs, check_finite=False)
            est[a:b] = w[:n].T @ vals
            var[a:b] = np.einsum("ij,ij->j", w, rhs)

    var = np.maximum(var, 0.0)
    return (est[:, 0] if one_d else est), var


def poly_design(uv, order: int, center, scale: float) -> np.ndarray:
    """Design matrix of a 2D polynomial of total degree <= `order`, on coordinates
    normalised as ``(uv - center) / scale``."""
    u, v = ((np.asarray(uv, dtype=float) - center) / scale).T
    cols = [u ** i * v ** (d - i) for d in range(order + 1) for i in range(d + 1)]
    return np.column_stack(cols)


# ---------------------------------------------------------------------------
# Holes on an occupancy grid
# ---------------------------------------------------------------------------
def find_holes(uv, cell: float, *, extra_uv=None, min_cells: int = 4):
    """Locate the missing parts of a 2D point set.

    The points are rasterised on a grid of square cells. A cell is missing if it
    is empty and either enclosed by occupied cells, or contains one of the
    `extra_uv` locations (e.g. points that were removed). Connected missing
    cells form a hole.

    Parameters
    ----------
    uv : (n, 2) array
        Point locations.
    cell : float
        Cell size.
    extra_uv : (r, 2) array, optional
        Locations known to be missing.
    min_cells : int
        Holes with fewer cells are ignored (sampling noise).

    Returns
    -------
    labels : (nu, nv) int array
        0 where nothing is missing, otherwise the hole number (1..n_holes).
    n_holes : int
    origin : (2,) array
        Coordinates of the corner of cell ``[0, 0]``.
    counts : (nu, nv) int array
        Number of points per cell.
    """
    uv = np.asarray(uv, dtype=float)
    if cell <= 0 or not np.isfinite(cell):
        raise ValueError("cell must be > 0.")
    lo, hi = uv.min(axis=0), uv.max(axis=0)
    extra = None
    if extra_uv is not None and len(extra_uv) > 0:
        extra = np.asarray(extra_uv, dtype=float)
        extra = extra[np.isfinite(extra).all(axis=1)]
        if extra.shape[0] > 0:
            lo, hi = np.minimum(lo, extra.min(axis=0)), np.maximum(hi, extra.max(axis=0))
        else:
            extra = None
    origin = lo - cell                                           # one cell of padding
    shape = tuple(np.floor((hi - origin) / cell).astype(np.int64) + 2)

    ij = np.floor((uv - origin) / cell).astype(np.int64)
    counts = np.bincount(ij[:, 0] * shape[1] + ij[:, 1], minlength=shape[0] * shape[1]).reshape(shape)
    occ = counts > 0

    missing = ndimage.binary_fill_holes(occ) & ~occ
    if extra is not None:
        rij = np.floor((extra - origin) / cell).astype(np.int64)
        rem = np.zeros(shape, dtype=bool)
        rem[rij[:, 0], rij[:, 1]] = True
        missing |= rem & ~occ

    labels, n = ndimage.label(missing, structure=np.ones((3, 3), dtype=int))
    if n > 0 and min_cells > 1:
        sizes = np.bincount(labels.ravel(), minlength=n + 1)
        small = sizes < min_cells
        small[0] = False
        if small.any():
            labels[small[labels]] = 0
            labels, n = ndimage.label(labels > 0, structure=np.ones((3, 3), dtype=int))
    return labels, int(n), origin, counts


def _thin(uv, idx, max_n: int, rng) -> np.ndarray:
    """Spatially even subset of `idx` with about `max_n` points: one random point
    per cell of a coarse grid."""
    if idx.size <= max_n:
        return idx
    idx = rng.permutation(idx)
    p = uv[idx]
    lo = p.min(axis=0)
    ext = np.maximum(p.max(axis=0) - lo, 1e-12)
    g = np.sqrt(ext[0] * ext[1] / max_n)
    pick = idx
    for _ in range(4):
        c = np.floor((p - lo) / g).astype(np.int64)
        _, first = np.unique(c[:, 0] * (c[:, 1].max() + 1) + c[:, 1], return_index=True)
        pick = idx[first]
        if 0.8 * max_n <= pick.size <= max_n:
            break
        g *= np.sqrt(pick.size / (0.9 * max_n))   # the points do not fill their bounding box
    if pick.size > max_n:
        pick = rng.choice(pick, size=max_n, replace=False)
    return np.sort(pick)


def _class_label(c) -> str:
    c = float(c)
    return f"{int(c)}" if c.is_integer() else f"{c:g}".replace(".", "_").replace("-", "m")


def fill_holes_uv(
    uvw,
    attrs: Optional[Dict[str, Tuple[np.ndarray, str]]] = None,
    *,
    cell: float,
    removed_uv=None,
    spacing: Optional[float] = None,
    min_cells: int = 4,
    ring_scale: float = 0.5,
    attr_ring_scale: float = 1.0,
    max_train: int = 2000,
    n_closest: Optional[int] = None,
    variogram: str = "spherical",
    attr_variogram: Optional[str] = None,
    trend_order: int = 1,
    nlags: int = 20,
    aniso: float = 10.0,
    angle: float = 0.0,
    w_aniso: float = 1.0,
    shared_variogram: bool = True,
    seed: int = 0,
) -> dict:
    """Create points in the holes of a point set and krige their height and attributes.

    Parameters
    ----------
    uvw : (n, 3) array
        Existing points: ``(u, v)`` in the plane and height ``w`` above it.
    attrs : dict
        ``name -> (values, kind)`` with ``values`` of shape ``(n,)`` or ``(n, k)``
        and ``kind`` either ``"categorical"`` (indicator kriging: a probability
        per class) or ``"continuous"`` (ordinary kriging).
    cell : float
        Cell size of the occupancy grid used to find the holes.
    removed_uv : (r, 2) array, optional
        Locations of points that were removed; lets holes that reach the edge of
        the point set be found.
    spacing : float, optional
        Spacing of the new points. Defaults to the spacing that matches the
        density of the existing points.
    min_cells : int
        Smallest hole, in cells.
    ring_scale, attr_ring_scale : float
        Width of the ring of existing points used for the height and for the
        attributes, as a fraction of the hole's equivalent diameter.
    max_train : int
        Largest number of ring points used per kriging system (the ring is thinned
        evenly to this number).
    n_closest : int, optional
        Moving neighbourhood size (see :func:`ordinary_kriging`).
    variogram, attr_variogram : str
        Variogram model for the height and for the attributes (default: the same).
    trend_order : int
        Order (0, 1 or 2) of the polynomial trend removed from ``w`` before kriging.
    aniso, angle : float
        Anisotropy of the attributes: ranges along the direction `angle` (degrees
        CCW from ``u``) are `aniso` times longer than across it.
    w_aniso : float
        Same ratio for the height ``w`` (1 = isotropic), along the same direction.
    shared_variogram : bool
        Use one variogram for all classes of a categorical attribute (fitted on the
        class whose proportion is closest to 0.5), so the probabilities sum to 1
        before any correction.

    Returns
    -------
    dict with keys

    ``uvw`` (m, 3), ``std`` (m,) kriging standard deviation of ``w``, ``hole``
    (m,) hole number of each new point, ``nearest`` (m,) index of the nearest
    existing point, ``attrs`` (``name -> dict``) and ``holes`` (one dict of
    diagnostics per hole).
    """
    uvw = np.asarray(uvw, dtype=float)
    uv, w = uvw[:, :2], uvw[:, 2]
    attrs = {} if attrs is None else attrs
    attr_variogram = variogram if attr_variogram is None else attr_variogram
    rng = np.random.default_rng(seed)

    labels, n_holes, origin, counts = find_holes(uv, cell, extra_uv=removed_uv, min_cells=min_cells)
    shape = labels.shape
    if spacing is not None and (spacing <= 0 or not np.isfinite(spacing)):
        raise ValueError("spacing must be > 0.")
    # Spacing that matches the density of the existing points (per cell of the grid)
    default_spacing = cell / np.sqrt(np.median(counts[counts > 0]))

    # Per-attribute bookkeeping
    spec = {}
    for name, (values, kind) in attrs.items():
        values = np.asarray(values)
        if values.shape[0] != uv.shape[0]:
            raise ValueError(f"Attribute '{name}' must have one row per point.")
        if kind == "categorical":
            if values.ndim != 1:
                raise ValueError(f"Categorical attribute '{name}' must be 1-D.")
            spec[name] = dict(kind=kind, values=values, classes=np.unique(values))
        elif kind == "continuous":
            spec[name] = dict(kind=kind, values=values.astype(float))
        else:
            raise ValueError("kind must be 'categorical' or 'continuous'.")

    out_uvw, out_std, out_hole, out_near = [], [], [], []
    out_attr = {name: dict(value=[], std=[], probs=[]) for name in spec}
    holes_info: List[dict] = []

    flat = np.floor((uv - origin) / cell).astype(np.int64)
    flat = flat[:, 0] * shape[1] + flat[:, 1]
    boxes = ndimage.find_objects(labels)

    for h in range(1, n_holes + 1):
        box = boxes[h - 1]
        n_cells = int(np.count_nonzero(labels[box] == h))
        area = n_cells * cell * cell
        d_eq = 2.0 * np.sqrt(area / np.pi)
        ring_g = max(ring_scale * d_eq, 3.0 * cell)
        ring_a = max(attr_ring_scale * d_eq, 3.0 * cell)
        pad = int(np.ceil(max(ring_g, ring_a) / cell)) + 2
        i0, i1 = max(box[0].start - pad, 0), min(box[0].stop + pad, shape[0])
        j0, j1 = max(box[1].start - pad, 0), min(box[1].stop + pad, shape[1])
        sub = labels[i0:i1, j0:j1] == h
        dist = ndimage.distance_transform_edt(~sub) * cell      # distance of each cell to the hole

        def points_within(radius):
            grid = np.zeros(shape, dtype=bool)
            grid[i0:i1, j0:j1] = (dist <= radius) & ~sub
            return np.flatnonzero(grid.ravel()[flat])

        # ---- spacing of the new points: match the density around the hole
        sp = spacing
        if sp is None:
            local = counts[i0:i1, j0:j1][(dist > 1.5 * cell) & (dist <= max(ring_g, 4.0 * cell))]
            local = local[local > 0]
            sp = cell / np.sqrt(np.median(local)) if local.size else default_spacing

        # ---- target points: a regular grid over the hole and its rim cells,
        #      away from the existing points
        grow = ndimage.binary_dilation(sub, structure=np.ones((3, 3), dtype=bool))
        win_origin = origin + np.array([i0, j0]) * cell
        gi, gj = np.nonzero(grow)
        lo = win_origin + np.array([gi.min(), gj.min()]) * cell
        hi = win_origin + np.array([gi.max() + 1, gj.max() + 1]) * cell
        gu = lo[0] + (np.arange(int(np.ceil((hi[0] - lo[0]) / sp))) + 0.5) * sp
        gv = lo[1] + (np.arange(int(np.ceil((hi[1] - lo[1]) / sp))) + 0.5) * sp
        GU, GV = np.meshgrid(gu, gv, indexing="ij")
        tgt = np.column_stack([GU.ravel(), GV.ravel()])
        c = np.floor((tgt - win_origin) / cell).astype(np.int64)
        inside = (c[:, 0] >= 0) & (c[:, 0] < sub.shape[0]) & (c[:, 1] >= 0) & (c[:, 1] < sub.shape[1])
        inside[inside] = grow[c[inside, 0], c[inside, 1]]
        tgt = tgt[inside]

        idx_near = points_within(2.5 * cell)
        nearest = np.full(tgt.shape[0], -1, dtype=np.int64)
        if idx_near.size > 0 and tgt.shape[0] > 0:
            dn, jn = cKDTree(uv[idx_near]).query(tgt, k=1)
            keep = dn >= 0.8 * sp
            tgt, nearest = tgt[keep], idx_near[jn[keep]]
        m = tgt.shape[0]

        info = dict(hole=h, n_cells=n_cells, area=float(area), diameter=float(d_eq), n_new=int(m), spacing=float(sp),
                    centre=(win_origin + (np.argwhere(sub).mean(axis=0) + 0.5) * cell).tolist())
        idx_g = _thin(uv, points_within(ring_g), max_train, rng)
        if m == 0 or idx_g.size < 3:
            info["skipped"] = "no target points" if m == 0 else "too few points around the hole"
            holes_info.append(info)
            continue

        # ---- height above the plane: trend + kriged residual
        ug, wg = uv[idx_g], w[idx_g]
        order = int(trend_order) if idx_g.size >= 10 else 0
        center = ug.mean(axis=0)
        scale = float(np.abs(ug - center).max()) or 1.0
        design = poly_design(ug, order, center, scale)
        coef, *_ = np.linalg.lstsq(design, wg, rcond=None)
        resid = wg - design @ coef
        vg = fit_variogram(ug, resid, variogram, nlags=nlags, aniso=w_aniso, angle=angle, seed=seed)
        r_new, var = ordinary_kriging(ug, resid, tgt, vg, n_closest=n_closest)
        w_new = poly_design(tgt, order, center, scale) @ coef + r_new
        std = np.sqrt(var)
        info.update(n_train=int(idx_g.size), ring=float(ring_g), w_variogram=vg.to_dict(),
                    w_residual_std=float(resid.std()))

        out_uvw.append(np.column_stack([tgt, w_new]))
        out_std.append(std)
        out_hole.append(np.full(m, h, dtype=np.int64))
        out_near.append(nearest)

        # ---- attributes
        if spec:
            idx_a = _thin(uv, points_within(ring_a), max_train, rng)
            ua = uv[idx_a]
            info["n_train_attr"] = int(idx_a.size)
            info["attr_variograms"] = {}
        for name, sp in spec.items():
            vals = sp["values"][idx_a]
            if sp["kind"] == "categorical":
                classes = sp["classes"]
                ind = (vals[:, None] == classes[None, :]).astype(float)
                present = ind.any(axis=0)
                P = np.zeros((m, classes.size))
                if present.sum() <= 1:
                    P[:, present] = 1.0
                elif shared_variogram:
                    props = ind.mean(axis=0)
                    ref = int(np.argmin(np.where(present, np.abs(props - 0.5), np.inf)))
                    va = fit_variogram(ua, ind[:, ref], attr_variogram, nlags=nlags,
                                       aniso=aniso, angle=angle, seed=seed)
                    P[:, present], _ = ordinary_kriging(ua, ind[:, present], tgt, va, n_closest=n_closest)
                    info["attr_variograms"][name] = dict(va.to_dict(), from_class=_class_label(classes[ref]))
                else:
                    info["attr_variograms"][name] = {}
                    for j in np.flatnonzero(present):
                        va = fit_variogram(ua, ind[:, j], attr_variogram, nlags=nlags,
                                           aniso=aniso, angle=angle, seed=seed)
                        P[:, j], _ = ordinary_kriging(ua, ind[:, j], tgt, va, n_closest=n_closest)
                        info["attr_variograms"][name][_class_label(classes[j])] = va.to_dict()
                # Order-relation correction: clip to [0, 1], renormalise to sum 1
                P_raw = P
                P = np.clip(P_raw, 0.0, 1.0)
                tot = P.sum(axis=1, keepdims=True)
                dead = tot[:, 0] <= 0
                if dead.any():                       # every class clipped to 0: raw argmax
                    P[dead] = np.eye(classes.size)[P_raw[dead].argmax(axis=1)]
                    tot[dead] = 1.0
                P /= tot
                out_attr[name]["probs"].append(P)
                out_attr[name]["value"].append(classes[P.argmax(axis=1)])
            else:
                col = vals if vals.ndim == 1 else vals.mean(axis=1)
                va = fit_variogram(ua, col, attr_variogram, nlags=nlags, aniso=aniso, angle=angle, seed=seed)
                est, var_a = ordinary_kriging(ua, vals, tgt, va, n_closest=n_closest)
                out_attr[name]["value"].append(est)
                out_attr[name]["std"].append(np.sqrt(var_a))
                info["attr_variograms"][name] = va.to_dict()
        holes_info.append(info)

    def cat(parts, empty):
        return np.concatenate(parts, axis=0) if parts else empty

    result_attrs = {}
    for name, sp in spec.items():
        o = out_attr[name]
        if sp["kind"] == "categorical":
            K = sp["classes"].size
            result_attrs[name] = dict(
                kind="categorical", classes=sp["classes"],
                value=cat(o["value"], np.empty(0, dtype=sp["values"].dtype)),
                probs=cat(o["probs"], np.empty((0, K))),
            )
        else:
            result_attrs[name] = dict(
                kind="continuous",
                value=cat(o["value"], np.empty((0,) + sp["values"].shape[1:])),
                std=cat(o["std"], np.empty(0)),
            )
    return dict(
        uvw=cat(out_uvw, np.empty((0, 3))),
        std=cat(out_std, np.empty(0)),
        hole=cat(out_hole, np.empty(0, dtype=np.int64)),
        nearest=cat(out_near, np.empty(0, dtype=np.int64)),
        attrs=result_attrs,
        holes=holes_info,
        spacing=float(default_spacing if spacing is None else spacing),
        cell=float(cell),
    )
