"""
_kriging.py
===========

Ordinary kriging of scattered 2D data and hole filling in the ``(u, v)`` plane
of a :class:`~._frame.CurvilinearFrame`.

- :class:`Variogram`, :func:`fit_variogram` and :func:`ordinary_kriging` are
  general-purpose and work on plain arrays.
- :class:`NestedVariogram` and :func:`fit_nested_variogram` describe surface
  texture: a nugget, a short-range and a long-range structure, fitted from all
  point pairs at short lags (so roughness below the spacing of a thinned
  training set is seen) and from a thinned set at long lags.
- :func:`surface_mask` flags points off the main surface (second layers, loose
  blocks, steps), which a height field cannot represent.
- :func:`find_holes` locates the missing parts of a point set on an occupancy
  grid and :func:`fill_holes_uv` creates new points there, kriging their height
  ``w`` above the ``(u, v)`` plane and any number of attributes.
- :class:`GaussianField`, :class:`RandomField` and
  :func:`conditional_simulation` draw spatially correlated realisations that
  reproduce a variogram and honour the data.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import ndimage
from scipy.linalg import lu_factor, lu_solve
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.fft import next_fast_len
from scipy.spatial.distance import cdist, pdist

__all__ = [
    "VARIOGRAM_MODELS",
    "Variogram",
    "NestedVariogram",
    "variogram_from_dict",
    "fit_variogram",
    "fit_nested_variogram",
    "surface_mask",
    "ordinary_kriging",
    "poly_design",
    "find_holes",
    "fill_holes_uv",
    "GaussianField",
    "RandomField",
    "conditional_simulation",
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

    def components(self) -> List["Variogram"]:
        """The structures without the nugget (here: one)."""
        return [Variogram(self.model, self.psill, self.range, 0.0, self.aniso, self.angle)]


@dataclass
class NestedVariogram:
    """Sum of variogram structures plus a nugget, with one geometric anisotropy.

    Attributes
    ----------
    structures : list of (model, psill, range)
        Each structure is a :class:`Variogram` without nugget.
    nugget, aniso, angle : float
        As in :class:`Variogram`.
    """
    structures: List[Tuple[str, float, float]] = field(default_factory=list)
    nugget: float = 0.0
    aniso: float = 1.0
    angle: float = 0.0

    def __post_init__(self):
        self.structures = [(str(m), float(p), float(r)) for m, p, r in self.structures]
        for m, _, _ in self.structures:
            if m not in VARIOGRAM_MODELS:
                raise ValueError(f"Unknown variogram model '{m}'. Use one of {VARIOGRAM_MODELS}.")

    @property
    def psill(self) -> float:
        """Sum of the partial sills of the structures."""
        return float(sum(p for _, p, _ in self.structures))

    def components(self) -> List[Variogram]:
        """The structures, as :class:`Variogram` objects without nugget."""
        return [Variogram(m, p, r, 0.0, self.aniso, self.angle) for m, p, r in self.structures]

    def gamma(self, h) -> np.ndarray:
        """Semivariance at (isotropic) distance ``h``."""
        h = np.asarray(h, dtype=float)
        g = np.zeros(h.shape)
        for c in self.components():
            g = g + c.gamma(h)
        return np.where(h > 0, g + self.nugget, 0.0)

    def transform(self, xy) -> np.ndarray:
        """Rotate and scale coordinates so that the model is isotropic in them."""
        return Variogram(aniso=self.aniso, angle=self.angle).transform(xy)

    def to_dict(self) -> dict:
        return dict(structures=[list(s) for s in self.structures], nugget=self.nugget,
                    aniso=self.aniso, angle=self.angle)


AnyVariogram = Union[Variogram, NestedVariogram]


def variogram_from_dict(d: dict) -> AnyVariogram:
    """Rebuild a :class:`Variogram` or :class:`NestedVariogram` from its ``to_dict()``
    (unknown keys, such as ``"from_class"``, are ignored)."""
    if "structures" in d:
        return NestedVariogram(**{k: d[k] for k in ("structures", "nugget", "aniso", "angle") if k in d})
    return Variogram(**{k: d[k] for k in ("model", "psill", "range", "nugget", "aniso", "angle") if k in d})


def surface_mask(uv, w, *, radius: Optional[float] = None, tol: float = 0.15, k: int = 24, n_iter: int = 3) -> np.ndarray:
    """Points on the main surface of a height field ``w(u, v)``.

    A point is off the surface when its ``w`` differs by more than `tol` from the
    median ``w`` of its `k` nearest neighbours in ``(u, v)`` (only those within
    `radius`, if given), so the neighbourhood adapts to the point density.
    This flags second layers of points (loose blocks, debris, vegetation),
    isolated strays and the points of steps where the face is nearly edge-on to
    the ``(u, v)`` plane: features a single-valued height field cannot represent.

    Each further pass (`n_iter` in all) takes the median over the neighbours found
    on the surface by the previous pass, which also catches layer points whose
    few neighbours happen to be mostly on the layer.

    Returns
    -------
    (n,) bool array, True on the main surface.
    """
    uv = np.asarray(uv, dtype=float)
    w = np.asarray(w, dtype=float).ravel()
    n = uv.shape[0]
    if n == 0:
        return np.zeros(0, dtype=bool)
    _, idx = cKDTree(uv).query(uv, k=min(int(k), n),
                               distance_upper_bound=np.inf if radius is None else float(radius))
    idx = idx.reshape(n, -1)
    vals = np.append(w, np.nan)[idx]                        # missing neighbours -> NaN
    on = np.abs(w - np.nanmedian(vals, axis=1)) <= tol      # the point itself is always there
    for _ in range(max(int(n_iter), 1) - 1):
        good = np.where(np.append(on, False)[idx], vals, np.nan)   # only neighbours on the surface
        has = np.isfinite(good).any(axis=1)
        med = np.full(n, np.nan)
        med[has] = np.nanmedian(good[has], axis=1)
        on = np.where(has, np.abs(w - med) <= tol, on)
    return on


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


def _binned_semivariance(d, adz, edges, robust: bool):
    """Experimental semivariance of pairs at distance `d` with ``|z_i - z_j| = adz``,
    in the bins `edges`. Robust: Cressie-Hawkins estimator."""
    nb = len(edges) - 1
    k = np.searchsorted(edges, d, side="right") - 1
    ok = (d > 0) & (k >= 0) & (k < nb)
    k, d, adz = k[ok], d[ok], adz[ok]
    cnt = np.bincount(k, minlength=nb).astype(float)
    m = cnt > 0
    lag = np.bincount(k, weights=d, minlength=nb)[m] / cnt[m]
    if robust:
        s = np.bincount(k, weights=np.sqrt(adz), minlength=nb)[m] / cnt[m]
        g = 0.5 * s ** 4 / (0.457 + 0.494 / cnt[m])
    else:
        g = np.bincount(k, weights=0.5 * adz ** 2, minlength=nb)[m] / cnt[m]
    return lag, g, cnt[m]


def fit_nested_variogram(
    xy,
    values,
    model: str = "spherical",
    *,
    short_model: str = "gaussian",
    dense_lag: float = 3.0,
    n_centres: int = 2000,
    nlags: int = 20,
    aniso: float = 1.0,
    angle: float = 0.0,
    max_points: int = 2000,
    max_lag: Optional[float] = None,
    robust: bool = True,
    min_pairs: int = 30,
    seed: int = 0,
) -> NestedVariogram:
    """Fit nugget + short-range structure + long-range structure to dense data.

    :func:`fit_variogram` works on a random subset of the data, so it cannot see
    variation over distances shorter than the spacing of that subset and
    extrapolates it (usually to zero): a model fitted that way is too smooth.
    Here the experimental variogram is computed

    - at lags up to `dense_lag` from **all** pairs between `n_centres` random
      points and every data point (fine bins, down to the point spacing), and
    - at longer lags from all pairs of `max_points` random points,

    and ``nugget + short_model(c1, a1) + model(c2, a2)`` is fitted to both by
    least squares on the relative misfit ``gamma_model / gamma_exp - 1``, so
    the short lags (small semivariances, the texture) weigh as much as the
    long ones, with ``a1 <= dense_lag <= a2 <= 2 * max_lag``. The default
    short structure is Gaussian (parabolic at the origin) and the long one
    spherical (linear): together they follow the ``h^1.3 - h^1.7`` growth that
    rough rock surfaces show at short lags.

    Parameters
    ----------
    xy : (n, 2) array
        Data locations.
    values : (n,) array
        Data values (e.g. the detrended height of a surface).
    model, short_model : str
        Models of the long- and short-range structures ("spherical",
        "exponential" or "gaussian").
    dense_lag : float
        Longest lag computed from all pairs; boundary between the two ranges.
    n_centres : int
        Number of random points whose pairs with every point are used below
        `dense_lag`.
    nlags : int
        Number of long-lag bins.
    aniso, angle : float
        Geometric anisotropy (imposed, not fitted).
    max_points : int
        Size of the random subset used for long lags.
    max_lag : float, optional
        Largest lag. Defaults to half the largest distance in the subset.
    robust : bool
        Cressie-Hawkins estimator, less sensitive to the odd large difference.
    min_pairs : int
        Bins with fewer pairs are ignored.

    Returns
    -------
    NestedVariogram
        Structures ``[(short_model, c1, a1), (model, c2, a2)]`` and the nugget.
    """
    for m in (model, short_model):
        if m not in VARIOGRAM_MODELS or m == "linear":
            raise ValueError(f"Nested variograms need bounded models: 'spherical', 'exponential' or 'gaussian', not '{m}'.")
    out = NestedVariogram(aniso=float(aniso), angle=float(angle))
    p = out.transform(np.asarray(xy, dtype=float))
    values = np.asarray(values, dtype=float).ravel()
    n = p.shape[0]
    var = float(values.var()) if n > 0 else 0.0
    if n < 3 or var <= 0:
        out.structures = [(model, max(var, 1e-12), 1.0)]
        return out
    rng = np.random.default_rng(seed)

    # ---- long lags: all pairs of a random subset
    sub = rng.choice(n, size=min(n, max_points), replace=False)
    d_l = pdist(p[sub])
    adz_l = pdist(values[sub, None])
    if max_lag is None:
        max_lag = 0.5 * float(d_l.max())
    if not np.isfinite(max_lag) or max_lag <= 0:
        out.structures = [(model, var, 1.0)]
        return out
    dense_lag = float(min(dense_lag, max_lag))

    # ---- short lags: every pair between random centres and all points
    cen = rng.choice(n, size=min(n, n_centres), replace=False)
    pairs = cKDTree(p[cen]).sparse_distance_matrix(cKDTree(p), dense_lag, output_type="ndarray")
    d_s = pairs["v"]
    adz_s = np.abs(values[cen[pairs["i"]]] - values[pairs["j"]])
    edges_s = dense_lag * np.r_[0.0, np.geomspace(0.03, 1.0, 12)]
    lag_s, g_s, c_s = _binned_semivariance(d_s, adz_s, edges_s, robust)
    if max_lag > dense_lag:
        lag_l, g_l, c_l = _binned_semivariance(d_l, adz_l, np.linspace(dense_lag, max_lag, nlags + 1), robust)
    else:
        lag_l = g_l = c_l = np.empty(0)
    lag, g, cnt = np.r_[lag_s, lag_l], np.r_[g_s, g_l], np.r_[c_s, c_l]
    keep = cnt >= min_pairs
    lag, g = lag[keep], g[keep]
    if lag.size < 5:
        single = fit_variogram(xy, values, model, nlags=nlags, aniso=aniso, angle=angle,
                               max_points=max_points, max_lag=max_lag, seed=seed)
        out.structures, out.nugget = [(model, single.psill, single.range)], single.nugget
        return out

    s_max = float(max(g.max(), 1e-12))
    g_floor = np.maximum(g, 1e-6 * s_max)
    a1_lo = float(max(lag.min(), 1e-6))
    a1_hi = float(max(dense_lag, 1.01 * a1_lo))
    a2_lo = float(dense_lag)
    # The data say nothing beyond max_lag; a longer range would also exceed the
    # size of the domain, which random fields cannot embed exactly (see GaussianField)
    a2_hi = float(max(2.0 * max_lag, 1.01 * a2_lo))
    lo = np.array([0.0, 0.0, a1_lo, 0.0, a2_lo])
    hi = np.array([s_max, 10 * s_max, a1_hi, 10 * s_max, a2_hi])
    x0 = np.clip([0.5 * g[0], 0.25 * s_max, 0.5 * (a1_lo + a1_hi), 0.75 * s_max, 0.5 * max_lag],
                 lo + 1e-9 * (hi - lo), hi - 1e-9 * (hi - lo))
    trial = NestedVariogram(aniso=1.0, angle=0.0)

    def resid(q):
        trial.nugget = q[0]
        trial.structures = [(short_model, q[1], q[2]), (model, q[3], q[4])]
        return trial.gamma(lag) / g_floor - 1.0        # relative misfit: short lags count as much as long

    q = least_squares(resid, x0, bounds=(lo, hi), loss="soft_l1").x
    out.nugget = float(q[0])
    out.structures = [(short_model, float(q[1]), float(q[2])), (model, float(q[3]), float(q[4]))]
    return out


# ---------------------------------------------------------------------------
# Ordinary kriging
# ---------------------------------------------------------------------------
def ordinary_kriging(
    xy,
    values,
    xy_new,
    variogram: AnyVariogram,
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
    variogram : Variogram or NestedVariogram
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
    dense_variogram: bool = True,
    dense_lag: float = 3.0,
    surface_tol: Optional[float] = 0.15,
    surface_radius: Optional[float] = None,
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
    dense_variogram : bool
        Fit the variogram of ``w`` with :func:`fit_nested_variogram` (nugget +
        short-range + long-range structure, short lags from all ring points), so
        it captures the roughness of the surface. False fits one structure on the
        thinned ring (:func:`fit_variogram`). A ``"linear"`` model always uses
        the latter.
    dense_lag : float
        Boundary between the short- and long-range structures.
    surface_tol, surface_radius : float
        Ring points more than `surface_tol` from the median ``w`` of their 24
        nearest neighbours (within `surface_radius`, if given; see
        :func:`surface_mask`) are left out of the height trend, variogram and
        kriging. ``surface_tol=None`` keeps every point.

    Returns
    -------
    dict with keys

    ``uvw`` (m, 3), ``std`` (m,) kriging standard deviation of ``w``, ``hole``
    (m,) hole number of each new point, ``nearest`` (m,) index of the nearest
    existing point, ``attrs`` (``name -> dict``) and ``holes`` (one dict of
    diagnostics per hole, including ``w_variogram`` and ``train_idx``, the
    indices of the points the height was kriged from, which
    :func:`conditional_simulation` needs to draw realisations consistent with
    the kriging).
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
        ring_pts = points_within(ring_g)
        if surface_tol is not None and ring_pts.size >= 3:
            on = surface_mask(uv[ring_pts], w[ring_pts], radius=surface_radius, tol=surface_tol)
            info["off_surface"] = float(1.0 - on.mean())
            if on.sum() >= 3:
                ring_pts = ring_pts[on]
        idx_g = _thin(uv, ring_pts, max_train, rng)
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
        if dense_variogram and variogram != "linear" and ring_pts.size >= 10:
            resid_all = w[ring_pts] - poly_design(uv[ring_pts], order, center, scale) @ coef
            vg = fit_nested_variogram(uv[ring_pts], resid_all, variogram, dense_lag=dense_lag, nlags=nlags,
                                      aniso=w_aniso, angle=angle, max_points=max_train, seed=seed)
        else:
            vg = fit_variogram(ug, resid, variogram, nlags=nlags, aniso=w_aniso, angle=angle, seed=seed)
        r_new, var = ordinary_kriging(ug, resid, tgt, vg, n_closest=n_closest)
        w_new = poly_design(tgt, order, center, scale) @ coef + r_new
        std = np.sqrt(var)
        info.update(n_train=int(idx_g.size), ring=float(ring_g), w_variogram=vg.to_dict(),
                    w_residual_std=float(resid.std()), train_idx=idx_g)

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


# ---------------------------------------------------------------------------
# Spatially correlated random fields
# ---------------------------------------------------------------------------
class GaussianField:
    """Unit-variance stationary Gaussian random field at fixed 2D points.

    The correlation is that of a :class:`Variogram` (shape, range, anisotropy
    and nugget): ``rho(h) = 1 - gamma(h) / (psill + nugget)`` away from the
    origin. The field is simulated on a regular grid by circulant embedding
    (FFT) and interpolated bilinearly to the points. A bilinear average of
    correlated grid values has a variance slightly below 1 (the variation inside
    a grid cell is lost); the missing variance is added back per point as
    independent noise, so the field has unit variance at every point and the
    model correlation between points more than a cell apart. (Rescaling the
    interpolated value instead would inflate those correlations, making the
    field too smooth.) The nugget is added as independent noise.

    A ``"linear"`` variogram has no stationary covariance; it is replaced by a
    spherical one with the same range.

    Inside a grid cell the field is the smooth bilinear part plus that
    independent noise, so the short-range structure is right only if the grid
    is about as fine as the points: pass their spacing as `resolution` when it
    matters (rough surfaces, spherical or exponential models).

    Parameters
    ----------
    xy : (n, 2) array
        Points at which the field is evaluated.
    variogram : Variogram
        Correlation model.
    cells_per_range : int
        Grid resolution, in cells per range (across the shortest axis).
    resolution : float, optional
        Finest grid spacing needed, in the units of `xy` (e.g. the spacing of the
        points). The grid spacing is the smaller of this and
        ``range / cells_per_range``, unless `max_cells` forces a coarser grid.
        Cells are stretched across the major axis by the anisotropy ratio.
    max_cells : int
        Largest number of grid cells along an axis (the grid is coarsened if
        needed).
    """

    def __init__(self, xy, variogram: "Variogram", *, cells_per_range: int = 20,
                 resolution: Optional[float] = None, max_cells: int = 1024):
        model = variogram.model if variogram.model != "linear" else "spherical"
        total = float(variogram.psill + variogram.nugget)
        self.nugget = float(np.clip(variogram.nugget / total, 0.0, 1.0)) if total > 0 else 0.0
        self.range = max(float(variogram.range), 1e-9)
        self._unit = Variogram(model=model, psill=1.0, range=self.range, nugget=0.0)
        pts = variogram.transform(np.asarray(xy, dtype=float))        # isotropic coordinates

        lo, hi = pts.min(axis=0), pts.max(axis=0)
        # Grid spacing per axis; in the isotropic coordinates the points are
        # `aniso` times further apart across the major axis
        want = np.full(2, self.range / cells_per_range)
        if resolution is not None and resolution > 0:
            want = np.minimum(want, float(resolution) * np.array([1.0, max(float(variogram.aniso), 1.0)]))
        dx = np.maximum(np.maximum(want, (hi - lo) / max_cells), 1e-9)
        n = np.ceil((hi - lo) / dx).astype(int) + 2                    # grid nodes per axis
        reach = self.range if model == "spherical" else 2.0 * self.range   # correlation ~ 0 beyond
        pad = np.minimum(np.ceil(reach / dx), n).astype(int)          # beyond the domain: at most its size
        M = [next_fast_len(int(2 * (ni + pi))) for ni, pi in zip(n, pad)]
        lags = [np.minimum(np.arange(m), m - np.arange(m)) * d for m, d in zip(M, dx)]
        c = self._rho(np.hypot(lags[0][:, None], lags[1][None, :]))
        lam = np.fft.fft2(c).real
        self.clipped = float(-lam[lam < 0].sum() / np.abs(lam).sum())   # share of negative spectrum
        self._sqrt_lam = np.sqrt(np.clip(lam, 0, None) / lam.size)
        self.grid_shape, self.dx = tuple(M), dx

        # Bilinear weights of the 4 surrounding grid nodes, and variance correction
        g = (pts - lo) / dx
        self._i0 = np.minimum(np.floor(g).astype(int), n - 2)
        t = g - self._i0
        self._w = np.stack([(1 - t[:, 0]) * (1 - t[:, 1]), t[:, 0] * (1 - t[:, 1]),
                            (1 - t[:, 0]) * t[:, 1], t[:, 0] * t[:, 1]], axis=1)
        offs = np.array([[0, 0], [1, 0], [0, 1], [1, 1]]) * dx
        C4 = self._rho(np.linalg.norm(offs[:, None] - offs[None], axis=-1))
        var_interp = np.einsum("pa,ab,pb->p", self._w, C4, self._w)
        self._sub_sd = np.sqrt(np.clip(1.0 - var_interp, 0.0, 1.0))    # sub-cell variation
        # one noise value per distinct location, so coincident points get the same value
        self._loc = np.unique(pts, axis=0, return_inverse=True)[1].ravel()
        self._n_loc = int(self._loc.max()) + 1 if self._loc.size else 0

    def _rho(self, h):
        return 1.0 - self._unit.gamma(h)

    def _interp(self, grid, rng):
        i, j = self._i0[:, 0], self._i0[:, 1]
        v = np.stack([grid[i, j], grid[i + 1, j], grid[i, j + 1], grid[i + 1, j + 1]], axis=1)
        return (v * self._w).sum(axis=1) + self._sub_sd * rng.standard_normal(self._n_loc)[self._loc]

    def sample(self, rng: np.random.Generator, k: int) -> np.ndarray:
        """`k` independent realisations of the field at the points, shape (k, n)."""
        out = []
        while len(out) < k:
            xi = rng.standard_normal(self.grid_shape) + 1j * rng.standard_normal(self.grid_shape)
            f = np.fft.fft2(self._sqrt_lam * xi)        # real and imaginary parts: 2 independent fields
            out += [self._interp(f.real, rng), self._interp(f.imag, rng)]
        z = np.array(out[:k])
        if self.nugget > 0:
            z = np.sqrt(1 - self.nugget) * z + np.sqrt(self.nugget) * rng.standard_normal(z.shape)
        return z


class RandomField:
    """Zero-mean stationary Gaussian field with the covariance of a (nested)
    variogram at fixed 2D points; its variance is the sill (``psill + nugget``).

    Each structure is simulated as an independent :class:`GaussianField`, scaled
    by the square root of its partial sill; the nugget is independent noise.

    Parameters
    ----------
    xy : (n, 2) array
        Points at which the field is evaluated.
    variogram : Variogram or NestedVariogram
        Covariance model.
    resolution : float, optional
        Spacing of the points (see :class:`GaussianField`), so that short-range
        variation is not smoothed by the grid.
    max_cells : int
        Largest number of grid cells along an axis.
    """

    def __init__(self, xy, variogram: AnyVariogram, *, resolution: Optional[float] = None, max_cells: int = 1024):
        xy = np.asarray(xy, dtype=float)
        self.n = xy.shape[0]
        self.nugget = float(variogram.nugget)
        total = float(variogram.psill + variogram.nugget)
        self.parts = [(np.sqrt(c.psill), GaussianField(xy, c, resolution=resolution, max_cells=max_cells))
                      for c in variogram.components() if c.psill > 1e-9 * total]   # skip negligible structures

    def sample(self, rng: np.random.Generator, k: int) -> np.ndarray:
        """`k` independent realisations at the points, shape (k, n)."""
        out = np.zeros((k, self.n))
        for scale, f in self.parts:
            out += scale * f.sample(rng, k)
        if self.nugget > 0:
            out += np.sqrt(self.nugget) * rng.standard_normal(out.shape)
        return out


def conditional_simulation(
    xy,
    values,
    xy_new,
    variogram: AnyVariogram,
    k: int,
    *,
    rng: Optional[np.random.Generator] = None,
    resolution: Optional[float] = None,
    n_closest: Optional[int] = None,
    max_cells: int = 1024,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Realisations of a random field that reproduce `variogram` and honour the data.

    For each realisation an unconditional field ``s`` is drawn at the data and
    the new points, and

        ``z = z_ok + (s - s_ok)``

    where ``z_ok`` is the ordinary kriging of the data and ``s_ok`` that of ``s``
    from its values at the data locations. ``s - s_ok`` is a draw of the kriging
    error, so over many realisations each new point has mean ``z_ok`` and
    variance equal to the kriging variance, while every realisation has the
    roughness of the variogram everywhere (unlike ``z_ok + sigma * noise``,
    which is smooth near the data).

    Parameters
    ----------
    xy : (n, 2) array
        Data locations.
    values : (n,) array
        Data values.
    xy_new : (m, 2) array
        Locations to simulate.
    variogram : Variogram or NestedVariogram
        Model of the field (the same as used for kriging).
    k : int
        Number of realisations.
    rng : numpy.random.Generator, optional
    resolution : float, optional
        Spacing of the new points (see :class:`GaussianField`).
    n_closest : int, optional
        Moving neighbourhood for the kriging (see :func:`ordinary_kriging`).
    max_cells : int
        Largest number of grid cells along an axis of the simulation grid.

    Returns
    -------
    realisations : (k, m) array
    estimate : (m,) array
        Kriging estimate ``z_ok``.
    variance : (m,) array
        Kriging variance.
    """
    rng = np.random.default_rng() if rng is None else rng
    xy = np.asarray(xy, dtype=float)
    xy_new = np.asarray(xy_new, dtype=float)
    values = np.asarray(values, dtype=float).ravel()
    n, m = xy.shape[0], xy_new.shape[0]
    if m == 0:
        return np.empty((k, 0)), np.empty(0), np.empty(0)
    sim = RandomField(np.vstack([xy, xy_new]), variogram, resolution=resolution, max_cells=max_cells).sample(rng, k)
    est, var = ordinary_kriging(xy, np.column_stack([values, sim[:, :n].T]), xy_new, variogram, n_closest=n_closest)
    real = est[:, :1].T + sim[:, n:] - est[:, 1:].T
    return real, est[:, 0], var
