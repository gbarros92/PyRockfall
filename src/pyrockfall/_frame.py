"""
_frame.py
=========

Curvilinear wall coordinates.

:class:`CurvilinearFrame` unrolls a wall along a plan-view polyline and gives
two extra sets of coordinates besides ``(x, y, z)``:

``(s, d, e)``
    ``s`` distance along the strike (arclength along the polyline), ``d``
    distance along the dip direction (signed horizontal distance from the
    polyline) and ``e`` elevation (equal to ``z``).

``(u, v, w)``
    Principal components of ``(s, d, e)``. ``u`` and ``v`` span the best-fit
    plane of the unrolled face, ``w`` is the height above that plane.

:func:`fit_polyline` fits the polyline (fewest straight segments within a
tolerance) through an unordered plan-view strip of points.
"""
from __future__ import annotations

import json
import warnings
from typing import Optional, Tuple, Union

import numpy as np

__all__ = ["CurvilinearFrame", "fit_polyline"]


# ---------------------------------------------------------------------------
# Polyline fit
# ---------------------------------------------------------------------------
def _bin_stats(s, x, y, res):
    """Per-bin sufficient statistics; ``s`` = along-profile coordinate.
    ``x``, ``y`` should be centred (see :func:`fit_polyline`) to avoid cancellation."""
    k = np.floor((s - s.min()) / res).astype(np.int64)
    _, k = np.unique(k, return_inverse=True)          # drop empty bins
    B = k.max() + 1
    f = lambda w: np.bincount(k, weights=w, minlength=B)
    n = f(np.ones_like(x))
    return np.stack([n, f(x), f(y), f(x * x), f(y * y), f(x * y)]), f(s) / n


def _segment_bins(st, tol, min_pts=3, max_bins=None):
    """Fewest segments with orthogonal RMSE <= tol each. Returns bin bounds
    and (centroid, direction) per segment."""
    n, sx, sy, sxx, syy, sxy = [np.concatenate([[0.0], np.cumsum(r)]) for r in st]
    M = st.shape[1]

    def seg(i, j):
        m = n[j] - n[i]
        a, b = sx[j] - sx[i], sy[j] - sy[i]
        cxx = sxx[j] - sxx[i] - a * a / m
        cyy = syy[j] - syy[i] - b * b / m
        cxy = sxy[j] - sxy[i] - a * b / m
        lam = 0.5 * (cxx + cyy) - np.sqrt(0.25 * (cxx - cyy) ** 2 + cxy ** 2)
        return np.maximum(lam, 0), m, a / m, b / m, cxx, cyy, cxy

    INF = 1 << 60
    nseg = np.full(M + 1, INF)
    nseg[0] = 0
    cost = np.full(M + 1, np.inf)
    cost[0] = 0
    prev = np.full(M + 1, -1)
    for j in range(1, M + 1):
        lo = 0 if max_bins is None else max(0, j - max_bins)
        i = np.arange(lo, j)
        e, m = seg(i, j)[:2]
        ok = (m >= min_pts) & (e <= tol ** 2 * m) & (nseg[i] < INF)
        if not ok.any():
            continue
        i, e = i[ok], e[ok]
        cn, cc = nseg[i] + 1, cost[i] + e
        b = np.lexsort((cc, cn))[0]
        nseg[j], cost[j], prev[j] = cn[b], cc[b], i[b]
    if nseg[M] >= INF:
        raise ValueError("infeasible: raise tol or max_bins")
    bounds, j = [M], M
    while j > 0:
        j = prev[j]
        bounds.append(j)
    bounds = bounds[::-1]
    lines = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        _, _, cx, cy, cxx, cyy, cxy = seg(a, b)
        w, v = np.linalg.eigh([[cxx, cxy], [cxy, cyy]])
        lines.append((np.array([cx, cy]), v[:, 1]))   # principal direction
    return bounds, lines


def _along(x, y, V):
    """Position of each point along polyline ``V`` (arclength of its projection) and
    its distance from ``V``. The first/last segments are extended, so points beyond
    the ends get ``s < 0`` or ``s > length``."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    D = np.diff(V, axis=0)
    L = np.maximum(np.hypot(*D.T), 1e-12)
    S0 = np.r_[0, np.cumsum(L)[:-1]]
    s, dist = np.zeros(x.shape), np.full(x.shape, np.inf)
    K = len(D)
    for i in range(K):
        t = ((x - V[i, 0]) * D[i, 0] + (y - V[i, 1]) * D[i, 1]) / L[i] ** 2
        t = np.clip(t, -np.inf if i == 0 else 0, np.inf if i == K - 1 else 1)
        d = np.hypot(x - V[i, 0] - t * D[i, 0], y - V[i, 1] - t * D[i, 1])
        m = d < dist
        dist[m], s[m] = d[m], S0[i] + t[m] * L[i]
    return s, dist


def _fit_once(s, x, y, tol, res, min_pts, max_bins, centreline):
    st, _ = _bin_stats(s, x, y, res)
    cen = (st[1:3] / st[0]).T                        # bin centroids
    if centreline:                                   # collapse each bin to its centroid
        n = st[0]
        st = np.stack([n, n * cen[:, 0], n * cen[:, 1],
                       n * cen[:, 0] ** 2, n * cen[:, 1] ** 2, n * cen[:, 0] * cen[:, 1]])
    bounds, lines = _segment_bins(st, tol, min_pts, max_bins)

    proj = lambda p, c, d: c + np.dot(p - c, d) * d
    V = [proj(np.array([x[s.argmin()], y[s.argmin()]]), *lines[0])]
    for k, ((c1, d1), (c2, d2)) in enumerate(zip(lines[:-1], lines[1:])):
        a, b, e = bounds[k], bounds[k + 1], bounds[k + 2]
        p = 0.5 * (cen[b - 1] + cen[b])              # break point between the two segments
        A = np.column_stack([d1, -d2])
        q = None
        if abs(np.linalg.det(A)) > 1e-9:             # intersect adjacent lines ...
            q = c1 + np.linalg.solve(A, c2 - c1)[0] * d1
            lim = 0.5 * min(np.hypot(*(cen[b - 1] - cen[a])), np.hypot(*(cen[e - 1] - cen[b])))
            if np.hypot(*(q - p)) > max(lim, res):   # ... unless it lands far from the break
                q = None
        if q is None:                                # near-parallel: midpoint at the break
            q = 0.5 * (proj(p, c1, d1) + proj(p, c2, d2))
        V.append(q)
    V.append(proj(np.array([x[s.argmax()], y[s.argmax()]]), *lines[-1]))
    return np.array(V), bounds


def fit_polyline(x, y, tol, res, min_pts=3, max_bins=None, centreline=True, n_iter=2):
    """Binned fewest-segment polyline through an unordered 2D profile/strip.

    Parameters
    ----------
    x, y : array-like
        Plan-view coordinates of the points.
    tol : float
        Largest orthogonal RMSE allowed per segment.
    res : float
        Bin width along the profile (breakpoints are snapped to bin edges).
    min_pts : int
        Minimum number of points per segment.
    max_bins : int, optional
        Longest segment, in bins.
    centreline : bool
        Measure the error of the bin centroids (count-weighted) instead of the
        individual points, so ``tol`` is how far the strip's centreline may stray
        from each segment, regardless of how wide the strip is.
    n_iter : int
        Refits that re-bin along the previous polyline instead of along one
        straight axis, so bins cut the strip square-on at bends and ends.

    Returns
    -------
    vertices : (k+1, 2) ndarray
        Polyline vertices in the input coordinates, ordered by increasing x.
    bounds : list[int]
        Bin index of each breakpoint.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    o = np.array([x.mean(), y.mean()])
    xc, yc = x - o[0], y - o[1]                      # centre: avoids cancellation in the sums
    u = np.linalg.eigh(np.cov(xc, yc))[1][:, 1]      # overall profile direction
    u = u if u[0] >= 0 else -u                       # vertices ordered by increasing x
    s = xc * u[0] + yc * u[1]                        # first pass: position along that axis
    for it in range(n_iter + 1):
        if it:
            s = _along(xc, yc, V)[0]                 # later passes: position along the polyline
        V_new, b_new = _fit_once(s, xc, yc, tol, res, min_pts, max_bins, centreline)
        bad_new = _n_folded(xc, yc, V_new)
        if it and bad_new > bad:                     # the strip is too wide for its bends:
            break                                    # re-binning would only diverge
        V, bounds, bad = V_new, b_new, bad_new
    return V + o, bounds


def _n_folded(x, y, V):
    """Points of a strip that are NaN or ambiguous in the frame of polyline ``V``."""
    try:
        f = CurvilinearFrame(V)
    except ValueError:
        return np.inf
    _, n_nan, n_amb = f.xyz_to_sde(np.column_stack([x, y, np.zeros_like(x)]), return_info=True)
    return n_nan + n_amb


def _principal_line(x, y):
    """Straight two-vertex polyline along the main plan-view direction of the points."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    o = np.array([x.mean(), y.mean()])
    xc, yc = x - o[0], y - o[1]
    C = np.cov(xc, yc) if x.size > 1 else np.zeros((2, 2))
    u = np.linalg.eigh(C)[1][:, 1] if np.all(np.isfinite(C)) else np.array([1.0, 0.0])
    u = u if u[0] >= 0 else -u
    s = xc * u[0] + yc * u[1]
    lo, hi = float(s.min()), float(s.max())
    if hi - lo <= 0:
        lo, hi = lo - 0.5, lo + 0.5
    return np.array([o + lo * u, o + hi * u])


# ---------------------------------------------------------------------------
# Frame
# ---------------------------------------------------------------------------
class CurvilinearFrame:
    """Unrolls a wall along a plan-view polyline, plus a PCA of the unrolled points.

    Curvilinear coordinates ``(s, d, e)``:

    - ``s`` strike: arclength along the polyline, extrapolated beyond its ends.
    - ``d`` dip direction: signed horizontal distance from the polyline, ``> 0`` to
      the left when walking along increasing ``s``. :meth:`fit` orients the polyline
      so that ``d`` increases down-dip.
    - ``e`` elevation, equal to ``z``.

    Corners are mitred: at each interior vertex the offset direction is the bisector,
    scaled so that ``d`` is the perpendicular distance to *both* adjacent segment
    lines. Inside segment ``i`` a point is
    ``p = V_i + t e_i + d ((1 - t) m_i + t m_{i+1})`` with ``t`` in ``[0, 1]``, which
    is continuous across vertices and invertible while ``d`` stays within
    :meth:`fold_limits` (offset lines on the inner side of a bend collapse beyond that).

    Principal coordinates ``(u, v, w)``: one global PCA of ``(s, d, e)``. ``u`` and
    ``v`` span the best-fit plane of the unrolled face (``u`` along ``+s``, ``v``
    upward) and ``w = u x v`` is the height above that plane.

    Parameters
    ----------
    polyline : (K+1, 2) array-like
        Plan-view vertices of the wall line.
    pca_mean, pca_axes, pca_var : array-like, optional
        A previously fitted PCA (see :meth:`fit_pca`).
    """

    def __init__(self, polyline, pca_mean=None, pca_axes=None, pca_var=None):
        self.polyline = np.asarray(polyline, float)
        if self.polyline.ndim != 2 or self.polyline.shape[0] < 2 or self.polyline.shape[1] != 2:
            raise ValueError("polyline must have shape (K+1, 2) with at least two vertices")
        E = np.diff(self.polyline, axis=0)
        self.L = np.hypot(*E.T)
        if np.any(self.L <= 0):
            raise ValueError("polyline has zero-length segments")
        self.Eh = E / self.L[:, None]                           # unit segment directions
        self.S0 = np.r_[0, np.cumsum(self.L)]                   # arclength at each vertex
        N = np.column_stack([-self.Eh[:, 1], self.Eh[:, 0]])    # left unit normals
        M = np.empty((len(self.polyline), 2))
        M[0], M[-1] = N[0], N[-1]
        c = 1 + np.einsum('ij,ij->i', N[:-1], N[1:])            # 1 + cos(turning angle)
        if np.any(c < 1e-6):
            raise ValueError("polyline turns back on itself (180 degree bend)")
        M[1:-1] = (N[:-1] + N[1:]) / c[:, None]                 # mitre: m . n = 1 on both sides
        self.N, self.M = N, M
        self.pca_mean = None if pca_mean is None else np.asarray(pca_mean, float)
        self.pca_axes = None if pca_axes is None else np.asarray(pca_axes, float)
        self.pca_var = None if pca_var is None else np.asarray(pca_var, float)

    # ---- construction ---------------------------------------------------------------
    @classmethod
    def fit(
        cls,
        xyz,
        steep=None,
        *,
        tol: float = 1.0,
        res: float = 0.5,
        corner_length: Union[str, float, None] = "auto",
        min_turn: float = 5.0,
        band_fraction: float = 0.05,
        min_pts: int = 3,
        n_iter: int = 2,
    ) -> "CurvilinearFrame":
        """Fit a frame to a wall.

        The polyline is fitted to the centreline of a thin elevation band around
        the median elevation of the face. With a ``steep`` mask the band is taken
        from the ``steep`` points only, so floor and berms at that elevation are
        left out; without one, from all the points.

        Parameters
        ----------
        xyz : (N, 3) array-like
            Point coordinates.
        steep : (N,) bool array, optional
            Points that belong to the face (ignored if fewer than 1% of the points).
        tol, res, min_pts, n_iter :
            Passed to :func:`fit_polyline`.
        corner_length : {"auto", float, None}
            Length either side of each bend over which the offset direction turns
            (see :meth:`refine_corners`). ``None`` keeps the fitted polyline as is,
            so the turn is spread along each whole segment.
        min_turn : float
            Bends below this angle (degrees) are not refined.
        band_fraction : float
            Fraction of the points (of the ``steep`` ones, if given) in the elevation band.
        """
        xyz = np.asarray(xyz, float)
        if xyz.ndim != 2 or xyz.shape[1] != 3 or xyz.shape[0] < 2:
            raise ValueError("fit() needs an (N, 3) array with at least two points")
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        n = xyz.shape[0]

        # The polyline follows the centreline of a plan-view strip, which must be thin
        # next to the bends of the wall. The whole steep face is not: a laid-back face
        # is as wide in plan as it is high, and binning a strip wider than the radius
        # of a bend mixes its two legs. So the steep mask only drops floor and berms,
        # and a thin elevation band is always taken from what is left.
        base = np.ones(n, bool)
        if steep is not None:
            steep = np.asarray(steep, bool)
            if steep.shape == (n,) and steep.sum() >= max(2 * min_pts, 0.01 * n):
                base = steep
        lo, hi = np.quantile(z[base], [0.5 - 0.5 * band_fraction, 0.5 + 0.5 * band_fraction])
        sel = base & (z >= lo) & (z <= hi)
        if sel.sum() < 2 * min_pts:
            sel = base

        try:
            V, _ = fit_polyline(x[sel], y[sel], tol, res, min_pts=min_pts, n_iter=n_iter)
            frame = cls(V)
        except (ValueError, np.linalg.LinAlgError):
            V = _principal_line(x[sel], y[sel])
            frame = cls(V)

        # Orientation: d increases down-dip (elevation falls as d grows)
        sde = frame.xyz_to_sde(xyz)
        ok = np.isfinite(sde).all(axis=1)
        d, e = sde[ok, 1], sde[ok, 2]
        d_max = float(np.abs(d).max()) if d.size else 0.0
        if d.size > 2 and d.std() > 0 and e.std() > 0 and np.corrcoef(d, e)[0, 1] > 0.1:
            frame = cls(V[::-1])

        if corner_length is not None:
            frame = frame.refine_corners(corner_length, d_max=d_max, min_turn=min_turn)

        sde = frame.xyz_to_sde(xyz)
        n_nan = int(np.isnan(sde[:, 0]).sum())
        if n_nan > 0.01 * n:
            warnings.warn(
                f"{n_nan} of {n} points fall outside the invertible region of the frame; "
                "raise tol for fewer, longer segments.", RuntimeWarning, stacklevel=2)
        frame.fit_pca(sde)
        return frame

    @classmethod
    def straight(cls, origin_xy=(0.0, 0.0), direction_xy=(1.0, 0.0)) -> "CurvilinearFrame":
        """One-segment frame: ``s`` along ``direction_xy`` from ``origin_xy``, ``d`` to its left."""
        o = np.asarray(origin_xy, float).ravel()[:2]
        u = np.asarray(direction_xy, float).ravel()[:2]
        nu = np.hypot(*u)
        if nu == 0:
            raise ValueError("direction cannot be zero.")
        return cls(np.array([o, o + u / nu]))

    def refine_corners(self, length: Union[str, float] = "auto", *, d_max: Optional[float] = None,
                       min_turn: float = 5.0, margin: float = 1.5) -> "CurvilinearFrame":
        """Confine the turn of the offset direction to a zone around each bend.

        Extra vertices are inserted ``length`` either side of every bend. The wall
        line is unchanged, but lines of constant ``s`` are then perpendicular to the
        wall everywhere except within ``length`` of a bend, instead of fanning out
        along each whole segment.

        Parameters
        ----------
        length : {"auto", float}
            Distance from each bend. ``"auto"`` uses ``margin * d_max * tan(turn / 2)``,
            the shortest length that keeps points up to ``d_max`` from the polyline
            inside the fold limit, with a safety ``margin``.
        d_max : float
            Largest ``|d|`` of the points (needed for ``"auto"``).
        min_turn : float
            Bends below this angle (degrees) are left alone.

        Returns
        -------
        CurvilinearFrame
            A new frame (without PCA; call :meth:`fit_pca`).
        """
        K = self.n_segments
        if K < 2:
            return CurvilinearFrame(self.polyline)
        turn = self.turning_angles()
        if isinstance(length, str):
            if length != "auto":
                raise ValueError("length must be 'auto' or a number")
            if d_max is None:
                raise ValueError("refine_corners('auto') needs d_max")
            r = margin * float(d_max) * np.tan(0.5 * turn)
        else:
            r = np.full(turn.shape, float(length))
        r[turn < np.radians(min_turn)] = 0.0

        V = self.polyline
        out = [V[0]]
        for i in range(K):
            L = self.L[i]
            r0 = r[i - 1] if i > 0 else 0.0          # bend at the start vertex
            r1 = r[i] if i < K - 1 else 0.0          # bend at the end vertex
            if r0 > 0 and r1 > 0 and r0 + r1 >= L:
                cuts = [L * r0 / (r0 + r1)]
            else:
                cuts = []
                if 0 < r0 < L:
                    cuts.append(r0)
                if 0 < r1 < L:
                    cuts.append(L - r1)
            for c in cuts:
                if 1e-9 * L < c < (1 - 1e-9) * L:
                    out.append(V[i] + c * self.Eh[i])
            out.append(V[i + 1])
        return CurvilinearFrame(np.array(out))

    # ---- properties ------------------------------------------------------------------
    @property
    def n_segments(self) -> int:
        return len(self.L)

    @property
    def length(self) -> float:
        """Total length of the polyline."""
        return float(self.S0[-1])

    def turning_angles(self) -> np.ndarray:
        """Turning angle (radians, >= 0) at each interior vertex."""
        c = np.einsum('ij,ij->i', self.Eh[:-1], self.Eh[1:])
        return np.arccos(np.clip(c, -1.0, 1.0))

    # ---- (x, y, z) <-> (s, d, e) -----------------------------------------------------
    def xyz_to_sde(self, xyz, return_info: bool = False):
        """``(N, 3)`` xyz -> ``(N, 3)`` sde. Points outside the invertible region get NaN.

        With ``return_info``, also returns the number of NaN points and the number of
        ambiguous points (valid for more than one segment)."""
        xyz = np.asarray(xyz, float)
        P = xyz[:, :2]
        K = self.n_segments
        s = np.full(len(P), np.nan)
        d = np.full(len(P), np.nan)
        n_valid = np.zeros(len(P), np.int8)
        for i in range(K):
            q = P - self.polyline[i]
            di = q @ self.N[i]                                  # distance to segment line
            a_e = self.M[i] @ self.Eh[i]
            b_e = (self.M[i + 1] - self.M[i]) @ self.Eh[i]
            den = self.L[i] + di * b_e                          # > 0 inside the fold limit
            with np.errstate(divide='ignore', invalid='ignore'):
                t = (q @ self.Eh[i] - di * a_e) / den
            ok = den > 0
            if i > 0:
                ok &= t >= 0
            if i < K - 1:
                ok &= t <= 1
            # a point exactly on a shared mitre line is valid for both segments: only
            # count it as ambiguous if the two solutions differ
            si = self.S0[i] + t * self.L[i]
            dup = ok & ~np.isnan(s) & (np.abs(si - s) < 1e-9) & (np.abs(di - d) < 1e-9)
            n_valid += (ok & ~dup)
            better = ok & ~dup & (np.isnan(d) | (np.abs(di) < np.abs(d)))
            s[better], d[better] = si[better], di[better]
        sde = np.column_stack([s, d, xyz[:, 2]])
        if return_info:
            return sde, int(np.isnan(s).sum()), int((n_valid > 1).sum())
        return sde

    def sde_to_xyz(self, sde):
        """``(N, 3)`` sde -> ``(N, 3)`` xyz."""
        sde = np.asarray(sde, float)
        s, d = sde[:, 0], sde[:, 1]
        i = np.clip(np.searchsorted(self.S0, s, side='right') - 1, 0, self.n_segments - 1)
        t = (s - self.S0[i]) / self.L[i]
        off = (1 - t)[:, None] * self.M[i] + t[:, None] * self.M[i + 1]
        xy = self.polyline[i] + (t * self.L[i])[:, None] * self.Eh[i] + d[:, None] * off
        return np.column_stack([xy, sde[:, 2]])

    def fold_limits(self) -> np.ndarray:
        """``(K, 2)`` largest ``|d|`` per segment on the left (``d > 0``) and right
        (``d < 0``) side before the offset lines fold over (inf where they never do)."""
        b_e = np.einsum('ij,ij->i', self.M[1:] - self.M[:-1], self.Eh)
        with np.errstate(divide='ignore'):
            dstar = -self.L / b_e                               # den = 0 here
        left = np.where(dstar > 0, dstar, np.inf)
        right = np.where(dstar < 0, -dstar, np.inf)
        return np.column_stack([left, right])

    def diagnostics(self, xyz) -> dict:
        """Quality of the frame for a set of points.

        Returns a dict with the number of points outside the invertible region
        (``n_nan``), valid for more than one segment (``n_ambiguous``) and past a fold
        limit (``n_past_fold``), plus the tightest fold limit (``fold_limit_min``)."""
        sde, n_nan, n_amb = self.xyz_to_sde(xyz, return_info=True)
        lim = self.fold_limits()
        d = sde[:, 1]
        seg = np.clip(np.searchsorted(self.S0, np.nan_to_num(sde[:, 0]), side='right') - 1,
                      0, self.n_segments - 1)
        over = np.isfinite(d) & (((d > 0) & (d > lim[seg, 0])) | ((d < 0) & (-d > lim[seg, 1])))
        return {"n_nan": n_nan, "n_ambiguous": n_amb, "n_past_fold": int(over.sum()),
                "fold_limit_min": float(lim.min())}

    # ---- (s, d, e) <-> (u, v, w) -----------------------------------------------------
    def fit_pca(self, sde) -> "CurvilinearFrame":
        """Global PCA of the unrolled points. Rows of ``pca_axes`` are U, V, W."""
        sde = np.asarray(sde, float)
        sde = sde[~np.isnan(sde).any(axis=1)]
        if sde.shape[0] < 3:
            raise ValueError("fit_pca() needs at least three valid points")
        mean = sde.mean(axis=0)
        var, vec = np.linalg.eigh(np.cov((sde - mean).T))      # ascending variance
        U, V = vec[:, 2], vec[:, 1]
        U = U if U[0] >= 0 else -U                              # U along +s
        V = V if V[2] >= 0 else -V                              # V upward
        self.pca_mean = mean
        self.pca_axes = np.vstack([U, V, np.cross(U, V)])       # right-handed
        total = var.sum()
        self.pca_var = var[::-1] / total if total > 0 else np.zeros(3)   # explained-variance ratio
        return self

    def _require_pca(self):
        if self.pca_axes is None or self.pca_mean is None:
            raise ValueError("The frame has no PCA yet; call fit_pca() first.")

    def sde_to_uvw(self, sde):
        self._require_pca()
        return (np.asarray(sde, float) - self.pca_mean) @ self.pca_axes.T

    def uvw_to_sde(self, uvw):
        self._require_pca()
        return np.asarray(uvw, float) @ self.pca_axes + self.pca_mean

    def xyz_to_uvw(self, xyz):
        return self.sde_to_uvw(self.xyz_to_sde(xyz))

    def uvw_to_xyz(self, uvw):
        return self.sde_to_xyz(self.uvw_to_sde(uvw))

    def horizontal_angle_uv(self) -> float:
        """Angle (degrees, CCW from U) of horizontal lines within the U-V plane.

        Lines of constant elevation on the best-fit plane run along this direction;
        it is the natural reference for horizontal layering."""
        self._require_pca()
        h = np.cross(self.pca_axes[2], np.array([0.0, 0.0, 1.0]))   # in-plane, e = const
        if np.linalg.norm(h) < 1e-12:
            return 0.0
        hu, hv = h @ self.pca_axes[0], h @ self.pca_axes[1]
        ang = np.degrees(np.arctan2(hv, hu))
        if ang > 90:
            ang -= 180
        elif ang <= -90:
            ang += 180
        return float(ang)

    # ---- persistence -----------------------------------------------------------------
    def save(self, path) -> None:
        """Write the polyline and PCA to a JSON file."""
        meta = {"polyline_xy": self.polyline.tolist(),
                "pca_mean_sde": None if self.pca_mean is None else self.pca_mean.tolist(),
                "pca_axes_sde": None if self.pca_axes is None else self.pca_axes.tolist(),
                "pca_explained_variance_ratio": None if self.pca_var is None else self.pca_var.tolist(),
                "fields": {"strike": "s, arclength along the polyline",
                           "dip_direction": "d, signed horizontal distance from the polyline, > 0 to the left",
                           "elevation": "e, equal to z",
                           "U, V, W": "(s, d, e) - pca_mean_sde projected on the rows of pca_axes_sde"}}
        with open(path, 'w') as f:
            json.dump(meta, f, indent=1)

    @classmethod
    def load(cls, path) -> "CurvilinearFrame":
        """Read a frame written by :meth:`save`."""
        with open(path) as f:
            m = json.load(f)
        mean = m.get("pca_mean_sde", m.get("pca_mean_sdz"))
        axes = m.get("pca_axes_sde", m.get("pca_axes_sdz"))
        return cls(m["polyline_xy"], mean, axes, m.get("pca_explained_variance_ratio"))

    def __repr__(self) -> str:
        return (f"CurvilinearFrame(n_segments={self.n_segments}, length={self.length:.3g}, "
                f"pca={'yes' if self.pca_axes is not None else 'no'})")
