import numpy as np
from scipy.stats import norm
from typing import Callable, Optional, Tuple, Union, Sequence
from numpy.typing import NDArray

from ._utils import build_neighbours_polygon, brentq

class Geometry:
    def __init__(
        self,
        nodes: NDArray[np.floating],
        nodes_std: Optional[NDArray[np.floating]] = None,
        elements: Optional[Union[Sequence[int], NDArray[np.integer]]] = None,
        attributes: Optional[Union[Sequence[int], NDArray[np.integer]]] = None,
        neighbours: Optional[NDArray[np.integer]] = None,
    ):
        # --- store nodes internally as (D, N) ---
        nodes = np.asarray(nodes, dtype=float)
        if nodes.ndim != 2:
            raise ValueError("`nodes` must have shape (N, D).")
        self._nodes = nodes.T  # (D, N)

        # --- set elements ---
        if elements is None:
            self._elements = np.column_stack([np.arange(self._nodes.shape[1] - 1), np.arange(1, self._nodes.shape[1])])
        else:
            self._elements = np.asarray(elements, dtype=int)

        # --- set attributes ---
        if attributes is None:
            self._attributes = attributes
        else:
            self._attributes = np.asarray(attributes, dtype=int)

        # --- consistency checks ---
        if self._elements.max() >= self._nodes.shape[1]:
            raise ValueError("Element connectivity references non-existent node index.")
        if self._attributes is not None:
            if len(self._elements) != len(self._attributes):
                raise ValueError("Number of elements and number of materials must match.")

        # --- stochastic node perturbation (optional, defaults to zero) ---
        if nodes_std is None:
            self._nodes_std = np.zeros_like(self._nodes)
        else:
            arr = np.asarray(nodes_std, dtype=float)
            if arr.shape != nodes.shape:
                raise ValueError("nodes_std must match nodes shape.")
            self._nodes_std = arr.T

        # Neighbours: store if provided, else compute on demand
        self._neighbours: Optional[np.ndarray] = None
        if neighbours is not None:
            self._neighbours = np.asarray(neighbours, dtype=np.int32)
            self._check_neighbours()
        elif elements is None:
            E = self._elements.shape[0]
            neighbours = np.column_stack((
                np.arange(-1, E - 1),
                np.arange(1, E + 1)
            ))
            neighbours[-1, 1] = -1


    def _check_neighbours(self) -> None:
        """Checks neighbours."""
        nb = self._neighbours
        if nb is None:
            raise ValueError("`neighbours` is None.")
        if nb.shape != (self._elements.shape[0], self._nodes.shape[1]):
            raise ValueError(
                f"`neighbours` must have shape ({self._elements.shape[0]}, {self._nodes.shape[1]}); got {nb.shape}."
            )
        if nb.size and (
            np.any(nb < -1) or np.any(nb >= self._elements.shape[0])
        ):
            raise ValueError("`neighbours` contains out-of-range element indices.")


    # --------------------------
    # Neighbours
    # --------------------------
    @property
    def neighbours(self) -> np.ndarray:
        """np.ndarray of shape (E, D): Directed-edge neighbours per element.

        Convention: for element ``[v0, v1]``,
        ``neighbours[e, 0]`` is before ``v0``, and
        ``neighbours[e, 1]`` is after ``v1``.
        """
        if self._neighbours is None:
            self._neighbours = build_neighbours_polygon(self._elements.astype(np.int32))
        return self._neighbours
    
    @neighbours.setter
    def neighbours(self, value: np.ndarray) -> None:
        """Sets the neighbours array."""
        self._neighbours = np.asarray(value, dtype=np.int32)
        self._check_neighbours()

    

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def nodes(self) -> np.ndarray:
        """np.ndarray of shape (N, D): Node coordinates (always returned as (N, D))."""
        return self._nodes.T

    @property
    def elements(self) -> np.ndarray:
        """np.ndarray of shape (E, M): Element connectivity."""
        return self._elements

    @property
    def attributes(self) -> np.ndarray | None:
        """np.ndarray of shape (E,): Element attributes."""
        return self._attributes

    @attributes.setter
    def attributes(self, attrs: np.ndarray) -> None:
        if len(self._elements) != len(attrs):
            raise ValueError("Number of elements and number of materials must match.")
        self._attributes = attrs

    @property
    def hasUncertainty(self) -> bool:
        return np.any(self._nodes_std != 0.0).astype(bool)

    @property
    def nodes_std(self) -> np.ndarray:
        """np.ndarray of shape (D, N): Standard deviations for stochastic perturbation."""
        return self._nodes_std.T

    @property
    def numRandomVariables(self) -> int:
        """Number of random variables (zero if no standard deviations)."""
        if np.all(self._nodes_std == 0.0):
            return 0
        else:
            return self._nodes_std.size

    @property
    def floor(self) -> float:
        """Lowest vertical coordinate across all slope nodes."""
        return float(np.min(self.nodes[:, -1]))

    # ------------------------------------------------------------------
    # Methods
    # ------------------------------------------------------------------
    def rvs(self, S: int = 1, *, random_state=None) -> np.ndarray:
        """
        Vectorised sampling of nodes.

        Inputs:
            self.nodes      : (D, M) float — mean coordinates
            self.nodes_std  : (D, M) float — per-node std dev

        Returns:
            (D, M, 1) if all std == 0, else (D, M, S)
        """
        mu = np.asarray(self.nodes, dtype=float)
        sd = np.asarray(self.nodes_std, dtype=float)
        if mu.shape != sd.shape or mu.ndim != 2:
            raise ValueError("`nodes` and `nodes_std` must both be (D, M).")
        if np.all(sd == 0):
            return mu[..., None]  # (D, M, 1)

        # Broadcast to (D, M, S) without copying
        muS = mu[..., None]
        sdS = sd[..., None]
        samples = norm.rvs(
            loc=muS,
            scale=sdS,
            size=(mu.shape[0], mu.shape[1], S),
            random_state=random_state
        )
        return np.asarray(samples, dtype=float)


    def ppf(self, q) -> np.ndarray:
        """
        Vectorised inverse-CDF for nodes.

        q can be:
            - (S,)           : same percentiles for every node → output (D, M, S)
            - (D, M, S)      : per-node percentiles           → output (D, M, S)

        Returns:
            (D, M, 1) if all std == 0, else (D, M, S)
        """
        mu = np.asarray(self.nodes, dtype=float)
        sd = np.asarray(self.nodes_std, dtype=float)
        if mu.shape != sd.shape or mu.ndim != 2:
            raise ValueError("`nodes` and `nodes_std` must both be (D, M).")

        if np.all(sd == 0):
            return mu[..., None]  # deterministic: ignore q

        q = np.asarray(q, dtype=float)
        if q.ndim == 1:
            # (S,) → (1,1,S) so it broadcasts with (D,M,1)
            q = q[None, None, :]
        elif q.ndim == 3:
            if q.shape[:2] != mu.shape:
                raise ValueError(f"`q` leading shape must be (D, M)={mu.shape} for 3D input.")
        else:
            raise ValueError("`q` must be shape (S,) or (D, M, S).")

        mu3 = mu[..., None]  # (D, M, 1)
        sd3 = sd[..., None]  # (D, M, 1)

        # norm.ppf(..., scale=0) yields NaN (scipy computes -inf * 0 for the
        # support bounds), so substitute a safe placeholder scale for
        # deterministic (std == 0) entries and overwrite them with the exact
        # node coordinate afterward, mirroring how rvs() already handles a
        # mix of zero and nonzero std correctly.
        safe_sd3 = np.where(sd3 == 0, 1.0, sd3)
        out = norm.ppf(q, loc=mu3, scale=safe_sd3)  # full broadcast to (D, M, S)
        out = np.where(sd3 == 0, mu3, out)

        # If numerically constant across S (can happen with tiny std), compress
        if out.shape[2] > 1 and np.allclose(out, out[..., :1], rtol=0, atol=0):
            return out[..., :1]
        return out.astype(float, copy=False)

    # ------------------------------------------------------------------
    # Contiguity checks and combination
    # ------------------------------------------------------------------
    def _contiguity_ok(self, other: "Geometry", atol: float = 1e-8) -> bool:
        """
        Check if two profiles are contiguous (last node of self equals first node of other).

        Parameters
        ----------
        other : Geometry
            Another geometry to check.
        atol : float
            Absolute tolerance for coordinates comparison.

        Returns
        -------
        bool
            True if contiguous, False otherwise.
        """
        return np.allclose(self.nodes[-1, :], other.nodes[0, :], atol=atol)


    def __add__(self, other: "Geometry") -> "Geometry":
        """
        Combine two contiguous profiles into a new one.

        Raises
        ------
        TypeError
            If `other` is not a Geometry.
        ValueError
            If profiles are not contiguous.
        """
        if not isinstance(other, Geometry):
            raise TypeError(f"Unsupported operand type for +: 'Geometry' and '{type(other).__name__}'")
        if not self._contiguity_ok(other):
            raise ValueError("Geometries must be contiguous to add.")        
        if not (
            (self.attributes is None and other.attributes is None)
            or
            (self.attributes is not None and other.attributes is not None)
            ):
            raise ValueError("Cannot add geometries if one has attributes and the other doesn't.")

        offset = self._nodes.shape[1] - 1
        new_nodes = np.concatenate([self._nodes, other._nodes[:, 1:]], axis=1)
        new_nodes_std = np.concatenate([self._nodes_std, other._nodes_std[:, 1:]], axis=1)
        new_elements = np.vstack([self._elements, other._elements + offset])
        new_attributes = np.vstack([self._attributes, other._attributes]) if (self._attributes is not None) and (other._attributes is not None) else None
        return Geometry(
            nodes=new_nodes.T,
            nodes_std=new_nodes_std.T,
            elements=new_elements,
            attributes=new_attributes
        )


    def __iadd__(self, other: "Geometry") -> "Geometry":
        """In-place version of :meth:`__add__`."""
        if not isinstance(other, Geometry):
            raise TypeError(f"Unsupported operand type for +: 'Geometry' and '{type(other).__name__}'")
        if not self._contiguity_ok(other):
            raise ValueError("Geometries must be contiguous to add.")        
        if not (
            (self.attributes is None and other.attributes is None)
            or
            (self.attributes is not None and other.attributes is not None)
            ):
            raise ValueError("Cannot add geometries if one has attributes and the other doesn't.")
        offset = self._nodes.shape[1] - 1
        self._nodes = np.concatenate([self._nodes, other._nodes[:, 1:]], axis=1)
        self._nodes_std = np.concatenate([self._nodes_std, other._nodes_std[:, 1:]], axis=1)
        self._elements = np.vstack([self._elements, other._elements + offset])
        self._attributes = np.vstack([self._attributes, other._attributes]) if (self._attributes is not None) and (other._attributes is not None) else None
        return self


    def __repr__(self) -> str:
        return (
            f"Geometry(num_nodes={self._nodes.shape[1]}, "
            f"num_elements={len(self._elements)})"
        )


    def exitTime(
        self,
        p: np.ndarray,          # (2, S)
        v: np.ndarray,          # (2, S)
        elem_id: np.ndarray,    # (S,)
        *,
        samples: Optional[np.ndarray] = None,   # None or (2, M, 1) or (2, M, S)
        tol: float = 1e-12,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        2D: time to exit current polyline segment along constant velocity.
        Returns (S,)-shaped arrays: next_el, t_exit
        """
        elems = self.elements.astype(np.int32)          # (E,2)
        S = p.shape[1]
        if elem_id.shape != (S,):
            raise ValueError("elem_id must have shape (S,) matching p,v.")

        # node indices per sample
        i_idx = elems[elem_id, 0]                       # (S,)
        j_idx = elems[elem_id, 1]                       # (S,)


        pts = self.getSamples(samples, S=S)  # (2, M, S)
        ar = np.arange(S, dtype=np.int32)
        Ni = pts[:, i_idx, ar]             # pairwise gather
        Nj = pts[:, j_idx, ar]

        # segment vectors & metrics
        Dvec = Nj - Ni                                  # (2,S)
        L2   = np.sum(Dvec * Dvec, axis=0)              # (S,)
        L    = np.sqrt(L2)
        ok   = L2 > tol

        d_hat = np.zeros_like(Dvec)
        d_hat[:, ok] = Dvec[:, ok] / L[ok]

        # param s along segment [0,1] from i→j
        s = np.zeros(S, dtype=float)
        s[ok] = np.sum((p[:, ok] - Ni[:, ok]) * Dvec[:, ok], axis=0) / L2[ok]
        s = np.clip(s, 0.0, 1.0)

        # signed speed along segment (+ toward j, − toward i)
        v_par = np.zeros(S, dtype=float)
        v_par[ok] = np.sum(v[:, ok] * d_hat[:, ok], axis=0)
        speed  = np.abs(v_par)
        moving = ok & (speed > tol)

        toward_j = v_par > 0.0
        end_node = np.where(toward_j, j_idx, i_idx).astype(np.int32)

        # remaining distance to endpoint
        dist_rem = np.empty(S, dtype=float)
        dist_rem[toward_j]  = (1.0 - s[toward_j]) * L[toward_j]
        dist_rem[~toward_j] = s[~toward_j]         * L[~toward_j]

        t_exit = np.full(S, np.nan, dtype=float)
        t_exit[moving] = np.where(dist_rem[moving] <= tol, 0.0, dist_rem[moving] / speed[moving])

        # guess next element by ±1; verify it shares the exit node
        guess = np.where(toward_j, elem_id + 1, elem_id - 1).astype(np.int32)
        next_el = np.full(S, -1, dtype=np.int32)

        in_range = (guess >= 0) & (guess < elems.shape[0])
        if np.any(in_range):
            a = elems[guess[in_range], 0]
            b = elems[guess[in_range], 1]
            shares = (a == end_node[in_range]) | (b == end_node[in_range])
            idx = np.flatnonzero(in_range)
            acc = idx[shares]
            next_el[acc] = guess[acc]

        # rare fallback: dumb search for those where ±1 failed
        need = moving & (next_el == -1)
        if np.any(need):
            for k in np.flatnonzero(need):
                nd = int(end_node[k])
                # all segments that touch node nd
                cand = np.flatnonzero((elems[:, 0] == nd) | (elems[:, 1] == nd))
                cand = cand[cand != elem_id[k]]
                if cand.size:
                    # pick the most aligned with v[:,k]
                    nodes2 = pts[:, :, k]  # (2,M)
                    other = np.where(elems[cand, 0] == nd, elems[cand, 1], elems[cand, 0])
                    vecs  = nodes2[:, other] - nodes2[:, nd][:, None]    # (2,Nc)
                    nrm   = np.linalg.norm(vecs, axis=0)
                    valid = nrm > tol
                    if np.any(valid):
                        dirs = vecs[:, valid] / nrm[valid][None, :]
                        dots = dirs.T @ v[:, k]
                        next_el[k] = cand[valid][np.argmax(dots)]

        return next_el, t_exit


    def intersectParabolaMatrix(
        self,
        p: np.ndarray,         # (2, S)
        v: np.ndarray,         # (2, S)
        a: np.ndarray,         # (2, S)
        elem_id: Optional[np.ndarray] = None,    # kept for API symmetry
        *,
        samples: Optional[np.ndarray] = None,    # (2, M, 1) or (2, M, S)
        tol: float = 1e-12,
    ) -> np.ndarray:
        """
        Candidate parabola/polyline support-line intersection times.

        Returns a matrix with shape (M-1, S). Entries are np.inf where the
        parabola has no finite intersection with the segment's supporting line.
        Use feasibleImpactTime to filter by segment bounds and choose one hit
        per sample.
        """
        p = np.asarray(p, dtype=float)
        v = np.asarray(v, dtype=float)
        a = np.asarray(a, dtype=float)
        if p.shape != v.shape or p.shape != a.shape or p.ndim != 2 or p.shape[0] != 2:
            raise ValueError("`p`, `v`, `a` must each be shape (2, S) with matching S.")
        S = p.shape[1]

        pts = self.getSamples(samples, S=S)  # (2, M, S)
        A = pts[:, :-1, :]               # (2, M-1, S)
        B = pts[:,  1:, :]               # (2, M-1, S)
        E = B - A                        # (2, M-1, S)
        len2 = np.sum(E * E, axis=0)     # (M-1, S)
        valid_seg = len2 > tol
        M1, S1 = len2.shape

        # Perpendicular vector to segment
        n_perp_x = -E[1]              # (M-1, S)
        n_perp_y =  E[0]              # (M-1, S)

        # Quadratic coefficients for n_perp · (x(t) - A) = 0
        # Broadcasting (M-1,S) * (S,) works: the (S,) broadcasts along columns.
        Acoef = (n_perp_x * a[0] + n_perp_y * a[1]) * 0.5  # (M-1, S)
        Bcoef = (n_perp_x * v[0] + n_perp_y * v[1])        # (M-1, S)
        Ccoef = (n_perp_x * (p[0][None, :] - A[0]) +
                 n_perp_y * (p[1][None, :] - A[1]))        # (M-1, S)

        # Candidate times (init to +inf)
        t = np.full((M1, S1), np.inf)

        # Quadratic roots where |A| > tol
        mask_quad = np.abs(Acoef) > tol
        if np.any(mask_quad):
            Ny = n_perp_y[mask_quad]
            Aq = Acoef[mask_quad]
            Bq = Bcoef[mask_quad]
            Cq = Ccoef[mask_quad]
            disc = Bq * Bq - 4.0 * Aq * Cq
            ok = disc >= -tol
            if np.any(ok):
                disc = np.maximum(disc, 0.0)
                sdisc = np.sqrt(disc)
                t1_vals = (-Bq - sdisc) / (2.0 * Aq)
                t2_vals = (-Bq + sdisc) / (2.0 * Aq)
                t_min_root = np.minimum(t1_vals, t2_vals)
                t_max_root = np.maximum(t1_vals, t2_vals)
                # Where element's normal points up get t_max
                t_root = np.where(Ny >= 0, t_max_root, t_min_root)
                t_root[~ok] = np.inf
                t[mask_quad] = t_root

        # Linear where ~quad and |B| > tol: B t + C = 0
        mask_lin = (~mask_quad) & (np.abs(Bcoef) > tol)
        if np.any(mask_lin):
            t[mask_lin] = -Ccoef[mask_lin] / Bcoef[mask_lin]

        v2 = np.sum(v * v, axis=0)
        a2 = np.sum(a * a, axis=0)
        stationary = (v2 <= tol) & (a2 <= tol)
        stationary_on_line = (
            stationary[None, :]
            & valid_seg
            & ~mask_quad
            & ~mask_lin
            & (np.abs(Ccoef) <= tol)
        )
        t[stationary_on_line] = 0.0

        return t


    def feasibleImpactTime(
        self,
        t: np.ndarray,          # (M-1, S)
        p: np.ndarray,          # (2, S)
        v: np.ndarray,          # (2, S)
        a: np.ndarray,          # (2, S)
        *,
        samples: Optional[np.ndarray] = None,    # (2, M, 1) or (2, M, S)
        t_min: float = 0.0,
        t_max: float = np.inf,
        tol: float = 1e-12,
        position: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter candidate impact times and choose the earliest feasible hit.

        The candidate matrix is tested against segment bounds, the time window,
        and then reduced to one segment/time pair per sample.
        """
        t = np.asarray(t, dtype=float)
        p = np.asarray(p, dtype=float)
        v = np.asarray(v, dtype=float)
        a = np.asarray(a, dtype=float)
        if p.shape != v.shape or p.shape != a.shape or p.ndim != 2 or p.shape[0] != 2:
            raise ValueError("`p`, `v`, `a` must each be shape (2, S) with matching S.")
        if t.ndim != 2 or t.shape[1] != p.shape[1]:
            raise ValueError("`t` must have shape (M-1, S).")

        S = p.shape[1]
        pts = self.getSamples(samples, S=S)  # (2, M, S)
        A = pts[:, :-1, :]                   # (2, M-1, S)
        B = pts[:,  1:, :]                   # (2, M-1, S)
        E = B - A                            # (2, M-1, S)
        len2 = np.sum(E * E, axis=0)         # (M-1, S)
        valid_seg = len2 > tol

        if t.shape != len2.shape:
            raise ValueError(f"`t` must have shape {len2.shape}; got {t.shape}.")

        if position is None:
            def position(tarr: np.ndarray) -> np.ndarray:
                tt = tarr[None, :, :]
                return p[:, None, :] + v[:, None, :] * tt + 0.5 * a[:, None, :] * (tt * tt)

        tmask = np.isfinite(t)
        if np.any(tmask):
            safe_t = np.where(tmask, t, 0.0)
            x = position(safe_t)
            if x.shape != (2, *t.shape):
                raise ValueError("`position(t)` must return shape (2, M-1, S).")

            w = x - A
            len2_safe = np.where(valid_seg, len2, np.inf)
            with np.errstate(divide="ignore", invalid="ignore"):
                u = np.sum(E * w, axis=0) / len2_safe

            u_ok = (u >= -tol) & (u <= 1.0 + tol) & valid_seg & tmask & np.isfinite(u)
            t = np.where(u_ok, t, np.inf)
        else:
            t = np.full_like(t, np.inf)
        
        # Time window filter
        def _window(tarr: np.ndarray, dlt: float) -> np.ndarray:
            # close to min
            if np.isfinite(t_min):
                tarr = np.where(tarr < t_min - dlt, np.inf, tarr)
            # close to max
            if np.isfinite(t_max):
                tarr = np.where(tarr > t_max + dlt, np.inf, tarr)
            return tarr
        t_s = _window(t, -tol)  # shrink window by tol
        if np.any(np.isfinite(t_s)):
            t = t_s
        else:
            t_e = _window(t, tol) # expand window by tol
            if np.isfinite(t_min):
                t = np.where(np.abs(t_e - t_min) < tol, t_min, t_e)
            else:
                t = t_e

        # Choose earliest candidate per sample
        seg_best = np.argmin(t, axis=0).astype(int)
        t_best = t[seg_best, np.arange(S)]   # shape (S,)

        # Write back, handling no-hit
        nohit = ~np.isfinite(t_best)
        seg_best[nohit] = -1
        t_best[nohit] = np.nan
        return seg_best, t_best


    def intersectFloor(
        self,
        p: np.ndarray,      # (D, S)
        v: np.ndarray,      # (D, S)
        a: np.ndarray,      # (D, S)
        floor: float,
        *,
        t_min: float = 0.0,
        tol: float = 1e-12,
    ) -> np.ndarray:
        """
        Smallest t >= t_min at which the parabola (p, v, a) reaches the floor: p[-1] + v[-1] t + 0.5 a[-1] t^2 = floor.

        Returns an (S,) array of times, np.inf where no such time exists.
        """
        p = np.asarray(p, dtype=float)
        v = np.asarray(v, dtype=float)
        a = np.asarray(a, dtype=float)

        py, vy, ay = p[-1], v[-1], a[-1]
        c = py - floor
        b = vy
        aq = 0.5 * ay

        t = np.full(p.shape[1], np.inf)

        mask_quad = np.abs(aq) > tol
        if np.any(mask_quad):
            Aq, Bq, Cq = aq[mask_quad], b[mask_quad], c[mask_quad]
            disc = Bq * Bq - 4.0 * Aq * Cq
            ok = disc >= -tol
            disc = np.maximum(disc, 0.0)
            sdisc = np.sqrt(disc)
            t1 = (-Bq - sdisc) / (2.0 * Aq)
            t2 = (-Bq + sdisc) / (2.0 * Aq)
            lo, hi = np.minimum(t1, t2), np.maximum(t1, t2)
            cand = np.where(lo >= t_min - tol, lo, hi)
            cand = np.where((cand >= t_min - tol) & ok, cand, np.inf)
            t[mask_quad] = cand

        mask_lin = (~mask_quad) & (np.abs(b) > tol)
        if np.any(mask_lin):
            tl = -c[mask_lin] / b[mask_lin]
            t[mask_lin] = np.where(tl >= t_min - tol, tl, np.inf)

        stationary_on_floor = (~mask_quad) & (~mask_lin) & (np.abs(c) <= tol)
        t[stationary_on_floor] = max(t_min, 0.0)

        return t


    def intersectParabola(
        self,
        p: np.ndarray,         # (2, S)
        v: np.ndarray,         # (2, S)
        a: np.ndarray,         # (2, S)
        elem_id: np.ndarray,    # (S,)
        *,
        samples: Optional[np.ndarray] = None,    # (2, M, 1) or (2, M, S)
        t_min: float = 0.0,
        t_max: float = np.inf,
        tol: float = 1e-12,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Vectorised 2D intersection between parabolic trajectories and a polyline.

        Returns (seg_id, t) with shapes (S,), (S,). seg_id=-1 and t is
        non-finite for no hit.
        """
        t = self.intersectParabolaMatrix(
            p,
            v,
            a,
            elem_id,
            samples=samples,
            tol=tol,
        )
        return self.feasibleImpactTime(
            t,
            p,
            v,
            a,
            samples=samples,
            t_min=t_min,
            t_max=t_max,
            tol=tol,
        )


    def intersectDamped(
        self,
        p: np.ndarray,          # (2, S)
        v: np.ndarray,          # (2, S)
        a: np.ndarray,          # (2, S), constant acceleration, e.g. [[0],[-g]]
        damping: np.ndarray,    # (S,), damping coefficients
        elem_id: np.ndarray,
        *,
        samples: Optional[np.ndarray] = None,
        t_min: float = 0.0,
        t_max: float = np.inf,
        t_ref: Optional[np.ndarray] = None,
        tol: float = 1e-12,
        max_expand: int = 32,
        maxiter: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Vectorised 2D intersection between damped trajectories and a polyline.

        Solves, for every segment m and trajectory sample s,

            n_m · (r_s(t) - A_m) = 0

        where

            r'(t) = v(t)
            v'(t) = a - damping * v

        Returns
        -------
        seg_id, t : Tuple[np.ndarray, np.ndarray]
            Shapes (S,), (S,). seg_id=-1 and t=nan where no valid segment
            intersection was found.
        """
        # ------------------------------------------------------------------
        # Linear damping trajectory:
        #
        #   v'(t) = a - D v
        #
        #   v(t) = a/D + (v0 - a/D) exp(-Dt)
        #
        #   r(t) = p0 + a/D * t
        #              + (v0 - a/D) / D * (1 - exp(-Dt))
        # ------------------------------------------------------------------
        p = np.asarray(p, dtype=float)
        v = np.asarray(v, dtype=float)
        a = np.asarray(a, dtype=float)

        if p.shape != v.shape or p.shape != a.shape or p.ndim != 2 or p.shape[0] != 2:
            raise ValueError("`p`, `v`, and `a` must have shape (2, S).")

        S = p.shape[1]
        D = np.asarray(damping, dtype=float)
        if D.shape != (S,):
            raise ValueError(f"`damping` must be a scalar or have shape (S,)={(S,)}.")
        if np.any(D <= 0.0):
            raise ValueError("`damping` must be positive for damped motion.")

        pts = self.getSamples(samples, S=S)  # (2, M, S)
        A = pts[:, :-1, :]                   # (2, M-1, S)
        B = pts[:,  1:, :]                   # (2, M-1, S)
        E = B - A                            # (2, M-1, S)

        len2 = np.sum(E * E, axis=0)          # (M-1, S)
        valid_seg = len2 > tol

        M1, S1 = len2.shape

        def r_damped(t: np.ndarray) -> np.ndarray:
            """
            t shape: (M-1, S)
            returns X shape: (2, M-1, S)
            """
            tt = t[None, :, :]                      # (1, M-1, S)
            Dv = D[None, None, :]
            e = np.exp(-Dv * tt)

            drift = a[:, None, :] / Dv
            transient = (v[:, None, :] - drift) / Dv

            return p[:, None, :] + drift * tt + transient * (1.0 - e)

        # Segment-line residual
        nx = -E[1]                                  # (M-1, S)
        ny =  E[0]                                  # (M-1, S)

        def f_mat(t: np.ndarray) -> np.ndarray:
            X = r_damped(t)
            return nx * (X[0] - A[0]) + ny * (X[1] - A[1])

        # brentq expects flat arrays
        shape = (M1, S1)

        def f_flat(x: np.ndarray) -> np.ndarray:
            return f_mat(x.reshape(shape)).ravel()

        # ------------------------------------------------------------------
        # Reference time
        # ------------------------------------------------------------------
        if t_ref is None:
            t_ref = self.intersectParabolaMatrix(
                p,
                v,
                a,
                elem_id,
                samples=samples,
                tol=tol,
            )
        else:
            t_ref = np.asarray(t_ref, dtype=float)

            if t_ref.shape == (S,):
                t_ref = np.broadcast_to(t_ref[None, :], shape).copy()
            elif t_ref.shape != shape:
                raise ValueError(f"`t_ref` must have shape (S,) or {shape}.")

        finite_ref = np.isfinite(t_ref) & (t_ref > tol)

        if np.any(finite_ref):
            dt0 = np.nanmedian(t_ref[finite_ref])
        else:
            dt0 = 1.0

        t_step = np.where(finite_ref, t_ref, dt0)
        t_step = np.maximum(t_step, tol)

        # ------------------------------------------------------------------
        # Bracket construction
        # ------------------------------------------------------------------
        t_l = np.full(shape, t_min, dtype=float)
        t_u = t_l + t_step

        if np.isfinite(t_max):
            t_u = np.minimum(t_u, t_max)

        f_l = f_mat(t_l)
        f_u = f_mat(t_u)

        active = valid_seg & np.isfinite(f_l) & np.isfinite(f_u)

        # Exact roots at lower/upper bounds
        root_at_l = active & (np.abs(f_l) <= tol)
        root_at_u = active & (np.abs(f_u) <= tol)

        bracketed = active & (f_l * f_u < 0.0)

        for _ in range(max_expand):
            bad = active & ~root_at_l & ~root_at_u & ~bracketed

            if np.isfinite(t_max):
                bad &= t_u < t_max

            if not np.any(bad):
                break

            t_l = np.where(bad, t_u, t_l)
            f_l = np.where(bad, f_u, f_l)

            t_u_new = t_u + t_step
            t_step = np.where(bad, 2.0 * t_step, t_step)

            if np.isfinite(t_max):
                t_u_new = np.minimum(t_u_new, t_max)

            t_u = np.where(bad, t_u_new, t_u)
            f_u = np.where(bad, f_mat(t_u), f_u)

            root_at_l = active & (np.abs(f_l) <= tol)
            root_at_u = active & (np.abs(f_u) <= tol)
            bracketed = active & (f_l * f_u < 0.0)

        # ------------------------------------------------------------------
        # Brent solve
        # ------------------------------------------------------------------
        t = np.full(shape, np.inf, dtype=float)

        t[root_at_l] = t_l[root_at_l]
        t[root_at_u] = t_u[root_at_u]

        solve = bracketed & ~root_at_l & ~root_at_u

        if np.any(solve):
            aa = np.where(solve, t_l, np.nan)
            bb = np.where(solve, t_u, np.nan)

            roots, converged, _ = brentq(
                f_flat,
                aa,
                bb,
                xtol=tol,
                ftol=tol,
                maxiter=maxiter,
            )

            t = np.where(solve & converged & np.isfinite(roots), roots, t)

        return self.feasibleImpactTime(
            t,
            p,
            v,
            a,
            samples=samples,
            t_min=t_min,
            t_max=t_max,
            tol=tol,
            position=r_damped,
        )


    def _dim(self) -> int:
        """Return geometry dimension D from self.nodes shape (N, D)."""
        pts = np.asarray(self.nodes)  # (N, D)
        if pts.ndim != 2:
            raise ValueError("`self.nodes` must be 2D with shape (N, D).")
        return int(pts.shape[1])


    def getSamples(
        self,
        samples: Optional[np.ndarray],
        *,
        S: Optional[int] = None,
        dtype: np.dtype | type = float,
    ) -> np.ndarray:
        """
        Return samples with shape (D, M, S). If `samples` is None, derive from self.points.

        - If samples is None -> use self.points.T[:, :, None]  -> (D, M, 1)
        - If last axis is 1 and S is provided -> repeat along last axis to (D, M, S)
        - If last axis is not in {1, S} -> error
        """
        D = self._dim()

        if samples is None:
            # self.nodes is (M, D); want (D, M, 1)
            pts = np.asarray(self.nodes, dtype=dtype, order="C")
            out = pts.T[:, :, None]
        else:
            out = np.asarray(samples, dtype=dtype, order="C")
            if out.ndim != 3 or out.shape[0] != D:
                raise ValueError(f"`samples` must have shape (D={D}, M, 1 or S). Got {out.shape}.")

        if S is not None:
            if out.shape[2] == 1:
                # broadcast single realisation to all S samples
                out = np.repeat(out, repeats=S, axis=2)
            elif out.shape[2] != S:
                raise ValueError(f"`samples.shape[2]` must be 1 or S={S}. Got {out.shape[2]}.")

        return np.ascontiguousarray(out)


    def _find_vertical_global(
        self,
        p: np.ndarray,
        direction: int,
        samples: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Global vertical search over all elements for each sample.

        Parameters
        ----------
        p : (2, S) float
            Query points.
        direction : int
            -1 under, 0 both, 1 above
        samples : (2, M, S) float
            Slope node coordinates.

        Returns
        -------
        elem_id : (S,) int
        point : (2, S) float
        """
        p = np.asarray(p, dtype=float)
        samples = np.asarray(samples, dtype=float)

        S = p.shape[1]
        E = self._elements.shape[0]

        x = p[0]  # (S,)
        y = p[1]  # (S,)

        ei = self._elements[:, 0]
        ej = self._elements[:, 1]

        xi = samples[0, ei, :]   # (E, S)
        yi = samples[1, ei, :]
        xj = samples[0, ej, :]
        yj = samples[1, ej, :]

        xmin = np.minimum(xi, xj)
        xmax = np.maximum(xi, xj)
        spans_x = (xmin <= x[None, :]) & (x[None, :] <= xmax)

        dx = xj - xi
        t = np.divide(
            x[None, :] - xi,
            dx,
            out=np.full((E, S), 0.5, dtype=float),
            where=np.abs(dx) > 0.0,
        )
        y_seg = yi + t * (yj - yi)

        if direction == -1:
            valid = spans_x & (y_seg <= y[None, :])
            dist = np.where(valid, y[None, :] - y_seg, np.inf)
        elif direction == 1:
            valid = spans_x & (y_seg >= y[None, :])
            dist = np.where(valid, y_seg - y[None, :], np.inf)
        else:  # direction == 0
            valid = spans_x
            dist = np.where(valid, np.abs(y_seg - y[None, :]), np.inf)

        idx = np.argmin(dist, axis=0)                  # (S,)
        best = dist[idx, np.arange(S)]                 # (S,)
        found = np.isfinite(best)

        out_elem = np.full(S, -1, dtype=np.int64)
        out_elem[found] = idx[found]

        out_point = np.full((2, S), np.nan, dtype=float)
        out_point[0, found] = x[found]
        out_point[1, found] = y_seg[idx[found], np.arange(S)[found]]

        return out_elem, out_point


    def _find_vertical_from_elem(
        self,
        p: np.ndarray,
        element_id: np.ndarray,
        neighbour_check: int,
        direction: int,
        samples: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Local vertical search from seed elements, with limited neighbour traversal.

        Parameters
        ----------
        p : (2, S) float
            Query points.
        element_id : (S,) int
            Seed elements. Must all be >= 0.
        neighbour_check : int
            Maximum number of neighbour traversals.
        direction : int
            -1 under, 0 both, 1 above
        samples : (2, M, S) float
            Slope node coordinates.

        Returns
        -------
        elem_id : (S,) int
            Found element ids, or -1 where unresolved locally.
        point : (2, S) float
            Projection points, or nan where unresolved.
        """
        p = np.asarray(p, dtype=float)
        curr = np.asarray(element_id, dtype=np.int64).copy()
        samples = np.asarray(samples, dtype=float)

        S = p.shape[1]
        x = p[0]
        y = p[1]

        out_elem = np.full(S, -1, dtype=np.int64)
        out_point = np.full((2, S), np.nan, dtype=float)

        active = curr >= 0

        # number of segment evaluations = initial + neighbour_check moves
        for _ in range(neighbour_check + 1):
            work = np.where(active)[0]
            if work.size == 0:
                break

            e = curr[work]

            ni = self._elements[e, 0]
            nj = self._elements[e, 1]

            xi = samples[0, ni, work]
            yi = samples[1, ni, work]
            xj = samples[0, nj, work]
            yj = samples[1, nj, work]

            xmin = np.minimum(xi, xj)
            xmax = np.maximum(xi, xj)

            left = x[work] < xmin
            right = x[work] > xmax
            inside = ~(left | right)

            # evaluate segments that span x
            if np.any(inside):
                idx = work[inside]
                e_in = curr[idx]

                ni = self._elements[e_in, 0]
                nj = self._elements[e_in, 1]

                xi = samples[0, ni, idx]
                yi = samples[1, ni, idx]
                xj = samples[0, nj, idx]
                yj = samples[1, nj, idx]

                dx = xj - xi
                t = np.divide(
                    x[idx] - xi,
                    dx,
                    out=np.full(idx.shape, 0.5, dtype=float),
                    where=np.abs(dx) > 0.0,
                )
                y_seg = yi + t * (yj - yi)

                if direction == -1:
                    ok = y_seg <= y[idx]
                elif direction == 1:
                    ok = y_seg >= y[idx]
                else:  # direction == 0
                    ok = np.ones(idx.shape, dtype=bool)

                if np.any(ok):
                    idx_ok = idx[ok]
                    out_elem[idx_ok] = curr[idx_ok]
                    out_point[0, idx_ok] = x[idx_ok]
                    out_point[1, idx_ok] = y_seg[ok]

                # whether accepted or rejected, local search ends here
                active[idx] = False

            # move left/right for unresolved active samples
            move_mask = active[work]
            if np.any(move_mask):
                idx_move = work[move_mask]
                e_move = curr[idx_move]

                ni = self._elements[e_move, 0]
                nj = self._elements[e_move, 1]

                xi = samples[0, ni, idx_move]
                xj = samples[0, nj, idx_move]

                xmin = np.minimum(xi, xj)
                xmax = np.maximum(xi, xj)

                left = x[idx_move] < xmin
                right = x[idx_move] > xmax

                if np.any(left):
                    idx_left = idx_move[left]
                    curr[idx_left] = self.neighbours[curr[idx_left], 0]
                if np.any(right):
                    idx_right = idx_move[right]
                    curr[idx_right] = self.neighbours[curr[idx_right], 1]

                active[idx_move] = curr[idx_move] >= 0

        return out_elem, out_point


    def verticalProjection(
        self,
        p: np.ndarray,
        element_id: np.ndarray,
        direction: int = 0,
        neighbour_check: int = 1,
        samples: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Find, for each sample, the closest segment vertically aligned with p.

        Parameters
        ----------
        p : (2, S) float
            Query points.
        element_id : (S,) int
            Seed element(s). If -1, performs a global search.
        direction : int
            Search direction:
            -1 -> only under
            0 -> both
            1 -> only above
        neighbour_check : int
            Maximum number of neighbour traversals before giving up and letting
            the caller trigger global search.
        samples : (2, M, 1) or (2, M, S) float, optional
            Slope node coordinates.

        Returns
        -------
        elem_id : (S,) int
            Segment id for each sample, or -1 if none.
        point : (2, S) float
            Vertical projection point on that segment, or nan if none.
        """
        if direction not in (-1, 0, 1):
            raise ValueError("direction must be -1, 0, or 1")

        p = np.asarray(p, dtype=float)
        element_id = np.asarray(element_id, dtype=np.int64)

        if p.ndim != 2 or p.shape[0] != 2:
            raise ValueError("p must have shape (2, S)")
        if element_id.shape != (p.shape[1],):
            raise ValueError("element_id must have shape (S,)")

        S = p.shape[1]
        pts = self.getSamples(samples, S=S)  # (2, M, S)

        out_element_id = np.full(S, -1, dtype=np.int64)
        out_point = np.full((2, S), np.nan, dtype=float)

        seeded = element_id > -1
        if np.any(seeded):
            e_loc, p_loc = self._find_vertical_from_elem(
                p[:, seeded],
                element_id[seeded],
                neighbour_check=neighbour_check,
                direction=direction,
                samples=pts[:, :, seeded],
            )
            out_element_id[seeded] = e_loc
            out_point[:, seeded] = p_loc

        unresolved = out_element_id == -1
        if np.any(unresolved):
            e_glob, p_glob = self._find_vertical_global(
                p[:, unresolved],
                direction=direction,
                samples=pts[:, :, unresolved],
            )
            out_element_id[unresolved] = e_glob
            out_point[:, unresolved] = p_glob

        return out_element_id, out_point
