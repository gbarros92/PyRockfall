# _geometry3d.py
# ===========
from __future__ import annotations

from typing import Optional, Union, Tuple, Sequence
from numpy.typing import NDArray

import numpy as np

from ._geometry import Geometry
from ._utils import (
    build_neighbours_mesh,
    getTriPoints,
    isInsideTriangle,
    timeClosest,
    timeParabolaPlane,
    timeRaySegment,
)

ArrayF = NDArray[np.floating]
ArrayI = NDArray[np.integer]


class Geometry3D(Geometry):
    """3D triangular geometry with 3D intersection helpers (exit time & parabola hit).

    This subclass keeps :class:`~.Geometry`’s nodes/sampling logic and
    only overrides:
      - :meth:`exitTime` (ray–directed-edge on the current triangle plane)
      - :meth:`intersectParabola` (parabola–triangle via neighbour walk)

    Args:
        nodes: Node coordinates of shape ``(N, 3)``.
        elements: Triangle connectivity of shape ``(E, 3)`` (0-based).
        materialIDs: Optional per-element IDs into `materials` (when `materials` is a table).
        nodes_std: Optional node std-devs, shape ``(N, 3)``.
        neighbours: Optional directed-edge neighbours array of shape ``(E, 3)``.
            If omitted, it will be computed lazily on first access.

    Raises:
        ValueError: If nodes are not 3D, elements are not triangles, or
            provided `neighbours` has an invalid shape/index range.
    """

    def __init__(
        self,
        nodes: NDArray[np.floating],
        *,
        elements: Optional[NDArray[np.integer]] = None,
        nodes_std: Optional[NDArray[np.floating]] = None,
        attributes: Optional[Union[Sequence[int], NDArray[np.integer]]] = None,
        neighbours: Optional[NDArray[np.integer]] = None,
    ) -> None:
        # Delegate core setup to Geometry
        super().__init__(
            nodes=nodes,
            nodes_std=nodes_std,
            elements=elements,
            attributes = attributes,
            neighbours = neighbours,
        )
            
    # --------------------------
    # Neighbours
    # --------------------------
    @property
    def neighbours(self) -> np.ndarray:
        """np.ndarray of shape (E, 3): Directed-edge neighbours per triangle.

        Convention: for triangle ``[v0, v1, v2]``,
        ``neighbours[e, 0]`` is across the directed edge ``v0→v1``,
        ``neighbours[e, 1]`` across ``v1→v2``, and
        ``neighbours[e, 2]`` across ``v2→v0``. Boundary → ``-1``.
        """
        if self._neighbours is None:
            self._neighbours = build_neighbours_mesh(self._elements.astype(np.int32))
        return self._neighbours

    # --------------------------
    # 3D exit along a triangle (constant velocity on plane)
    # --------------------------
    def exitTime(
        self,
        p: np.ndarray,            # (3, S)
        v: np.ndarray,            # (3, S)
        elem_id: np.ndarray,      # (S,)
        *,
        samples: Optional[np.ndarray] = None,  # None or (3, M, 1) or (3, M, S)
        tol: float = 1e-12,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Earliest t>0 where x(t)=p+v t hits a **directed edge** of the current triangle.

        Args:
            p: Positions (3, S).
            v: Velocities (3, S).
            elem_id: Current triangle per sample (S,).
            samples: Optional sampled nodes (3, M, 1) or (3, M, S).
            tol: Numerical tolerance.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                next_elem (S,), t_exit (S,) with ``-1``/``nan`` for no exit.
        """
        p = np.asarray(p, dtype=float)
        v = np.asarray(v, dtype=float)
        if p.shape != v.shape or p.ndim != 2 or p.shape[0] != 3:
            raise ValueError("`p` and `v` must be (3, S).")
        S = p.shape[1]
        if elem_id.shape != (S,):
            raise ValueError("`elem_id` must be shape (S,).")

        elems = self.elements.astype(np.int32)
        tri = elems[elem_id]  # (S,3)
        i0, i1, i2 = tri[:, 0], tri[:, 1], tri[:, 2]

        pts = self.getSamples(samples, S=S)  # (3, M, S)
        ar = np.arange(S, dtype=np.int32)
        P0 = pts[:, i0, ar]
        P1 = pts[:, i1, ar]
        P2 = pts[:, i2, ar]

        # Plane basis / normal
        E0 = P1 - P0
        E1 = P2 - P0
        n = np.cross(E0.T, E1.T).T
        n_norm = np.linalg.norm(n, axis=0)
        good = n_norm > tol
        n[:, good] /= n_norm[good]

        # Project pos/vel to plane
        dp = p - P0
        p_plane = p - n * np.sum(dp * n, axis=0)
        v_tan = v - n * np.sum(v * n, axis=0)
        moving = np.linalg.norm(v_tan, axis=0) > tol
        ok = good & moving

        next_elem = np.full(S, -1, dtype=np.int32)
        t_exit = np.full(S, np.nan, dtype=float)
        if not np.any(ok):
            return next_elem, t_exit

        ids = np.flatnonzero(ok)

        def _edge_points(A: ArrayF, B: ArrayF) -> np.ndarray:
            # (3, 2, K) with columns [A, B]
            return np.stack((A, B), axis=1)

        t0 = timeRaySegment(p_plane[:, ids], v_tan[:, ids], _edge_points(P0[:, ids], P1[:, ids]), tol=tol)
        t1 = timeRaySegment(p_plane[:, ids], v_tan[:, ids], _edge_points(P1[:, ids], P2[:, ids]), tol=tol)
        t2 = timeRaySegment(p_plane[:, ids], v_tan[:, ids], _edge_points(P2[:, ids], P0[:, ids]), tol=tol)

        T = np.vstack((t0, t1, t2))          # (3, K)
        e_idx = np.argmin(T, axis=0)
        t_best = T[e_idx, np.arange(T.shape[1])]

        hit = np.isfinite(t_best)
        if np.any(hit):
            sel = ids[hit]
            t_exit[sel] = t_best[hit]
            next_elem[sel] = self.neighbours[elem_id[sel], e_idx[hit]]
        return next_elem, t_exit

    # --------------------------
    # 3D parabola–triangle walk
    # --------------------------
    def intersectParabola(
        self,
        p: np.ndarray,                 # (3, S)
        v: np.ndarray,                 # (3, S)
        a: np.ndarray,                 # (3, S)
        current: np.ndarray,           # (S,)
        *,
        samples: Optional[np.ndarray] = None,   # None or (3, M, 1) or (3, M, S)
        t_min: float = 0.0,
        t_max: float = np.inf,
        tol: float = 1e-12,
        max_steps: int = 32,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """First impact for x(t)=p + v t + 0.5 a t² using directed-edge neighbour walking.

        Args:
            p: Initial positions (3, S).
            v: Initial velocities (3, S).
            a: Constant accelerations (3, S).
            current: Starting triangle per sample (S,) (use -1 for “unknown”).
            samples: Optional sampled nodes (3, M, 1) or (3, M, S).
            t_min: Minimum accepted time (inclusive).
            t_max: Maximum accepted time (inclusive).
            tol: Numerical tolerance.
            max_steps: Maximum neighbour-walk iterations.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (hit_elem (S,), t_hit (S,))
                with ``-1``/``nan`` on no impact.
        """
        p = np.asarray(p, dtype=float)
        v = np.asarray(v, dtype=float)
        a = np.asarray(a, dtype=float)
        if p.shape != v.shape or p.shape != a.shape or p.ndim != 2 or p.shape[0] != 3:
            raise ValueError("`p`, `v`, `a` must be (3, S).")
        S = p.shape[1]

        points = self.getSamples(samples, S=S)                 # (3, M, S)
        triangles = np.asarray(self.elements, dtype=np.int32)  # (E, 3)
        neighbours = self.neighbours.astype(np.int32)          # (E, 3)

        m = triangles.shape[0]
        if np.any((triangles < 0) | (triangles >= points.shape[1])):
            raise ValueError("`elements` contain invalid vertex indices.")
        if neighbours.shape != (m, 3) or np.any((neighbours < -1) | (neighbours >= m)):
            raise ValueError("`neighbours` invalid or mismatched with elements.")

        cur = np.asarray(current, dtype=np.int32).copy()
        cur[cur < 0] = 0

        hit_elem = np.full(S, -1, dtype=np.int32)
        t_hit = np.full(S, np.nan, dtype=float)
        cum_time = np.zeros(S, dtype=float)

        p_cur = np.array(p)
        v_cur = np.array(v)
        alive = np.ones(S, dtype=bool)

        for _ in range(max_steps):
            if not np.any(alive):
                break

            ids = np.flatnonzero(alive)
            tri = triangles[cur[ids]]                       # (K, 3)
            tri_points = getTriPoints(points, tri, ids)     # (3, 3, K)

            # 1) time to current planes
            time = timeParabolaPlane(
                p_cur[:, ids], v_cur[:, ids], a[:, ids], tri_points,
                tol=tol, t_min=t_min, t_max=t_max
            )  # (K,)

            ok_time = np.isfinite(time)
            if not np.any(ok_time):
                alive[ids] = False
                continue

            good = ids[ok_time]
            time_good = time[ok_time]

            # 2) hit points
            p_good = p_cur[:, good] + v_cur[:, good] * time_good + 0.5 * a[:, good] * (time_good * time_good)
            tri_points_good = tri_points[:, :, ok_time]  # (3, 3, |good|)

            # 3) inside / step decision
            edge_idx = isInsideTriangle(p_good, tri_points_good, tol=tol)  # -1 or 0/1/2
            inside = (edge_idx == -1)

            if np.any(inside):
                acc = good[inside]
                cum_time[acc] += time_good[inside]
                hit_elem[acc] = cur[acc]
                t_hit[acc] = cum_time[acc]
                alive[acc] = False

            rem_mask = ~inside
            if np.any(rem_mask):
                rem = good[rem_mask]
                eidx = edge_idx[rem_mask]
                nxt = neighbours[cur[rem], eidx]
                cur[rem] = nxt

                # leave mesh → stop
                dead = nxt < 0
                if np.any(dead):
                    alive[rem[dead]] = False

                # advance to edge midpoint (stable local step)
                mid = 0.5 * (
                    tri_points[:, eidx, rem] + tri_points[:, (eidx + 1) % 3, rem]
                )
                dt = timeClosest(p_cur[:, rem], v_cur[:, rem], a[:, rem], mid)

                ok_dt = np.isfinite(dt)
                if np.any(ok_dt):
                    good2 = rem[ok_dt]
                    dt2 = dt[ok_dt]
                    p_cur[:, good2] += v_cur[:, good2] * dt2 + 0.5 * a[:, good2] * (dt2 * dt2)
                    v_cur[:, good2] += a[:, good2] * dt2
                    cum_time[good2] += dt2

        return hit_elem, t_hit
