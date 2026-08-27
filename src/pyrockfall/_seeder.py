"""
Seeder for Rockfall Simulations
===============================

This module defines the :class:`Seeder` class and related enumerations
used to initialise rockfall simulations in 2D and 3D. A Seeder specifies
the release locations, initial velocities, and associated rock groups.

Main Features
-------------
* Unified support for **2D and 3D simulations** via a ``points`` argument.
* **Point seeder**: defined by a single point ``(D,)`` or
  a list of points ``(D,N)``. If multiple points are provided, positions
  are sampled to be one of the provided points.
* **Line seeder**: defined by a polyline with at least two points
  ``(D, N>=2)``, with positions sampled uniformly along arc length.
* **Area seeder**: defined by multiple points ``(D, N>=2)``, with positions
  sampled as linear combinations of the provided points.
* Dimension-aware velocity distributions:
  - Translational velocity: D components (vx, vy[, vz]).
  - Angular velocity: 1 component in 2D (ωz), 3 components in 3D (ωx, ωy, ωz).

Contents
--------
- :class:`SeederRocksThrown`: Enum specifying seeding strategies.
- :class:`Seeder`: Main class for defining point seeders.
- :class:`LineSeeder`: Main class for defining line seeders.
- :class:`AreaSeeder`: Main class for defining area seeders.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import List, Sequence, Tuple, Union

import numpy as np

from ._rock import Rock
from . import stats


class SeederRocksThrown(Enum):
    """Enum specifying how rocks are distributed across types when seeding."""

    Overall = auto()
    """Rocks distributed across all types collectively."""

    PerRockType = auto()
    """Rocks assigned per rock type."""


class Seeder:
    """Defines a Seeder (rock thrower) in 2D or 3D rockfall simulations.

    A Seeder represents the location(s) where rocks are released, the
    distributions of their initial velocities, and the associated rock types.

    Seeder types:
        * **Point seeder**: defined by a single point (``points.shape == (D,)`` or ``(D,N)``).
          When multiple points are provided, positions are sampled to be one of the provided points.

    Args:
        points (array-like): Seeder coordinates of shape ``(D,)`` or ``(D, N)``.
            - ``D=2``: 2D seeder.
            - ``D=3``: 3D seeder.
            - ``N=1``: point seeder.
            - ``N>=2``: point seeder with multiple points.
        rocks (list[Rock]): Rock instances associated with the seeder.
        name (str, optional): Name of the seeder. Auto-generated if not provided.

    Attributes:
        rocks (list[Rock]): Rock types/groups associated with the seeder.
        rockThrowMode (SeederRocksThrown): Strategy for distributing rocks.
        numberOfRocks (int): Number of rocks to throw.
        translationalVelocity (list): Distributions for translational velocity
            components (length ``D``).
        angularVelocity (list): Distributions for angular velocity components:
            length 1 in 2D (scalar ωz), length 3 in 3D (ωx, ωy, ωz).
    """

    _instance_count = 0

    # --------------------------------------------------------------------- #
    # Constructor and core geometry
    # --------------------------------------------------------------------- #
    def __init__(self, points: np.ndarray, rocks: List[Rock], name: str = "") -> None:
        Seeder._instance_count += 1
        self._name: str = name or f"Seeder {Seeder._instance_count}"
        self.rockThrowMode: SeederRocksThrown = SeederRocksThrown.Overall
        self.numberOfRocks: int = 0
        self.rocks: List[Rock] = rocks

        P = np.asarray(points, dtype=float)
        if P.ndim == 1:
            P = P.reshape(-1, 1)
        if P.ndim != 2:
            raise ValueError("points must be shape (D,) or (D, N).")

        D, N = P.shape
        if D not in (2, 3):
            raise ValueError(f"D must be 2 or 3; got D={D}.")
        if N < 1:
            raise ValueError("points must contain at least one point.")
        
        self._D: int = D
        self._points: np.ndarray = P  # shape (D, N)

        # Default velocity distributions (dimension-aware).
        # Use asDistribution for scalars; asDistVector for explicit vectors.
        self._translationalVelocity: stats.DistributionVector = stats.DistributionVector(0.0, length=D)
        self._angularVelocity: stats.DistributionVector = stats.DistributionVector(0.0, length=1 if D == 2 else 3)

    # --------------------------------------------------------------------- #
    # Properties
    # --------------------------------------------------------------------- #
    @property
    def name(self) -> str:
        """str: Name of the seeder."""
        return self._name

    @property
    def D(self) -> int:
        """int: Dimension of the seeder (2 or 3)."""
        return self._D

    @property
    def points(self) -> np.ndarray:
        """ndarray: Seeder coordinates of shape ``(D, N)``."""
        return self._points

    @property
    def isPointSeeder(self) -> bool:
        """bool: True if seeder is a point seeder."""
        return True

    @property
    def isLineSeeder(self) -> bool:
        """bool: True if seeder is a line seeder."""
        return False

    @property
    def isAreaSeeder(self) -> bool:
        """bool: True if seeder is an area seeder."""
        return False

    @property
    def numRVsPosition(self) -> int:
        """int: Number of random variables for position sampling."""
        return 0 if self.points.shape[1] == 1 else 1

    @property
    def numRandomVariables(self) -> int:
        """int: Number of random variables.

        Count:
            * ``D`` translational components
            * 1 angular in 2D, or 3 angular in 3D
            * + number of position RVs
        """
        base = self.D + (1 if self.D == 2 else 3)
        return base + self.numRVsPosition

    # --------------------- Velocity distributions --------------------- #
    @property
    def translationalVelocity(
        self,
    ) -> stats.DistributionVector:
        """list: Distributions for ``D`` translational velocity components."""
        return self._translationalVelocity

    @translationalVelocity.setter
    def translationalVelocity(self, value: stats.DistributionVectorLike) -> None:
        """Set translational velocity distributions.

        Accepts either:
            * a **vector** (list/tuple/1D ndarray) converted via :func:`asDistVector`
              (length must equal ``D``), or
            * a **scalar/distribution** to be **broadcast** to all ``D`` components
              using :func:`asDistribution`.

        Raises:
            ValueError: If a vector is provided but its length ≠ ``D``.
        """
        self._translationalVelocity = stats.asDistributionVector(value, length=self.D)
        

    @property
    def angularVelocity(self) -> stats.DistributionVector:
        """list: Distributions for angular velocity (1 in 2D, 3 in 3D)."""
        return self._angularVelocity

    @angularVelocity.setter
    def angularVelocity(self, value: stats.DistributionVectorLike) -> None:
        """Set angular velocity distributions.

        Accepts either:
            * a **vector** (list/tuple/1D ndarray) converted via :func:`asDistVector`
              (length must be 1 in 2D or 3 in 3D), or
            * a **scalar/distribution** to be **broadcast** to the angular length
              using :func:`asDistribution`.

        Raises:
            ValueError: If a vector is provided but its length is invalid for the dimension.
        """
        required = 1 if self.D == 2 else 3
        self._angularVelocity = stats.asDistributionVector(value, length=required)

    # --------------------------------------------------------------------- #
    # Sampling helpers
    # --------------------------------------------------------------------- #
    def _sample_positions(self, u: np.ndarray) -> np.ndarray:
        """Sample positions to be one of the provided points.

        Args:
            u (np.ndarray): Samples in ``[0,1]``, shape ``(1, M)``.

        Returns:
            np.ndarray: Sampled positions of shape ``(D, M)``.
        """
        # Check shape of u
        if u.ndim == 1:
            u = u.reshape(1, -1)
        if u.ndim != 2 or u.shape[0] > 1:
            raise ValueError("Position sampling quantiles u must have shape (1, M).")
        
        # Check for deterministic case
        if u.shape[0] == 0:
            return np.repeat(self.points[:, 0:1], repeats=u.shape[1], axis=1)

        # Sanity check: all quantiles must be in [0, 1]
        if ((u < 0) | (u > 1)).any():
            raise ValueError("Position sampling quantiles must be in [0, 1].")

        D = self.points.shape[0]
        num_pts = self.points.shape[1]

        # Compute indices of points corresponding to each u
        # Intervals: (0, 1/num_pts], (1/num_pts, 2/num_pts], ..., ((n-1)/n, 1]
        # so index = ceil(u * num_pts) - 1
        idx = np.ceil(u * num_pts).astype(int) - 1

        # Clip just in case u == 0.0; this maps it to the first interval.
        # (If you want to treat u == 0 as invalid, you can check and raise instead.)
        idx = np.clip(idx, 0, num_pts - 1)

        # Get samples' position
        samples = self.points[:, idx.ravel()]           # shape (D, M)

        return samples


    def _ppf_stack(self, dists: Sequence, qs: np.ndarray) -> np.ndarray:
        """Evaluate percent-point function for a list of distributions.

        Args:
            dists (Sequence): List of distributions.
            qs (np.ndarray): Quantiles, shape ``(len(dists), M)``.

        Returns:
            np.ndarray: Samples, shape ``(len(dists), M)``.
        """
        return np.vstack([dists[i].ppf(qs[i]) for i in range(len(dists))])

    def _rvs_stack(self, dists: Sequence, n: int) -> np.ndarray:
        """Draw random samples from a list of distributions.

        Args:
            dists (Sequence): List of distributions.
            n (int): Number of samples.

        Returns:
            np.ndarray: Samples, shape ``(len(dists), n)``.
        """
        return np.vstack([d.rvs(n) for d in dists])

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def ppf(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate deterministic samples from quantiles.

        Args:
            q (np.ndarray): Percentiles, shape ``(numRandomVariables, M)``.
                - First ``D`` rows: translational velocity components.
                - Next rows: angular velocity (``1`` row in 2D, ``3`` rows in 3D).
                - Last row: position of samples.

        Returns:
            tuple:
                - **positions** (np.ndarray): Seeder positions, shape ``(D, M)``.
                - **translational_velocity** (np.ndarray): Velocities, shape ``(D, M)``.
                - **angular_velocity** (np.ndarray): Angular velocities,
                  shape ``(1, M)`` in 2D or ``(3, M)`` in 3D.

        Raises:
            ValueError: If the shape of ``q`` is inconsistent with the seeder definition.
        """
        q = np.asarray(q, dtype=float)
        if q.ndim != 2 or q.shape[0] != self.numRandomVariables:
            raise ValueError(
                f"q must have shape (numRandomVariables, M) with numRandomVariables={self.numRandomVariables}, "
                f"got {q.shape}."
            )

        M = q.shape[1]
        D = self.D
        A = 1 if D == 2 else 3

        q_trans = q[0:D, :]
        q_ang = q[D:D + A, :]
        q_pos = q[D + A:, :]

        translational_velocity = self._ppf_stack(self._translationalVelocity, q_trans)  # (D, M)
        angular_velocity = self._ppf_stack(self._angularVelocity, q_ang)                # (A, M)
        positions = self._sample_positions(q_pos)  # (D, M)

        return positions, translational_velocity, angular_velocity

    def rvs(self, num_samples: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate random samples of positions and velocities.

        Args:
            num_samples (int): Number of samples.

        Returns:
            tuple:
                - **positions** (np.ndarray): Seeder positions, shape ``(D, N)``.
                - **translational_velocity** (np.ndarray): Velocities, shape ``(D, N)``.
                - **angular_velocity** (np.ndarray): Angular velocities,
                  shape ``(1, N)`` in 2D or ``(3, N)`` in 3D.
        """
        N = int(num_samples)
        D = self.D
        A = 1 if D == 2 else 3

        translational_velocity = self._rvs_stack(self._translationalVelocity, N)    # (D, N)
        angular_velocity = self._rvs_stack(self._angularVelocity, N)                # (A, N)
        positions = self._sample_positions(np.random.rand(self.numRVsPosition, N))  # (D, N)

        return positions, translational_velocity, angular_velocity

    # --------------------------------------------------------------------- #
    # Plot helper
    # --------------------------------------------------------------------- #
    def plot(self, ax=None, **kwargs):
        """Visualize the seeder geometry.

        Args:
            ax (matplotlib.axes.Axes, optional): Axes to plot on.
                If ``None``, a new figure/axes is created.
            **kwargs: Forwarded to matplotlib ``plot``/``scatter``.

        Returns:
            matplotlib.axes.Axes: The axes with the Seeder drawn.
        """
        import matplotlib.pyplot as plt

        if self.D == 2:
            if ax is None:
                _, ax = plt.subplots()
            P = self._points
            if self.isPointSeeder:
                ax.scatter(P[0, 0], P[1, 0], **kwargs)
            else:
                ax.plot(P[0, :], P[1, :], **kwargs)
            return ax

        # 3D plotting
        from mpl_toolkits.mplot3d import Axes3D  # type: ignore
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")

        P = self._points
        if self.isPointSeeder:
            ax.scatter(P[0, 0], P[1, 0], P[2, 0], **kwargs)
        else:
            ax.plot(P[0, :], P[1, :], P[2, :], **kwargs)
        return ax

class LineSeeder(Seeder):
    """Defines a Line Seeder (rock thrower) in 2D or 3D rockfall simulations.

    Line Seeder is defined by a polyline with at least 2 points
    (``points.shape == (D, N>=2)``). Positions are sampled along the polyline
    proportional to arc length.

    Args:
        points (array-like): Seeder coordinates of shape ``(D,)`` or ``(D, N)``.
            - ``D=2``: 2D seeder.
            - ``D=3``: 3D seeder.
            - ``N=1``: single point seeder.
            - ``N>=2``: line seeder along polyline.
        rocks (list[Rock]): Rock instances associated with the seeder.
        name (str, optional): Name of the seeder. Auto-generated if not provided.

    Attributes:
        rocks (list[Rock]): Rock types/groups associated with the seeder.
        rockThrowMode (SeederRocksThrown): Strategy for distributing rocks.
        numberOfRocks (int): Number of rocks to throw.
        translationalVelocity (list): Distributions for translational velocity
            components (length ``D``).
        angularVelocity (list): Distributions for angular velocity components:
            length 1 in 2D (scalar ωz), length 3 in 3D (ωx, ωy, ωz).
    """

    @property
    def isPointSeeder(self) -> bool:
        """bool: True if seeder is a point seeder."""
        return False

    @property
    def isLineSeeder(self) -> bool:
        """bool: True if seeder is a line seeder."""
        return True

    @property
    def isAreaSeeder(self) -> bool:
        """bool: True if seeder is an area seeder."""
        return False

    @property
    def numRVsPosition(self) -> int:
        """int: Number of random variables for position sampling."""
        return 1

    def _sample_positions(self, u: np.ndarray) -> np.ndarray:
        """Sample positions uniformly along a polyline by arc length.

        Args:
            u (np.ndarray): Samples in ``[0,1]``, shape ``(M,)``.

        Returns:
            np.ndarray: Sampled positions of shape ``(D, M)``.
        """
        if u.shape[0] != 1:
            raise ValueError("Position sampling quantiles u with shape (1, M) must be provided for LineSeeder.")
        P = self._points  # (D, N)
        if P.shape[1] == 1:
            return np.repeat(P, repeats=u.size, axis=1)

        seg = P[:, 1:] - P[:, :-1]  # (D, N-1)
        lengths = np.linalg.norm(seg, axis=0)  # (N-1,)
        if not np.all(lengths > 0):
            raise ValueError("Degenerate segments (zero length) detected.")

        cum = np.cumsum(lengths)
        total = cum[-1]
        ends = cum / total
        starts = np.zeros_like(ends)
        starts[1:] = ends[:-1]

        # Segment index for each u
        idx = np.searchsorted(ends, u.ravel(), side="right")
        idx = np.clip(idx, 0, len(lengths) - 1)

        # Local interpolation parameter
        seg_start = starts[idx]
        seg_end = ends[idx]
        w = (u - seg_start) / (seg_end - seg_start)

        Ps = P[:, idx]  # (D, M)
        Ss = seg[:, idx]  # (D, M)

        return Ps + Ss * w


class AreaSeeder(Seeder):
    @property
    def isPointSeeder(self) -> bool:
        """bool: True if seeder is a point seeder."""
        return False

    @property
    def isLineSeeder(self) -> bool:
        """bool: True if seeder is a line seeder."""
        return False

    @property
    def isAreaSeeder(self) -> bool:
        """bool: True if seeder is an area seeder."""
        return True

    @property
    def numRVsPosition(self) -> int:
        """int: Number of random variables for position sampling."""
        return self.points.shape[1]

    def _sample_positions(self, u: np.ndarray) -> np.ndarray:
        """Sample positions uniformly with a linear combination.

        Args:
            u (np.ndarray): Samples in ``[0,1]``, shape ``(N, M)``.

        Returns:
            np.ndarray: Sampled positions of shape ``(D, M)``.
        """
        if u.ndim != 2 or u.shape[0] != self.points.shape[1]:
            raise ValueError("Position sampling quantiles u with shape (N, M) must be provided for AreaSeeder.")

        if self._points.shape[1] < 2:
            raise ValueError("AreaSeeder requires at least 2 points.")

        # Normalize u to sum to 1
        u_sum = np.sum(u, axis=0, keepdims=True)  # (1, M)
        u_norm = u / u_sum                        # (N, M)

        # Compute positions as linear combinations
        positions = self._points @ u_norm  # (D, M)

        return positions


