"""
Rock definition for pyrockfall
==============================

This module defines the :class:`Rock` class, representing a rock type or group
to be thrown in rockfall simulations. Each :class:`Rock` encapsulates material
properties as probability distributions (or deterministic constants) to support
uncertainty quantification during sampling.

Main Features
-------------
* Material properties as distributions:
  - ``mass`` and ``density`` can be deterministic or random (SciPy frozen RVs
    or custom distributions).
* Consistent sampling API:
  - :meth:`Rock.ppf` for inverse-CDF sampling from quantiles.
  - :meth:`Rock.rvs` for random variate draws.

Notes
-----
Use :func:`stats.asDistribution` to convert numeric inputs into deterministic
distributions and to accept frozen SciPy distributions seamlessly.
"""

from typing import List, Optional, Tuple

import numpy as np

from . import stats


class Rock:
    """Represents a rock type or group in a rockfall simulation.

    Each instance stores the probabilistic description of material properties,
    currently **mass** and **density**. Properties may be deterministic
    (e.g., :class:`Deterministic`) or random (e.g., SciPy ``rv_frozen`` or
    custom :class:`TruncatedRandomVariable`).

    Args:
        name (str, optional): Name of the rock group. If not provided,
            names are auto-generated as ``"Group <i>"``.

    Attributes:
        name (str): Name of the rock group.
        mass: stats.Distribution object for mass.
        density: stats.Distribution object for density.
    """

    _instance_count = 0

    def __init__(self, name: str = '',
                 mass: stats.DistributionLike = 0.0,
                 density: stats.DistributionLike = 0.0,
                 color: Optional[Tuple[int, int, int]] = None) -> None:
        """Initialize a new :class:`Rock`."""
        Rock._instance_count += 1
        self._name = name or f'Group {Rock._instance_count}'

        # Defaults are deterministic zeros; users should override as needed.
        self._mass: stats.Distribution = stats.asDistribution(mass)
        self._density: stats.Distribution = stats.asDistribution(density)
        self._color: Optional[Tuple[int, int, int]] = color

    # ---------------------- Properties ----------------------

    @property
    def name(self) -> str:
        """str: Name of the rock group."""
        return self._name

    @property
    def mass(self) -> stats.Distribution:
        """stats.Distribution for mass (deterministic or random)."""
        return self._mass

    @mass.setter
    def mass(self, value: stats.DistributionLike) -> None:
        """Set mass as a value or distribution.

        Args:
            value: Numeric value (wrapped via :func:`stats.asDistribution`) or a
                distribution .
        """
        self._mass = stats.asDistribution(value)

    @property
    def density(self) -> stats.Distribution:
        """stats.Distribution for density (deterministic or random)."""
        return self._density

    @density.setter
    def density(self, value: stats.DistributionLike) -> None:
        """Set density as a value or distribution.

        Args:
            value: Numeric value (wrapped via :func:`stats.asDistribution`) or a
                distribution.
        """
        self._density = stats.asDistribution(value)

    @property
    def color(self) -> Optional[Tuple[int, int, int]]:
        """Optional RGB color for visualization/export."""
        return self._color

    @color.setter
    def color(self, value: Optional[Tuple[int, int, int]]) -> None:
        self._color = value

    @property
    def numRandomVariables(self) -> int:
        """int: Number of random variables (currently 2: mass and density)."""
        return 2

    # ---------------------- Sampling API ----------------------

    def ppf(self, q: np.ndarray) -> List[np.ndarray]:
        """Evaluate the percent-point function (inverse CDF).

        Args:
            q (np.ndarray): Quantiles for each random variable, shape ``(2, N)``.
                - ``q[0, :]``: quantiles for mass
                - ``q[1, :]``: quantiles for density

        Returns:
            list[np.ndarray]: ``[mass_samples, density_samples]``, each of shape ``(N,)``.
        """
        q = np.asarray(q)
        if q.ndim != 2 or q.shape[0] != 2:
            raise ValueError(f"`q` must have shape (2, N). Got {q.shape}.")
        mass_samples = self._mass.ppf(q[0])
        density_samples = self._density.ppf(q[1])
        return [np.asarray(mass_samples), np.asarray(density_samples)]

    def rvs(self, num_samples: int) -> List[np.ndarray]:
        """Draw random samples for mass and density.

        Args:
            num_samples (int): Number of samples ``N``.

        Returns:
            list[np.ndarray]: ``[mass_samples, density_samples]``, each of shape ``(N,)``.
        """
        N = int(num_samples)
        mass_samples = self._mass.rvs(N)
        density_samples = self._density.rvs(N)
        return [np.asarray(mass_samples), np.asarray(density_samples)]
