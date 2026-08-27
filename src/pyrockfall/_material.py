"""
Materials for pyrockfall
========================

Defines the :class:`Material` abstraction used in rockfall simulations.

The class encapsulates three stochastic (or deterministic) parameters:

- **normalRestitution** – normal coefficient of restitution (typically in ``[0, 1]``)
- **tangentialRestitution** – tangential coefficient of restitution (typically in ``[0, 1]``)
- **frictionAngle** – friction angle in degrees

Each parameter may be either a deterministic value (wrapped by
:class:`~. _distribution.Deterministic`) or a random variable that exposes
``.ppf(q)`` and ``.rvs(n)`` (e.g., a SciPy ``rv_frozen``).

Notes
-----
- This module does **not** require SciPy at import time; any object with
  ``ppf`` and ``rvs`` methods is accepted (duck typing).
- Returned arrays use the convention ``(3, ...)`` where the first axis
  is ordered as: ``[normal, tangential, friction]``.
"""

from __future__ import annotations

from typing import Sequence, Union

import numpy as np

from . import stats


__all__ = ["Material"]


class Material:
    """Material with stochastic properties for slope/rockfall simulation.

    A :class:`Material` holds three RV-like parameters. Each can be assigned a
    scalar (wrapped as :class:`Deterministic`) or any object that supports
    ``ppf`` and ``rvs`` (e.g., SciPy's ``rv_frozen``).

    Attributes
    ----------
    name : str
        Name/label of the material.
    normalRestitution : Deterministic or RV-like
        Normal restitution coefficient (typ. in ``[0, 1]``).
    tangentialRestitution : Deterministic or RV-like
        Tangential restitution coefficient (typ. in ``[0, 1]``).
    frictionAngle : Deterministic or RV-like
        Friction angle in degrees.

    Examples
    --------
    >>> m = Material("Basalt")
    >>> m.normalRestitution = 0.45
    >>> m.tangentialRestitution = 0.6
    >>> m.frictionAngle = 35.0
    >>> m.numRandomVariables
    3
    >>> q = [np.array([0.5]), np.array([0.5]), np.array([0.5])]
    >>> m.ppf(q).shape
    (3, 1)
    >>> m.rvs(4).shape
    (3, 4)
    """

    _instance_count = 0

    def __init__(
            self,
            name: str = "",
            normalRestitution: stats.DistributionLike = 0.0,
            tangentialRestitution: stats.DistributionLike = 0.0,
            frictionAngle: stats.DistributionLike = 0.0,
            roughness: stats.DistributionLike = 0.0,
        ) -> None:
        """Initialise a :class:`Material`.

        If no name is provided, a unique default (``"Material <n>"``) is assigned.

        Args:
            name: Optional name for the material.
        """
        Material._instance_count += 1
        self._name: str = name or f"Material {Material._instance_count}"

        # Default to deterministic zeros
        self._normalRestitution: stats.Distribution = stats.asDistribution(normalRestitution)
        self._tangentialRestitution: stats.Distribution = stats.asDistribution(tangentialRestitution)
        self._frictionAngle: stats.Distribution = stats.asDistribution(frictionAngle)
        self._roughness: stats.Distribution = stats.asDistribution(roughness)

    # ----------------------------
    # Properties
    # ----------------------------
    @property
    def name(self) -> str:
        """Name of the material."""
        return self._name

    @property
    def normalRestitution(self) -> stats.Distribution:
        """Normal restitution coefficient (Deterministic or RV-like)."""
        return self._normalRestitution

    @normalRestitution.setter
    def normalRestitution(self, value: stats.DistributionLike) -> None:
        """Set the normal restitution coefficient.

        Accepts a scalar (wrapped as :class:`Deterministic`) or an RV-like object.
        """
        self._normalRestitution = stats.asDistribution(value)

    @property
    def tangentialRestitution(self) -> stats.Distribution:
        """Tangential restitution coefficient (Deterministic or RV-like)."""
        return self._tangentialRestitution

    @tangentialRestitution.setter
    def tangentialRestitution(self, value: stats.DistributionLike) -> None:
        """Set the tangential restitution coefficient.

        Accepts a scalar (wrapped as :class:`Deterministic`) or an RV-like object.
        """
        self._tangentialRestitution = stats.asDistribution(value)


    @property
    def frictionAngle(self) -> stats.Distribution:
        """Friction angle in degrees (Deterministic or RV-like)."""
        return self._frictionAngle

    @frictionAngle.setter
    def frictionAngle(self, value: stats.DistributionLike) -> None:
        """Set the friction angle (degrees).

        Accepts a scalar (wrapped as :class:`Deterministic`) or an RV-like object.
        """
        self._frictionAngle = stats.asDistribution(value)

    @property
    def roughness(self) -> stats.Distribution:
        return self._roughness

    @roughness.setter
    def roughness(self, value: stats.DistributionLike) -> None:
        self._roughness = stats.asDistribution(value)

    @property
    def numRandomVariables(self) -> int:
        """Number of stochastic variables (always ``3``)."""
        return 3

    # ----------------------------
    # Sampling / quantiles
    # ----------------------------
    def ppf(self, q: Union[Sequence[np.ndarray], np.ndarray]) -> np.ndarray:
        """Percent point function (inverse CDF) for each variable.

        Args:
            q:
                Quantiles for each random variable. Provide three arrays (or
                array‑likes) in the order:
                ``[normalRestitution, tangentialRestitution, frictionAngle]``.

        Returns:
            ndarray:
                Array of shape ``(3, K)`` (or ``(3, ...)``), where the first axis is
                ordered as ``[normal, tangential, friction]`` and the remaining
                dimensions match the broadcast/shape of the provided quantiles.

        Raises:
            ValueError: If ``q`` does not contain three quantile arrays.
        """
        if not isinstance(q, (list, tuple, np.ndarray)) or len(q) != 3:
            raise ValueError("`q` must be a sequence of three quantile arrays.")
        n, t, f = q  # type: ignore[assignment]
        out = [
            self.normalRestitution.ppf(np.asarray(n)),
            self.tangentialRestitution.ppf(np.asarray(t)),
            self.frictionAngle.ppf(np.asarray(f)),
        ]
        return np.asarray(out)

    def rvs(self, num_samples: int) -> np.ndarray:
        """Draw random samples for the three stochastic parameters.

        Args:
            num_samples:
                Number of samples to draw per variable (``>= 0``).

        Returns:
            ndarray:
                Array of shape ``(3, num_samples)`` with samples ordered as
                ``[normal, tangential, friction]`` along the first axis.

        Notes
        -----
        - If any parameter is deterministic, its samples are just repeated constants.
        """
        if num_samples < 0:
            raise ValueError("`num_samples` must be non-negative.")

        samples = [
            self.normalRestitution.rvs(num_samples),
            self.tangentialRestitution.rvs(num_samples),
            self.frictionAngle.rvs(num_samples),
        ]
        return np.asarray(samples)

    # ----------------------------
    # Representation
    # ----------------------------
    def __repr__(self) -> str:  # pragma: no cover - convenience only
        return (
            f"Material(name='{self._name}', "
            f"normalRestitution={type(self._normalRestitution).__name__}, "
            f"tangentialRestitution={type(self._tangentialRestitution).__name__}, "
            f"frictionAngle={type(self._frictionAngle).__name__})"
        )
