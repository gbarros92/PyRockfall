"""
Analysis module for pyrockfall
==============================

This module defines the :class:`Analysis` class — an orchestration layer that
coordinates sampling, physics kernels, and interaction with the slope/geometry
backend in a **vectorised** manner for both 2D and 3D simulations.

Design Philosophy
-----------------
- Classes (e.g., :class:`Seeder`, :class:`Rock`, :class:`Material`, :class:`SlopeModel`)
  are used for **configuration**.
- Before simulation, all classes are **sampled** into **NumPy arrays** (and a
  sampled slope batch), after which the :class:`Analysis` methods operate as
  **pure, argument-in / value-out kernels** to preserve vectorisation and
  enable easy testing/parallelism.
- Intersection queries are delegated to the sampled slope object (e.g.,
  ``SlopeBatch``), which provides **batched**, **broadcastable** kernels.

Contents
--------
- :class:`Analysis`: Public API comprising validation, sampling, frame transforms,
  impact and sliding mechanics, free-fall integration, metrics, and high-level
  preprocessing/processing/postprocessing orchestration.

Notes
-----
This file contains **signatures and docstrings only** (no implementations) to
clarify the public API and expected array shapes. Implementations should be
vectorised and side-effect free (no mutation of ``self``), except where a method
explicitly writes into an external container (e.g., ``Trajectories``).
"""

from __future__ import annotations

from enum import Enum, auto
from dataclasses import dataclass
from scipy.stats.qmc import LatinHypercube
import numpy as np

from typing import Tuple, List, Dict, Optional

from ._rock import Rock
from ._seeder import Seeder, SeederRocksThrown
from ._slope import Slope
from ._vegetation import Vegetation
from ._trajectories import Trajectories
from ._utils import (
    decompose,
    sampleNormals,
    getSubSamples,
    setSubSamples,
)

class Sampling(Enum):
    """
    Enumeration of stochastic sampling methods.

    Attributes:
        LatinHypercube: Use Latin Hypercube Sampling.
        MonteCarlo: Use simple Monte Carlo sampling.
    """
    LatinHypercube = auto()
    MonteCarlo = auto()

FLOOR_ELEMENT_ID = -2  # sentinel: block reached the slope's lowest elevation with no real segment hit

class AnalysisRocksThrown(Enum):
    """
    Enumeration of rock-throwing strategies in simulation.

    Attributes:
        DistributedFromNumberOfRocks: Rocks distributed globally among all seeders.
        IndividuallyPerSeeder: Each seeder throws its specified number of rocks.
    """
    DistributedFromNumberOfRocks = auto()
    IndividuallyPerSeeder = auto()

@dataclass
class SampleMaps:
    rockSamples: Dict[Rock, List[int]]
    seederSamples: Dict[Seeder, List[int]]
    numSamples: int


class Analysis:
    """
    Refactored Analysis class that avoids adding new instance attributes at runtime.
    - All configuration stays on `self`.
    - Every method receives exactly the arrays/values it needs and returns outputs explicitly.
    - The main loop threads the necessary arrays between steps.
    """

    def __init__(self, copy: Optional["Analysis"] = None):
        # --- Configuration / fixed attributes only ---
        self.seeders: List[Seeder] = []
        self._slope: Optional[Slope] = None
        self._vegetation: Optional[Vegetation] = None
        self.rockThrowMode = AnalysisRocksThrown.IndividuallyPerSeeder
        self.numberOfRocks = 0
        self.useSpecificSeed = False
        self.specificSeed = 12345234
        self.samplingMethod = Sampling.LatinHypercube

        self.normalVelocityThreshold = 0.1
        self.stoppedVelocity = 1e-5

        self.scaleByVelocity = False
        self.K = 9.144   # velocity scale
        self.scaleByMass = False
        self.C = 1000.0  # mass scale

        self.considerRotationalVelocity = False
        self.A = 6.096
        self.B = 76.2

        self.gravity = -9.80665
        self.timeStep = 1e-2
        self.tolerance = 1e-7
        self.maxIter = 1000
        self._ndim = 0

        self._maps: Optional[SampleMaps] = None

        if copy is not None:
            for attr, value in copy.__dict__.items():
                if attr in {
                    "seeders","_slope","_vegetation","rockThrowMode","numberOfRocks","useSpecificSeed","specificSeed",
                    "samplingMethod","normalVelocityThreshold","stoppedVelocity","scaleByVelocity","K",
                    "scaleByMass","C","considerRotationalVelocity","A","B","gravity","tolerance","maxIter","_ndim",
                    "timeStep"
                }:
                    setattr(self, attr, value)

        self.trajectories = Trajectories()

    @property
    def slope(self) -> Slope:
        if self._slope is None:
            raise RuntimeError("Slope not set yet")
        return self._slope

    @slope.setter
    def slope(self, value: Slope) -> None:
        self._slope = value

    @property
    def hasVegetation(self) -> bool:
        return self._vegetation is not None

    @property
    def vegetation(self) -> Vegetation:
        if self._vegetation is None:
            raise RuntimeError("Slope has no vegetation")
        return self._vegetation

    @vegetation.setter
    def vegetation(self, value: Vegetation) -> None:
        self._vegetation = value

    @property
    def maps(self) -> SampleMaps:
        if self._maps is None:
            raise RuntimeError("Sample maps not set yet")
        return self._maps

    @maps.setter
    def maps(self, value: SampleMaps) -> None:
        self._maps = value

    # ----------------- Validation & setup helpers -----------------

    def _checkBeforeRun(self) -> None:
        if len(self.seeders) == 0:
            raise ValueError('No seeders defined')
        for s in self.seeders:
            if not isinstance(s, Seeder):
                raise ValueError('All seeders must be of type Seeder')
        self._ndim = self.seeders[0].points.shape[0]
        if self.slope is None:
            raise ValueError('No slope defined')
        if not isinstance(self.slope, Slope):
            raise ValueError('Slope must be of type Slope')
        if self.slope.nodes.shape[1] != self._ndim:
            raise ValueError('Slope dimensionality must match seeder dimensionality')
        if self.rockThrowMode == AnalysisRocksThrown.DistributedFromNumberOfRocks and self.numberOfRocks == 0:
            raise ValueError('Number of rocks thrown must be greater than 0')
        if self.scaleByVelocity and self.K <= 0.0:
            raise ValueError('Scaling parameter for velocity must be greater than 0')
        if self.scaleByMass and self.C <= 0.0:
            raise ValueError('Scaling parameter for mass must be greater than 0')

    def _distribute_rocks(self) -> None:
        if self.rockThrowMode == AnalysisRocksThrown.DistributedFromNumberOfRocks:
            for s in self.seeders:
                s.numberOfRocks = self.numberOfRocks // len(self.seeders)
                s.rockThrowMode = SeederRocksThrown.Overall

    def _prepare_sample_maps(self) -> None:
        rocktypes = list({r for s in self.seeders for r in s.rocks})
        samples_per_rock: Dict[Rock, List[int]] = {r: [] for r in rocktypes}
        samples_per_seeder: Dict[Seeder, List[int]] = {s: [] for s in self.seeders}

        num_samples = 0
        for s in self.seeders:
            num_samples_init = num_samples
            if s.rockThrowMode == SeederRocksThrown.Overall:
                num_rocks_per_type = s.numberOfRocks // len(s.rocks)
                for r in s.rocks:
                    samples_per_rock[r].extend(range(num_samples, num_samples + num_rocks_per_type))
                    num_samples += num_rocks_per_type
            elif s.rockThrowMode == SeederRocksThrown.PerRockType:
                for r in s.rocks:
                    samples_per_rock[r].extend(range(num_samples, num_samples + s.numberOfRocks))
                    num_samples += s.numberOfRocks
            samples_per_seeder[s].extend(range(num_samples_init, num_samples))

        self.maps = SampleMaps(samples_per_rock, samples_per_seeder, num_samples)

    def _draw_percentiles(self, num_vars: int, num_samples: int) -> np.ndarray:
        if self.samplingMethod == Sampling.LatinHypercube:
            lhc = LatinHypercube(num_vars, rng=self.specificSeed) if self.useSpecificSeed else LatinHypercube(num_vars)
            percentiles = lhc.random(num_samples)
        else:  # MonteCarlo
            if self.useSpecificSeed:
                np.random.seed(self.specificSeed)
            percentiles = np.random.rand(num_samples, num_vars)
        return percentiles.T  # (num_vars, num_samples)

    def _calcSlopeNormals(self, profiles: np.ndarray) -> np.ndarray:
        """
        Compute per-element unit normals using self.slope.elements.

        Args:
            profiles: array of node coordinates. Accepts:
                - (D, N)              -> single geometry
                - (D, N, S>=1)        -> multiple samples; uses sample 0

        Returns:
            normals: (D, M) unit normals, where:
                D is 2 (segments) or 3 (triangles), M is number of elements.
        """
        D = profiles.shape[0]
        elements = self.slope.elements  # (M, 2) for 2D, (M, 3) for 3D
        M = elements.shape[0]

        if self._ndim == 2:
            # 2D segments → perpendicular vector
            if D != 2:
                raise ValueError("2-node elements require D=2 nodes.")
            i, j = elements[:, 0], elements[:, 1]
            edge = profiles[:, j] - profiles[:, i]     # (2, M)
            n = np.stack((-edge[1], edge[0]), axis=0)  # rotate 90° CCW
            norm = np.linalg.norm(n, axis=0)
            norm = np.where(norm == 0.0, 1.0, norm)    # avoid div-by-zero
            return n / norm

        elif self._ndim == 3:
            # 3D triangles → cross product of two edges
            if D != 3:
                raise ValueError("3-node elements require D=3 nodes.")
            i, j, k = elements[:, 0], elements[:, 1], elements[:, 2]
            p1, p2, p3 = profiles[:, i], profiles[:, j], profiles[:, k]  # (3, M) each
            e1 = p2 - p1
            e2 = p3 - p1
            n = np.cross(e1.T, e2.T).T                # (3, M)
            norm = np.linalg.norm(n, axis=0)
            norm = np.where(norm == 0.0, 1.0, norm)
            return n / norm

        else:
            raise ValueError("Elements must have 2 (2D) or 3 (3D) nodes.")

    # ----------------- Generation phase -----------------
    def _generate_samples(self) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray,  # position, velocity, angular_velocity
        np.ndarray, np.ndarray,              # rock_params, material_params
        np.ndarray, np.ndarray,              # time, acceleration
        np.ndarray, np.ndarray,              # profiles, slope_normals
        Optional[np.ndarray], Optional[np.ndarray],  # drag_params, canopy
    ]:
        self._distribute_rocks()
        self._prepare_sample_maps()

        rocktypes = list(self.maps.rockSamples.keys())
        num_vars = sum(s.numRandomVariables for s in self.seeders) \
                 + sum(r.numRandomVariables for r in rocktypes) \
                 + sum(m.numRandomVariables for m in self.slope.materialTable) \
                 + self.slope.numRandomVariables
        
        if self._vegetation is not None:
            num_vars += sum(d.numRandomVariables for d in self.vegetation.dragTable) \
                      + self.vegetation.numRandomVariables

        percentiles = self._draw_percentiles(num_vars, self.maps.numSamples)

        # Initial kinematics
        position = np.zeros((self._ndim, self.maps.numSamples))
        velocity = np.zeros((self._ndim, self.maps.numSamples))
        angular_velocity = np.zeros((1 if self._ndim == 2 else self._ndim, self.maps.numSamples))

        count_vars = 0
        count_samples = 0
        for s in self.seeders:
            nv = s.numRandomVariables
            ns = len(self.maps.seederSamples[s])
            q = percentiles[count_vars:count_vars+nv, count_samples:count_samples+ns]
            pos, tra_vel, ang_vel = s.ppf(q)
            position[:, self.maps.seederSamples[s]] = pos
            velocity[:, self.maps.seederSamples[s]] = tra_vel
            angular_velocity[:, self.maps.seederSamples[s]] = -np.radians(ang_vel)
            count_vars += nv
            count_samples += ns

        # Rock properties
        max_n_params_r = max([r.numRandomVariables for r in rocktypes]) if rocktypes else 0
        rock_params = np.zeros((max_n_params_r, self.maps.numSamples))
        count_samples = 0
        for r in rocktypes:
            nv = r.numRandomVariables
            ns = len(self.maps.rockSamples[r])
            q = percentiles[count_vars:count_vars+nv, count_samples:count_samples+ns]
            rock_params[:nv, self.maps.rockSamples[r]] = r.ppf(q)
            count_vars += nv
            count_samples += ns

        # Material properties (per-material, per-block)
        materials = self.slope.materialTable
        num_materials = len(materials)
        max_n_params_m = max([m.numRandomVariables for m in materials]) if materials else 0
        material_params = np.zeros((max_n_params_m, num_materials, self.maps.numSamples))
        for m_id, m in enumerate(materials):
            nv = m.numRandomVariables
            material_params[:nv, m_id, :] = m.ppf(percentiles[count_vars:count_vars+nv])
            count_vars += nv

        # Time/acc/geometry
        time = np.zeros(self.maps.numSamples)
        acceleration = np.zeros((self._ndim, self.maps.numSamples))
        acceleration[-1, :] = self.gravity

        # Slope samples
        nv = self.slope.numRandomVariables
        profiles = self.slope.ppf(percentiles[count_vars:count_vars+nv]).transpose(1, 0, 2)
        slope_normals = self._calcSlopeNormals(profiles)
        count_vars += nv

        drag_params = np.empty((1, 0, position.shape[1]))
        canopy = np.empty((position.shape[1], 0, 1))
        if self._vegetation is not None:
            drags = self.vegetation.dragTable
            num_drags = len(drags)
            max_n_params_d = max([d.numRandomVariables for d in drags]) if drags else 0
            drag_params = np.zeros((max_n_params_d, num_drags, self.maps.numSamples))
            for d_id, d in enumerate(drags):
                nv = d.numRandomVariables
                drag_params[:nv, d_id, :] = d.ppf(percentiles[count_vars:count_vars+nv])
                count_vars += nv

            nv = self.vegetation.numRandomVariables
            canopy = self.vegetation.ppf(percentiles[count_vars:count_vars+nv]).transpose(1, 0, 2)
            count_vars += nv

        return (
            position, velocity, angular_velocity,
            rock_params, material_params,
            time, acceleration,
            profiles, slope_normals,
            drag_params, canopy,
        )
    
    def _sampleRoughness(self, numSamplesPerMaterial: int):
        delta = np.empty((len(self.slope.materialTable), numSamplesPerMaterial), dtype=float)
        for i, material in enumerate(self.slope.materialTable):
            delta[i] = material.roughness.rvs(numSamplesPerMaterial)
        return delta

    # ----------------- Physics kernels (argument-pure) -----------------
    def _addRoughness(
        self,
        normal,
        element_id,
        roughness_samples,
        cum_impacts,
    ) -> np.ndarray:
        # Impacted material id for each sample
        mat_id = self.slope.materialIDs[element_id]

        # batch ocurrences by element_id to efficiently sample roughness perturbations per impact
        order = np.argsort(mat_id, kind="stable")
        mat_sorted = mat_id[order]
        is_new = np.r_[True, mat_sorted[1:] != mat_sorted[:-1]]
        group_start = np.flatnonzero(is_new)
        group_size = np.diff(np.r_[group_start, len(mat_sorted)])
        local_sorted = np.arange(len(mat_sorted)) - np.repeat(group_start, group_size)
        local_idx = np.empty_like(local_sorted)
        local_idx[order] = local_sorted

        # base normal angle for each impacted sample
        theta0 = np.arctan2(normal[1], normal[0])   # (S,)

        # sample angular perturbation
        imp_id = cum_impacts[mat_id] + local_idx
        imp_id %= roughness_samples.shape[1]  # wrap around if run out of samples for a material
        delta = roughness_samples[mat_id, imp_id]
        np.add.at(cum_impacts, mat_id, 1)
        theta = theta0 + np.radians(delta)

        # rebuild perturbed unit normals
        return np.stack((np.cos(theta), np.sin(theta)), axis=0)  # (2, S)

    def _impactMaterial(
        self,
        element_id: np.ndarray,    # (S,)  -1 means no touch
        material_params: np.ndarray,  # (A, B, S)
    ) -> np.ndarray:
        """
        Returns per-sample material parameters with shape (A, S),
        where A = #params per material, S = #samples.
        Non-touching samples (element_id == -1) are set to NaN.
        """
        out = np.full((material_params.shape[0], element_id.shape[0]), np.nan, dtype=material_params.dtype)

        hit = element_id >= 0
        if not np.any(hit):
            return out

        # material id for each *hit* sample
        mat_ids = np.asarray(self.slope.materialIDs, dtype=int)[element_id[hit]]  # (H,)
        s_idx   = np.nonzero(hit)[0]                                              # (H,)

        # Gather: for each hit sample s, take column (material=mat_ids[s], sample=s)
        # Resulting shape: (A, H)
        out[:, s_idx] = material_params[:, mat_ids, s_idx]

        return out

    def _impactDrag(
        self,
        segment_id: np.ndarray,
        drag_params: np.ndarray,
    ) -> np.ndarray:
        """Return per-sample vegetation drag parameters with shape (A, S)."""
        out = np.full((drag_params.shape[0], segment_id.shape[0]), np.nan, dtype=drag_params.dtype)

        hit = segment_id >= 0
        if not np.any(hit):
            return out

        drag_ids = np.asarray(self.vegetation.dragIdentities, dtype=int)[segment_id[hit]]
        s_idx = np.nonzero(hit)[0]
        # if drag_params.shape[-1] == 1:
        #     sample_idx = np.zeros_like(s_idx)
        # elif drag_params.shape[-1] == segment_id.shape[0]:
        #     sample_idx = s_idx
        # else:
        #     raise ValueError("`drag_params` last axis must be 1 or match `segment_id`.")
        out[:, s_idx] = drag_params[:, drag_ids, s_idx]

        return out

    def _isImpacting(
        self,
        normal_velocity: np.ndarray,
    ) -> np.ndarray:
        """Return (is_impacting, is_velocity_lt_threshold)."""
        return normal_velocity < 0.0

    def _impacts(
        self,
        vn: np.ndarray,       # (S,)  normal velocity scalar
        vt: np.ndarray,       # (D, S)  tangential velocity vector
        w: np.ndarray,        # (A, S)  angular velocity vector (A=1 if D=2 else D)
        mass: np.ndarray,     # (S,)  mass per sample
        density: np.ndarray,  # (S,)  density per sample
        rn: np.ndarray,       # (S,)  normal restitution per sample
        rt: np.ndarray,       # (S,)  tangential restitution per sample
        normal: np.ndarray,   # (D, S)  unit facet normal per sample
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (normal_velocity, tangential_velocity, angular_velocity) after impact."""
        if self.considerRotationalVelocity:
            r = np.cbrt(3 * mass / (4 * np.pi * density))
            I = 2/5 * mass * r**2
            den = I + mass * r**2
            if self._ndim == 2:
                normal = np.array([normal[0], normal[1], np.zeros_like(normal[0])])  # (3,S)
                vt = np.array([vt[0], vt[1], np.zeros_like(vt[0])])  # (3,S)
                w = np.array([np.zeros_like(w[0]), np.zeros_like(w[0]), w[0]])  # (3,S)

            wn, wt = decompose(w, normal)
            v_slip = vt + r * np.cross(normal.T, wt.T).T   # (D,N)
            v_slip_norm = np.linalg.norm(v_slip, axis=0)
            vt_norm = np.linalg.norm(vt, axis=0) # (S,)

            # Friction
            frict = (1 - rt) / ((v_slip_norm / self.A)**2 + 1.2)
            frict += rt

            # Normal scaling
            scl = np.full_like(vn, 0.0)
            mask = rn != 0.0
            scl[mask] = rt[mask] / ((vn[mask] / (self.B * rn[mask]))**2 + 1.0)

            num = r**2 * (I * np.sum(wt**2, axis=0) + mass * np.sum(vt**2, axis=0)) * frict * scl
            vt_post_norm = np.sqrt(num / den)

            # direction preserved from slip
            den_t = np.where(vt_norm > 0.0, vt_norm, 1.0)  # avoid div-by-zero
            num_t = np.where(vt_norm > 0.0, vt_post_norm, 0.0)  # zero if no slip
            vt_post = vt * num_t / den_t

            wt_post = (1/r) * np.cross(normal.T, vt_post.T).T
            w_post = wn + wt_post
            if self._ndim == 2:
                w_post = w_post[2][np.newaxis, :]
                vt_post = vt_post[:2, :]
        else:
            vt_post = vt * rt
            w_post = w

        scln = np.full_like(vn, 1.0)
        if self.scaleByVelocity:
            scln *= 1 + (vn / self.K) ** 2
        if self.scaleByMass:
            scln *= 1 + (mass / self.C) ** 2
        vn_post = -rn / scln * vn

        return vn_post, vt_post, w_post
    
    def _isSliding(
        self,
        vt: np.ndarray,        # (3, S)  tangential velocity vector on the facet
        normal: np.ndarray,    # (3, S)  unit facet normal per sample
        phi: np.ndarray,       # (S,)  friction angle [rad] per sample
    ) -> np.ndarray:
        """
        Decide stick–slip state on the current facet using a per-step Coulomb test.

        Returns:
            is_sliding: (S,) bool  -> true if in contact and sliding this step

        Assumes:
            - `normal` columns are unit vectors.
            - Tangential driving is due to self.gravity projected onto the tangent plane.
            - Static friction coefficient mu_s = tan(phi).
        """
        # Gravity vector in world coords
        g = np.zeros(self._ndim, dtype=float)        # (D,)
        g[self._ndim - 1] = self.gravity             # (D,)

        # Tangential speed norm
        vt_norm = np.linalg.norm(vt, axis=0)         # (S,)
        is_stopped = vt_norm < self.stoppedVelocity  # (S,) bool

        # Tangential component of gravity: a_t = g - (g·n) n
        gn = np.sum(g[:, None] * normal, axis=0)     # (S,)
        a_t = g[:, None] - normal * gn               # (D,S)
        at_norm = np.linalg.norm(a_t, axis=0)        # (S,)

        # Normal load per unit mass (>=0): N/m = max(0, -(g·n))
        N_over_m = np.maximum(0.0, -gn)              # (S,)

        # Static friction threshold
        mu_s = np.tan(np.radians(phi))               # (1,S)

        # Start sliding if required tangential > mu_s * N
        start_sliding = at_norm > mu_s * N_over_m    # (1,S) bool

        # Stick–slip decision (elementwise)
        is_sliding = ~is_stopped | (is_stopped & start_sliding)  # (1,S)

        return is_sliding  # return as (S,)

    def _slide(
        self,
        pos,     # (D,S) positions at contact
        vt,      # (D,S) tangential velocities; assumed dot(vt, n)==0
        n,       # (D,S) unit normals of contacted facets
        phi,     # friction angle in degrees (float or (S,))
        profiles,
        elem_id,
    ):
        """
        One RK2 (midpoint) step confined to the tangent plane. Works for D=2 or 3.

        Returns:
            pos_new (D,S), vt_new (D,S), is_falling (S,), is_sticking (S,)
        """
        dt = np.full(vt.shape[1], self.timeStep, dtype=float)            # (S,)
        # Gravity vector in world coords (last axis is vertical). Shape (D,)
        g = np.zeros(self._ndim, dtype=float)
        g[self._ndim - 1] = self.gravity

        # Calculate friction coefficient
        mu = np.tan(np.radians(phi))                                     # (S,)

        # Ensure vt is strictly tangential (cheap safety)
        vt = vt - n * np.sum(vt * n, axis=0, keepdims=True)              # (D,S)

        # Tangential component of gravity: g_t = g - (g·n) n
        gn = np.sum(g[:, None] * n, axis=0)                               # (S,)
        gt = g[:, None] - n * gn                                          # (D,S)
        gt_norm = np.linalg.norm(gt, axis=0)                              # (S,)

        # Normal load per unit mass (>=0): N/m = max(0, -(g·n))
        N_over_m = np.maximum(0.0, -gn)                                   # (S,)

        # Direction for friction:
        vt_norm = np.linalg.norm(vt, axis=0)                              # (S,)
        moving = vt_norm > self.stoppedVelocity
        dir_t = np.zeros_like(vt)                                         # (D,S)
        if np.any(moving):
            dir_t[:, moving] = vt[:, moving] / vt_norm[moving]

        # If nearly at rest, use downhill direction (gt) so friction opposes motion correctly if it slips at all
        near_rest = ~moving
        use_downhill = near_rest & (gt_norm > 0)
        if np.any(use_downhill):
            h = gt[:, use_downhill]
            hnorm = np.linalg.norm(h, axis=0)
            dir_t[:, use_downhill] = h / hnorm

        # Tangential acceleration with kinetic friction magnitude μ*N/m opposing dir_t
        a0 = gt - dir_t * (mu * N_over_m)[None, :]                        # (D,S)

        # ---- Projected midpoint (RK2) in the tangent plane ----
        v_half = vt + 0.5 * a0 * dt[None, :]
        # Re-project to tangent (numerical cleanliness)
        v_half = v_half - n * np.sum(v_half * n, axis=0, keepdims=True)

        # Friction direction at half step
        v_half_norm = np.linalg.norm(v_half, axis=0)
        half_moving = v_half_norm > self.stoppedVelocity
        dir_half = np.zeros_like(v_half)
        if np.any(half_moving):
            dir_half[:, half_moving] = v_half[:, half_moving] / v_half_norm[half_moving]
        use_downhill2 = (~half_moving) & (gt_norm > 0)
        if np.any(use_downhill2):
            h2 = gt[:, use_downhill2]
            h2n = np.linalg.norm(h2, axis=0)
            dir_half[:, use_downhill2] = h2 / h2n

        a_half = gt - dir_half * (mu * N_over_m)[None, :]                 # (D,S)

        vt_new = vt + a_half * dt[None, :]
        vt_new = vt_new - n * np.sum(vt_new * n, axis=0, keepdims=True)   # enforce tangency

        pos_new = pos + v_half * dt[None, :]
        pos_new = pos_new - n * np.sum((pos_new - pos) * n, axis=0, keepdims=True)

        # ---- Guaranteed-to-stop clamp (exact stop inside the step when applicable) ----
        # Predict stop time using initial parallel deceleration along dir_t.
        # a_parallel0 < 0 ensures deceleration along motion; t_stop = v0 / |a_parallel0|.
        dir_norm = np.linalg.norm(dir_t, axis=0)
        has_dir = dir_norm > 0
        a_parallel0 = np.sum(a0 * dir_t, axis=0)                          # (S,)
        t_stop = np.full(vt.shape[1], np.inf, dtype=float)
        stop_cand = has_dir & (a_parallel0 < -1e-14)
        if np.any(stop_cand):
            t_stop[stop_cand] = vt_norm[stop_cand] / (-a_parallel0[stop_cand])

        stop_in_step = stop_cand & (t_stop <= dt)
        if np.any(stop_in_step):
            ts = t_stop[stop_in_step]
            p0 = pos[:, stop_in_step]
            v0 = vt[:, stop_in_step]
            a0s = a0[:, stop_in_step]
            # Exact position up to stop time with constant a0 (good approximation over small dt)
            pos_new[:, stop_in_step] = p0 + v0 * ts + 0.5 * a0s * (ts**2)
            vt_new[:, stop_in_step] = 0.0
            dt[stop_in_step] = ts

        # Sticking mask: anything not detached and effectively at rest *after
        # this step* (stopped mid-step, or ended it near zero speed). Do not
        # gate this on the pre-step speed: a block starting the step at rest
        # must still be allowed to accelerate away from rest this step.
        vt_new_norm = np.linalg.norm(vt_new, axis=0)
        is_sticking = (stop_in_step | (vt_new_norm <= self.stoppedVelocity))

        # Treat blocks sliding off to other elements
        disp = pos_new - pos
        new_element_id, alpha = self.slope.exitTime(
            pos,
            disp,
            elem_id,
            samples=profiles,
            tol=self.tolerance
        )
        gone_to_new = (alpha < 1.0) & (alpha > 0.0)
        new_element_id[~gone_to_new] = elem_id[~gone_to_new]
        if np.any(gone_to_new):
            alpha = alpha[gone_to_new]
            pos_new[:, gone_to_new] = pos[:, gone_to_new] + disp[:, gone_to_new] * alpha
            vt_new[:, gone_to_new] = vt_new[:, gone_to_new] + (vt_new[:, gone_to_new] - vt[:, gone_to_new]) * alpha
            dt[gone_to_new] *= alpha

        # Zero-out tiny residual tangential velocity for sticking samples
        if np.any(is_sticking):
            vt_new[:, is_sticking] = 0.0

        return pos_new, vt_new, a0, dt, new_element_id

    def _isFalling(
        self,
        vn: np.ndarray,
        normals: np.ndarray
    ) -> np.ndarray:
        is_falling = (vn > self.normalVelocityThreshold) | (normals[-1] <= 0.0)
        return is_falling

    def _fall(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        profiles: np.ndarray,
        element_id: np.ndarray,
        segment_id: np.ndarray,
        canopy: Optional[np.ndarray] = None,
        drag_params: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Update ballistic blocks. Returns (position, velocity, dt)."""                
        g = np.zeros_like(velocity)
        g[-1, :] = self.gravity
        elapsed_time = np.full(position.shape[1], np.nan, dtype=float)

        no_vegetation = segment_id < 0
        in_vegetation = segment_id >= 0
        if np.any(no_vegetation):
            pos, vel, acc, pro, elm, can, seg = getSubSamples(
                no_vegetation,
                position,
                velocity,
                g,
                profiles,
                element_id,
                canopy,
                segment_id
            )
            elm2, dt_elm = self.slope.intersectParabola(
                pos,
                vel,
                acc,
                elm,
                samples=pro,
                tol=self.tolerance,
            )
            good_elm = np.isfinite(dt_elm)

            if self.hasVegetation:
                seg2, dt_seg = self.vegetation.intersectParabola(
                    pos,
                    vel,
                    acc,
                    seg,
                    samples=can,
                    tol=self.tolerance,
                )
                good_seg = np.isfinite(dt_seg)

                hit_elm = good_elm & (~good_seg | (dt_elm <= dt_seg))
                hit_seg = good_seg & (~good_elm | (dt_seg < dt_elm))

                dt = np.full_like(dt_elm, np.inf)
                dt[hit_elm] = dt_elm[hit_elm]
                dt[hit_seg] = dt_seg[hit_seg]

                good = hit_elm | hit_seg

                seg_out = seg.copy()
                seg_out[hit_seg] = seg2[hit_seg]
            else:
                good = np.isfinite(dt_elm)
                hit_elm = good_elm
                dt = dt_elm
                seg_out = seg

            elm_out = elm.copy()
            elm_out[hit_elm] = elm2[hit_elm]

            no_hit = ~good
            if np.any(no_hit):
                dt_floor = self.slope.intersectFloor(
                    pos[:, no_hit], vel[:, no_hit], acc[:, no_hit],
                    self.slope.floor, t_min=0.0, tol=self.tolerance,
                )
                floor_hit = np.isfinite(dt_floor)
                if np.any(floor_hit):
                    idx_floor = np.flatnonzero(no_hit)[floor_hit]
                    dt[idx_floor] = dt_floor[floor_hit]
                    elm_out[idx_floor] = FLOOR_ELEMENT_ID
                    good[idx_floor] = True

            if np.any(good):
                pos[:, good] += vel[:, good] * dt[good] + 0.5 * acc[:, good] * dt[good]**2
                vel[:, good] += acc[:, good] * dt[good]
            elapsed_time[no_vegetation] = dt
            setSubSamples(
                no_vegetation,
                (pos, vel, elm_out, seg_out),
                position,
                velocity,
                element_id,
                segment_id,
            )

        if np.any(in_vegetation):
            pos, vel, acc, pro, elm, can, seg, drag = getSubSamples(
                in_vegetation,
                position,
                velocity,
                g,
                profiles,
                element_id,
                canopy,
                segment_id,
                drag_params,
            )
            elm2, dt_elm = self.slope.intersectDamped(
                pos,
                vel,
                acc,
                drag,
                elm,
                samples=pro,
                tol=self.tolerance,
            )
            seg2, dt_seg = self.vegetation.intersectDamped(
                pos,
                vel,
                acc,
                drag,
                seg,
                samples=can,
                tol=self.tolerance,
                t_min=self.tolerance
            )
            good_elm = np.isfinite(dt_elm)
            good_seg = np.isfinite(dt_seg)

            hit_elm = good_elm & (~good_seg | (dt_elm <= dt_seg))
            hit_seg = good_seg & (~good_elm | (dt_seg < dt_elm))

            dt = np.full_like(dt_elm, np.inf)
            dt[hit_elm] = dt_elm[hit_elm]
            dt[hit_seg] = dt_seg[hit_seg]

            good = hit_elm | hit_seg
            if np.any(good):
                d = drag[good]  # or damping[good]
                t = dt[good]
                e = np.exp(-d * t)

                a_d = acc[:, good] / d[None, :]
                v0_minus_ad = vel[:, good] - a_d

                pos[:, good] += a_d * t[None, :] + v0_minus_ad / d[None, :] * (1.0 - e)[None, :]
                vel[:, good] = a_d + v0_minus_ad * e[None, :]

            elapsed_time[in_vegetation] = dt
            elm_out = elm.copy()
            seg_out = seg.copy()
            elm_out[hit_elm] = elm2[hit_elm]
            seg_out[hit_seg] = seg2[hit_seg]
            setSubSamples(
                in_vegetation,
                (pos, vel, elm_out, seg_out),
                position,
                velocity,
                element_id,
                segment_id,
            )

        return position, velocity, elapsed_time, element_id, segment_id

    # ----------------- Orchestration -----------------

    def _saveTrajectories(
        self,
        time: np.ndarray,
        position: np.ndarray,
        velocity: np.ndarray,
        angular_velocity: np.ndarray,
        acceleration: np.ndarray,
    ) -> None:
        self.trajectories.addData(time, position, velocity, angular_velocity, acceleration)

    def preprocess(self):
        self._checkBeforeRun()
        return self._generate_samples()

    def process(self, *state_tuple):
        (
            position, velocity, angular_velocity,
            rock_params, material_params,
            time, acceleration,
            profiles, slope_normals,
            drag_params, canopy,
        ) = state_tuple

        down_vec = np.zeros_like(position)
        down_vec[-1] = -1.0
        element_id, time_hit = self.slope.intersectParabola(
            position, down_vec, np.zeros_like(position),
            np.full(position.shape[1], -1, dtype=int), samples=profiles,
            t_min=-np.inf, tol=self.tolerance)
        
        # Check for block inside slope
        is_on_el = element_id >= 0
        if np.any(is_on_el):
            pos_on_el, vel_on_el, id_on_el = getSubSamples(is_on_el,
                position, velocity,
                element_id
            )
            normal_on_el = sampleNormals(pos_on_el, id_on_el, slope_normals)
            is_inside = (normal_on_el[-1] < 0) & (time_hit[is_on_el] > self.tolerance)
            if np.any(is_inside):
                dlt_h = -time_hit[is_on_el][is_inside]
                dlt_v = np.sqrt(2*self.gravity*dlt_h)
                pos_on_el[-1, is_inside] += dlt_h
                vel_on_el[-1, is_inside] += dlt_v
            position[:, is_on_el] = pos_on_el
            velocity[:, is_on_el] = vel_on_el
        
        # Elements under blocks
        element_id[np.abs(time_hit) > self.tolerance] = -1

        # Segments above blocks
        segment_id = np.full(position.shape[1], -1, dtype=int)
        if self._vegetation is not None:
            segment_id, _ = self.vegetation.verticalProjection(
                position,
                segment_id,
                direction=1,
                samples=canopy,
            )

        # Sample roughness
        roughness_samples = self._sampleRoughness(numSamplesPerMaterial=position.shape[1]*100//len(self.slope.materialTable))  # oversample to avoid running out of roughness samples for highly impacted materials
        cum_impacts = np.zeros(len(self.slope.materialTable), dtype=int)

        it = -1
        while True:
            it += 1
            # save
            self._saveTrajectories(time, position, velocity, angular_velocity, acceleration)
            # mapping to segments
            drag_samples = self._impactDrag(segment_id, drag_params)[0]
            # mapping to elements
            is_on_el = element_id >= 0
            is_falling = (~is_on_el) & (element_id != FLOOR_ELEMENT_ID)
            is_sliding = np.full_like(is_on_el, False, dtype=bool)
            if np.any(is_on_el):
                # get samples on elements
                pos_on_el, vel_on_el, acc_on_el, w_on_el, rp_on_el, t_on_el, id_on_el = getSubSamples(is_on_el,
                    position, velocity, acceleration, angular_velocity,
                    rock_params, time, element_id,
                )

                # materials at impact locations
                mp_on_el = self._impactMaterial(id_on_el, material_params)

                # get normal and tangential velocities
                normal_on_el = sampleNormals(pos_on_el, id_on_el, slope_normals)
                normal_on_el = self._addRoughness(normal_on_el, id_on_el, roughness_samples, cum_impacts)
                vn_on_el, vt_on_el = decompose(vel_on_el, normal_on_el)
                an_on_el, at_on_el = decompose(acc_on_el, normal_on_el)
                vn_on_el[np.abs(vn_on_el) < self.stoppedVelocity] = 0.0  # force parallel when vn is nearly zero

                # check if impacting
                is_imp = self._isImpacting(vn_on_el)
                is_low = np.abs(vn_on_el) < self.normalVelocityThreshold  # check if velocity before impact is less then threshold
                
                if np.any(is_imp):
                    # get impacting samples
                    vn_imp, vt_imp, w_imp, m_imp, d_imp, rn_imp, rt_imp, normal_imp = getSubSamples(is_imp,
                        vn_on_el, vt_on_el, w_on_el,
                        rp_on_el[0], rp_on_el[1], mp_on_el[0], mp_on_el[1],
                        normal_on_el
                    )

                    # impact update
                    vn_post, vt_post, w_post = self._impacts(
                        vn_imp, vt_imp, w_imp,
                        m_imp, d_imp,
                        rn_imp, rt_imp,
                        normal_imp
                    )
                    setSubSamples(is_imp, (vn_post, vt_post, w_post), vn_on_el, vt_on_el, w_on_el)

                # check if sliding
                is_low |= np.abs(vn_on_el) < self.stoppedVelocity  # check if post impact velocity is mearly zero
                is_sliding_on_el = np.full_like(is_low, False, dtype=bool)
                if np.any(is_low):
                    vt_low, normal_low, phi_low = getSubSamples(is_low, vt_on_el, normal_on_el, mp_on_el[2])
                    setSubSamples(is_low, (self._isSliding(vt_low, normal_low, phi_low),), is_sliding_on_el)

                # slide update (apply only to sliding blocks by slicing inputs)
                if np.any(is_sliding_on_el):
                    pos_s, vt_s, t_s, normal_s, phi_s, pro_s, id_s = getSubSamples(is_sliding_on_el,
                        pos_on_el, vt_on_el, t_on_el,
                        normal_on_el, mp_on_el[2],
                        profiles, id_on_el
                    )
                    pos_s, vt_s, at_s, dt_s, id_s = self._slide(pos_s, vt_s, normal_s, phi_s, pro_s, id_s)
                    t_s += dt_s
                    setSubSamples(is_sliding_on_el,
                        (pos_s, vt_s, at_s, .0, .0, t_s, id_s),
                        pos_on_el, vt_on_el, at_on_el, vn_on_el, an_on_el, t_on_el, id_on_el
                    )

                # rotate back to Cartesian
                vel_on_el = vt_on_el + normal_on_el * vn_on_el[None, :]
                acc_on_el = at_on_el + normal_on_el * an_on_el[None, :]

                # check if falling
                is_falling_on_el = (~is_low & (vn_on_el > 0.0)) | (normal_on_el[-1] <= 0.0)
                is_falling_on_el &= ~is_sliding_on_el

                setSubSamples(is_on_el,
                    (is_sliding_on_el, is_falling_on_el, pos_on_el, vel_on_el, acc_on_el, w_on_el, t_on_el, id_on_el),
                    is_sliding, is_falling, position, velocity, acceleration, angular_velocity, time, element_id
                )

            # fall update
            if np.any(is_falling):
                pos_fall, vel_fall, pro_fall, elem_fall, seg_fall, can_fall, drag_fall = getSubSamples(
                    is_falling,
                    position,
                    velocity,
                    profiles,
                    element_id,
                    segment_id,
                    canopy,
                    drag_samples,
                )
                pos_fall, vel_fall, dt, elem_fall, seg_fall = self._fall(
                    pos_fall,
                    vel_fall,
                    pro_fall,
                    elem_fall,
                    seg_fall,
                    canopy=can_fall,
                    drag_params=drag_fall,
                )
                good = np.isfinite(dt)
                idx = np.flatnonzero(is_falling)
                idx_keep = idx[good]
                idx_drop = idx[~good]
                is_falling[idx_drop] = False  # stop falling if numerical issue or end of slope

                position[...,idx_keep] = pos_fall[...,good]
                velocity[...,idx_keep] = vel_fall[...,good]
                acceleration[...,idx_keep] = 0.0
                acceleration[-1, idx_keep] = self.gravity
                time[...,idx_keep] += dt[...,good]
                element_id[...,idx_keep] = elem_fall[...,good]
                segment_id[...,idx_keep] = seg_fall[...,good]

            # termination
            if not (np.any(is_falling) or np.any(is_sliding)):
                break
            if self.maxIter > 0 and it >= self.maxIter:
                break

        return (
            position, velocity, angular_velocity,
            rock_params, material_params,
            time, acceleration,
            profiles, slope_normals,
        )

    def postprocess(self, rock_params: np.ndarray) -> None:
        self.trajectories.simulationDone()
        mass = rock_params[0, :]
        density = rock_params[1, :]
        r = np.cbrt(3 * mass / (4 * np.pi * density))
        inertia = 2/5 * mass * r**2
        self.trajectories.mass = mass
        self.trajectories.inertia = inertia
        self.trajectories.floor = self.slope.floor

    def run(self) -> None:
        state_tuple = self.preprocess()
        state_tuple = self.process(*state_tuple)
        self.postprocess(state_tuple[3])
