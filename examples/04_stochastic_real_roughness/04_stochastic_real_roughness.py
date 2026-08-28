"""
Example 4 - Stochastic rockfall from real project files
==========================================================

Unlike examples 1-3, which build the slope/material/rock/seeder in the
script itself, this example loads everything from files exported by a
real rockfall analysis project (RocFall-style .ini/.txt files):

    materials.ini  - material library (rn, rt, phi, roughness, ... per material)
    rocks.ini      - rock groups (mass, density)
    profile.txt    - slope geometry with a material ID per segment
    seeders.txt    - seeder location(s), velocities and number of rocks
    settings.ini   - analysis settings (sampling method, physics options, ...)

The slope here has several different materials along its length (each
with its own, possibly random, roughness), so this is the closest of the
four examples to a real FYP rockfall study. The plots are the same as in
examples 1-3: all trajectories, and the mean/90th-percentile kinetic
energy vs x.

Run with:
    python 04_stochastic_real_roughness.py
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Patch

import pyrockfall as pr
from pyrockfall import stats


# ---------------------------------------------------------------------
# 1. Load materials, rocks, slope, seeders and settings from file
# ---------------------------------------------------------------------
folder = os.path.dirname(os.path.abspath(__file__))

rocks = pr.importRocks(os.path.join(folder, "rocks.ini"))
materials = pr.importMaterials(os.path.join(folder, "materials.ini"))
mColors = np.array([m.color for m in materials], dtype=float) / 255.0

analysis = pr.importSettings(os.path.join(folder, "settings.ini"))
slope = pr.importSlope(os.path.join(folder, "profile.txt"), materials=materials)
seeders = pr.importSeeders(os.path.join(folder, "seeders.txt"), rockTypes=rocks)

analysis.slope = slope
analysis.seeders = seeders

# ---------------------------------------------------------------------
# 2. Run the analysis
# ---------------------------------------------------------------------
analysis.run()
traj = analysis.trajectories

# ---------------------------------------------------------------------
# 3. Kinetic energy along every trajectory (see example 2 for details)
# ---------------------------------------------------------------------
numPoints = 300
time = np.linspace(traj.startTime, traj.stopTime, numPoints)
pos = traj.position(time)
vel, w = traj.velocity(time, angVel=True)

x = pos[0]
translational_KE = 0.5 * traj.mass * np.sum(vel ** 2, axis=0)
rotational_KE = 0.5 * traj.inertia * w[0] ** 2
kinetic_energy = (translational_KE + rotational_KE) / 1000.0  # [kJ]

# Resample every trajectory onto a common x-grid so the samples can be
# averaged even though each block stops at a different x.
x0 = np.nanmin(x[0, :])
x_grid = np.linspace(x0, np.nanmax(x), 200)
energy_on_grid = np.full((x_grid.size, x.shape[1]), np.nan)
for s in range(x.shape[1]):
    energy_on_grid[:, s] = np.interp(x_grid, x[:, s], kinetic_energy[:, s], left=np.nan, right=np.nan)

mean_energy = np.nanmean(energy_on_grid, axis=1)
p90_energy = np.nanpercentile(energy_on_grid, 90, axis=1)

# ---------------------------------------------------------------------
# 4. Plots
# ---------------------------------------------------------------------
fig, (ax_traj, ax_ke) = plt.subplots(1, 2, figsize=(11, 4.5))

# Slope, coloured by material
nodes = slope.nodes
segments = np.stack([nodes[:-1], nodes[1:]], axis=1)
colors = [mColors[m] for m in slope.materialIDs]
ax_traj.add_collection(LineCollection(segments, colors=colors, linewidths=1.5, zorder=3))
for seeder in seeders:
    ax_traj.scatter(seeder.points[0], seeder.points[1], color="k", marker="x", zorder=4)

ax_traj.plot(x, pos[1], "r-", linewidth=0.3, alpha=0.1)

handles = [
    Patch(facecolor=mColors[i], edgecolor="k", label=materials[i].name)
    for i in np.unique(slope.materialIDs).astype(int)
]
ax_traj.legend(handles=handles, fontsize=7, loc="upper right")
ax_traj.set_xlabel("x [m]")
ax_traj.set_ylabel("y [m]")
ax_traj.set_xlim(nodes[:, 0].min(), nodes[:, 0].max())
ax_traj.set_aspect("equal", adjustable="box")
ax_traj.grid(True)

ax_ke.plot(x_grid, mean_energy, "b-", label="Mean")
ax_ke.plot(x_grid, p90_energy, "r--", label="90th percentile")
ax_ke.set_xlabel("x [m]")
ax_ke.set_ylabel("Kinetic energy [kJ]")
ax_ke.legend()
ax_ke.grid(True)

fig.tight_layout()
plt.show()
