"""
Example 2 - Stochastic rockfall, smooth slope (no roughness)
==============================================================

Same slope, rock and seeder as example 1, but now rn, rt and phi are
random variables instead of fixed numbers:

    rn  ~ TruncatedNormal(mean=0.5, std=0.1) in [0, 1]
    rt  ~ TruncatedNormal(mean=0.9, std=0.1) in [0, 1]
    phi ~ TruncatedNormal(mean=30,  std=2)   in [24, 36]

1000 independent rocks are thrown from the same seed point. Because each
one samples its own rn/rt/phi, every trajectory is a bit different. We
plot all of them, then summarise the kinetic energy with its mean and
90th percentile as a function of x.

Run with:
    python 02_stochastic_material.py
"""
import numpy as np
import matplotlib.pyplot as plt
import pyrockfall as pr
from pyrockfall import stats

# ---------------------------------------------------------------------
# 1. Slope geometry (identical to example 1)
# ---------------------------------------------------------------------
height = 45.0
slope_angle = 75.0
runout_length = 40.0

run = height / np.tan(np.radians(slope_angle))
nodes = np.array([
    [0.0, height],
    [run, 0.0],
    [run + runout_length, 0.0],
])
geometry = pr.Geometry(nodes)

# ---------------------------------------------------------------------
# 2. Material: every parameter is now a truncated normal distribution
# ---------------------------------------------------------------------
normal_restitution = stats.Truncate(stats.Normal(0.5, 0.1), lower=0.0, upper=1.0)
tangential_restitution = stats.Truncate(stats.Normal(0.9, 0.1), lower=0.0, upper=1.0)
friction_angle = stats.Truncate(stats.Normal(30.0, 2.0), lower=24.0, upper=36.0)

material = pr.Material(
    name="Rock face",
    normalRestitution=normal_restitution,
    tangentialRestitution=tangential_restitution,
    frictionAngle=friction_angle,
    roughness=0.0,   # still perfectly smooth - see example 3
)
slope = pr.Slope(geometry, materials=material)

# ---------------------------------------------------------------------
# 3. Rock and seeder: same as example 1, thrown 1000 times
# ---------------------------------------------------------------------
rock = pr.Rock(name="Boulder", mass=1000.0, density=2700.0)

seed_point = np.array([0.1, height + 0.5])
seeder = pr.Seeder(seed_point, rocks=[rock])
seeder.numberOfRocks = 1000

# ---------------------------------------------------------------------
# 4. Run the analysis
# ---------------------------------------------------------------------
analysis = pr.Analysis()
analysis.slope = slope
analysis.seeders = [seeder]
analysis.scaleByVelocity = True
analysis.considerRotationalVelocity = True
analysis.useSpecificSeed = True     # reproducible sampling
analysis.specificSeed = 42
analysis.run()

traj = analysis.trajectories

# ---------------------------------------------------------------------
# 5. Kinetic energy along every trajectory
# ---------------------------------------------------------------------
# Each of the 1000 blocks stops at its own time, so `traj.startTime`/
# `traj.stopTime` are arrays (one value per block) and `time` below is a
# (numPoints, numSamples) grid: one normalised time axis per block.
numPoints = 300
time = np.linspace(traj.startTime, traj.stopTime, numPoints)  # (numPoints, 1000)
pos = traj.position(time)                                     # (2, numPoints, 1000)
vel, w = traj.velocity(time, angVel=True)                     # (2, ...), (1, ...)

x = pos[0]                                                     # (numPoints, 1000)
translational_KE = 0.5 * traj.mass * np.sum(vel ** 2, axis=0)  # (numPoints, 1000)
rotational_KE = 0.5 * traj.inertia * w[0] ** 2
kinetic_energy = (translational_KE + rotational_KE) / 1000.0  # [kJ]

# ---------------------------------------------------------------------
# 6. Resample every trajectory onto a common x-grid, then summarise
# ---------------------------------------------------------------------
x_grid = np.linspace(seed_point[0], np.nanmax(x), 200)
energy_on_grid = np.full((x_grid.size, x.shape[1]), np.nan)
for s in range(x.shape[1]):
    energy_on_grid[:, s] = np.interp(x_grid, x[:, s], kinetic_energy[:, s], left=np.nan, right=np.nan)

mean_energy = np.nanmean(energy_on_grid, axis=1)
p90_energy = np.nanpercentile(energy_on_grid, 90, axis=1)

# ---------------------------------------------------------------------
# 7. Plots
# ---------------------------------------------------------------------
fig, (ax_traj, ax_ke) = plt.subplots(1, 2, figsize=(11, 4.5))

ax_traj.plot(slope.nodes[:, 0], slope.nodes[:, 1], "k-", linewidth=1.5, zorder=3, label="Slope")
ax_traj.plot(x, pos[1], "r-", linewidth=0.3, alpha=0.1)
ax_traj.set_xlabel("x [m]")
ax_traj.set_ylabel("y [m]")
ax_traj.set_aspect("equal", adjustable="box")
ax_traj.legend()
ax_traj.grid(True)

ax_ke.plot(x_grid, mean_energy, "b-", label="Mean")
ax_ke.plot(x_grid, p90_energy, "r--", label="90th percentile")
ax_ke.set_xlabel("x [m]")
ax_ke.set_ylabel("Kinetic energy [kJ]")
ax_ke.legend()
ax_ke.grid(True)

fig.tight_layout()
plt.show()
