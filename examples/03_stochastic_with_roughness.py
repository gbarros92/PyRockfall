"""
Example 3 - Stochastic rockfall with surface roughness
=========================================================

Identical to example 2, except the material now also has a `roughness`:
a small random perturbation (in degrees) applied to the slope's surface
normal at each impact, used to emulate small-scale irregularities of the
real rock face that are not captured by the coarse slope geometry.

    roughness ~ TruncatedNormal(mean=3, std=1.5) in [0, 10] degrees

Everything else - slope, rock, seeder, number of samples, plots - is the
same as example 2, so you can directly compare the two figures to see
the effect roughness has on the spread of trajectories and impact energies.

Run with:
    python 03_stochastic_with_roughness.py
"""
import numpy as np
import matplotlib.pyplot as plt
import pyrockfall as pr
from pyrockfall import stats

# ---------------------------------------------------------------------
# 1. Slope geometry (identical to examples 1 and 2)
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
# 2. Material: same distributions as example 2, plus roughness
# ---------------------------------------------------------------------
normal_restitution = stats.Truncate(stats.Normal(0.5, 0.1), lower=0.0, upper=1.0)
tangential_restitution = stats.Truncate(stats.Normal(0.9, 0.1), lower=0.0, upper=1.0)
friction_angle = stats.Truncate(stats.Normal(30.0, 2.0), lower=24.0, upper=36.0)
roughness = stats.Truncate(stats.Normal(3.0, 1.5), lower=0.0, upper=10.0)  # [deg]

material = pr.Material(
    name="Rock face",
    normalRestitution=normal_restitution,
    tangentialRestitution=tangential_restitution,
    frictionAngle=friction_angle,
    roughness=roughness,
)
slope = pr.Slope(geometry, materials=material)

# ---------------------------------------------------------------------
# 3. Rock and seeder: same as example 2
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
analysis.useSpecificSeed = True
analysis.specificSeed = 42
analysis.run()

traj = analysis.trajectories

# ---------------------------------------------------------------------
# 5. Kinetic energy along every trajectory
# ---------------------------------------------------------------------
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
