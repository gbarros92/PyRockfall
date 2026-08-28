"""
Example 1 - Deterministic rockfall, smooth slope (no roughness)
=================================================================

The simplest possible PyRockfall simulation:
  - a single straight rock face followed by a flat run-out,
  - a single material with fixed (deterministic) rn, rt, phi,
  - a single rock, thrown once.

The block is seeded 0.5 m above and 0.1 m to the right of the crest, so it
free-falls that 0.5 m before its first impact - this is what gives it its
initial kinetic energy, instead of setting an initial velocity by hand.

Run with:
    python 01_deterministic_no_roughness.py
"""
import numpy as np
import matplotlib.pyplot as plt
import pyrockfall as pr

# ---------------------------------------------------------------------
# 1. Slope geometry
# ---------------------------------------------------------------------
# A rock face of a given height and angle, followed by a flat run-out
# where the block can roll/slide to a stop.
height = 45.0          # [m] height of the rock face
slope_angle = 75.0     # [deg] angle of the face from horizontal
runout_length = 40.0   # [m] flat ground at the toe of the slope

run = height / np.tan(np.radians(slope_angle))
nodes = np.array([
    [0.0, height],               # crest
    [run, 0.0],                  # toe of the face
    [run + runout_length, 0.0],  # end of the flat run-out
])
geometry = pr.Geometry(nodes)

# ---------------------------------------------------------------------
# 2. Material: deterministic rn, rt, phi, no roughness
# ---------------------------------------------------------------------
material = pr.Material(
    name="Rock face",
    normalRestitution=0.5,       # rn
    tangentialRestitution=0.9,   # rt
    frictionAngle=30.0,          # phi [deg]
    roughness=0.0,               # perfectly smooth surface
)

# A single material applies to the whole slope (every segment).
slope = pr.Slope(geometry, materials=material)

# ---------------------------------------------------------------------
# 3. Rock and seeder
# ---------------------------------------------------------------------
rock = pr.Rock(name="Boulder", mass=1000.0, density=2700.0)

seed_point = np.array([0.1, height + 0.5])
seeder = pr.Seeder(seed_point, rocks=[rock])
seeder.numberOfRocks = 1
seeder.translationalVelocity = [0.0, 0.0]   # released from rest
seeder.angularVelocity = [0.0]

# ---------------------------------------------------------------------
# 4. Run the analysis
# ---------------------------------------------------------------------
analysis = pr.Analysis()
analysis.slope = slope
analysis.seeders = [seeder]
analysis.scaleByVelocity = True             # restitution scales with impact speed
analysis.considerRotationalVelocity = True  # rocks can roll/spin
analysis.run()

traj = analysis.trajectories

# ---------------------------------------------------------------------
# 5. Kinetic energy along the trajectory
# ---------------------------------------------------------------------
numPoints = 300
time = np.linspace(traj.startTime, traj.stopTime, numPoints)  # (numPoints, 1)
pos = traj.position(time)                                     # (2, numPoints, 1)
vel, w = traj.velocity(time, angVel=True)                     # (2, numPoints, 1), (1, numPoints, 1)

x, y = pos[0, :, 0], pos[1, :, 0]
translational_KE = 0.5 * traj.mass * np.sum(vel[:, :, 0] ** 2, axis=0)
rotational_KE = 0.5 * traj.inertia * w[0, :, 0] ** 2
kinetic_energy = (translational_KE + rotational_KE) / 1000.0  # [kJ]

# ---------------------------------------------------------------------
# 6. Plots
# ---------------------------------------------------------------------
fig, (ax_traj, ax_ke) = plt.subplots(1, 2, figsize=(11, 4.5))

ax_traj.plot(slope.nodes[:, 0], slope.nodes[:, 1], "k-", linewidth=1.5, label="Slope")
ax_traj.plot(x, y, "r-", linewidth=1.0, label="Trajectory")
ax_traj.scatter(*seed_point, color="b", marker="x", label="Seeder")
ax_traj.set_xlabel("x [m]")
ax_traj.set_ylabel("y [m]")
ax_traj.set_aspect("equal", adjustable="box")
ax_traj.legend()
ax_traj.grid(True)

ax_ke.plot(x, kinetic_energy, "r-")
ax_ke.set_xlabel("x [m]")
ax_ke.set_ylabel("Kinetic energy [kJ]")
ax_ke.grid(True)

fig.tight_layout()
plt.show()
