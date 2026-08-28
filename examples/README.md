# PyRockfall examples

Four self-contained scripts of increasing complexity, meant as a starting
point for PyRockfall. Examples 1-3 share the same synthetic slope
(45 m high, 75 deg face) and the same 1000 kg / 2700 kg/m^3 rock.
Example 4 switches to a slope with realistic roughness.

- **01_deterministic_no_roughness.py** - one rock, one run. Fixed
  `rn=0.5`, `rt=0.9`, `phi=30 deg`, no roughness. Plots the trajectory and
  the kinetic energy along it.
- **02_stochastic_material.py** - `rn`, `rt`, `phi` become truncated
  normal distributions. 1000 rocks are thrown; all trajectories are
  plotted, and the mean/90th-percentile kinetic energy vs. x is computed.
- **03_stochastic_with_roughness.py** - same as (2), with a random
  `roughness` added to the material to represent small-scale surface
  irregularities.
- **04_stochastic_real_roughness/** - instead of building the slope and
  materials in the script, everything (`materials.ini`, `rocks.ini`,
  `profile.txt`, `seeders.txt`, `settings.ini`) is loaded from files.
  Same trajectory and kinetic energy plots as (2) and (3), now on a multi-
  material slope.

Run any of them directly, e.g.:

```bash
python 01_deterministic_no_roughness.py
```

Requires `pyrockfall`, `numpy` and `matplotlib`.
