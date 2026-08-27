# PyRockfall

[![Tests](https://github.com/gbarros92/PyRockfall/actions/workflows/tests.yml/badge.svg)](https://github.com/gbarros92/PyRockfall/actions/workflows/tests.yml)

**PyRockfall** is an open-source Python package for stochastic rockfall simulation. It runs **vectorised Monte Carlo rockfall simulations** on user-defined or 3D-extracted 2D profiles, as well as directly on 3D models, for rockfall hazard assessment and mitigation design.

Rather than simulating Monte Carlo realisations sequentially, PyRockfall represents the ensemble using NumPy arrays and advances all samples simultaneously through vectorised operations. This enables efficient probabilistic analysis of rockfall trajectories and associated engineering design quantities.

## Key Features

- **Vectorised Monte Carlo simulation**  
  Rockfall samples are propagated simultaneously using vectorised numerical operations rather than sample-by-sample Python loops.

- **2D rockfall simulation**  
  Run deterministic or stochastic simulations on user-defined slope profiles.

- **2D simulation from 3D geometry**  
  Extract representative 2D sections from 3D rock-face models and use them for rockfall simulation.

- **3D rockfall simulation**  
  Simulate rockfall trajectories directly on three-dimensional slope geometries.

- **Spatially varying materials**  
  Represent slopes containing multiple geological or surface materials with different rock-slope interaction properties.

- **Probabilistic material properties**  
  Sample uncertain interaction parameters for Monte Carlo analysis.

- **Engineering-oriented outputs**  
  Evaluate quantities relevant to rockfall hazard assessment and mitigation design, including kinetic energy, impact locations, and run-out distances.

- **Result visualisation**  
  Analyse simulation outputs and map calculated quantities onto the original slope geometry.

- **Open-source Python implementation**  
  Designed to support reproducible research, extension of the numerical framework, and integration into other rockfall-analysis workflows.

## Installation

Install PyRockfall from PyPI:

```bash
pip install pyrockfall