.. pyrockfall documentation master file, created by
   sphinx-quickstart on Wed Jul 30 10:21:48 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pyrockfall's documentation!
======================================

**pyrockfall** is an open-source Python package for stochastic rockfall
simulation. It runs vectorised Monte Carlo rockfall simulations on
user-defined or 3D-extracted 2D profiles, as well as directly on 3D models,
for rockfall hazard assessment and mitigation design.

Rather than simulating Monte Carlo realisations sequentially, pyrockfall
represents the ensemble using NumPy arrays and advances all samples
simultaneously through vectorised operations. This enables efficient
probabilistic analysis of rockfall trajectories and associated engineering
design quantities.

Key features
------------

- **Vectorised Monte Carlo simulation** -- rockfall samples are propagated
  simultaneously using vectorised numerical operations rather than
  sample-by-sample Python loops.
- **2D rockfall simulation** -- run deterministic or stochastic simulations
  on user-defined slope profiles.
- **2D simulation from 3D geometry** -- extract representative 2D sections
  from 3D rock-face models and use them for rockfall simulation.
- **3D rockfall simulation** -- simulate rockfall trajectories directly on
  three-dimensional slope geometries.
- **Spatially varying materials** -- represent slopes containing multiple
  geological or surface materials with different rock-slope interaction
  properties.
- **Probabilistic material properties** -- sample uncertain interaction
  parameters for Monte Carlo analysis.
- **Engineering-oriented outputs** -- evaluate quantities relevant to
  rockfall hazard assessment and mitigation design, including kinetic
  energy, impact locations, and run-out distances.
- **Result visualisation** -- analyse simulation outputs and map calculated
  quantities onto the original slope geometry.

.. toctree::
   :maxdepth: 1
   :caption: Getting started

   examples
   installation
   manual

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
