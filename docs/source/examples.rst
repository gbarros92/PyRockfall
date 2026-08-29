Examples
========

Four self-contained scripts of increasing complexity, meant as a starting
point for pyrockfall. They live in the ``examples/`` directory of the
repository. Examples 1-3 share the same synthetic slope (45 m high, 75 deg
face) and the same 1000 kg / 2700 kg/m\ :sup:`3` rock. Example 4 switches to
a slope with realistic roughness loaded from data files.

Each script is runnable directly, e.g.:

.. code-block:: bash

   python 01_deterministic_no_roughness.py

Requires ``pyrockfall``, ``numpy`` and ``matplotlib``.

1. Deterministic rockfall, no roughness
-----------------------------------------

One rock, one run. Fixed ``rn=0.5``, ``rt=0.9``, ``phi=30 deg``, no
roughness. Plots the trajectory and the kinetic energy along it.

.. literalinclude:: ../../examples/01_deterministic_no_roughness.py
   :language: python

2. Stochastic material properties
------------------------------------

``rn``, ``rt`` and ``phi`` become truncated normal distributions. 1000
rocks are thrown; all trajectories are plotted, and the mean/90th-percentile
kinetic energy vs. x is computed.

.. literalinclude:: ../../examples/02_stochastic_material.py
   :language: python

3. Stochastic material with surface roughness
------------------------------------------------

Same as example 2, with a random ``roughness`` added to the material to
represent small-scale surface irregularities.

.. literalinclude:: ../../examples/03_stochastic_with_roughness.py
   :language: python

4. Stochastic simulation from input files
---------------------------------------------

Instead of building the slope and materials in the script, everything
(``materials.ini``, ``rocks.ini``, ``profile.txt``, ``seeders.txt``,
``settings.ini``) is loaded from files. Same trajectory and kinetic energy
plots as examples 2 and 3, now on a multi-material slope.

.. literalinclude:: ../../examples/04_stochastic_real_roughness/04_stochastic_real_roughness.py
   :language: python
