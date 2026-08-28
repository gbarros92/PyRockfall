Command-line interface
=======================

Installing pyrockfall also installs the ``pyrockfall`` command, a thin
wrapper around the entry scripts in :mod:`pyrockfall.scripts`:

.. code-block:: bash

   pyrockfall {create,extract,section} ...

``pyrockfall create``
----------------------

Create input files for rockfall simulations from profiles, given an
initial height above the floor, a height increment between seeders, and an
initial drop height used to trigger the fall.

.. code-block:: bash

   pyrockfall create profiles.xyz [options]

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Argument
     - Description
   * - ``filename``
     - Input profiles file (``.xyz`` with material and profile attributes).
   * - ``-m``, ``--material-lib``
     - Material library file. If omitted, uses the bundled default.
   * - ``-n``, ``--material-name``
     - Attribute name for material IDs in the profiles file. Default: ``Material``.
   * - ``-p``, ``--profile-name``
     - Attribute name for profile IDs in the profiles file. Default: ``Profile``.
   * - ``-H``, ``--height-start``
     - Starting height for simulations. Default: ``1.0``.
   * - ``-D``, ``--height-delta``
     - Delta height between seeders. Default: ``1.0``.
   * - ``-i``, ``--drop-init``
     - Initial drop height for simulations. Default: ``0.5``.
   * - ``-N``, ``--number-of-rocks``
     - Number of rocks per seeder. Default: ``1000``.

``pyrockfall extract``
------------------------

Extract representative 2D profiles from a 3D model (point cloud or mesh),
for use as input to ``pyrockfall create``.

.. code-block:: bash

   pyrockfall extract model.ply [options]

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Argument
     - Description
   * - ``filename``
     - Input point cloud file (``.ply`` with material attributes).
   * - ``-L``, ``--segment-length``
     - Minimum segment length, in metres. Default: ``10.0``.
   * - ``-S``, ``--profile-spacing``
     - Spacing between profiles, in metres. Default: ``1.0``.
   * - ``-q``, ``--profile-resolution``
     - Profile resolution, in metres. Default: ``0.15``.
   * - ``-n``, ``--material-name``
     - Attribute name for material IDs in the point cloud. Default: ``Material``.
   * - ``-R``, ``--remove-materials``
     - Remove one or more materials by ID, e.g. ``-R 6 7``.
   * - ``--save-profiles``
     - Save profiles: ``yes``, ``no``, or ``separate``. Default: ``yes``.
   * - ``--save-segments``
     - Save segments: ``yes``, ``no``, or ``aligned``. Default: ``no``.

``pyrockfall section``
------------------------

Extract representative 2D profiles from a 3D point cloud by tracing
sections along strike (:meth:`pyrockfall.PointCloud.section`), for use as
input to ``pyrockfall create``. Unlike ``pyrockfall extract``, this does not
require segmenting or pre-aligning the model: the section-tracing handles
curved or non-planar walls directly.

.. code-block:: bash

   pyrockfall section model.ply [options]

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Argument
     - Description
   * - ``filename``
     - Input point cloud file (``.ply`` with material attributes).
   * - ``-S``, ``--profile-spacing``
     - Spacing between profiles/sections, in metres. Default: ``1.0``.
   * - ``-n``, ``--material-name``
     - Attribute name for material IDs in the point cloud. Default: ``Material``.
   * - ``-r``, ``--transverse-radius``
     - Half-width of the strip used to assign points to each section, in
       metres. Defaults to half the point cloud's vertical extent.
   * - ``--min-points``
     - Minimum number of points required for a section to be valid. Default: ``20``.
   * - ``--max-turn-angle``
     - Maximum change in marching direction between consecutive nodes, in
       degrees. Default: ``45.0``.
   * - ``--save-profiles``
     - Save profiles: ``yes`` (combined), ``no``, or ``separate``. Default: ``yes``.

All subcommands are also runnable programmatically via
``main_from_namespace`` -- see :mod:`pyrockfall.scripts.create`,
:mod:`pyrockfall.scripts.extract`, and :mod:`pyrockfall.scripts.section`.
