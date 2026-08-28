Installation
============

Install pyrockfall from PyPI:

.. code-block:: bash

   pip install pyrockfall

Development install
--------------------

To work on pyrockfall itself, clone the repository and install it in
editable mode with the development extras:

.. code-block:: bash

   git clone https://github.com/gbarros92/PyRockfall.git
   cd PyRockfall
   pip install -e ".[dev]"

The development extras include ``pytest``, ``ipdb``, ``black``, ``ruff``,
``mypy``, ``build``, ``twine`` and ``rich``, which are used for testing,
debugging, linting and packaging.

Requirements
------------

pyrockfall requires Python 3.10 or newer, and depends on:

- ``numpy``
- ``scipy``
- ``pandas``
- ``matplotlib``
- ``seaborn``
- ``scikit-learn``
- ``plyfile``
- ``open3d``
- ``tensorflow``
- ``xgboost``
