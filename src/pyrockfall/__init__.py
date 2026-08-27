"""
pyrockfall package
==================

Core interface for the pyrockfall rockfall simulation engine.

Exposes main simulation classes (Slope, Material, Seeder, Analysis, Trajectories)
and data import utilities at the package level for user convenience.
"""
from ._model import Model
from ._pointcloud import PointCloud
from ._mesh import Mesh
from ._geometry import Geometry
from ._geometry3d import Geometry3D
from ._material import Material
from ._slope import Slope
from ._vegetation import Drag, Vegetation
from ._rock import Rock
from ._seeder import SeederRocksThrown, Seeder, LineSeeder, AreaSeeder
from ._analysis import Analysis, Sampling, AnalysisRocksThrown
from ._trajectories import Trajectories
from ._functionalities import *
from ._filesystem import (
    readMaterials, writeMaterials,  # Read/write material libraries as csv
    exportMaterials, importMaterials,  # export/import material libraries .ini files
    exportSlope, importSlope,  # export/import slope geometry and material IDs comma-separated values file
    exportRocks, importRocks,  # export/import rock definitions .ini files
    exportSeeders, importSeeders,  # export/import seeder properties .txt files
    exportSettings, importSettings,  # export/import project settings .ini files
)
from . import stats

__all__ = [
    "stats",
]

