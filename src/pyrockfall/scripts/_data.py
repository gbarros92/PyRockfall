"""Bundled data resources for CLI scripts (no import-time side effects)."""
from __future__ import annotations
from contextlib import contextmanager
from importlib.resources import files, as_file
from typing import Iterator

__all__ = [
    "DEFAULT_MATERIAL_LIB",
    "material_lib_path",
]

DEFAULT_MATERIAL_LIB = "material_lib.csv"


@contextmanager
def material_lib_path() -> Iterator[str]:
    """Yield a filesystem path to the bundled default material library CSV."""
    ref = files("pyrockfall.scripts") / DEFAULT_MATERIAL_LIB
    with as_file(ref) as tmp_path:
        yield str(tmp_path)
