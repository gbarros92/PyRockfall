"""Plotting utilities for CLI scripts (no import-time side effects)."""
from __future__ import annotations
import re
import json
import numpy
import shutil
import warnings
import itertools
import matplotlib as mpl
import matplotlib.pyplot as plt
from importlib.resources import files, as_file
from pathlib import Path
from typing import Iterable, Mapping

__all__ = [
    "configure_matplotlib",
    "fmt_text",
    "to_latex",
    "latex_palette",
    "MATERIAL_COLORS",
    "get_material_colors",
    "ensure_palette_for",
    "fill_nans_linear",
    "line_with_fill",
    "wrap_text",
]

DEFAULT_MPLSTYLE = "mplstyle.json"

# Common axis/label text used across scripts
fmt_text: dict[str, str] = {
    'E1_90': r'$E_{1,90}$ [J]',
    'd1_95': r'$d_{1,95}$ [m]',
    'df_95': r'$d_{f,95}$ [m]',
    'E1': r'$E_{1}$ [J]',
    'd1': r'$d_{1}$ [m]',
    'df': r'$d_{f}$ [m]',
    'first_impact': r'$d_{1}$ [m]',
    'runout': r'$d_{f}$ [m]',
}

def to_latex(s: str) -> str:
    """Escape LaTeX special characters in plain text labels."""
    repl = {
        '\\': r'\textbackslash{}', '{': r'\{', '}': r'\}',
        '%': r'\%', '$': r'\$', '&': r'\&', '#': r'\#',
        '_': r'\_', '~': r'\textasciitilde{}', '^': r'\textasciicircum{}',
    }
    return re.sub(r'[\\{}%$&#_^~^]', lambda m: repl[m.group()], s)

def latex_palette(name_to_hex: dict[str, str]) -> dict[str, str]:
    """Map a {name: color} dict to a LaTeX-escaped version for plotting."""
    return {to_latex(k): v for k, v in name_to_hex.items()}

def _has_latex() -> bool:
    return shutil.which("latex") is not None


# Canonical, repo-wide material colors (raw names, not LaTeX-escaped)
MATERIAL_COLORS: dict[str, str] = {
    "Weathered Sandstone": "#FFFF80",
    "Weathered Mudstone":  "#FF8080",
    "Sandstone":           "#FFFF00",
    "Mudstone":            "#FF0000",
    "Coal":                "#000000",
    "Coarse Sandstone":    "#D2D200",
    r"Talus\Debris":       "#785448",
    "Interbedded Sandstone & Mudstone": "#FFC800",
    "Interbedded Mudstone & Sandstone": "#FF4B00",
    "Interbedded Coal & Sandstone":     "#808000",
    "Interbedded Coal & Mudstone":      "#800000",
    "Extremely Weathered Rock":         "#646464",
    "All layers": "#0000FF",
}

def get_material_colors(
    *,
    latex_escape: bool = True,
    overrides: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """
    Return the (optionally LaTeX-escaped) material->color mapping.
    `overrides` lets callers tweak colors without changing the defaults.
    """
    palette = dict(MATERIAL_COLORS)
    if overrides:
        palette.update(overrides)
    return {to_latex(k): v for k, v in palette.items()} if latex_escape else palette

def ensure_palette_for(
    labels: Iterable[str],
    base_palette: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """
    Ensure every label has a colour. Uses `base_palette` for known labels and
    falls back to the Matplotlib color cycle for unknowns. Assumes `labels`
    are already LaTeX-escaped if `base_palette` keys are escaped.
    """
    if base_palette is None:
        base_palette = get_material_colors(latex_escape=True)

    cycle = mpl.rcParams.get("axes.prop_cycle", None)
    cycle_colors = (cycle.by_key().get("color", []) if cycle else []) or ["#999999"]
    fallback = itertools.cycle(cycle_colors)

    out = {}
    for lab in labels:
        out[lab] = base_palette.get(lab, next(fallback))
    return out

def configure_matplotlib(
    use_latex: bool = True,
    figure_dpi: int = 600,
    figure_width_in: float | None = 3.0,
    figure_height_in: float | None = 3.0,
    serif_family: str = "serif",
    latex_preamble: str = r"\usepackage{newtxtext,newtxmath,amsmath}",
    style_path: str | None = None,
) -> None:
    """
    Apply consistent Matplotlib rcParams across all high-level scripts.

    Base rcParams are loaded from a bundled JSON file (`mplstyle.json`, or
    `style_path` if given) so plotting defaults can be tweaked without
    touching code. Runtime-dependent settings (LaTeX availability, figure
    size, DPI) stay as function arguments.

    - If LaTeX is not available, gracefully falls back to Matplotlib's mathtext.
    """
    if use_latex and not _has_latex():
        warnings.warn("LaTeX not found on PATH; falling back to mathtext (usetex=False).")
        use_latex = False

    if style_path is None:
        with as_file(files("pyrockfall.scripts") / DEFAULT_MPLSTYLE) as p:
            style = json.loads(Path(p).read_text())
    else:
        style = json.loads(Path(style_path).read_text())

    rc = mpl.rcParams
    rc.update(style)

    # Text/Fonts
    rc["text.usetex"] = bool(use_latex)
    rc["font.family"] = serif_family
    if use_latex:
        rc["text.latex.preamble"] = latex_preamble

    # Sizes
    if figure_width_in is not None and figure_height_in is not None:
        rc["figure.figsize"] = [float(figure_width_in), float(figure_height_in)]
    rc["figure.dpi"] = int(figure_dpi)
    rc["savefig.dpi"] = int(figure_dpi)

def wrap_text(s: str, width: int) -> str:
    """Lightweight word wrap that preserves words."""
    return re.sub(rf'(.{{1,{width}}})(\s+|$)', r'\1\n', s).strip()

def fill_nans_linear(arr: numpy.ndarray) -> numpy.ndarray:
    """
    Replace interior NaNs in a 1D array with linear interpolation;
    preserve leading and trailing NaNs.
    """
    x = numpy.arange(len(arr))
    mask = numpy.isfinite(arr)
    if mask.sum() < 2:
        return arr
    filled = numpy.interp(x, x[mask], arr[mask])
    first, last = numpy.where(mask)[0][[0, -1]]
    filled[:first] = numpy.nan
    filled[last + 1:] = numpy.nan
    return filled

def line_with_fill(data, x, y, color=None, **kws):
    ax = plt.gca()
    linestyles = ["-", "--", "-.", ":"]
    layer_ids = data["layer"].unique()
    ls_map = {lid: ls for lid, ls in zip(sorted(layer_ids), itertools.cycle(linestyles))}
    for lid, grp in data.groupby("layer"):
        grp = grp.sort_values(x)
        ax.plot(grp[x], grp[y], color=color, linestyle=ls_map[lid])
        ax.fill_between(grp[x], grp[y], color=color, alpha=0.3)
