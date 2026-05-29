"""Shared publication style for analysis figures (analysis.*).

Single source of truth for figure width, per-category font sizes, line widths,
and the save format, so the style of every paper figure can be changed by
editing this one file (handy when matching a journal's formatting rules).

Two style groups:
- The four single-column plots (``pomdp_gap``, ``action_profile``,
  ``framestack_ablation``, ``distortion_ablation``) share one width and a
  moderate font scale. They call :func:`apply_style`, which sets
  ``matplotlib.rcParams``, so the scripts stay free of per-call ``fontsize=``.
- ``seir`` is a wide ``1xN`` panel grid; it keeps its own larger scale
  (``SEIR_*``) because it is physically wider. All of its numbers live here too.
"""

import os
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt

# --- Four single-column plots (shared width) ---
FIG_WIDTH = 10.0           # inches; height stays content-dependent
TITLE_FONTSIZE = 20        # agent names / panel titles + suptitle
AXIS_LABEL_FONTSIZE = 16   # axis labels
TICK_FONTSIZE = 13         # tick labels
LEGEND_FONTSIZE = 14       # legend
ANNOTATION_FONTSIZE = 14   # in-plot text (distortion heatmap cells)
LINEWIDTH = 2.5            # data lines

# --- seir: wide panel grid (keeps its own larger scale) ---
SEIR_PANEL_WIDTH = 7
SEIR_PANEL_HEIGHT = 6
SEIR_TWO_ROW_HEIGHT = 12
SEIR_TITLE_FONTSIZE = 40
SEIR_TICK_FONTSIZE = 32
SEIR_AXIS_LABEL_FONTSIZE = 46
SEIR_LEGEND_FONTSIZE = 38
SEIR_LINEWIDTH = 7.0
SEIR_LEGEND_HANDLE_LINEWIDTH = 10
SEIR_ROW_HSPACE = 0.14

SAVE_DPI = 300


def apply_style() -> None:
    """Apply the moderate style (for the four single-column plots) to rcParams.

    Call once before creating a figure. Covers per-category font sizes and the
    line width, so scripts stay free of per-call ``fontsize=``. ``seir`` does
    NOT use this function (it has its own explicit ``SEIR_*`` sizes).
    """
    plt.rcParams.update({
        "axes.titlesize": TITLE_FONTSIZE,
        "figure.titlesize": TITLE_FONTSIZE,
        "axes.labelsize": AXIS_LABEL_FONTSIZE,
        "xtick.labelsize": TICK_FONTSIZE,
        "ytick.labelsize": TICK_FONTSIZE,
        "legend.fontsize": LEGEND_FONTSIZE,
        "lines.linewidth": LINEWIDTH,
    })


def save_figure(save_path: Optional[Union[str, os.PathLike]]) -> None:
    """Save the current figure as PDF (publication format) and close it.

    The extension is forced to ``.pdf``. If ``save_path`` is ``None``, show
    interactively instead. Mirrors the ``src.utils._save_or_show`` contract but
    PDF-only.

    Args:
        save_path: Output path, ``str`` or ``Path`` (any extension is coerced
            to ``.pdf``). If ``None``, the figure is shown interactively.
    """
    if save_path:
        pdf_path = Path(save_path).with_suffix(".pdf")
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(pdf_path, dpi=SAVE_DPI, bbox_inches="tight")
    else:
        plt.show()
    plt.close()
