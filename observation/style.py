"""Shared figure styling so every panel is measured with the same ruler."""
import numpy as np

BENIGN_COLOR = "#546E7A"
AXIS_COLOR = "#37474F"
DEFAULT_PALETTE = ["#D81B60", "#00897B", "#F4511E", "#5E35B1", "#1565C0",
                   "#00838F", "#6D4C41", "#AD1457"]

FS_TICK, FS_LABEL, FS_TITLE, FS_LEGEND = 20, 20, 21, 18
LW_AXIS = 2.6
SYMLOG_LINTHRESH = 10.0


def pow10_label(v):
    """symlog tick label: 0, 10^k, -10^k."""
    if v == 0:
        return "0"
    return rf"${'-' if v < 0 else ''}10^{{{int(np.log10(abs(v)))}}}$"


def decorate_axes(ax, xlabel, ylabel, title):
    """Arrow axes, no spines, no grid, no origin lines."""
    for sp in ("top", "right", "bottom", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="both", width=LW_AXIS, length=7)
    for xy in ((1.0, 0), (0, 1.0)):
        ax.annotate("", xy=xy, xycoords="axes fraction", xytext=(0, 0),
                    textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>", color=AXIS_COLOR, lw=LW_AXIS,
                                    shrinkA=0, shrinkB=0, mutation_scale=26))
    ax.set_xlabel(xlabel, fontsize=FS_LABEL, labelpad=9)
    ax.set_ylabel(ylabel, fontsize=FS_LABEL, labelpad=9)
    ax.set_title(title, fontsize=FS_TITLE, pad=12)
