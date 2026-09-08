#!/usr/bin/env python
"""Single-axis view: benign density plus per-attack points and removal rate.

  python plot_axis1d.py --config config.json --cell cifar10_resnet18 --axis par --out figs
"""
import argparse, os
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms as mtr
from scipy.stats import gaussian_kde
import axes as AX, style as ST, config as CF

LABEL = {
    "rep": (r"Representation deviation    $z(\log\|F_i\|)$", "representation axis"),
    "par": (r"Parameter movement    $z(\log\|\Delta w_i\|)$", "parameter axis"),
}
LINTHRESH = 5.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--cell", required=True)
    ap.add_argument("--axis", required=True, choices=["rep", "par"])
    ap.add_argument("--out", default="figs")
    a = ap.parse_args()
    cfg = CF.load(a.config)
    cell = cfg["cells"][a.cell]
    k_mal, k_ben = ("mal_a", "ben_a") if a.axis == "rep" else ("mal_b", "ben_b")

    ben, series = [], []
    for atk, col, label in CF.attack_list(cfg, a.cell):
        fps = CF.dumps_for(cfg, a.cell, atk)
        if not fps: continue
        c = AX.collect(fps, cfg["nlast"], cfg["keep_ratio"])
        if not c["n_rounds"]: continue
        ben += list(c[k_ben])
        series.append((label, col, c[k_mal], c["removal"][a.axis]))
    if not series:
        raise SystemExit(f"no data for cell '{a.cell}'")
    ben = np.asarray(ben)
    q95 = np.percentile(np.abs(ben), 95)

    allv = np.concatenate([ben] + [s[2] for s in series])
    E = max(1, int(np.ceil(np.log10(max(np.abs(allv).max(), 20)))))
    ticks = sorted({-10 ** k for k in range(1, E + 1)} | {0} | {10 ** k for k in range(1, E + 1)})
    XL = (-1.9 * 10 ** E, 1.9 * 10 ** E)

    fig, ax = plt.subplots(figsize=(8.0, 1.05 * len(series) + 1.9))
    fig.subplots_adjust(left=.125, right=.775, top=.875, bottom=.145)
    ax.set_xscale("symlog", linthresh=LINTHRESH, linscale=1.15)
    DH, ROW = 1.20, -0.52
    BOT = ROW * (len(series) + 1.2)
    grid = np.concatenate([-np.logspace(np.log10(LINTHRESH), E + .28, 240)[::-1],
                           np.linspace(-LINTHRESH, LINTHRESH, 240),
                           np.logspace(np.log10(LINTHRESH), E + .28, 240)])
    dv = gaussian_kde(ben)(grid); dv = dv / dv.max() * DH
    ax.fill_between(grid, 0, dv, color=ST.BENIGN_COLOR, alpha=.18, lw=0, zorder=1)
    ax.plot(grid, dv, color=ST.AXIS_COLOR, lw=1.4, zorder=2)
    for s in (-1, 1):
        ax.plot([s * q95] * 2, [BOT, DH], color="#78909C", lw=1, ls=(0, (4, 3)), alpha=.65, zorder=0)
    ax.text(q95, DH * 1.06, "benign 95%", fontsize=8.5, color="#546E7A", ha="center")

    blend = mtr.blended_transform_factory(ax.transAxes, ax.transData)
    ax.text(-.018, DH * .50, f"Benign\nn={len(ben)}", transform=blend, fontsize=9.5,
            color=ST.AXIS_COLOR, ha="right", va="center", fontweight="bold")
    rng = np.random.default_rng(0)
    for j, (label, col, X, rem) in enumerate(series):
        y0 = ROW * (j + 1)
        ax.plot(XL, [y0, y0], color="#ECEFF1", lw=1.0, zorder=0)
        ax.scatter(X, y0 + rng.uniform(-.115, .115, len(X)), s=28, c=col, alpha=.8,
                   edgecolors="white", linewidths=.45, zorder=4)
        ax.text(-.018, y0, label, transform=blend, fontsize=11, color=col,
                ha="right", va="center", fontweight="bold")
        ax.text(1.015, y0 + .11, f"median {np.median(X):+.1f}" + r"$\sigma$", transform=blend,
                fontsize=8.8, color="#546E7A", ha="left", va="center")
        rr = 100 * rem
        ax.text(1.015, y0 - .14, f"{rr:.0f}% removed", transform=blend, fontsize=8.8,
                color=("#C62828" if rr < 90 else "#2E7D32"), ha="left", va="center", fontweight="bold")
    ax.set_xlim(*XL); ax.set_ylim(BOT - .16, DH * 1.26)
    ax.set_xticks(ticks); ax.set_xticklabels([ST.pow10_label(v) for v in ticks], fontsize=10)
    ax.set_xticks([], minor=True); ax.set_yticks([])
    for sp in ("top", "right", "left", "bottom"): ax.spines[sp].set_visible(False)
    ax.annotate("", xy=(1.0, BOT - .10), xycoords=blend, xytext=(0, BOT - .10),
                textcoords=blend, arrowprops=dict(arrowstyle="-|>", color=ST.AXIS_COLOR, lw=1.3))
    lab, sub = LABEL[a.axis]
    ax.set_xlabel(lab + r"    [benign $\sigma$]", fontsize=11, labelpad=7)
    ax.set_title(f"{cell['title']}   ·   {sub}", fontsize=10.5, pad=9)
    os.makedirs(a.out, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(a.out, f"axis1d_{a.axis}_{a.cell}.{ext}"), dpi=230, bbox_inches="tight")
    print(f"  {a.axis} {a.cell}: " +
          "  ".join(f"{l} {np.median(X):+6.1f}/{100*r:3.0f}%" for l, _, X, r in series))


if __name__ == "__main__":
    main()
