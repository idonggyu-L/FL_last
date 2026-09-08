#!/usr/bin/env python
"""2D plane: benign cloud with per-attack malicious points.

  python plot_plane.py --config config.json --cell cifar10_resnet18 --out figs
"""
import argparse, os
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import axes as AX, style as ST, config as CF

FALLBACK = dict(xt=[-10, 0, 10], yt=[-1e4, -1e2, -10, 0, 10, 1e2, 1e4],
                xl=[-25, 25], yl=[-2e4, 1e4])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--cell", required=True)
    ap.add_argument("--out", default="figs")
    ap.add_argument("--legend", action="store_true", help="override config's legend flag")
    a = ap.parse_args()
    cfg = CF.load(a.config)
    cell = cfg["cells"][a.cell]

    ben_a, ben_b, mal = [], [], {}
    for atk, col, label in CF.attack_list(cfg, a.cell):
        fps = CF.dumps_for(cfg, a.cell, atk)
        if not fps:
            print(f"  [skip] {label}: no dumps matched"); continue
        c = AX.collect(fps, cfg["nlast"], cfg["keep_ratio"])
        if not c["n_rounds"]:
            print(f"  [skip] {label}: no usable rounds"); continue
        ben_a += list(c["ben_a"]); ben_b += list(c["ben_b"])
        mal[atk] = (c["mal_a"], c["mal_b"], col, label, c["n_files"])
    if not mal:
        raise SystemExit(f"no data for cell '{a.cell}' — check the globs in {a.config}")
    ben_a, ben_b = np.asarray(ben_a), np.asarray(ben_b)

    L = cell["limits"] or FALLBACK
    fig, ax = plt.subplots(figsize=(7.8, 6.2))
    ax.set_xscale("symlog", linthresh=ST.SYMLOG_LINTHRESH, linscale=1.0)
    ax.set_yscale("symlog", linthresh=ST.SYMLOG_LINTHRESH, linscale=1.0)
    ax.scatter(ben_a, ben_b, s=17, c=ST.BENIGN_COLOR, alpha=.22,
               edgecolors="none", label="Benign", zorder=2)
    for atk, (X, Y, col, label, _) in mal.items():
        ax.scatter(X, Y, s=42, c=col, alpha=.75, edgecolors="white",
                   linewidths=.7, zorder=3, label=label)
    ax.set_xticks(L["xt"]); ax.set_xticklabels([ST.pow10_label(v) for v in L["xt"]], fontsize=ST.FS_TICK)
    ax.set_yticks(L["yt"]); ax.set_yticklabels([ST.pow10_label(v) for v in L["yt"]], fontsize=ST.FS_TICK)
    ax.set_xticks([], minor=True); ax.set_yticks([], minor=True)
    ax.set_xlim(*L["xl"]); ax.set_ylim(*L["yl"])
    ST.decorate_axes(ax, "Representation deviation", "Parameter deviation", cell["title"])
    if a.legend or cell.get("legend"):
        ax.legend(loc="center left", bbox_to_anchor=(1.02, .5), fontsize=ST.FS_LEGEND,
                  framealpha=1, handletextpad=.4, borderpad=.7, labelspacing=.6, markerscale=1.6)
    os.makedirs(a.out, exist_ok=True)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(a.out, f"plane_{a.cell}.{ext}"), dpi=230, bbox_inches="tight")
    print(f"  {a.cell}: benign n={len(ben_a)}  " +
          "  ".join(f"{v[3]} n={len(v[0])}({v[4]} files)" for v in mal.values()))


if __name__ == "__main__":
    main()
