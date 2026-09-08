#!/usr/bin/env python
"""Per-axis separation table backing the figures.

  python report_table.py --config config.json --out figs
"""
import argparse, csv, os
import numpy as np
import axes as AX, config as CF


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--out", default="figs")
    a = ap.parse_args()
    cfg = CF.load(a.config)
    rows = []
    for cell in cfg["cells"]:
        for atk, _, label in CF.attack_list(cfg, cell):
            fps = CF.dumps_for(cfg, cell, atk)
            if not fps: continue
            c = AX.collect(fps, cfg["nlast"], cfg["keep_ratio"])
            if not c["n_rounds"]: continue
            rows.append(dict(cell=cell, attack=label, files=c["n_files"], rounds=c["n_rounds"],
                             dw_ratio=c["ratio"]["dw"], F_ratio=c["ratio"]["F"],
                             par_z=float(np.median(c["mal_b"])), rep_z=float(np.median(c["mal_a"])),
                             removed_par=100 * c["removal"]["par"],
                             removed_rep=100 * c["removal"]["rep"],
                             removed_2d=100 * c["removal"]["2d"]))
    if not rows:
        raise SystemExit("no data — check the globs in the config")
    hdr = (f"  {'cell':22s} {'attack':10s} {'n':>3s} | {'|dw|':>7s} {'par z':>9s} {'par':>5s} | "
           f"{'|F|':>6s} {'rep z':>8s} {'rep':>5s} | {'2D':>5s}")
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    last = None
    for r in rows:
        if last and r["cell"] != last: print("  " + "-" * (len(hdr) - 2))
        last = r["cell"]
        print(f"  {r['cell']:22s} {r['attack']:10s} {r['files']:>3d} | {r['dw_ratio']:6.2f}x "
              f"{r['par_z']:+9.1f} {r['removed_par']:4.0f}% | {r['F_ratio']:5.2f}x "
              f"{r['rep_z']:+8.1f} {r['removed_rep']:4.0f}% | {r['removed_2d']:4.0f}%")
    os.makedirs(a.out, exist_ok=True)
    p = os.path.join(a.out, "axis_separation.csv")
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\n  -> {p}")


if __name__ == "__main__":
    main()
