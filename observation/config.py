"""Configuration loading.

A config is JSON describing, per cell (dataset x architecture), where the dumps
for each attack live.  Globs are resolved relative to `root` (default: the
config file's directory).  See config.example.json.
"""
import glob as _glob
import json, os
import style


def load(path):
    with open(path) as f:
        cfg = json.load(f)
    cfg.setdefault("root", os.path.dirname(os.path.abspath(path)) or ".")
    cfg.setdefault("keep_ratio", 0.7)
    cfg.setdefault("nlast", 10)
    for name, cell in cfg["cells"].items():
        cell.setdefault("title", name)
        cell.setdefault("limits", None)
    return cfg


def dumps_for(cfg, cell, attack):
    """Resolve an attack's glob into existing file paths, sorted."""
    pat = cfg["cells"][cell]["attacks"][attack]
    return sorted(_glob.glob(os.path.join(cfg["root"], pat)))


def attack_list(cfg, cell):
    """[(attack, color, label)] in config order, with palette fallback."""
    colors = cfg.get("colors", {})
    out = []
    for k, atk in enumerate(cfg["cells"][cell]["attacks"]):
        out.append((atk, colors.get(atk, style.DEFAULT_PALETTE[k % len(style.DEFAULT_PALETTE)]), atk))
    return out
