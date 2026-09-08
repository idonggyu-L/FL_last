#!/usr/bin/env python
"""Write synthetic dumps so the pipeline runs without the real experiment data.

Three attack archetypes are simulated, matching what the real signatures look
like: one that shrinks its update, one that inflates it, and one that keeps the
update size benign but distorts the representation geometry.

  python example/make_synthetic.py --out dumps
"""
import argparse, os
import numpy as np

# (name, update-norm multiplier, prototype-perturbation strength)
ARCHETYPES = {
    "LGA":      (0.11, 0.30),   # shrinks the update; also distorts geometry
    "BadNet":   (3.05, 0.10),   # inflates the update; mild geometric change
    "LPA":      (1.00, 0.35),   # benign-sized update, strong geometric change
    "Adaptive": (1.00, 0.40),   # same, tuned to sit inside the benign norm band
    "DAA":      (2.03, 0.20),
}


def make(path, K, D, n_clients, rounds, dw_mult, proto_kick, seed):
    rng = np.random.default_rng(seed)
    protos, dw, is_mal, rnd = [], [], [], []
    base = rng.normal(size=(K, D))                      # shared class structure
    for r in range(rounds):
        for i in range(n_clients):
            mal = (i == 0)
            p = base + rng.normal(scale=0.25, size=(K, D))
            if mal:                                     # tilt a few classes
                idx = rng.choice(K, max(1, K // 5), replace=False)
                p[idx] += rng.normal(scale=proto_kick * 4.0, size=(len(idx), D))
            protos.append(p.reshape(-1))
            n = rng.lognormal(mean=0.0, sigma=0.12)
            dw.append(n * (dw_mult if mal else 1.0))
            is_mal.append(int(mal)); rnd.append(r + 1)
    np.savez(path, protos=np.stack(protos).astype(np.float32),
             dw=np.asarray(dw, dtype=np.float32),
             is_mal=np.asarray(is_mal, dtype=np.int64),
             round=np.asarray(rnd, dtype=np.int64), K=np.int64(K))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dumps")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--clients", type=int, default=10)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cells = {"c10_RN": (10, 64), "c100_RN": (100, 64)}
    names = {"LGA": "LGA", "LPA": "LPA", "BadNet": "badnet",
             "DAA": "daa2", "Adaptive": "adaptive"}
    for cell, (K, D) in cells.items():
        for atk, (dwm, kick) in ARCHETYPES.items():
            for s in range(1, a.seeds + 1):
                p = os.path.join(a.out, f"{cell}_{names[atk]}_s{s}.npz")
                make(p, K, D, a.clients, a.rounds, dwm, kick, seed=hash((cell, atk, s)) % 2**31)
    print(f"  wrote {len(cells)*len(ARCHETYPES)*a.seeds} dumps to {a.out}/")


if __name__ == "__main__":
    main()
