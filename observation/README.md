# Two-Axis Observation

Plotting tools for inspecting federated-learning client updates on two axes:
how far a client's **class-prototype geometry** drifts from its peers, and how
far its **parameters** move. Backdoor attacks separate along different
directions of this plane, and some are invisible on one axis while obvious on
the other.

<!-- figs/plane_cifar10_resnet18.png -->

## What it measures

For every client `i` participating in a round:

| axis | quantity | what it captures |
|---|---|---|
| representation | `a_i = log ‖F_i‖` | distortion of the class-prototype relation matrix |
| parameter | `b_i = log ‖Δw_i‖` | size of the weight update |

```
R_i   = cosine relation matrix of the client's class prototypes  (K × K)
Z_i   = upper triangle of R_i, standardised column-wise across the round's clients
F_i   = Z_i − median_j Z_j
Δw_i  = w_i − w_global          (BatchNorm running statistics excluded)
```

The cosine relation matrix is invariant to rotations of the embedding space,
so models that never shared a coordinate system stay comparable. Only the
upper triangle is used — the diagonal is constant and the lower triangle is
redundant, leaving `K(K−1)/2` unique class pairs.

A client is scored by the Mahalanobis distance of `(a_i, b_i)` to its peers,
with the covariance estimated after dropping the two most distant points so a
malicious client cannot inflate its own direction:

```
u   = x − median(x)
S   = cov(u over the n−2 closest clients) + 1e-6 · tr(S) · I
M   = sqrt(uᵀ S⁻¹ u)
```

Every coordinate is **round-local**: points from different rounds or seeds are
comparable only because each is standardised against its own peers. The unit
is therefore "benign σ".

## Install

```bash
pip install -r requirements.txt      # numpy, matplotlib, scipy
```

## Try it without data

```bash
python example/make_synthetic.py --out dumps
cp config.example.json config.json
bash run_all.sh
```

This writes synthetic dumps for three attack archetypes — one that shrinks its
update, one that inflates it, and one that keeps the update benign-sized while
distorting the geometry — then renders every figure into `figs/`.

## Use with your own runs

### 1. Dump format

One `.npz` per run, holding every participating client of every logged round:

| array | shape | dtype | meaning |
|---|---|---|---|
| `protos` | `(N, K*D)` | float | flattened class prototypes, `K` classes × `D` features |
| `dw` | `(N,)` | float | `‖w_i − w_global‖₂`, positive |
| `is_mal` | `(N,)` | int | 1 for the malicious client, 0 otherwise |
| `round` | `(N,)` | int | round index each row belongs to |
| `K` | scalar | int | number of classes |

`N` is the total number of (round, client) pairs. Rows of the same round must
be contiguous in the sense that `round` identifies them; ordering within a
round does not matter. Rounds without a malicious client, or with non-positive
`dw`, are skipped.

A class prototype is the mean feature vector of a class over a fixed probe set
that is held out from every client. Record the values your defense actually
computed during the run rather than recomputing them afterwards — a different
probe changes the scale.

### 2. Point the config at them

```bash
cp config.example.json config.json
```

```json
{
  "nlast": 10,
  "keep_ratio": 0.7,
  "colors": {"LGA": "#D81B60"},
  "cells": {
    "cifar10_resnet18": {
      "title": "CIFAR-10 / ResNet-18",
      "legend": false,
      "limits": {"xt": [-10, 0, 10], "yt": [-10000, -100, -10, 0, 10, 100, 10000],
                 "xl": [-25, 25], "yl": [-20000, 10000]},
      "attacks": {"LGA": "dumps/c10_RN_LGA_s*.npz"}
    }
  }
}
```

| key | meaning |
|---|---|
| `nlast` | use only the last N rounds of each run (0 = all) |
| `keep_ratio` | fraction of clients an aggregator keeps; sets the removal cut-off |
| `limits` | axis ticks and ranges — the plane spans several decades, so these are per-cell |
| `legend` | draw the legend on this cell only |

Globs resolve relative to `root` (default: the config's directory) and are
expanded per attack, so one entry covers all seeds.

### 3. Render

```bash
bash run_all.sh                                    # everything
python plot_plane.py  --cell cifar10_resnet18      # 2D plane
python plot_axis1d.py --cell cifar10_resnet18 --axis par
python report_table.py                             # numbers behind the figures
```

## Output

| file | content |
|---|---|
| `plane_<cell>.{png,pdf}` | benign cloud with per-attack malicious points |
| `axis1d_{rep,par}_<cell>.{png,pdf}` | one axis alone: benign density, points, removal rate |
| `axis_separation.csv` | update-norm ratio, z-scores and removal rate per axis |

The table is the point of the exercise. An attack removed 100% of the time by
one axis and 10% by the other is what a single-axis detector would miss:

```
cell                attack      |dw|    par z   par |   |F|   rep z   rep |   2D
cifar10_vgg19       BadNet     3.05x   +710.7  100% | 1.48x   +1.8   56% | 100%
cifar10_vgg19       Adaptive   1.00x     +0.2   10% | 5.11x   +8.0  100% | 100%
```

## Files

| file | role |
|---|---|
| `axes.py` | axis coordinates, Mahalanobis score, removal decision — **the method lives here** |
| `style.py` | palette, fonts, axis decoration |
| `config.py` | config loading and glob resolution |
| `plot_plane.py` | 2D plane figure |
| `plot_axis1d.py` | single-axis figure |
| `report_table.py` | separation table → CSV |
| `run_all.sh` | render everything from one config |
| `example/make_synthetic.py` | synthetic dumps so the pipeline runs without real data |
