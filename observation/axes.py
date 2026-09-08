"""Two-axis coordinates for federated-learning client updates.

Each client i in a round is placed at

    a_i = log ||F_i||       representation axis
    b_i = log ||dw_i||      parameter axis

where

    R_i        = cosine relation matrix of class prototypes (K x K)
    Z_i        = column-wise standardisation of the upper triangle of R_i
                 across the clients of that round
    F_i        = Z_i - median_j Z_j
    dw_i       = w_i - w_global   (BatchNorm running statistics excluded)

A client is scored by the Mahalanobis distance of (a_i, b_i) to the peer
distribution of the same round, with the covariance estimated after trimming
the two most distant points so a malicious client cannot inflate its own
direction.  The lowest-scoring `keep` fraction is aggregated.

Everything here is a *round-local* statistic: points from different rounds or
seeds are only comparable because each is standardised against its own peers.
"""
import numpy as np

EPS = 1e-12


def trim2_std(u):
    """Robust scale: standard deviation after dropping the two largest |u|."""
    keep = np.argsort(np.abs(u))[: max(3, len(u) - 2)]
    return u[keep].std() + EPS


def robust_z(values):
    """Median-centred, trim-2-scaled score. Units are 'benign sigma'."""
    u = np.asarray(values, dtype=np.float64)
    u = u - np.median(u)
    return u / trim2_std(u)


def rel_deviation(protos, K):
    """(n, K*D) prototypes -> per-client relation-matrix deviation ||F_i||, shape (n,).

    The cosine relation matrix is invariant to rotations of the embedding
    space, so models that never shared a coordinate system remain comparable.
    Only the upper triangle is used: the diagonal is constant and the lower
    triangle is redundant, leaving K(K-1)/2 unique class pairs.
    """
    n = protos.shape[0]
    D = protos.shape[1] // K
    iu0, iu1 = np.triu_indices(K, k=1)
    P = protos.reshape(n, K, D).astype(np.float64)
    P = P / (np.linalg.norm(P, axis=2, keepdims=True) + EPS)
    C = (P @ np.transpose(P, (0, 2, 1)))[:, iu0, iu1]
    Z = (C - C.mean(0)) / (C.std(0) + EPS)
    return np.linalg.norm(Z - np.median(Z, axis=0), axis=1)


def maha2d(a, b, keep_ratio=0.7):
    """Mahalanobis score on the (a, b) plane and the removal cut-off.

    Returns (scores, cut) where the `cut` highest-scoring clients are removed.
    """
    n = len(a)
    X = np.stack([a, b], 1)
    U = X - np.median(X, axis=0)
    d0 = np.linalg.norm(U / (U.std(0) + EPS), axis=1)
    kp = np.argsort(d0)[: max(3, n - 2)]                       # trim 2
    S = np.cov(U[kp].T)
    S = S + np.eye(2) * 1e-6 * max(float(np.trace(S)), EPS)    # ridge
    M = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", U, np.linalg.inv(S), U), 0.0))
    return M, n - int(round(keep_ratio * n))


def rounds_from_dump(npz_path, nlast=10, keep_ratio=0.7):
    """Read one dump and return a list of per-round dicts.

    Expected arrays (see README for the exact contract):
      protos (N, K*D) | dw (N,) | is_mal (N,) | round (N,) | K (scalar)
    Rounds without a malicious participant, or with non-finite update norms,
    are skipped.
    """
    d = np.load(npz_path)
    protos, rd, is_mal, dw = d["protos"], d["round"], d["is_mal"], d["dw"]
    K = int(d["K"])
    all_rounds = sorted(set(rd.tolist()))
    out = []
    for r in (all_rounds[-nlast:] if nlast else all_rounds):
        m = rd == r
        y = is_mal[m].astype(int)
        w = dw[m].astype(np.float64)
        if y.sum() == 0 or not np.isfinite(w).all() or (w <= 0).any():
            continue
        Fn = rel_deviation(protos[m], K)
        a, b = np.log(Fn + EPS), np.log(w)
        M, cut = maha2d(a, b, keep_ratio)
        za, zb = robust_z(a), robust_z(b)
        i = int(np.where(y == 1)[0][0])
        out.append(dict(
            round=int(r), y=y, za=za, zb=zb, maha=M,
            F_ratio=float(Fn[i] / np.median(Fn[y == 0])),
            dw_ratio=float(w[i] / np.median(w[y == 0])),
            removed_2d=int((M > M[i]).sum()) < cut,
            removed_rep=int((np.abs(za) > abs(za[i])).sum()) < cut,
            removed_par=int((np.abs(zb) > abs(zb[i])).sum()) < cut,
        ))
    return out


def collect(npz_paths, nlast=10, keep_ratio=0.7):
    """Pool several dumps (e.g. seeds) into benign/malicious coordinates."""
    ben_a, ben_b, mal_a, mal_b = [], [], [], []
    rem = {"2d": [], "rep": [], "par": []}
    ratio = {"F": [], "dw": []}
    for p in npz_paths:
        for R in rounds_from_dump(p, nlast, keep_ratio):
            b_ = R["y"] == 0
            i = int(np.where(R["y"] == 1)[0][0])
            ben_a += list(R["za"][b_]); ben_b += list(R["zb"][b_])
            mal_a.append(R["za"][i]);   mal_b.append(R["zb"][i])
            rem["2d"].append(R["removed_2d"])
            rem["rep"].append(R["removed_rep"])
            rem["par"].append(R["removed_par"])
            ratio["F"].append(R["F_ratio"]); ratio["dw"].append(R["dw_ratio"])
    f = lambda v: np.asarray(v, dtype=np.float64)
    nan = float("nan")
    return dict(
        ben_a=f(ben_a), ben_b=f(ben_b), mal_a=f(mal_a), mal_b=f(mal_b),
        removal={k: (float(np.mean(v)) if v else nan) for k, v in rem.items()},
        ratio={k: (float(np.mean(v)) if v else nan) for k, v in ratio.items()},
        n_rounds=len(mal_a), n_files=len(npz_paths))
