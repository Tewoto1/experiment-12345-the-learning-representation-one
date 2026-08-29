"""
Geometry of a moving representation cloud.

Nothing in here touches a model, a probe, or a classifier. Every function takes
plain arrays, so the same code reads Qwen activations, Gemma activations, or a
synthetic test cloud, and the analysis never needs a GPU.

Shapes
------
X       (N, D)      one cloud at one checkpoint: N words, D hidden dims
traj    (T, N, D)   the same N words at T checkpoints, rows aligned across T
background          the gauge words, same layout, disjoint from MEM and FILL

Everything is per (layer, read position). Loop over those outside.

Units
-----
gauge_apply returns coordinates divided by the background cloud's RMS radius,
so a displacement of 1.0 means "moved one background-cloud radius". Raw
activation norms differ by an order of magnitude across depth; gauged ones are
comparable layer to layer, which is the only reason a depth x time heatmap
means anything.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPS = 1e-12


def _f64(X):
    return np.asarray(X, dtype = np.float64)


def rms_radius(Xc):
    """
    Root-mean-square distance of already-centred rows from the origin.
    Args:
        Xc (ndarray): (N, D) centred cloud.
    Returns:
        float: the cloud's RMS radius.
    """
    return float(np.sqrt((_f64(Xc) ** 2).sum(axis = 1).mean())) + EPS


def procrustes(A, B):
    """
    Orthogonal R minimising ||A R - B||_F, by SVD of A^T B.
    Args:
        A (ndarray): (N, D) cloud to rotate.
        B (ndarray): (N, D) cloud to rotate onto.
    Returns:
        ndarray: (D, D) orthogonal matrix.
    """
    U, _, Vt = np.linalg.svd(_f64(A).T @ _f64(B), full_matrices = False)
    return U @ Vt


def top_pcs(X, k):
    """
    Top-k principal directions of a cloud, as orthonormal rows.
    Args:
        X (ndarray): (N, D) cloud; centred internally.
        k (int): how many directions.
    Returns:
        ndarray: (k, D) orthonormal rows, k capped at min(N, D).
    """
    Xc = _f64(X)
    Xc = Xc - Xc.mean(axis = 0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices = False)
    return Vt[: min(k, Vt.shape[0])]


# ---------------------------------------------------------------- gauge fixing

@dataclass
class Gauge:
    """The rigid map taking one checkpoint's activations into the reference frame."""
    mu: np.ndarray          # (D,) background mean at this checkpoint
    scale: float            # background RMS radius at this checkpoint
    R: np.ndarray | None    # (D, D) rotation onto the reference frame, None for identity


def gauge_fit(background_ref, background_cur, rotate = True, rotate_dim = None):
    """
    Fit the frame that removes global drift, using only the background words.

    Fine-tuning translates, rescales and rotates the whole residual stream. Raw
    displacement is dominated by that, so MEM motion is only meaningful as the
    residual after the background has been aligned. The background words never
    appear in the SFT data, so they carry drift and nothing else.

    Args:
        background_ref (ndarray): (M, D) background cloud at the reference checkpoint.
        background_cur (ndarray): (M, D) background cloud at this checkpoint, same rows.
        rotate (bool): fit the rotation too. Off means centre-and-scale only, which is
            weaker but cheap and never ill-conditioned.
        rotate_dim (int | None): fit the rotation inside the reference cloud's top-k
            principal subspace and leave the orthogonal complement untouched. A full
            2048x2048 SVD per (layer, checkpoint) is a few seconds; k=256 is near
            instant. None means full.
    Returns:
        Gauge: the map, to be handed to gauge_apply.
    """
    cur = _f64(background_cur)
    ref = _f64(background_ref)
    mu = cur.mean(axis = 0)
    cur_c = cur - mu
    scale = rms_radius(cur_c)
    ref_c = ref - ref.mean(axis = 0)

    if not rotate:
        return Gauge(mu = mu, scale = scale, R = None)

    A = cur_c / scale
    B = ref_c / rms_radius(ref_c)
    if rotate_dim is None:
        return Gauge(mu = mu, scale = scale, R = procrustes(A, B))

    Q = top_pcs(ref_c, rotate_dim)                  # (k, D)
    Rk = procrustes(A @ Q.T, B @ Q.T)               # (k, k)
    P = Q.T @ Q                                     # projector onto the subspace
    R = (np.eye(Q.shape[1]) - P) + Q.T @ Rk @ Q     # rotate inside, identity outside
    return Gauge(mu = mu, scale = scale, R = R)


def gauge_apply(X, g):
    """
    Put a cloud into the reference frame.
    Args:
        X (ndarray): (N, D) raw activations from the checkpoint g was fitted on.
        g (Gauge): the map from gauge_fit.
    Returns:
        ndarray: (N, D) gauged coordinates, in units of background radius.
    """
    Y = (_f64(X) - g.mu) / g.scale
    return Y if g.R is None else Y @ g.R


def gauge_residual(background_ref, background_cur, g):
    """
    How much of the background's motion the gauge failed to absorb.

    This is the honesty check on the whole method. A rigid map can only remove
    drift that is actually rigid; if the fine-tune is reshaping the background
    cloud itself then no frame exists and every downstream number is suspect.
    Log it per (layer, checkpoint) and plot it before anything else. Past roughly
    0.3, fall back to rotate = False and say so in the writeup.

    Args:
        background_ref (ndarray): (M, D) background at the reference checkpoint.
        background_cur (ndarray): (M, D) background at this checkpoint.
        g (Gauge): the map fitted for this checkpoint.
    Returns:
        float: relative residual, 0 being a perfect rigid match.
    """
    ref_c = _f64(background_ref)
    ref_c = ref_c - ref_c.mean(axis = 0)
    ref_n = ref_c / rms_radius(ref_c)
    cur_n = gauge_apply(background_cur, g)
    return float(np.linalg.norm(cur_n - ref_n) / (np.linalg.norm(ref_n) + EPS))


def gauge_trajectory(raw_background, raw_cloud, rotate = True, rotate_dim = None, ref = 0):
    """
    Gauge a whole trajectory against one reference checkpoint.
    Args:
        raw_background (ndarray): (T, M, D) background cloud over checkpoints.
        raw_cloud (ndarray): (T, N, D) the cloud of interest over the same checkpoints.
        rotate (bool): passed to gauge_fit.
        rotate_dim (int | None): passed to gauge_fit.
        ref (int): index of the reference checkpoint, normally 0.
    Returns:
        traj (ndarray): (T, N, D) gauged trajectory.
        residuals (ndarray): (T,) gauge residual per checkpoint.
    """
    ref_bg = raw_background[ref]
    traj, residuals = [], []
    for t in range(raw_background.shape[0]):
        g = gauge_fit(ref_bg, raw_background[t], rotate = rotate, rotate_dim = rotate_dim)
        traj.append(gauge_apply(raw_cloud[t], g))
        residuals.append(gauge_residual(ref_bg, raw_background[t], g))
    return np.stack(traj), np.array(residuals)


def gauge_all(raw_background, clouds, rotate = True, rotate_dim = None, ref = 0):
    """
    Gauge several clouds against one background, fitting each map only once.
    Args:
        raw_background (ndarray): (T, M, D) background cloud over checkpoints.
        clouds (dict[str, ndarray]): name -> (T, N, D) raw trajectory.
        rotate (bool): passed to gauge_fit.
        rotate_dim (int | None): passed to gauge_fit.
        ref (int): index of the reference checkpoint.
    Returns:
        gauged (dict[str, ndarray]): name -> (T, N, D) gauged trajectory, plus
            "background" for the gauged background itself.
        residuals (ndarray): (T,) gauge residual per checkpoint.
    """
    ref_bg = raw_background[ref]
    out = {name: [] for name in clouds}
    out["background"] = []
    residuals = []
    for t in range(raw_background.shape[0]):
        g = gauge_fit(ref_bg, raw_background[t], rotate = rotate, rotate_dim = rotate_dim)
        for name, traj in clouds.items():
            out[name].append(gauge_apply(traj[t], g))
        out["background"].append(gauge_apply(raw_background[t], g))
        residuals.append(gauge_residual(ref_bg, raw_background[t], g))
    return {name: np.stack(v) for name, v in out.items()}, np.array(residuals)


# ------------------------------------------------------------------ statistics

def spectrum(M):
    """
    Eigenvalues of the Gram matrix of a set of vectors, descending.
    Args:
        M (ndarray): (N, D) rows are the vectors.
    Returns:
        ndarray: (min(N, D),) squared singular values.
    """
    return np.linalg.svd(_f64(M), compute_uv = False) ** 2


def top_fraction(lam):
    """Share of the spectrum in its leading eigenvalue; 1/N means no shared direction."""
    return float(lam[0] / (lam.sum() + EPS))


def participation_ratio(lam):
    """Effective number of directions carrying the variance: (sum L)^2 / sum L^2."""
    return float((lam.sum() ** 2) / ((lam ** 2).sum() + EPS))


def velocity_coherence(traj, unit = True):
    """
    S1. Are the words all moving the same way?

    The bucket-versus-private-paths question asked of the dynamics rather than of
    a classifier. One shared thing being built pushes every word along a common
    direction and the leading eigenvalue swallows the spectrum. N private lookups
    leave the velocities mutually unaligned and top_fraction sits near 1/N.
    Unit-normalising first makes the statistic read direction agreement rather
    than "a few words moved a lot", and makes it invariant to checkpoint spacing.

    Args:
        traj (ndarray): (T, N, D) gauged trajectory.
        unit (bool): normalise each word's velocity to unit length first.
    Returns:
        dict: "top" (T-1,) leading-eigenvalue share, "pr" (T-1,) participation ratio,
            each aligned to the interval ending at checkpoint t+1.
    """
    V = np.diff(_f64(traj), axis = 0)
    top, pr = [], []
    for Vt in V:
        if unit:
            Vt = Vt / (np.linalg.norm(Vt, axis = 1, keepdims = True) + EPS)
        lam = spectrum(Vt)
        top.append(top_fraction(lam))
        pr.append(participation_ratio(lam))
    return {"top": np.array(top), "pr": np.array(pr)}


def novel_fraction(traj, background_ref_gauged, k = 64):
    """
    S2. Is the motion going somewhere the pretrained model was not already using?

    Project total displacement onto the top-k principal subspace of the frozen
    background cloud and report what falls outside. Rising over training is the
    geometric reading of "the model allocated new space" rather than "the model
    reused a distinction it already had".

    Args:
        traj (ndarray): (T, N, D) gauged trajectory.
        background_ref_gauged (ndarray): (M, D) gauged background at the reference checkpoint.
        k (int): size of the reference subspace.
    Returns:
        ndarray: (T,) fraction outside the subspace; entry 0 is nan, displacement being zero.
    """
    Q = top_pcs(background_ref_gauged, k)
    D = _f64(traj) - _f64(traj)[0]
    inside = np.einsum("tnd,kd->tnk", D, Q)
    total = (D ** 2).sum(axis = (1, 2))
    out = 1.0 - (inside ** 2).sum(axis = (1, 2)) / (total + EPS)
    out[0] = np.nan
    return out


def tortuosity(traj):
    """
    S3a. Path length over net displacement, per word.

    1.0 is a straight shot from where the word started to where it ended. Large
    means the representation wandered before settling. Only comparable between
    runs sharing a checkpoint schedule: coarser sampling shortcuts the path.

    Args:
        traj (ndarray): (T, N, D) gauged trajectory.
    Returns:
        ndarray: (N,) ratio per word.
    """
    X = _f64(traj)
    path = np.linalg.norm(np.diff(X, axis = 0), axis = 2).sum(axis = 0)
    net = np.linalg.norm(X[-1] - X[0], axis = 1)
    return path / (net + EPS)


def incremental_cosine(traj):
    """
    S3b. Angle between one step of motion and the next, averaged over words.

    Near 0 early and climbing toward 1 is wander-then-commit. Flat and high from
    the start means the direction was fixed by the first update.

    Args:
        traj (ndarray): (T, N, D) gauged trajectory.
    Returns:
        ndarray: (T-2,) mean cosine.
    """
    V = np.diff(_f64(traj), axis = 0)
    a, b = V[:-1], V[1:]
    num = (a * b).sum(axis = -1)
    den = np.linalg.norm(a, axis = -1) * np.linalg.norm(b, axis = -1)
    return (num / (den + EPS)).mean(axis = 1)


def direction_to_final(traj):
    """
    S3c. Cosine between each word's displacement so far and its final displacement.

    The convergence curve. Reaching 1 early means the endpoint was decided early
    and the rest of training only scaled it.

    Args:
        traj (ndarray): (T, N, D) gauged trajectory.
    Returns:
        ndarray: (T,) mean cosine; entry 0 is nan.
    """
    X = _f64(traj)
    D = X - X[0]
    final = D[-1]
    num = (D * final).sum(axis = -1)
    den = np.linalg.norm(D, axis = -1) * np.linalg.norm(final, axis = -1)
    out = (num / (den + EPS)).mean(axis = 1)
    out[0] = np.nan
    return out


def cloud_separation(mem, fill):
    """
    S4a. Centroid gap in units of the pooled within-cloud radius.

    Descriptive geometry, not a classifier: nothing is fitted, so it cannot be
    inflated by having more dimensions than words the way probe accuracy can.

    Args:
        mem (ndarray): (N, D) gauged MEM cloud at one checkpoint.
        fill (ndarray): (N, D) gauged FILL cloud at the same checkpoint.
    Returns:
        float: separation.
    """
    m, f = _f64(mem), _f64(fill)
    gap = np.linalg.norm(m.mean(axis = 0) - f.mean(axis = 0))
    within = np.sqrt(0.5 * (rms_radius(m - m.mean(0)) ** 2 + rms_radius(f - f.mean(0)) ** 2))
    return float(gap / (within + EPS))


def cloud_shape(X):
    """
    S4b. How many directions the cloud actually spreads over.
    Args:
        X (ndarray): (N, D) gauged cloud at one checkpoint.
    Returns:
        dict: "pr" participation ratio, "top" leading-eigenvalue share.
    """
    Xc = _f64(X)
    lam = spectrum(Xc - Xc.mean(axis = 0))
    return {"pr": participation_ratio(lam), "top": top_fraction(lam)}


def displacement(traj):
    """
    S5. Mean distance travelled from the reference checkpoint.
    Args:
        traj (ndarray): (T, N, D) gauged trajectory.
    Returns:
        ndarray: (T,) mean displacement, in background radii.
    """
    X = _f64(traj)
    return np.linalg.norm(X - X[0], axis = 2).mean(axis = 1)


def summarise(mem_traj, fill_traj, background_ref_gauged, k = 64):
    """
    Every statistic for one (layer, read position), MEM against its FILL control.
    Args:
        mem_traj (ndarray): (T, N, D) gauged MEM trajectory.
        fill_traj (ndarray): (T, N, D) gauged FILL trajectory.
        background_ref_gauged (ndarray): (M, D) gauged background at the reference checkpoint.
        k (int): reference subspace size for novel_fraction.
    Returns:
        dict: statistic name -> array or float. MEM keys are bare, the FILL control
            is the same key suffixed "_fill". Every FILL number is the null: a MEM
            curve only means something where it parts from its own control.
    """
    out = {}
    for tag, traj in (("", mem_traj), ("_fill", fill_traj)):
        coh = velocity_coherence(traj)
        out["coherence" + tag] = coh["top"]
        out["coherence_pr" + tag] = coh["pr"]
        out["novel" + tag] = novel_fraction(traj, background_ref_gauged, k = k)
        out["tortuosity" + tag] = float(np.median(tortuosity(traj)))
        out["inc_cos" + tag] = incremental_cosine(traj)
        out["to_final" + tag] = direction_to_final(traj)
        out["displacement" + tag] = displacement(traj)
        out["pr" + tag] = np.array([cloud_shape(X)["pr"] for X in traj])
    out["separation"] = np.array([cloud_separation(m, f) for m, f in zip(mem_traj, fill_traj)])
    return out


# ------------------------------------------------------------------ scheduling

def checkpoint_schedule(total_steps, n_points = 19, dense_until = 4):
    """
    Log-spaced harvest steps.

    Almost all of the interesting motion happens in the first few dozen optimizer
    steps. Linear spacing spends one frame on that and the rest on a plateau, and
    the spacing cannot be fixed after the run has finished. Every step up to
    dense_until is kept outright, the remainder is geometric.

    Args:
        total_steps (int): last step of training.
        n_points (int): roughly how many harvest points to return.
        dense_until (int): keep every step up to and including this one.
    Returns:
        list[int]: strictly increasing, always starting at 0 and ending at total_steps.
    """
    head = list(range(0, min(dense_until, total_steps) + 1))
    remaining = max(n_points - len(head), 2)
    tail = np.unique(np.geomspace(max(head[-1], 1) + 1, total_steps, remaining).round().astype(int))
    return sorted(set(head) | {int(s) for s in tail} | {total_steps})
