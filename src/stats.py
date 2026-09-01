"""
What a gauged trajectory is asked. One statistic per question, MEM against its FILL null.

Every function takes a (T, N, D) trajectory already in the reference frame and returns
numbers, so none of this needs a model, a tokenizer or a GPU -- analysis runs on a laptop
against the saved activations. A MEM curve means nothing except where it parts from its
own control, which is why summarise computes both and suffixes the control "_fill".
"""
from __future__ import annotations

import numpy as np

from src.geometry import (EPS, _f64, participation_ratio, rms_radius, spectrum,
                          top_fraction, top_pcs)


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

def velocity_coherence_scales(traj, windows = None, unit = True):
    """
    S1b. Coherence of displacement measured over a range of time scales.

    velocity_coherence asks whether the words move together between ADJACENT
    checkpoints, which is the harshest version of the question and the one most
    easily destroyed by noise. Writing a word's position as

        h_i(t) = c_i + a_i(t) u + r_i(t)

    with u a shared feature and r_i private movement, the one-step difference is
    (da_i)u + dr_i. If the private part is fast and the shared part is slow, dr_i
    dominates every single step while the shared displacement accumulates: over a
    window of W steps the a_i term grows with the drift while the r_i term, being
    unbiased, grows only as sqrt(W). Coherence should therefore RISE with window
    length if u is real, and stay flat if it is not.

    That rise, or its absence, is the measurement. A single number at one time
    scale cannot tell the two cases apart.

    The largest window, T - 1, is the endpoint statistic: total displacement from
    the reference checkpoint, one sample, no averaging.

    Args:
        traj (ndarray): (T, N, D) gauged trajectory.
        windows (list[int] | None): window lengths in checkpoint steps. None uses
            a log-spaced ladder from 1 to T - 1.
        unit (bool): normalise each displacement to unit length first, so this reads
            direction agreement rather than "a few words moved a lot".
    Returns:
        dict: "windows" the lengths used, "coherence" mean leading-eigenvalue share at
            each, "pr" the matching participation ratios, "n" how many start points
            were averaged at each window.
    """
    X = _f64(traj)
    T = X.shape[0]
    if windows is None:
        ladder = np.unique(np.geomspace(1, max(T - 1, 1), min(8, max(T - 1, 1))).round().astype(int))
        windows = [int(w) for w in ladder if 1 <= w <= T - 1]

    out = {"windows": [], "coherence": [], "pr": [], "n": []}
    for w in windows:
        tops, prs = [], []
        for t in range(0, T - w):
            V = X[t + w] - X[t]
            if unit:
                V = V / (np.linalg.norm(V, axis = 1, keepdims = True) + EPS)
            lam = spectrum(V)
            tops.append(top_fraction(lam))
            prs.append(participation_ratio(lam))
        if not tops:
            continue
        out["windows"].append(int(w))
        out["coherence"].append(float(np.mean(tops)))
        out["pr"].append(float(np.mean(prs)))
        out["n"].append(len(tops))
    return out

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
        scales = velocity_coherence_scales(traj)
        out["scale_windows" + tag] = np.array(scales["windows"])
        out["scale_coherence" + tag] = np.array(scales["coherence"])
        out["scale_pr" + tag] = np.array(scales["pr"])
        out["novel" + tag] = novel_fraction(traj, background_ref_gauged, k = k)
        out["tortuosity" + tag] = float(np.median(tortuosity(traj)))
        out["inc_cos" + tag] = incremental_cosine(traj)
        out["to_final" + tag] = direction_to_final(traj)
        out["displacement" + tag] = displacement(traj)
        out["pr" + tag] = np.array([cloud_shape(X)["pr"] for X in traj])
    out["separation"] = np.array([cloud_separation(m, f) for m, f in zip(mem_traj, fill_traj)])
    return out
