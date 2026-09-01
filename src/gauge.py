"""
The rigid frame that removes global drift, fitted on the background words alone.

Fine-tuning translates, rescales and rotates the whole residual stream. Raw displacement
is dominated by that, so MEM motion is only meaningful as what is left after the
background has been aligned. The background words never appear in the SFT data, so they
carry drift and nothing else.

Everything downstream works in the units this module defines: one unit is one background
RMS radius at that checkpoint.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.geometry import EPS, _f64, procrustes, rms_radius, top_pcs


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
