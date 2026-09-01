"""
Linear-algebra primitives shared by the gauge and the statistics.

Nothing here knows what a checkpoint or a word is. Every function takes an array and
returns a number or an array, so it can be tested against a case whose answer is known
by construction -- which is what test/test_geometry.py does.
"""
from __future__ import annotations

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
