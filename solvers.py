"""Routines for solving a linear system of equations."""
import numpy as np


def gaussian_eliminate(aa, bb):
    """Solves a linear system of equations (Ax = b) by Gauss-elimination

    Args:
        aa: Matrix with the coefficients. Shape: (n, n).
        bb: Right hand side of the equation. Shape: (n,)

    Returns:
        Vector xx with the solution of the linear equation or None
        if the equations are linearly dependent.
    """
    nn = aa.shape[0]
    for ii in range(nn - 1):
        for jj in range(ii + 1, nn):
            coeff = -aa[jj, ii] / aa[ii, ii]
            aa[jj, ii:] += coeff * aa[ii, ii:]
            bb[jj] += coeff * bb[ii]

    xx = np.empty((nn,), dtype=float)
    for ii in range(nn - 1, -1, -1):
        xx[ii] = (bb[ii] - np.dot(aa[ii, ii + 1:], xx[ii + 1:])) / aa[ii, ii]
    return xx
