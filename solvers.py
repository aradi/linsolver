"""Routines for solving a linear system of equations."""
import numpy as np

# Tolerance value when checking for linear dependency
_TOLERANCE = 1e-12


def gaussian_eliminate(aa, bb):
    """Solves a linear system of equations (Ax = b) by Gauss-elimination

    Args:
        aa: Matrix with the coefficients. Shape: (n, n).
        bb: Right hand side of the equation. Shape: (n,)

    Returns:
        Vector xx with the solution of the linear equation.

    Raises:
        ValueError: if the system of equation is close to linear dependency.
    """
    nn = aa.shape[0]
    for ii in range(nn):
        _make_partial_pivot(aa[ii:, :], bb[ii:], ii)
        if abs(aa[ii, ii]) < _TOLERANCE:
            msg = "System of equations is linearly dependent"
            raise ValueError(msg)
        for jj in range(ii + 1, nn):
            coeff = -aa[jj, ii] / aa[ii, ii]
            aa[jj, ii:] += coeff * aa[ii, ii:]
            bb[jj] += coeff * bb[ii]

    xx = np.empty((nn,), dtype=float)
    for ii in range(nn - 1, -1, -1):
        xx[ii] = (bb[ii] - np.dot(aa[ii, ii + 1:], xx[ii + 1:])) / aa[ii, ii]
    return xx


def _make_partial_pivot(aa, bb, icol):
    """Makes pivot changes in matrix A and vector b based on a given column of
    the matrix A.

    It looks for the element with the biggest absolute number in the given
    columnt of A and exchanges the rows of A, so that the biggest number
    is in the zeroth row.

    Args:
        aa: Matirx A on entry, pivoted matrix on exit. Shape (m, n)
        bb: Vector b on entry, pivoted vector on exit. Shape: (m,)
        icol: Column number to investigate for the pivoting (0 <= icol < m)
    """
    imax = np.argmax(abs(aa[:, icol]))
    if imax:
        tmp = np.array(aa[0])
        aa[0] = aa[imax]
        aa[imax] = tmp
        tmp = np.array(bb[0])
        bb[0] = bb[imax]
        bb[imax] = tmp
