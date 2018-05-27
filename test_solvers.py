#!/usr/bin/env python3
"""Contains routines to test the solvers module"""

import numpy as np
import solvers


TOLERANCE = 1e-10


def test_elimination():
    "Tests simple elimination"
    aa = np.array([[2.0, 4.0, 4.0], [5.0, 4.0, 2.0], [1.0, 2.0, -1.0]])
    bb = np.array([1.0, 4.0, 2.0])
    xx_expected = np.array([0.666666666666667, 0.416666666666667, -0.5])
    xx_gauss = solvers.gaussian_eliminate(aa, bb)
    assert np.all(np.abs(xx_gauss - xx_expected) < TOLERANCE)


def test_elimination_needing_pivot():
    "Tests elimination where pivot is needed"
    aa = np.array([[2.0, 4.0, 4.0], [1.0, 2.0, -1.0], [5.0, 4.0, 2.0]])
    bb = np.array([1.0, 2.0, 4.0])
    xx_expected = np.array([0.666666666666667, 0.416666666666667, -0.5])
    xx_gauss = solvers.gaussian_eliminate(aa, bb)
    assert np.all(np.abs(xx_gauss - xx_expected) < TOLERANCE)


def test_elimination_linear_dep():
    "Tests detection of linear dependency"
    aa = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    bb = np.array([1.0, 2.0, 3.0])
    xx_gauss = solvers.gaussian_eliminate(aa, bb)
    assert xx_gauss is None

