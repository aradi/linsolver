#!/usr/bin/env python3
"""Contains routines to test the solvers module"""

import os.path
import pytest
import numpy as np
import solvers
import linsolveio as io


ABSOLUTE_TOLERANCE = 1e-10
RELATIVE_TOLERANCE = 1e-10

TESTDATADIR = 'testdata'

TESTS = ['simple', 'needs_pivot', 'linearly_dependant']


def get_test_input(testname):
    "Reads the input for a given test."
    testinfile = os.path.join(TESTDATADIR, testname + '.in')
    aa, bb = io.read_input(testinfile)
    return aa, bb


def get_test_output(testname):
    "Reads the reference ouput for a given test."
    testoutfile = os.path.join(TESTDATADIR, testname + '.out')
    result = io.read_result(testoutfile)
    return result


@pytest.mark.parametrize("testname", TESTS)
def test_elimination(testname):
    "Tests elimination."
    aa, bb = get_test_input(testname)
    xx_expected = get_test_output(testname)
    if xx_expected is None:
        # Linear system of equations can not be solved -> expecting exception
        with pytest.raises(ValueError):
            solvers.gaussian_eliminate(aa, bb)
    else:
        xx_gauss = solvers.gaussian_eliminate(aa, bb)
        assert np.allclose(xx_gauss, xx_expected, atol=ABSOLUTE_TOLERANCE,
                           rtol=RELATIVE_TOLERANCE)


if __name__ == '__main__':
    pytest.main()
