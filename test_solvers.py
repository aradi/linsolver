#!/usr/bin/env python3
"""Contains routines to test the solvers module"""

import os.path
import pytest
import numpy as np
import solvers


ABSOLUTE_TOLERANCE = 1e-10
RELATIVE_TOLERANCE = 1e-10

TESTDATADIR = 'testdata'

TESTS_SUCCESSFUL = ['simple', 'needs_pivot']

TESTS_LINEARDEP = ['linearly_dependant']


def get_test_input(testname):
    "Reads the input for a given test."
    testinfile = os.path.join(TESTDATADIR, testname + '.in')
    data = np.loadtxt(testinfile)
    nn = data.shape[1]
    aa = data[:nn, :]
    bb = data[nn, :]
    return aa, bb


def get_test_output(testname):
    "Reads the ouput for a given test."
    testoutfile = os.path.join(TESTDATADIR, testname + '.out')
    data = np.loadtxt(testoutfile)
    return data


@pytest.mark.parametrize("testname", TESTS_SUCCESSFUL)
def test_successful_elimination(testname):
    "Tests successful elimination."
    aa, bb = get_test_input(testname)
    xx_expected = get_test_output(testname)
    xx_gauss = solvers.gaussian_eliminate(aa, bb)
    assert np.allclose(xx_gauss, xx_expected, atol=ABSOLUTE_TOLERANCE,
                       rtol=RELATIVE_TOLERANCE)


@pytest.mark.parametrize("testname", TESTS_LINEARDEP)
def test_linear_dependancy(testname):
    "Tests linear dependancy"
    aa, bb = get_test_input(testname)
    with pytest.raises(ValueError):
        solvers.gaussian_eliminate(aa, bb)


if __name__ == '__main__':
    pytest.main()
