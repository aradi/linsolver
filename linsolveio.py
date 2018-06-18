'''Contains input output routines for the equation solver.'''
import sys
import numpy as np


_ERROR_PREFIX = "ERROR::"


def read_input(inputfile):
    '''Reads the input for the solver.

    Note: It calls sys.exit() if the reading the input fails for any reasons.

    Args:
        inputfile: Name of the input file.

    Returns:
        Content of the input file.
    '''
    inparray = np.loadtxt(inputfile)
    ndim = inparray.shape[1]
    aa = np.array(inparray[0:ndim, 0:ndim])
    bb = np.array(inparray[ndim, 0:ndim])
    return aa, bb


def write_error(filename, errortype, errormsg):
    '''Writes an output file with an error message.
    
    Args:
        filename: Name of the file to write.
        errortype: Short string identifying the type of the error.
        errormsg: Longer string containing the error message.
    '''
    with open(filename, 'w') as fp:
        fp.write('{}{}: {}\n'.format(_ERROR_PREFIX, errortype, errormsg))


def write_result(filename, xx):
    '''Writes the result of the solver into a file.

    Args:
        filename: Name of the file where the results should be written to.
        xx: Solution vector.
    '''
    # Converting to (1, n) matrix to ensure that all numbers of the solution
    # vector appear in one row.
    xrow = np.reshape(xx, (1, -1))
    np.savetxt(filename, xrow)


def read_result(resultfile):
    '''Reads the result written by the solver (used for testing).

    Args:
        resultfile: Result file to read.

    Returns:
        Result vector x.
    '''
    with open(resultfile, 'r') as fp:
        line = fp.readline()
    if line.startswith(_ERROR_PREFIX):
        xx = None
    else:
        xx = np.loadtxt(resultfile)
    return xx
