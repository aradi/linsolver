'''Contains input output routines for the solver program.'''
import sys
import numpy as np


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
    '''Writes an output file with an error message
    
    Args:
        filename: Name of the file to write.
        errortype: Short string identifying the type of the error.
        errormsg: Longer string containing the error message.
    '''
    with open(filename, 'w') as fp:
        fp.write('ERROR::{}: {}\n'.format(errortype, errormsg))


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
