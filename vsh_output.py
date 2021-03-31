# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 14:12:08 2017

Write the output information (estimates associated with their formal errors
and covariance) into a text file or on screen.

@author: Neo
"""

from astropy.table import Table
from astropy import units as u
import numpy as np
import sys

__all__ = ["write_htable", "write_vtable",
           "print_vsh_corr", "print_vsh_result"]


#################### Block of functions ####################
def init_parnames(parnb):
    """Initialize the names of parameters if they are not given.

    Parameter
    ---------
    parnb : int
        number of parameters

    Returns
    -------
    parnames: list of string
        names of VSH parameters of first 2-degree
    """

    if parnb == 6:
        parnames = np.array(["D1", "D2", "D3", "R1", "R2", "R3"])
    elif parnb == 16:
        parnames = np.array(["D1", "D2", "D3", "R1", "R2", "R3",
                             "E22R", "E22I", "E21R", "E21I", "E20",
                             "M22R", "M22I", "M21R", "M21I", "M20"])
    elif parnb == 10:
        parnames = np.array(["E22R", "E22I", "E21R", "E21I", "E20",
                             "M22R", "M22I", "M21R", "M21I", "M20"])
    else:
        print("Sorry this length of array is not recongnized :-(")
        sys.exit()

    return parnames


# ----------- Horizontal Table -------------------
def write_htable(catnames, pmts, sigs, names=None, opt=sys.stdout, fmt="8.2f"):
    '''Print estmates and corresponding formal errors into a horizontal table.

    Parameters
    ----------
    names : array of string
        names or labels of parameters
    pmts : array of float
        estimates of parameters
    sigs : array of float
        formal errors of estimates
    opt : file handling
        Default to print all the output on the screen.
    fmt : string
        specifier of the output format
    '''

    if names is None:
        names = init_parnames(len(pmts[0]))

    # Table header
    head_line = []
    for name in names:
        head_line.append("  &${:10s}$".format(name))
    head_line.append(" \\\\")
    print("".join(head_line), file=opt)
    print("\\hline", file=opt)

    # data line
    data_fmt = "  &${0:s} \\pm {0:s}$".format(fmt, fmt)

    for i in range(len(catnames)):
        data_line = ["{:10s}".format(catnames[i])]
        pmt, sig = pmts[i], sigs[i]

        for j in range(len(pmt)):
            data_line.append(data_fmt % (pmt[j], sig[j]))

        data_line.append(" \\\\")

        print("".join(data_line), file=opt)


# ----------- Vertical Table -------------------
def write_vtable(names, pmts, sigs, opt=sys.stdout, fmt="8.2f"):
    '''Print estmates and corresponding formal errors into a horizontal table.

    Parameters
    ----------
    names : array of string
        names or labels of the parameters
    pmts : array of float
        estimates of parameters
    sigs : array of float
        formal errors of estimates
    opt : file handling
        Default to print all the output on the screen.
    fmt : string
        specifier of the output format
    '''

    line_fmt = "$%10s$  &${0:s} \\pm {0:s}$ \\\\".format(fmt)

    for pmt_namei, pmti, pmt_erri in (names, pmts, sigs):
        print(line_fmt % (pmt_namei, pmti, pmt_erri), file=opt)


# ----------- Correlation coefficients -------------------
def print_corr(names, parcorrs, deci_digit=2, included_one=True,
               opt=sys.stdout):
    """Print the correlation coefficient in the screen.

    Parameters
    ----------
    names : array
        names or labels of the parameters
    parcorrs : array of N * N
        matrix or correlation coefficient
    deci_digit : int
        decimal digits. Only 1 and 2 are supported. Default is 2.
    included_one : boolean
        to include the correlation between the same parameters (of course the
        value equals to 1) or not. True for yes.
    opt : file handling
        Default to print all the output on the screen.
    """

    parnb = names.size

    print("\nCorrelation coefficient", file=opt)
    print("---------------------------------------------------------", file=opt)

    # If the correlation matrix is an object of np.mat type,
    # convert it to np.ndarray
    if type(parcorrs) is np.mat or type(parcorrs) is np.matrix:
        parcorrs = np.array(parcorrs)

    if deci_digit == 1:
        # Now begin to print the correlation coefficient
        if included_one:
            # The first line
            print(("  %4s" * (parnb+1)) % ("    ", *names), file=opt)

            # Include the correlation coefficient of one
            for i, pmt_namei in enumerate(names):
                line_fmt = "  %4s" + "  %+4.1f" * (i + 1)
                print(line_fmt % (pmt_namei, *parcorrs[i, :i+1]), file=opt)
        else:
            # The first line
            print(("  %4s" * parnb) % ("    ", *names[:-1]), file=opt)

            for i in range(1, parnb):
                line_fmt = "  %4s" + "  %+4.1f" * i
                print(line_fmt % (names[i], *parcorrs[i, :i]), file=opt)

    elif deci_digit == 2:

        # Now begin to print the correlation coefficient
        if included_one:
            # The first line
            print(("  %5s" * (parnb+1)) % ("    ", *names), file=opt)

            for i, pmt_namei in enumerate(names):
                line_fmt = "  %5s" + "  %+5.2f" * (i + 1)
                print(line_fmt % (pmt_namei, *parcorrs[i, :i+1]), file=opt)
        else:
            # The first line
            print(("  %5s" * parnb) % ("    ", *names[:-1]), file=opt)

            for i in range(1, parnb):
                line_fmt = "  %5s" + "  %+5.2f" * i
                print(line_fmt % (names[i], *parcorrs[i, :i]), file=opt)
    else:
        print("The decimal digit of only 1 or 2 is supported!")
        sys.exit()


def print_vsh_corr(parcorrs, parnames=None,
                   deci_digit=1, included_one=True, opt=sys.stdout):
    """Print the correlation coefficient of VSH02 parameters in the screen.

    Parameters
    ----------
    parcorrs : array of N * N
        matrix or correlation coefficient
    deci_digit : int
        decimal digits. Only 1 and 2 are supported. Default is 2.
    included_one : boolean
        to include the correlation between the same parameters (of course the
        value equals to 1) or not. True for yes.
    """

    if parnames is None:
        parnb = parcorrs[0].size
        if parnb == 6:
            parnames = np.array(["D1", "D2", "D3", "R1", "R2", "R3"])
        elif parnb == 16:
            parnames = np.array(["D1", "D2", "D3", "R1", "R2", "R3",
                                 "E22R", "E22I", "E21R", "E21I", "E20",
                                 "M22R", "M22I", "M21R", "M21I", "M20"])
    else:
        parnb = parnames.size

    # Check the shape of the matrix
    a, b = parcorrs.shape
    if a != b or a != parnb:
        print("The shape of the correlation matrix should be (N, N)(N={:d})!".format(parnb))
        sys.exit()

    print_corr(parnames, parcorrs, deci_digit, included_one, opt)


def print_vsh_result(pmts, sigs, parcorrs,
                     parnames=None, opt=sys.stdout, fmt="%5.0f"):
    '''Print estmates and corresponding formal errors of vsh01 parameters.

    Parameters
    ----------
    parnames : array of string
        parnames or labels of the parameters
    pmts : array of float
        estimates of parameters
    sigs : array of float
        formal errors of estimates
    opt : file handling
        Default to print all the output on the screen.
    fmt : string
        specifier of the output format
    '''

    parnb = len(pmts)

    if parnames is None:
        parnames = init_parnames(parnb)

    tvsh = Table([parnames, pmts, sigs], names=[
        "VSH parameter", "Estimate", "Error"])
    tvsh["Estimate"].format = fmt
    tvsh["Error"].format = fmt
    tvsh["Estimate"].unit = u.uas
    tvsh["Error"].unit = u.uas

    print(tvsh, file=opt)

    print_vsh_corr(parcorrs, parnames,
                   deci_digit=1, included_one=True, opt=opt)


def save_vsh_result(pmts, sigs, ofile, parnames=None, fmt="%5.0f", comment=""):
    """Save the VSH results into a text file.

    Parameters
    ----------
    pmts : array of float
        estimates of parameters
    sigs : array of float
        formal errors of estimates
    opt : file handle
        Default to print all the output on the screen.
    fmt : string
        specifier of the output format
    """

    parnb = len(pmts)

    if parnames is None:
        parnames = init_parnames(parnb)

    tvsh = Table([parnames, pmts, sigs],
                 names=["Names", "Estimate", "Error"])
    tvsh["Estimate"].format = fmt
    tvsh["Error"].format = fmt
    tvsh["Estimate"].unit = u.uas
    tvsh["Error"].unit = u.uas
    tvsh.meta["comment"] = comment

    tvsh.write(ofile, format="ascii", overwrite=True)


def write_textable(catnames, pmts, sigs, ofile=sys.stdout, parnames=None, fmt="%5.0f"):
    """Save the VSH results into a tex table.

    Parameters
    ----------
    pmts : array of float
        estimates of parameters
    sigs : array of float
        formal errors of estimates
    opt : file handle
        Default to print all the output on the screen.
    fmt : string
        specifier of the output format
    """

    parnb = len(pmts[0])

    if parnames is None:
        parnames = init_parnames(parnb)

    write_htable(catnames, pmts, sigs, parnames, ofile, fmt)


# ------------------------- MAIN ---------------------------------------
if __name__ == "__main__":
    print("Sorry but there is nothing to do.")
# ------------------------- END ----------------------------------------
