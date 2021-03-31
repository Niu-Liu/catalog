#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File name: stats_calc.py
"""
Created on Wed Feb 14 11:02:00 2018

@author: Neo(liuniu@smail.nju.edu.cn)

This script is used for calculating the pre-fit wrms,
post-fit wrms, reduced-chi square, and standard deviation.

3 Mar 2018, Niu : now function 'calc_wrms' also computes the standard
                  deviation

"""

from functools import reduce

import numpy as np

__all__ = ["calc_wrms", "calc_wrms2", "calc_chi2", "calc_chi2_2d", "calc_gof"]


# -----------------------------  FUNCTIONS -----------------------------
def calc_mean(x, err=None):
    """Calculate the mean value.

    Parameters
    ----------
    x : array, float
        Series
    err : array, float, default is None.

    Returns
    ----------
    mean : float
        mean
    """

    if err is None:
        mean = np.mean(x)
    else:
        p = 1. / err
        mean = np.dot(x*p, p) / np.dot(p, p)

    return mean


def calc_wrms(x, err=None):
    '''Calculate the (weighted) wrms of x series after removing
    the bias.

    Standard deviation
    std = sqrt(sum( (xi-mean)^2/erri^2 ) / sum( 1.0/erri^2 ))
         if weighted,
         = sqrt(sum( (xi-mean)^2/erri^2 ) / (N-1))
         otherwise.

    Weighted root mean square
    wrms = sqrt(sum( xi^2/erri^2 ) / sum( 1.0/erri^2 ))
         if weighted,
         = sqrt(sum( xi^2/erri^2 ) / (N-1))
         otherwise.

    Parameters
    ----------
    x : array, float
        Series
    err : array, float, default is None.

    Returns
    ----------
    wrms : float
        weighted rms
    '''

    if err is None:
        mean = np.mean(x)
        xn = x - mean
        wrms = np.sqrt(np.dot(xn, xn) / (x.size - 1))
    else:
        p = 1. / err
        mean = np.dot(x, p**2) / np.dot(p, p)
        xn = (x - mean) * p
        wrms = np.sqrt(np.dot(xn, xn) / np.dot(p, p))

    return wrms


def calc_wrms2(x, err=None):
    '''Calculate the (weighted) wrms of x series.

    Weighted root mean square
    wrms = sqrt(sum( xi^2/erri^2 ) / sum( 1.0/erri^2 ))
         if weighted,
         = sqrt(sum( xi^2/erri^2 ) / (N-1))
         otherwise.

    Parameters
    ----------
    x : array, float
        Series
    err : array, float, default is None.

    Returns
    ----------
    wrms : float
        weighted rms
    '''

    if err is None:
        wrms = np.sqrt(np.dot(x, x) / (x.size - 1))
    else:
        p = 1. / err
        wrms = np.sqrt(np.dot(x*p, x*p) / np.dot(p, p))

    return wrms


def calc_chi2(x, err, reduced=False, deg=0):
    '''Calculate the (reduced) Chi-square.


    Parameters
    ----------
    x : array, float
        residuals
    err : array, float
        formal errors of residuals
    reduced : boolean
        True for calculating the reduced chi-square
    deg : int
        degree of freedom

    Returns
    ----------
    (reduced) chi-square
    '''

    wx = x / err
    chi2 = np.dot(wx, wx)

    if reduced:
        if deg:
            return chi2 / (x.size - deg)
        else:
            print("# ERROR: the degree of freedom cannot be 0!")
    else:
        return chi2


def calc_chi2_2d(x, errx, y, erry, covxy=None, reduced=False, num_fdm=0):
    '''Calculate the 2-Dimension (reduced) Chi-square.


    Parameters
    ----------
    x : array, float
        residuals of x
    errx : array, float
        formal errors of x
    x : array, float
        residuals of x
    errx : array, float
        formal errors of x
    covxy : array, float
        summation of covariance between x and y
    reduced : boolean
        True for calculating the reduced chi-square

    Returns
    ----------
    (reduced) chi-square
    '''

    Qxy = np.zeros_like(x)

    if covxy is None:
        covxy = np.zeros_like(x)

    for i, (xi, errxi, yi, erryi, covxyi) in enumerate(
            zip(x, errx, y, erry, covxy)):

        wgt = np.linalg.inv(np.array([[errxi**2, covxyi],
                                      [covxyi, erryi**2]]))

        Xmat = np.array([xi, yi])

        Qxy[i] = reduce(np.dot, (Xmat, wgt, Xmat))

    if num_fdm == 0:
        num_fdm = 2 * x.size - 1

    if reduced:
        return np.sum(Qxy) / num_fdm
    else:
        return np.sum(Qxy)


def calc_gof(fdm_num, chi2):
    """Calculate the goodness-of-fit.

    The formula is expressed as below.

    Q = gammq(fdm_num / 2, chi2 / 2). (Numerical Recipes)

    gammq is the incomplete gamma function.

    Parameters
    ----------
    fdm_num : int
        number of freedom
    chi2 : float
        chi-square

    Return
    ------
    Q : float
        goodness-of-fit
    """

    from scipy.special import gammaincc

    Q = gammaincc(fdm_num/2., chi2/2.)

    return Q
# --------------------------------- END --------------------------------
