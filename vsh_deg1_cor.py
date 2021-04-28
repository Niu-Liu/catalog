# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 22:54:38 2017

@author: Neo

VSH function.
The full covariance matrix is used.

# Notice !!!
# unit for RA and DE are rad.

History

N.Liu, 22/02/2018 : add some comments;
                    add new funtion 'test_code';
                    calculate the rms of residuals and reduced chi-squares.
N.Liu, 31/03/2018 : add new elimination criterior of outliers,
                      i.e., functions 'elim_angsep' and 'elim_norsep';
                    add some comments to understand these codes;
N.Liu, 04/04/2018 : add a new input parameter 'wgt' to function
                      'elim_nsigma';
                    add 2 new input parameters to functions
                      'vsh_deg01_fitting' and 'VSHdeg02_fitting';
                    add a new function 'find_good_obs';
                    functions 'vsh_deg01_fitting' and 'VSHdeg02_fitting'
                      now can use different outlier elimination algorithm;
N.Liu, 30/04/2018 : divide this code into two files "vsh_deg1_cor" and
                    "vsh_deg2_cor";
N.Liu, 03/05/2018 : add "fit_type" parameter to function "vsh_deg01_fitting";
N.Liu, 16/07/2018 : printing output to a file is optional.
N.Liu, 27/09/2018 : add a new output "gof" of the function "vsh_deg01_fitting";
                    change the function name: "VSHdeg01" -> "vsh_deg01_solve",
                             "VSHdeg01_fitting" -> "vsh_deg01_fitting";
                    change the order of inputs for the function "vsh_deg01_fitting";
N.Liu, 09/09/2019: add new argument 'return_aux' to function
                    'vsh_deg01_fitting'

"""

import numpy as np
from numpy import sin, cos, pi, concatenate
import sys
# My modules
from .stats_calc import calc_wrms, calc_chi2_2d, calc_gof, calc_mean
from .pos_diff import nor_sep_calc


__all__ = ["residual_calc01", "vsh_deg01_solve", "vsh_deg01_fitting"]


# ------------------ FUNCTION --------------------------------
def elim_nsigma(y1r, y2r, n=3.0, wgt_flag=False):
    '''An outlier elimination using n-sigma criteria.

    Parameters
    ----------
    y1r/y2r : array of float
        residuals of RA and DC
    n : float
        the strength of elimination, default value 3.0
    wgt_flag : True or False, default False
        use the rms or wrms as the unit of n

    Returns
    ----------
    ind_go : array of int
        index of good observations
    '''

    if wgt_flag:
        # wrms
        std1 = np.sqrt(np.sum(y1r**2 / y1_err**2) / np.sum(y1_err**-2))
        std2 = np.sqrt(np.sum(y2r**2 / y2_err**2) / np.sum(y2_err**-2))
    else:
        # rms
        std1 = np.sqrt(np.sum(y1r**2) / (y1r.size - 1))
        std2 = np.sqrt(np.sum(y2r**2) / (y2r.size - 1))

    indice1 = np.where(np.fabs(y1r) - n * std1 <= 0)
    indice2 = np.where(np.fabs(y2r) - n * std2 <= 0)
    ind_go = np.intersect1d(indice1, indice2)

    # return ind_go, std1, std2
    return ind_go


# ----------------------------------------------------
def elim_angsep(angsep, pho_max=10.0e3):
    '''An outlier elimiantion based optic-radio angular seperation.

    Parameters
    ----------
    ang_sep : array of float
        angular seperation, in micro-as
    pho_max : float
        accepted maximum angular seperation, default 10.0 mas

    Returns
    ----------
    ind_go : array of int
        index of good observations
    '''

    # print(pho_max)
    ind_go = np.where(angsep <= pho_max)

    return ind_go


# ----------------------------------------------------
def elim_norsep(X, X_max=10.0):
    '''A outlier elimiantion based the normalized optic-radio seperation.

    Parameters
    ----------
    X : array of float
        normalized separations, unit-less.
    X_max : float
        accepted maximum X, default 10.0

    Returns
    ----------
    ind_go : array of int
        index of good observations
    '''

    ind_go = np.where(X <= X_max)

    return ind_go


def find_good_obs(dRA, dDE, e_dRA, e_dDE, cov, RA, DE, ind_go):
    '''Find the good observations based on index.

    Parameters
    ----------
    dRA/dDE : array of float
        R.A.(*cos(Dec.))/Dec. differences in uas
    e_dRA/e_dDE : array of float
        formal uncertainty of dRA(*cos(DE))/dDE in uas
    cov : array of float
        covariance between dRA and dDE in uas^2
    RA/DE : array of float
        Right ascension/Declination in radian
    ind_go : array of int
        index of good observations

    Returns
    ----------
    dRAn/dDEn : array of float
        R.A.(*cos(Dec.))/Dec. differences for good obsevations in uas
    e_dRAn/e_dDEn : array of float
        formal uncertainty of dRA(*cos(DE))/dDE good obsevations in uas
    covn : array of float
        covariance between dRA and dDE good obsevations in uas^2
    RAn/DEn : array of float
        Right ascension/Declination good obsevations in radian
    '''

    dRAn, dDEn, e_dRAn, e_dDEn = [np.take(dRA, ind_go)[0],
                                  np.take(dDE, ind_go)[0],
                                  np.take(e_dRA, ind_go)[0],
                                  np.take(e_dDE, ind_go)[0]]
    if cov is None:
        covn = None
    else:
        covn = np.take(cov, ind_go)[0]

    RAn, DEn = np.take(RA, ind_go)[0], np.take(DE, ind_go)[0]

    return dRAn, dDEn, e_dRAn, e_dDEn, covn, RAn, DEn


# ----------------------------------------------------
def wgt_mat(e_dRA, e_dDE, cov=None):
    '''Generate the weighted matrix.

    Parameters
    ----------
    e_dRA/e_dDE : array of float
        formal uncertainty of dRA(*cos(DE))/dDE in uas
    cov : array of float
        covariance between dRA and dDE in uas^2, default is None

    Returns
    ----------
    wgt : matrix
        weighted matrix used in the least squares fitting.
    '''

    err = concatenate((e_dRA, e_dDE), axis=0)

    # Covariance matrix.
    covmat = np.diag(err**2)

    if cov is not None:
        # Take the correlation into consideration.
        num = e_dRA.size
        for i, C in enumerate(cov):
            covmat[i, i + num] = C
            covmat[i + num, i] = C

    # Inverse it to obtain weighted matrix.
    wgt = np.linalg.inv(covmat)

    return wgt


# ---------------------------------------------------
def jac_mat_deg01(RA, DE, fit_type="full"):
    '''Generate the Jacobian matrix of 1st VSH function.

    Parameters
    ----------
    RA : array of float
        right ascension in radian
    DE : array of float
        declination in radian
    fit_type : string
        flag to determine which parameters to be fitted
        "full" for full 6 parameters
        "rotation" for only 3 rotation parameters
        "glide" for only 3 glide parameters

    Returns
    ----------
    JacMat/JacMatT : matrix
        Jacobian matrix and its transpose matrix
    '''

    # Partial array dRA and dDE, respectively.
    if fit_type is "full":
        # For RA
        # glide
        par1_d1 = -sin(RA)
        par1_d2 = cos(RA)
        par1_d3 = np.zeros_like(RA)
        # rotation
        par1_r1 = -cos(RA) * sin(DE)
        par1_r2 = -sin(RA) * sin(DE)
        par1_r3 = cos(DE)

        # For DE
        # glide
        par2_d1 = -cos(RA) * sin(DE)
        par2_d2 = -sin(RA) * sin(DE)
        par2_d3 = cos(DE)
        # rotation
        par2_r1 = sin(RA)
        par2_r2 = -cos(RA)
        par2_r3 = np.zeros_like(RA)

        # (dRA, dDE).
        pard1 = concatenate((par1_d1, par2_d1), axis=0)
        pard2 = concatenate((par1_d2, par2_d2), axis=0)
        pard3 = concatenate((par1_d3, par2_d3), axis=0)
        parr1 = concatenate((par1_r1, par2_r1), axis=0)
        parr2 = concatenate((par1_r2, par2_r2), axis=0)
        parr3 = concatenate((par1_r3, par2_r3), axis=0)

        # transpose of Jacobian matrix.
        # JacMatT = np.vstack((pard1, pard2, pard3,
        #                      parr1, parr2, parr3))
        N = pard1.size
        JacMatT = concatenate(
            (pard1.reshape(1, N), pard2.reshape(1, N), pard3.reshape(1, N),
             parr1.reshape(1, N), parr2.reshape(1, N), parr3.reshape(1, N)),
            axis=0)

    elif fit_type is "rotation":
        # For RA
        par1_r1 = -cos(RA) * sin(DE)
        par1_r2 = -sin(RA) * sin(DE)
        par1_r3 = cos(DE)

        # For DE
        par2_r1 = sin(RA)
        par2_r2 = -cos(RA)
        par2_r3 = np.zeros_like(RA)

        # (dRA, dDE).
        parr1 = concatenate((par1_r1, par2_r1), axis=0)
        parr2 = concatenate((par1_r2, par2_r2), axis=0)
        parr3 = concatenate((par1_r3, par2_r3), axis=0)

        # transpose of Jacobian matrix.
        N = parr1.size
        JacMatT = concatenate(
            (parr1.reshape(1, N), parr2.reshape(1, N), parr3.reshape(1, N)),
            axis=0)

    elif fit_type is "glide":
        # For RA
        par1_d1 = -sin(RA)
        par1_d2 = cos(RA)
        par1_d3 = np.zeros_like(RA)

        # For DE
        par2_d1 = -cos(RA) * sin(DE)
        par2_d2 = -sin(RA) * sin(DE)
        par2_d3 = cos(DE)

        # (dRA, dDE).
        pard1 = concatenate((par1_d1, par2_d1), axis=0)
        pard2 = concatenate((par1_d2, par2_d2), axis=0)
        pard3 = concatenate((par1_d3, par2_d3), axis=0)

        # transpose of Jacobian matrix.
        # JacMatT = np.vstack((pard1, pard2, pard3))
        N = pard1.size
        JacMatT = concatenate(
            (pard1.reshape(1, N), pard2.reshape(1, N), pard3.reshape(1, N)),
            axis=0)

    else:
        print("Error in the fit_type parameter.")
        exit()

    # Jacobian matrix.
    JacMat = np.transpose(JacMatT)

    return JacMat, JacMatT


# ---------------------------------------------------
def vsh_func01(ra, dec, X, fit_type="full"):
    # r1, r2, r3, g1, g2, g3):
    '''VSH function of the first degree.

    Parameters
    ----------
    ra/dec : array of float
        Right ascension/Declination in radian
    X :
        r1, r2, r3 : float
            rotation parameters
        g1, g2, g3 : float
            glide parameters
    fit_type : string
        flag to determine which parameters to be fitted
        "full" for full 6 parameters
        "rotation" for only 3 rotation parameters
        "glide" for only 3 glide parameters

    Returns
    ----------
    dra/ddec : array of float
        R.A.(*cos(Dec.))/Dec. differences in uas
    '''

    if fit_type is "full":
        g1, g2, g3, r1, r2, r3 = X
        dra = [-r1 * cos(ra) * sin(dec) - r2 * sin(ra) * sin(dec)
               + r3 * cos(dec)
               - g1 * sin(ra) + g2 * cos(ra)][0]
        ddec = [+ r1 * sin(ra) - r2 * cos(ra)
                - g1 * cos(ra) * sin(dec) - g2 * sin(ra) * sin(dec)
                + g3 * cos(dec)][0]
    elif fit_type is "rotation":
        r1, r2, r3 = X
        dra = [-r1 * cos(ra) * sin(dec) - r2 * sin(ra) * sin(dec)
               + r3 * cos(dec)][0]
        ddec = [+ r1 * sin(ra) - r2 * cos(ra)][0]
    elif fit_type is "glide":
        g1, g2, g3 = X
        dra = [- g1 * sin(ra) + g2 * cos(ra)][0]
        ddec = [- g1 * cos(ra) * sin(dec) - g2 * sin(ra) * sin(dec)
                + g3 * cos(dec)][0]
    else:
        print("ERROR in parameter fit_type(function 'vsh_func01')!")
        exit()
    return dra, ddec


# ---------------------------------------------------
def residual_calc01(dRA, dDE, RA, DE, param01, fit_type="full"):
    '''Calculate the residuals of RA/Dec

    Parameters
    ----------
    dRA/dDE : array of float
        R.A.(*cos(Dec.))/Dec. differences in uas
    RA/DE : array of float
        Right ascension/Declination in radian
    param01 : array of float
        estimation of rotation and glide parameters
    fit_type : string
        flag to determine which parameters to be fitted
        "full" for full 6 parameters
        "rotation" for only 3 rotation parameters
        "glide" for only 3 glide parameters

    Returns
    ----------
    ResRA/ResDE : array of float
        residual array of dRA(*cos(Dec))/dDec in uas.
    '''

    # Theoritical value
    dra, ddec = vsh_func01(RA, DE, param01, fit_type)

    # Calculate the residual. ( O - C )
    ResRA, ResDE = dRA - dra, dDE - ddec

    return ResRA, ResDE


# ---------------------------------------------------
def normal_matrix_calc(dRA, dDE, e_dRA, e_dDE, RA, DE,
                       cov=None, fit_type="full"):
    '''Calculate the normal matrix

    Parameters
    ----------
    dRA/dDE : array of float
        R.A.(*cos(Dec.))/Dec. differences in uas
    e_dRA/e_dDE : array of float
        formal uncertainty of dRA(*cos(DE))/dDE in uas
    RA/DE : array of float
        Right ascension/Declination in radian
    cov : array of float
        covariance between dRA and dDE in uas^2, default is None
    fit_type : string
        flag to determine which parameters to be fitted
        "full" for full 6 parameters
        "rotation" for only 3 rotation parameters
        "glide" for only 3 glide parameters

    Returns
    ----------
    A : array of float
        normal matrix
    b : array of float
        observational matrix
    '''

    # print("normal_matrix_calc")
    # print(cov, cov.shape)

    # Jacobian matrix and its transpose.
    JacMat, JacMatT = jac_mat_deg01(RA, DE, fit_type)

    # Weighted matrix.
    WgtMat = wgt_mat(e_dRA, e_dDE, cov)

    # Calculate matrix A and b of matrix equation:
    # A * x = b.
    A = np.dot(np.dot(JacMatT, WgtMat), JacMat)

    dPos = concatenate((dRA, dDE), axis=0)
    b = np.dot(np.dot(JacMatT, WgtMat),  dPos)

    return A, b


# ---------------------------------------------------
def vsh_deg01_solve(dRA, dDE, e_dRA, e_dDE, RA, DE, cov=None, fit_type="full"):
    # ), max_num=int(1e3)):
    ''' The 1st degree of VSH function: glide and rotation.

    Parameters
    ----------
    dRA/dDE : array of float
        R.A.(*cos(Dec.))/Dec. differences in uas
    e_dRA/e_dDE : array of float
        formal uncertainty of dRA(*cos(DE))/dDE in uas
    RA/DE : array of float
        Right ascension/Declination in radian
    cov : array of float
        covariance between dRA and dDE in uas^2, default is None
    fit_type : string
        flag to determine which parameters to be fitted
        "full" for full 6 parameters
        "rotation" for only 3 rotation parameters
        "glide" for only 3 glide parameters

    Returns
    ----------
    x : array of float
        estimaation of (d1, d2, d3, r1, r2, r3) in uas
    sig : array of float
        uncertainty of x in uas
    corr_mat : matrix
        matrix of correlation coefficient.
    '''

    # Maxium number of calculation the matrix
    max_num = 1000

    if dRA.size > max_num:
        div = dRA.size // max_num
        rem = dRA.size % max_num

        if cov is None:
            A, b = normal_matrix_calc(dRA[:rem], dDE[:rem],
                                      e_dRA[:rem], e_dDE[:rem],
                                      RA[:rem], DE[:rem],
                                      cov, fit_type)
        else:
            A, b = normal_matrix_calc(dRA[:rem], dDE[:rem],
                                      e_dRA[:rem], e_dDE[:rem],
                                      RA[:rem], DE[:rem],
                                      cov[:rem], fit_type)

        for i in range(div):
            ista = rem + i * max_num
            iend = ista + max_num

            if cov is None:
                An, bn = normal_matrix_calc(dRA[ista:iend], dDE[ista:iend],
                                            e_dRA[ista:iend], e_dDE[ista:iend],
                                            RA[ista:iend], DE[ista:iend],
                                            cov, fit_type)
            else:
                An, bn = normal_matrix_calc(dRA[ista:iend], dDE[ista:iend],
                                            e_dRA[ista:iend], e_dDE[ista:iend],
                                            RA[ista:iend], DE[ista:iend],
                                            cov[ista:iend], fit_type)
            A = A + An
            b = b + bn
    else:

        A, b = normal_matrix_calc(dRA, dDE, e_dRA, e_dDE, RA, DE,
                                  cov, fit_type)

    # Solve the equations.
    '''Components of estimation
    fit_type     |           x
    "full"       |(d1, d2, d3, r1, r2, r3)
    "rotation"   |      (r1, r2, r3)
    "glide"      |      (d1, d2, d3)
    '''
    x = np.linalg.solve(A, b)

    # Covariance.
    pcov = np.linalg.inv(A)
    sig = np.sqrt(pcov.diagonal())

    # Correlation coefficient.
    corr_mat = np.array([pcov[i, j] / sig[i] / sig[j]
                         for j in range(len(x))
                         for i in range(len(x))])
    corr_mat.resize((len(x), len(x)))

    # Return the result.
    return x, sig, corr_mat


# ----------------------------------------------------
def vsh_deg01_fitting(dRA, dDE, RA, DE, e_dRA=None, e_dDE=None,
                      cov=None, flog=None,
                      elim_flag="sigma", N=3.0, ang_sep=None, X=None,
                      fit_type="full", return_aux=False):
    '''1st-degree vsh fitting.

    Parameters
    ----------
    dRA/dDE : array of float
        R.A.(*cos(Dec.))/Dec. differences in uas
    e_dRA/e_dDE : array of float
        formal uncertainty of dRA(*cos(DE))/dDE in uas
    RA/DE : array of float
        Right ascension/Declination in radian
    flog :
        handlings of output file. If None, print to the screen
    cov : array of float
        covariance between dRA and dDE in uas^2, default is None
    elim_flag : string
        "sigma" uses n-sigma principle
        "angsep" uses angular seperation as the criteria
        "norsep" uses normalized seperation as the criteria
        "nor_ang" uses both normalized and angular seperation as the criteria
        "None" or "none" doesn't use any criteria
    N : float
        N-sigma principle for eliminating the outliers
        or
        Maximum seperation
    fit_type : string
        flag to determine which parameters to be fitted
        "full" for full 6 parameters
        "rotation" for only 3 rotation parameters
        "glide" for only 3 glide parameters
   return_aux : Boolean
        If true, return the post-fit residuals besides the parameter estimates

    Returns
    ----------
    x : array of float
        estimaation of (d1, d2, d3, r1, r2, r3) in uas
    sig : array of float
        uncertainty of x in uas
    cofmat : matrix
        matrix of correlation coefficient.
    ind_outl : array of int
        index of outliers
    dRAres/dDEres: array of float
        residual array of dRA(*cos(Dec))/dDec in uas.
    '''

    # If we don't know the individual error
    if e_dRA is None:
        e_dRA = np.ones_like(dRA)
        gof_known = True  # Assume the gof is 1
    else:
        gof_known = False

    if e_dDE is None:
        e_dDE = np.ones_like(dDE)
        gof_known = True  # Assume the gof is 1
    else:
        gof_known = False

    # Calculate the apriori wrms
    if flog is not None:
        meanRA = calc_mean(dRA)
        rmsRA = calc_wrms(dRA)
        wrmsRA = calc_wrms(dRA, e_dRA)
        stdRA = np.std(dRA)
        meanDE = calc_mean(dDE)
        rmsDE = calc_wrms(dDE)
        wrmsDE = calc_wrms(dDE, e_dDE)
        stdDE = np.std(dDE)
        print("# apriori statistics (weighted)\n"
              "#         mean for RA: %10.3f \n"
              "#          rms for RA: %10.3f \n"
              "#         wrms for RA: %10.3f \n"
              "#          std for RA: %10.3f \n"
              "#        mean for Dec: %10.3f \n"
              "#         rms for Dec: %10.3f \n"
              "#        wrms for Dec: %10.3f \n"
              "#         std for Dec: %10.3f   " %
              (meanRA, rmsRA, wrmsRA, stdRA, meanDE, rmsDE, wrmsDE, stdDE), file=flog)

    # Calculate the reduced Chi-square
    if flog is not None:
        apr_chi2 = calc_chi2_2d(dRA, e_dRA, dDE, e_dDE, cov, reduced=True)
        print("# apriori reduced Chi-square for: %10.3f" % apr_chi2, file=flog)

    # Now we can use different criteria of elimination.
    if elim_flag is None or elim_flag is "None":
        x, sig, cofmat = vsh_deg01_solve(dRA, dDE, e_dRA, e_dDE, RA, DE, cov,
                                         fit_type)
        # fit_type, max_num)
        ind_go = np.arange(dRA.size)

    elif elim_flag is "sigma":
        x, sig, cofmat = vsh_deg01_solve(dRA, dDE, e_dRA, e_dDE, RA, DE,
                                         cov, fit_type)
        # Iteration.
        num1 = 1
        num2 = 0
        while(num1 != num2):
            num1 = num2

            # Calculate the residual. ( O - C )
            rRA, rDE = residual_calc01(dRA, dDE, RA, DE, x, fit_type)
            ind_go = elim_nsigma(rRA, rDE, N)
            num2 = dRA.size - ind_go.size

            dRAn, dDEn, e_dRAn, e_dDEn, covn, RAn, DEn = find_good_obs(
                dRA, dDE, e_dRA, e_dDE, cov, RA, DE, ind_go)
            # [dRAn, dDEn, e_dRAn, e_dDEn] = [np.take(dRA, ind_go),
            #                                 np.take(dDE, ind_go),
            #                                 np.take(e_dRA, ind_go),
            #                                 np.take(e_dDE, ind_go)]
            # covn = np.take(cov, ind_go)
            # RAn, DEn = np.take(RA, ind_go), np.take(DE, ind_go)

            xn, sign, cofmatn = vsh_deg01_solve(dRAn, dDEn, e_dRAn, e_dDEn,
                                                RAn, DEn, covn, fit_type)

            x, sig, cofmat = xn, sign, cofmatn

            if flog is not None:
                print("# Number of sample: %d" % (dRA.size-num2),
                      file=flog)
    else:
        ang_sep, X_a, X_d, X = nor_sep_calc(dRA, dDE, e_dRA, e_dDE, cov)

        if elim_flag is "angsep":
            ind_go = elim_angsep(ang_sep)
        elif elim_flag is "norsep":
            ind_go = elim_norsep(X)
        elif elim_flag is "nor_ang":
            ind_go_nor = elim_norsep(X, N)
            ind_go_ang = elim_angsep(ang_sep, N)
            ind_go = np.intersect1d(ind_go_nor, ind_go_ang)
        else:
            print("ERROR: elim_flag can only be sigma, angsep, or norsep!")
            exit()

        # Find all good observations
        dRAn, dDEn, e_dRAn, e_dDEn, covn, RAn, DEn = find_good_obs(
            dRA, dDE, e_dRA, e_dDE, cov, RA, DE, ind_go)
        x, sig, cofmat = vsh_deg01_solve(dRAn, dDEn, e_dRAn, e_dDEn, RAn, DEn,
                                         covn, fit_type)

        if flog is not None:
            print("# Number of sample: %d" % dRAn.size,
                  file=flog)

    ind_outl = np.setxor1d(np.arange(dRA.size), ind_go)
    # dRAres, dDEres RA, dDE, RA, DE, xn)
    dRAres, dDEres = residual_calc01(dRA, dDE, RA, DE, x, fit_type)

    if flog is not None:
        # Calculate the posteriori wrms
        meanRA = calc_mean(dRAres)
        rmsRA = calc_wrms(dRAres)
        wrmsRA = calc_wrms(dRAres, e_dRA)
        stdRA = np.std(dRAres)
        meanDE = calc_mean(dDEres)
        rmsDE = calc_wrms(dDEres)
        wrmsDE = calc_wrms(dDEres, e_dDE)
        stdDE = np.std(dDEres)
        print("# posteriori statistics  of vsh01 fit (weighted)\n"
              "#         mean for RA: %10.3f \n"
              "#          rms for RA: %10.3f \n"
              "#         wrms for RA: %10.3f \n"
              "#          std for RA: %10.3f \n"
              "#        mean for Dec: %10.3f \n"
              "#         rms for Dec: %10.3f \n"
              "#        wrms for Dec: %10.3f \n"
              "#         std for Dec: %10.3f   " %
              (meanRA, rmsRA, wrmsRA, stdRA, meanDE, rmsDE, wrmsDE, stdDE), file=flog)

    # Calculate the reduced Chi-square
    M = 6
    pos_chi2_rdc = calc_chi2_2d(dRAres, e_dRA, dDEres, e_dDE, cov,
                                reduced=True, num_fdm=2*dRAres.size-1-M)
    if flog is not None:
        print("# posteriori reduced Chi-square for: %10.3f\n" %
              pos_chi2_rdc, file=flog)

    # Calculate the goodness-of-fit
    if flog is not None:
        pos_chi2 = calc_chi2_2d(dRAres, e_dRA, dDEres, e_dDE, cov)
        print("# goodness-of-fit is %10.3f\n" %
              calc_gof(2*dRAres.size-1-M, pos_chi2), file=flog)

    # Rescale the formal errors
    sig = sig * np.sqrt(pos_chi2_rdc)

    # Return the result
    if return_aux:
        return x, sig, cofmat, ind_outl, dRAres, dDEres
    else:
        return x, sig, cofmat


# ----------------------------------------------------
def test_code():
    '''Code testing
    '''

    # Check the result with Titov &Lambert (2013)
    # Log file.
    flog = open("../logs/Titov_Lambert2013_check_vsh01.log", "w")

    # Read data
    RAdeg, DEdeg, pmRA, pmDE, e_pmRA, e_pmDE = np.genfromtxt(
        "/Users/Neo/Astronomy/Works/201711_GDR2_ICRF3/data/list429.dat",
        usecols=range(2, 8), unpack=True)

    # degree -> rad
    RArad, DErad = np.deg2rad(RAdeg), np.deg2rad(DEdeg)
    cor = np.zeros_like(RArad)

    # Using VSH degree 01.
    print("VSH deg01:")
    w, sig, corrcoef, _, _, _ = vsh_deg01_fitting(
        pmRA, pmDE, e_pmRA, e_pmDE, cor, RArad, DErad,
        flog, elim_flag="None")
    print("Estimations: ")
    print("Dipole: ", w[:3])
    print("Rotation: ", w[3:])
    print("sigma: ", sig)
    print("correlation: ", corrcoef)
    flog.close()
    print("Done!")

    '''
    Result in paper:
    glide:  (-0.4 +/- 0.7, -5.7 +/- 0.8, -2.8 +/- 0.9)
    rotation: -(-1.1 +/- 0.9, +1.4 +/- 0.8, +0.7 +/- 0.6)

    My result
    Estimations:
    Dipole:  [-0.42655902 -5.74274308 -2.7982711 ]
    Rotation:  [ 1.1286308  -1.40004549 -0.71103623]
    sigma:  [0.72724984 0.79054006 0.90500037
             0.93086674 0.84906682 0.64386228]

    # Same result
    '''


if __name__ == "__main__":
    test_code()
# -------------------- END -----------------------------------
