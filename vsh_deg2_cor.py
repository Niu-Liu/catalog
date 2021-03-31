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
                      'VSHdeg01_fitting' and 'VSHdeg02_fitting';
                    add a new function 'find_good_obs';
                    functions 'VSHdeg01_fitting' and 'VSHdeg02_fitting'
                      now can use different outlier elimination algorithm;
N.Liu, 04/04/2018 : divide this code into two files "vsh_deg1_cor" and
                    "vsh_deg2_cor";
N.Liu, 16/07/2018 : printing output to a file is optional.
N.Liu, 27/09/2018 : add a new output "gof" of the function "vsh_deg02_fitting";
                    change the function name: "VSHdeg02" -> "vsh_deg02_solve",
                             "VSHdeg02_fitting" -> "vsh_deg02_fitting";
                    change the order of inputs for the function "vsh_deg01_fitting";
N.Liu, 09/09/2019: add new argument 'return_aux' to function
                    'vsh_deg02_fitting'
"""

import numpy as np
from numpy import sin, cos, pi, concatenate
import sys
# My modules
from .stats_calc import calc_wrms, calc_chi2_2d, calc_gof, calc_mean
from .pos_diff import nor_sep_calc


__all__ = ["residual_calc02", "vsh_deg02_solve", "vsh_deg02_fitting"]


# ------------------ FUNCTION --------------------------------
def elim_nsigma(y1r, y2r, n=3.0, wgt_flag=False,
                y1_err=None, y2_err=None):
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
        # std1 = np.sqrt(np.sum(y1r**2 / y1_err**2) / np.sum(y1_err**-2))
        # std2 = np.sqrt(np.sum(y2r**2 / y2_err**2) / np.sum(y2_err**-2))
        indice1 = np.where(np.fabs(y1r) - n * y1_err <= 0)
        indice2 = np.where(np.fabs(y2r) - n * y2_err <= 0)
        # y = np.sqrt(y1r**2 + y2r ** 2)
        # y_err = np.sqrt(y1_err**2 + y2_err**2)
        # ind_go = np.where(y - n * y_err <= 0)[0]
    else:
        # rms
        std1 = np.sqrt(np.sum(y1r**2) / (y1r.size - 1))
        std2 = np.sqrt(np.sum(y2r**2) / (y2r.size - 1))

    # indice1 = np.where(np.fabs(y1r) - n * std1 <= 0)
    # indice2 = np.where(np.fabs(y2r) - n * std2 <= 0)
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

    dRAn, dDEn, e_dRAn, e_dDEn = [np.take(dRA, ind_go),
                                  np.take(dDE, ind_go),
                                  np.take(e_dRA, ind_go),
                                  np.take(e_dDE, ind_go)]
    if cov is not None:
        covn = np.take(cov, ind_go)
    else:
        covn = None
    RAn, DEn = np.take(RA, ind_go), np.take(DE, ind_go)

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
    # print(covmat.shape)

    if cov is not None:
        # Take the correlation into consideration.
        num = e_dRA.size
        for i, covi in enumerate(cov):
            covmat[i, i + num] = covi
            covmat[i + num, i] = covi

    # Inverse it to obtain weighted matrix.
    wgt = np.linalg.inv(covmat)

    return wgt


# ---------------------------------------------------
def jac_mat_deg02(RA, DE):
    '''Generate the Jacobian matrix

    Parameters
    ----------
    RA : array of float
        right ascension in radian
    DE : array of float
        declination in radian

    Returns
    ----------
    jacMat/jacMatT : matrix
        Jacobian matrix and its transpose matrix
    '''

    # Partial array dRA and dDE, respectively.
    # For RA
    # glide
    par1_11ER = -sin(RA)
    par1_11EI = cos(RA)  # a_{1,-1}^E
    par1_10E = np.zeros_like(RA)
    # rotation
    par1_11MR = -cos(RA) * sin(DE)
    par1_11MI = -sin(RA) * sin(DE)  # a_{1,-1}^M
    par1_10M = cos(DE)

    # # --------------- TEST 20180924 ----------------
    # # rotation
    # par1_11MR = cos(RA) * sin(DE)
    # par1_11MI = sin(RA) * sin(DE)  # a_{1,-1}^M
    # par1_10M = -cos(DE)
    # # --------------- END ----------------

    # quadrupole
    par1_22ER = -2 * sin(2 * RA) * cos(DE)
    par1_22EI = -2 * cos(2 * RA) * cos(DE)
    par1_21ER = sin(RA) * sin(DE)
    par1_21EI = cos(RA) * sin(DE)
    par1_20E = np.zeros_like(RA)
    par1_22MR = -cos(2 * RA) * sin(2 * DE)
    par1_22MI = sin(2 * RA) * sin(2 * DE)
    par1_21MR = -cos(RA) * cos(2 * DE)
    par1_21MI = sin(RA) * cos(2 * DE)
    par1_20M = sin(2 * DE)

    # For DE
    # glide
    par2_11ER = par1_11MR
    par2_11EI = par1_11MI
    par2_10E = par1_10M
    # rotation
    par2_11MR = -par1_11ER
    par2_11MI = -par1_11EI
    par2_10M = -par1_10E

    # quadrupole
    # ----- Bugs found on Sep. 25, 2018 -----
    # par2_22ER = par1_22MI
    # par2_22EI = par1_22MR
    # par2_21ER = par1_21MI
    # par2_21EI = par1_21MR
    # par2_20E = par1_20M
    # par2_22MR = -par1_22EI
    # par2_22MI = -par1_22ER
    # par2_21MR = -par1_21EI
    # par2_21MI = -par1_21ER
    # par2_20M = -par1_20E
    # --------------- END ----------------

    par2_22ER = par1_22MR
    par2_22EI = par1_22MI
    par2_21ER = par1_21MR
    par2_21EI = par1_21MI
    par2_20E = par1_20M
    par2_22MR = -par1_22ER
    par2_22MI = -par1_22EI
    par2_21MR = -par1_21ER
    par2_21MI = -par1_21EI
    par2_20M = -par1_20E

    # (dRA, dDE).
    # dipole glide
    par11ER = concatenate((par1_11ER, par2_11ER), axis=0)
    par11EI = concatenate((par1_11EI, par2_11EI), axis=0)
    par10E = concatenate((par1_10E, par2_10E), axis=0)
    # dipole rotation
    par11MR = concatenate((par1_11MR, par2_11MR), axis=0)
    par11MI = concatenate((par1_11MI, par2_11MI), axis=0)
    par10M = concatenate((par1_10M, par2_10M), axis=0)
    # quadrupole
    par22ER = concatenate((par1_22ER, par2_22ER), axis=0)
    par22EI = concatenate((par1_22EI, par2_22EI), axis=0)
    par21ER = concatenate((par1_21ER, par2_21ER), axis=0)
    par21EI = concatenate((par1_21EI, par2_21EI), axis=0)
    par20E = concatenate((par1_20E, par2_20E), axis=0)
    par22MR = concatenate((par1_22MR, par2_22MR), axis=0)
    par22MI = concatenate((par1_22MI, par2_22MI), axis=0)
    par21MR = concatenate((par1_21MR, par2_21MR), axis=0)
    par21MI = concatenate((par1_21MI, par2_21MI), axis=0)
    par20M = concatenate((par1_20M, par2_20M), axis=0)

    # # Jacobian matrix.
    # jacMatT = np.vstack((
    #     # dipole glide
    #     par11ER, par11EI, par10E,
    #     # dipole rotation
    #     par11MR, par11MI, par10M,
    #     # quadrupole
    #     par22ER, par22EI, par21ER, par21EI, par20E,
    #     par22MR, par22MI, par21MR, par21MI, par20M))

    N = par11ER.size
    jacMatT = concatenate((
        # dipole glide
        par11ER.reshape(1, N), par11EI.reshape(1, N), par10E.reshape(1, N),
        # dipole rotation
        par11MR.reshape(1, N), par11MI.reshape(1, N), par10M.reshape(1, N),
        # quadrupole
        par22ER.reshape(1, N),
        par22EI.reshape(1, N), par21ER.reshape(1, N),
        par21EI.reshape(1, N), par20E.reshape(1, N),
        par22MR.reshape(1, N),
        par22MI.reshape(1, N), par21MR.reshape(1, N),
        par21MI.reshape(1, N), par20M.reshape(1, N)), axis=0)

    jacMat = np.transpose(jacMatT)

    return jacMat, jacMatT


# ---------------------------------------------------
def vsh_func01(ra, dec,
               g1, g2, g3, r1, r2, r3):
    '''VSH function of the first degree.

    Parameters
    ----------
    ra/dec : array of float
        Right ascension/Declination in radian
    r1, r2, r3 : float
        rotation parameters
    g1, g2, g3 : float
        glide parameters

    Returns
    ----------
    dra/ddec : array of float
        R.A.(*cos(Dec.))/Dec. differences in uas
    '''

    dra = [-r1 * cos(ra) * sin(dec) - r2 * sin(ra) * sin(dec)
           + r3 * cos(dec)
           - g1 * sin(ra) + g2 * cos(ra)][0]
    ddec = [+ r1 * sin(ra) - r2 * cos(ra)
            - g1 * cos(ra) * sin(dec) - g2 * sin(ra) * sin(dec)
            + g3 * cos(dec)][0]

    return dra, ddec


# ---------------------------------------------------
def vsh_func02(ra, dec,
               ER_22, EI_22, ER_21, EI_21, E_20,
               MR_22, MI_22, MR_21, MI_21, M_20):
    '''VSH function of the second degree.

    Parameters
    ----------
    ra/dec : array of float
        Right ascension/Declination in radian
    E_20, ER_21, EI_21, ER_22, EI_22 : float
        quadrupolar parameters of electric type
    M_20, MR_21, MI_21, MR_22, MI_22 : float
        quadrupolar parameters of magnetic type

    Returns
    ----------
    dra/ddec : array of float
        R.A.(*cos(Dec.))/Dec. differences in uas
    '''

    dra = [+M_20 * sin(2 * dec)
           - (MR_21 * cos(ra) - MI_21 * sin(ra)) * cos(2*dec)
           + (ER_21 * sin(ra) + EI_21 * cos(ra)) * sin(dec)
           - (MR_22 * cos(2*ra) - MI_22 * sin(2*ra)) * sin(2*dec)
           - 2*(ER_22 * sin(2*ra) + EI_22 * cos(2*ra)) * cos(dec)][0]
    ddec = [+E_20 * sin(2 * dec)
            - (MR_21 * sin(ra) + MI_21 * cos(ra)) * sin(dec)
            - (ER_21 * cos(ra) - EI_21 * sin(ra)) * cos(2*dec)
            + 2*(MR_22 * sin(2*ra) + MI_22 * cos(2*ra)) * cos(dec)
            - (ER_22 * cos(2*ra) - EI_22 * sin(2*ra)) * sin(2*dec)][0]

    return dra, ddec


# ---------------------------------------------------
def vsh_func_calc02(ra, dec, param02):
    '''VSH function of the second degree.

    Parameters
    ----------
    ra/dec : array of float
        Right ascension/Declination in radian
    param02 : array of float
        estimation of rotation, glide, and quadrupolar parameters

    Returns
    ----------
    dra/ddec : array of float
        R.A.(*cos(Dec.))/Dec. differences in uas
    '''

    dra1, ddec1 = vsh_func01(ra, dec, *param02[:6])
    dra2, ddec2 = vsh_func02(ra, dec, *param02[6:])

    return dra1 + dra2, ddec1 + ddec2


# ---------------------------------------------------
def residual_calc02(dRA, dDE, RA, DE, param02):
    '''Calculate the residuals of RA/Dec

    Parameters
    ----------
    dRA/dDE : array of float
        R.A.(*cos(Dec.))/Dec. differences in uas
    RA/DE : array of float
        Right ascension/Declination in radian
    param02 : array of float
        estimation of rotation, glide, and quadrupolar parameters

    Returns
    ----------
    ResRA/ResDE : array of float
        residual array of dRA(*cos(Dec))/dDec in uas.
    '''

    # Theoritical value
    dra, ddec = vsh_func_calc02(RA, DE, param02)

    # Calculate the residual. ( O - C )
    ResRA, ResDE = dRA - dra, dDE - ddec

    return ResRA, ResDE


# ---------------------------------------------------
def normal_matrix_calc(dRA, dDE, e_dRA, e_dDE, RA, DE,
                       cov=None):
    '''Calculate the normal matrix

    dRA/dDE : array of float
        R.A.(*cos(Dec.))/Dec. differences in uas
    e_dRA/e_dDE : array of float
        formal uncertainty of dRA(*cos(DE))/dDE in uas
    RA/DE : array of float
        Right ascension/Declination in radian
    cov : array of float
        covariance between dRA and dDE in uas^2, default is None

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
    # jacMat, jacMatT = jac_mat_deg02(RA, DE, fit_type)
    jacMat, jacMatT = jac_mat_deg02(RA, DE)

    # Weighted matrix.
    WgtMat = wgt_mat(e_dRA, e_dDE, cov)

    # Calculate matrix A and b of matrix equation:
    # A * x = b.
    mat_tmp = np.dot(jacMatT, WgtMat)
    A = np.dot(mat_tmp, jacMat)

    dPos = concatenate((dRA, dDE), axis=0)
    b = np.dot(mat_tmp,  dPos)

    return A, b


# ---------------------------------------------------
def vsh_deg02_solve(dRA, dDE, e_dRA, e_dDE, RA, DE, cov=None):
    '''2rd degree of VSH function: glide and rotation.

    The 2nd degree of VSH function: glide and rotation + quadrupole.

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

    Returns
    ----------
    x : array of float
        estimaation of (d1, d2, d3,
                        r1, r2, r3,
                        ER_22, EI_22, ER_21, EI_21, E_20,
                        MR_22, MI_22, MR_21, MI_21, M_20) in uas
    sig : array of float
        uncertainty of x in uas
    corrmat : matrix
        matrix of correlation coefficient.
    '''

    # Maxium number of calculation the matrix
    max_num = 500

    if dRA.size > max_num:
        div = dRA.size // max_num
        rem = dRA.size % max_num

        if cov is None:
            A, b = normal_matrix_calc(dRA[:rem], dDE[:rem],
                                      e_dRA[:rem], e_dDE[:rem],
                                      RA[:rem], DE[:rem], cov)
        else:
            A, b = normal_matrix_calc(dRA[:rem], dDE[:rem],
                                      e_dRA[:rem], e_dDE[:rem],
                                      RA[:rem], DE[:rem], cov[:rem])

        for i in range(div):
            ista = rem + i * max_num
            iend = ista + max_num

            if cov is None:
                An, bn = normal_matrix_calc(dRA[ista:iend], dDE[ista:iend],
                                            e_dRA[ista:iend], e_dDE[ista:iend],
                                            RA[ista:iend], DE[ista:iend],
                                            cov)
            else:
                An, bn = normal_matrix_calc(dRA[ista:iend], dDE[ista:iend],
                                            e_dRA[ista:iend], e_dDE[ista:iend],
                                            RA[ista:iend], DE[ista:iend],
                                            cov[ista:iend])
            A = A + An
            b = b + bn
    else:

        A, b = normal_matrix_calc(dRA, dDE, e_dRA, e_dDE, RA, DE, cov)

    # Solve the equations.
    # x = (d1, d2, d3,
    #      r1, r2, r3,
    #      ER_22, EI_22, ER_21, EI_21, E_20,
    #      MR_22, MI_22, MR_21, MI_21, M_20)
    x = np.linalg.solve(A, b)

    # Covariance.
    pcov = np.linalg.inv(A)
    sig = np.sqrt(pcov.diagonal())

    # Correlation coefficient.
    corrmat = np.array([pcov[i, j] / sig[i] / sig[j]
                        for j in range(len(x))
                        for i in range(len(x))])
    corrmat.resize((len(x), len(x)))

    # Return the result.
    return x, sig, corrmat


# ----------------------------------------------------
def vsh_deg02_fitting(dRA, dDE, RA, DE, e_dRA=None, e_dDE=None,
                      cov=None, flog=None,
                      elim_flag="sigma", N=3.0, return_aux=False):
    '''2rd-degree vsh fitting.

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
    flog :
        handlings of output file.
    elim_flag : string
        "sigma" uses n-sigma principle
        "angsep" uses angular seperation as the criteria
        "norsep" uses normalized seperation as the criteria
        "nor_ang" uses both normalized and angular seperation as the criteria
        "None" or "none" doesn't use any criteria
    N : float
        N-sigma principle for eliminating the outliers
        or
        the maximum seperation (uas)
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
        print("# apriori reduced Chi-square for: %10.3f\n" % apr_chi2, file=flog)

    # Now we can use different criteria of elimination.
    if elim_flag is "None" or elim_flag is "none":
        x, sig, cofmat = vsh_deg02_solve(dRA, dDE, e_dRA, e_dDE, RA, DE, cov)
        ind_go = np.arange(dRA.size)

    elif elim_flag is "sigma":
        x, sig, cofmat = vsh_deg02_solve(dRA, dDE, e_dRA, e_dDE, cov, RA, DE)
        # Iteration.
        num1 = 1
        num2 = 0
        while(num1 != num2):
            num1 = num2

            # Calculate the residual. ( O - C )
            rRA, rDE = residual_calc02(dRA, dDE, RA, DE, x)
            ind_go = elim_nsigma(rRA, rDE, N, wgt_flag=True,
                                 y1_err=e_dRA, y2_err=e_dDE)
            # ind_go = elim_nsigma(rRA, rDE, N)
            num2 = dRA.size - ind_go.size
            # num2 = dRA.size - len(ind_go)

            dRAn, dDEn, e_dRAn, e_dDEn, covn, RAn, DEn = find_good_obs(
                dRA, dDE, e_dRA, e_dDE, cov, RA, DE, ind_go)

            xn, sign, corrmatn = vsh_deg02_solve(dRAn, dDEn, e_dRAn, e_dDEn,
                                                 covn, RAn, DEn)

            x, sig, cofmat = xn, sign, corrmatn

            if flog is not None:
                print("# Number of sample: %d" % (dRA.size-num2),
                      file=flog)

    else:
        ang_sep, X_a, X_d, X = nor_sep_calc(dRA, dDE, e_dRA, e_dDE, cov)

        if elim_flag is "angsep":
            ind_go = elim_angsep(ang_sep, N)
        elif elim_flag is "norsep":
            ind_go = elim_norsep(X, N)
        elif elim_flag is "nor_ang":
            ind_go_nor = elim_norsep(X, N)
            ind_go_ang = elim_angsep(ang_sep, N)
            ind_go = np.intersect1d(ind_go_nor, ind_go_ang)
        else:
            print("ERROR: elim_flag can only be sigma, angsep, norsep,"
                  " or nor_ang!")
            exit()

        # Find all good observations
        dRAn, dDEn, e_dRAn, e_dDEn, covn, RAn, DEn = find_good_obs(
            dRA, dDE, e_dRA, e_dDE, cov, RA, DE, ind_go)
        x, sig, cofmat = vsh_deg02_solve(dRAn, dDEn, e_dRAn, e_dDEn, covn,
                                         RAn, DEn)

        if flog is not None:
            print("# Number of sample: %d" % dRAn.size,
                  file=flog)

    ind_outl = np.setxor1d(np.arange(dRA.size), ind_go)
    dRAres, dDEres = residual_calc02(dRA, dDE, RA, DE, x)

    # Calculate the posteriori wrms
    if flog is not None:
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
    M = 16
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


# ------------------------- MAIN ---------------------------------------
if __name__ == "__main__":
    print("Nothing to do!")
# ------------------------- END ----------------------------------------
