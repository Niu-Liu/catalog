# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 16:00:08 2017

Positional transformation.

@author: Neo

Oct 29, 2018: re-write all the codes
"""

import numpy as np
from numpy import sin, cos, pi, concatenate
# My modules
from .cov_mat import calc_wgt_mat, read_cov_mat
from .stats_calc import calc_chi2_2d, calc_gof


__all__ = ["tran_fitting1", "tran_fitting2",
           "tran_fitting3", "tran_fitting3_0"]


##################  IESR transformation version 01 ##################
def tran_func1(ra, dec, r1, r2, r3):
    """Coordinate transformation function.

    The transformation function considering onl the rigid rotation
    is given by
    d_RA^* = -r_x*sin(dec)*cos(ra) - r_y*sin(dec)*sin(ra) + r_z*cos(dec)
    d_DE   = +r_x*sin(ra) - r_y*cos(ra)

    Parameters
    ----------
    ra/dec : array of float
        right ascension/declination in radian
    r1/r2/r3 : float
        rotational angles around X-, Y-, and Z-axis

    Returns
    -------
    dra/ddec : array of float
        R.A.(*cos(Dec.))/Dec. differences
    """

    dra = -r1 * cos(ra) * sin(dec) - r2 * sin(ra) * sin(dec) + r3 * cos(dec)
    ddec = r1 * sin(ra) - r2 * cos(ra)

    return dra, ddec


def jac_mat1(ra, dec):
    """Calculate the Jacobian matrix.

    The transformation function considering onl the rigid rotation
    is given by
    d_RA^* = -r_x*sin(dec)*cos(ra) - r_y*sin(dec)*sin(ra) + r_z*cos(dec)
    d_DE   = +r_x*sin(ra) - r_y*cos(ra)

    Parameters
    ----------
    ra : array of float
        right ascension in radian
    dec : array of float
        declination in radian

    Returns
    -------
    jac_mat : Jacobian matrix
    """

    # Partial array dra and ddec, respectively.
    parx1 = -sin(dec)*cos(ra)
    parx2 = sin(ra)
    pary1 = -sin(dec)*sin(ra)
    pary2 = -cos(ra)
    parz1 = cos(dec)
    parz2 = np.zeros_like(dec)

    # (dra, ddec).
    parx = concatenate((parx1, parx2))
    pary = concatenate((pary1, pary2))
    parz = concatenate((parz1, parz2))

    # Jacobian matrix.
    jac_mat = np.stack((parx, pary, parz), axis=-1)

    return jac_mat


def solve_tran_neq1(dra, ddec, ra, dec,
                    dra_err=None, ddec_err=None, corr=None):
    """Solve normal equation.

    The transformation function considering onl the rigid rotation
    is given by
    d_RA^* = -r_x*sin(dec)*cos(ra) - r_y*sin(dec)*sin(ra) + r_z*cos(dec)
    d_DE   = +r_x*sin(ra) - r_y*cos(ra)

    Parameters
    ----------
    dra/ddec : array of float
        offset in right ascension/declination
    ra/dec : array of float
        right ascension/declination in radian
    dra_err/ddec_err : array of float
        formal uncertainties of dra/ddec, default value is None
    corr : array of float
        correlation between dra and ddec, default value is None

    Returns
    -------
    w : array of (3,)
        estimation of three rotational angles
    sig : array of (3,)
        formal uncertainty of estimations
    corr_mat : array of (3,3)
        correlation coefficients between estimations
    """

    # Jacobian matrix and its transpose.
    jac_mat = jac_mat1(ra, dec)
    jac_mat_t = np.transpose(jac_mat)

    # Weighted matrix.
    wgt_mat = calc_wgt_mat(dra_err, ddec_err, corr)

    # Calculate matrix A and b of matrix equation:
    # A * w = b.
    mat_tmp = np.dot(jac_mat_t, wgt_mat)
    A = np.dot(mat_tmp, jac_mat)
    dpos = concatenate((dra, ddec))
    b = np.dot(mat_tmp,  dpos)

    # Solve the equations.
    #  w = (r_x, r_y, r_z)^T
    w = np.linalg.solve(A, b)

    # Covariance.
    cov_mat = np.linalg.inv(A)
    sig, corr_mat = read_cov_mat(cov_mat)

    # Return the result.
    return w, sig, corr_mat


def residual_calc1(dRA, dDE, RA, DE, param):
    """Calculate the residuals of RA/Dec

    The transformation function considering onl the rigid rotation
    is given by
    d_RA^* = -r_x*sin(dec)*cos(ra) - r_y*sin(dec)*sin(ra) + r_z*cos(dec)
    d_DE   = +r_x*sin(ra) - r_y*cos(ra)

    Parameters
    ----------
    dRA/dDE : array of float
        differences in R.A.(*cos(Dec.))/Dec.
    RA/DE : array of float
        Right ascension/Declination in radian
    param : array of float
        estimation of three rotation angles

    Returns
    ----------
    ResRA/ResDE : array of float
        residual array of dRA(*cos(Dec))/dDec in uas.
    """

    # Theoritical value
    dra, ddec = tran_func1(RA, DE, *param)

    # Calculate the residual. ( O - C )
    ResRA, ResDE = dRA - dra, dDE - ddec

    return ResRA, ResDE


def tran_fitting1(dra, ddec, ra, dec,
                  dra_err=None, ddec_err=None, corr_arr=None, flog=sys.stdout):
    """Least square fitting of transformation equation.

    The transformation function considering onl the rigid rotation
    is given by
    d_RA^* = -r_x*sin(dec)*cos(ra) - r_y*sin(dec)*sin(ra) + r_z*cos(dec)
    d_DE   = +r_x*sin(ra) - r_y*cos(ra)

    Parameters
    ----------
    dra/ddec : array of float
        offset in right ascension/declination
    ra/dec : array of float
        right ascension/declination in radian
    dra_err/ddec_err : array of float
        formal uncertainties of dra/ddec, default value is None
    corr_arr : array of float
        correlation between dra and ddec, default value is None

    Returns
    -------
    opt : array of (3,)
        estimation of three rotational angles
    sig : array of (3,)
        formal uncertainty of estimations
    corr_mat : array of (3,3)
        correlation coefficients between estimations
    """

    opt, sig, corr_mat = solve_tran_neq1(
        dra, ddec, ra, dec, dra_err, ddec_err, corr_arr)

    # Calculate the residual. ( O - C )
    rsd_ra, rsd_dec = residual_calc1(dRA, dDE, RA, DE, opt)

    if dra_err is None and ddec_err is None:
        # Calculate the a priori reduced Chi-square
        cov_arr = corr_arr * dra_err * ddec_err
        apr_chi2 = calc_chi2_2d(
            dra, dra_err, ddec, ddec_err, cov_arr, reduced=True)

        # Calculate the post-fitting reduced Chi-square
        M = opt.size
        pos_chi2_rdc = calc_chi2_2d(rsd_ra, dra_err, rsd_ddec, ddec_err,
                                    cov_arr, reduced=True,
                                    num_fdm=2*rsd_ra.size-1-M)

        print("# apriori reduced Chi-square for: %10.3f\n"
              "# posteriori reduced Chi-square for: %10.3f" %
              (apr_chi2, pos_chi2_rdc), file=flog)

        # Calculate the goodness-of-fit
        pos_chi2 = calc_chi2_2d(rsd_ra, dra_err, rsd_ddec, ddec_err, cov_arr)
        print("# goodness-of-fit is %10.3f" %
              calc_gof(2*rsd_ra.size-1-M, pos_chi2), file=flog)

        # Rescale the formal errors
        sig = sig * np.sqrt(pos_chi2_rdc)

    return opt, sig, corr_mat


##################  IESR transformation version 03 ##################
def tran_func2(ra, dec, r1, r2, r3, dz):
    """Coordinate transformation function.

    The transformation equation considers a rigid rotation together with
    one declination bias, which could be given by the following
    d_RA^* = -r_x*sin(dec)*cos(ra) - r_y*sin(dec)*sin(ra) + r_z*cos(dec)
    d_DE   = +r_x*sin(ra)          - r_y*cos(ra) + dz

    Parameters
    ----------
    ra/dec : array of float
        right ascension/declination in radian
    r1/r2/r3 : float
        rotational angles around X-, Y-, and Z-axis
    dz : float
        bias in declination

    Returns
    -------
    dra/ddec : array of float
        R.A.(*cos(Dec.))/Dec. differences
    """

    dra = [-r1 * cos(ra) * sin(dec) - r2 *
           sin(ra) * sin(dec) + r3 * cos(dec)][0]
    ddec = [r1 * sin(ra) - r2 * cos(ra) + dz][0]

    return dra, ddec


def jac_mat2(ra, dec):
    """Calculate the Jacobian matrix.

    The transformation equation considers a rigid rotation together with
    one declination bias, which could be given by the following
    d_RA^* = -r_x*sin(dec)*cos(ra) - r_y*sin(dec)*sin(ra) + r_z*cos(dec)
    d_DE   = +r_x*sin(ra)          - r_y*cos(ra) + dz

    Parameters
    ----------
    ra : array of float
        right ascension in radian
    dec : array of float
        declination in radian

    Returns
    -------
    jac_mat : Jacobian matrix
    """

    # Partial deviation of dra and ddec.
    par11 = -sin(dec) * cos(ra)
    par12 = sin(ra)
    par21 = -sin(dec) * sin(ra)
    par22 = -cos(ra)
    par31 = cos(dec)
    par32 = np.zeros_like(dec)
    par41 = np.rad2deg(dec)  # unit: deg
    par42 = np.zeros_like(dec)
    par51 = np.zeros_like(dec)
    par52 = np.rad2deg(dec)  # unit: deg
    par61 = np.zeros_like(dec)
    par62 = np.ones_like(dec)

    par1 = concatenate((par11, par12))
    par2 = concatenate((par21, par22))
    par3 = concatenate((par31, par32))
    par4 = concatenate((par41, par42))
    par5 = concatenate((par51, par52))
    par6 = concatenate((par61, par62))

    # Jacobian matrix.
    jac_mat = np.stack((par1, par2, par3, par4, par5, par6), axis=-1)

    return jac_mat


def solve_tran_neq2(dra, ddec, ra, dec,
                    dra_err=None, ddec_err=None, corr=None):
    """Solve normal equation.

    The transformation equation considers a rigid rotation together with
    one declination bias, which could be given by the following
    d_RA^* = -r_x*sin(dec)*cos(ra) - r_y*sin(dec)*sin(ra) + r_z*cos(dec)
    d_DE   = +r_x*sin(ra)          - r_y*cos(ra) + dz

    Parameters
    ----------
    dra/ddec : array of float
        offset in right ascension/declination
    ra/dec : array of float
        right ascension/declination in radian
    dra_err/ddec_err : array of float
        formal uncertainties of dra/ddec, default value is None
    corr : array of float
        correlation between dra and ddec, default value is None

    Returns
    -------
    opt : array of (4,)
        estimation of three rotational angles and one bias
    sig : array of (4,)
        formal uncertainty of estimations
    corr_mat : array of (4,4)
        correlation coefficients between estimations
    """

    # Jacobian matrix and its transpose.
    jac_mat = jac_mat2(ra, dec)
    jac_mat_t = np.transpose(jac_mat)

    # Weighted matrix.
    wgt_mat = calc_wgt_mat(dra_err, ddec_err, corr)

    # Calculate matrix A and b of matrix equation:
    # A * w = b.
    mat_tmp = np.dot(jac_mat_t, wgt_mat)
    A = np.dot(mat_tmp, jac_mat)
    dpos = concatenate((dra, ddec))
    b = np.dot(mat_tmp,  dpos)

    # Solve the equations.
    #  w = (r1, r2, r3, dz)^T
    w = np.linalg.solve(A, b)

    # Covariance.
    cov_mat = np.linalg.inv(A)
    sig, corr_mat = read_cov_mat(cov_mat)

    # Return the result.
    return w, sig, corr_mat


def residual_calc2(dRA, dDE, RA, DE, param):
    """Calculate the residuals of RA/Dec

    The transformation equation considers a rigid rotation together with
    one declination bias, which could be given by the following
    d_RA^* = -r_x*sin(dec)*cos(ra) - r_y*sin(dec)*sin(ra) + r_z*cos(dec)
    d_DE   = +r_x*sin(ra)          - r_y*cos(ra) + dz

    Parameters
    ----------
    dRA/dDE : array of float
        differences in R.A.(*cos(Dec.))/Dec.
    RA/DE : array of float
        Right ascension/Declination in radian
    param : array of float
        estimation of three rotation angles and one bias

    Returns
    ----------
    ResRA/ResDE : array of float
        residual array of dRA(*cos(Dec))/dDec in uas.
    """

    # Theoritical value
    dra, ddec = tran_func2(RA, DE, *param)

    # Calculate the residual. ( O - C )
    ResRA, ResDE = dRA - dra, dDE - ddec

    return ResRA, ResDE


def tran_fitting2(dra, ddec, ra, dec,
                  dra_err=None, ddec_err=None, corr_arr=None, flog=sys.stdout):
    """Least square fitting of transformation equation.

    The transformation equation considers a rigid rotation together with
    one declination bias, which could be given by the following
    d_RA^* = -r_x*sin(dec)*cos(ra) - r_y*sin(dec)*sin(ra) + r_z*cos(dec)
    d_DE   = +r_x*sin(ra)          - r_y*cos(ra) + dz

    Parameters
    ----------
    dra/ddec : array of float
        offset in right ascension/declination
    ra/dec : array of float
        right ascension/declination in radian
    dra_err/ddec_err : array of float
        formal uncertainties of dra/ddec, default value is None
    corr_arr : array of float
        correlation between dra and ddec, default value is None

    Returns
    -------
    opt : array of (4,)
        estimation of three rotational angles and one bias
    sig : array of (4,)
        formal uncertainty of estimations
    corr_mat : array of (4,4)
        correlation coefficients between estimations
    """

    opt, sig, corr_mat = solve_tran_neq2(
        dra, ddec, ra, dec, dra_err, ddec_err, corr_arr)

    # Calculate the residual. ( O - C )
    rsd_ra, rsd_dec = residual_calc2(dRA, dDE, RA, DE, opt)

    if dra_err is None and ddec_err is None:
        # Calculate the a priori reduced Chi-square
        cov_arr = corr_arr * dra_err * ddec_err
        apr_chi2 = calc_chi2_2d(
            dra, dra_err, ddec, ddec_err, cov_arr, reduced=True)

        # Calculate the post-fitting reduced Chi-square
        M = opt.size
        pos_chi2_rdc = calc_chi2_2d(rsd_ra, dra_err, rsd_ddec, ddec_err,
                                    cov_arr, reduced=True,
                                    num_fdm=2*rsd_ra.size-1-M)

        print("# apriori reduced Chi-square for: %10.3f\n"
              "# posteriori reduced Chi-square for: %10.3f" %
              (apr_chi2, pos_chi2_rdc), file=flog)

        # Calculate the goodness-of-fit
        pos_chi2 = calc_chi2_2d(rsd_ra, dra_err, rsd_ddec, ddec_err, cov_arr)
        print("# goodness-of-fit is %10.3f" %
              calc_gof(2*rsd_ra.size-1-M, pos_chi2), file=flog)

        # Rescale the formal errors
        sig = sig * np.sqrt(pos_chi2_rdc)

    return opt, sig, corr_mat


##################  IESR transformation version 03-00 ##################
def tran_func3_0(ra, dec, r1, r2, r3, d1, d2, b2):
    """Coordinate transformation function.

    The transformation equation considers a rigid rotation together with
    two declination-dependent slopes and one declination bias, which could
    be given by the following
    d_RA^* = -r_x*sin(dec)*cos(ra) - r_y*sin(dec)*sin(ra) + r_z*cos(dec)
             + D_1*(dec-DE0)*cos(dec)
    d_DE   = +r_x*sin(ra)         - r_y*cos(ra)
             + D_2*(dec-DE0) + B_2
    where the reference declination is choosen as DE0 = 0.0.

    Parameters
    ----------
    ra/dec : array of float
        right ascension/declination in radian
    r1/r2/r3 : float
        rotational angles around X-, Y-, and Z-axis
    d1/d2 : float
        two declination-dependent slopes in right ascension/declination
    b2 : float
        one bias in declination

    Returns
    -------
    dra/ddec : array of float
        R.A.(*cos(Dec.))/Dec. differences
    """

    dec0 = 0
    delta_dec = dec - dec0

    dra = [-r1 * cos(ra) * sin(dec) - r2 *
           sin(ra) * sin(dec) + r3 * cos(dec)
           + d1 * delta_dec * cos(dec)][0]
    ddec = [r1 * sin(ra) - r2 * cos(ra)
            + d2 * delta_dec + b2][0]

    return dra, ddec


def jac_mat3_0(ra, dec):
    """Calculate the Jacobian matrix.

    The transformation equation considers a rigid rotation together with
    two declination-dependent slopes and one declination bias, which could
    be given by the following
    d_RA^* = -r_x*sin(dec)*cos(ra) - r_y*sin(dec)*sin(ra) + r_z*cos(dec)
             + D_1*(dec-DE0)*cos(dec)
    d_DE   = +r_x*sin(ra)         - r_y*cos(ra)
             + D_2*(dec-DE0) + B_2
    where the reference declination is choosen as DE0 = 0.0.

    Parameters
    ----------
    ra : array of float
        right ascension in radian
    dec : array of float
        declination in radian

    Returns
    -------
    jac_mat : Jacobian matrix
    """

    # Partial deviation of dra and ddec.
    par11 = -sin(dec) * cos(ra)
    par12 = sin(ra)
    par21 = -sin(dec) * sin(ra)
    par22 = -cos(ra)
    par31 = cos(dec)
    par32 = np.zeros_like(dec)
    par41 = np.rad2deg(dec) * cos(dec)  # unit: deg
    par42 = np.zeros_like(dec)
    par51 = np.zeros_like(dec)
    par52 = np.rad2deg(dec)  # unit: deg
    par61 = np.zeros_like(dec)
    par62 = np.ones_like(dec)

    par1 = concatenate((par11, par12))
    par2 = concatenate((par21, par22))
    par3 = concatenate((par31, par32))
    par4 = concatenate((par41, par42))
    par5 = concatenate((par51, par52))
    par6 = concatenate((par61, par62))

    # Jacobian matrix.
    jac_mat = np.stack((par1, par2, par3, par4, par5, par6), axis=-1)

    return jac_mat


def solve_tran_neq3_0(dra, ddec, ra, dec,
                      dra_err=None, ddec_err=None, corr=None):
    """Solve normal equation.

    The transformation equation considers a rigid rotation together with
    two declination-dependent slopes and one declination bias, which could
    be given by the following
    d_RA^* = -r_x*sin(dec)*cos(ra) - r_y*sin(dec)*sin(ra) + r_z*cos(dec)
             + D_1*(dec-DE0)*cos(dec)
    d_DE   = +r_x*sin(ra)         - r_y*cos(ra)
             + D_2*(dec-DE0) + B_2
    where the reference declination is choosen as DE0 = 0.0.

    Parameters
    ----------
    dra/ddec : array of float
        offset in right ascension/declination
    ra/dec : array of float
        right ascension/declination in radian
    dra_err/ddec_err : array of float
        formal uncertainties of dra/ddec, default value is None
    corr : array of float
        correlation between dra and ddec, default value is None

    Returns
    -------
    opt : array of (6,)
        estimation of three rotational angles, two slopes, and one bias
    sig : array of (6,)
        formal uncertainty of estimations
    corr_mat : array of (6,6)
        correlation coefficients between estimations
    """

    # Jacobian matrix and its transpose.
    jac_mat = jac_mat3_0(ra, dec)
    jac_mat_t = np.transpose(jac_mat)

    # Weighted matrix.
    wgt_mat = calc_wgt_mat(dra_err, ddec_err, corr)

    # Calculate matrix A and b of matrix equation:
    # A * w = b.
    mat_tmp = np.dot(jac_mat_t, wgt_mat)
    A = np.dot(mat_tmp, jac_mat)
    dpos = concatenate((dra, ddec))
    b = np.dot(mat_tmp,  dpos)

    # Solve the equations.
    #  w = (r_x, r_y, r_z, D_1, D_2, B_2)^T
    opt = np.linalg.solve(A, b)

    # Covariance.
    cov_mat = np.linalg.inv(A)
    sig, corr_mat = read_cov_mat(cov_mat)

    # Return the result.
    return opt, sig, corr_mat


def residual_calc3_0(dRA, dDE, RA, DE, param):
    """Calculate the residuals of RA/Dec

    The transformation equation considers a rigid rotation together with
    two declination-dependent slopes and one declination bias, which could
    be given by the following
    d_RA^* = -r_x*sin(dec)*cos(ra) - r_y*sin(dec)*sin(ra) + r_z*cos(dec)
             + D_1*(dec-DE0)*cos(dec)
    d_DE   = +r_x*sin(ra)         - r_y*cos(ra)
             + D_2*(dec-DE0) + B_2
    where the reference declination is choosen as DE0 = 0.0.

    Parameters
    ----------
    dRA/dDE : array of float
        differences in R.A.(*cos(Dec.))/Dec.
    RA/DE : array of float
        Right ascension/Declination in radian
    param : array of float
        estimation of three rotation angles, two solpes, and one bias.

    Returns
    ----------
    ResRA/ResDE : array of float
        residual array of dRA(*cos(Dec))/dDec in uas.
    """

    # Theoritical value
    dra, ddec = tran_func3_0(RA, DE, *param)

    # Calculate the residual. ( O - C )
    ResRA, ResDE = dRA - dra, dDE - ddec

    return ResRA, ResDE


def tran_fitting3_0(dra, ddec, ra, dec,
                    dra_err=None, ddec_err=None, corr_arr=None, flog=sys.stdout):
    """Least square fitting of transformation equation.

    The transformation equation considers a rigid rotation together with
    two declination-dependent slopes and one declination bias, which could
    be given by the following
    d_RA^* = -r_x*sin(dec)*cos(ra) - r_y*sin(dec)*sin(ra) + r_z*cos(dec)
             + D_1*(dec-DE0)*cos(dec)
    d_DE   = +r_x*sin(ra)         - r_y*cos(ra)
             + D_2*(dec-DE0) + B_2
    where the reference declination is choosen as DE0 = 0.0.

    Parameters
    ----------
    dra/ddec : array of float
        offset in right ascension/declination
    ra/dec : array of float
        right ascension/declination in radian
    dra_err/ddec_err : array of float
        formal uncertainties of dra/ddec, default value is None
    corr_arr : array of float
        correlation between dra and ddec, default value is None

    Returns
    -------
    opt : array of (6,)
        estimation of three rotational angles, two slopes, and one bias
    sig : array of (6,)
        formal uncertainty of estimations
    corr_mat : array of (6,6)
        correlation coefficients between estimations
    """

    opt, sig, corr_mat = solve_tran_neq3_0(
        dra, ddec, ra, dec, dra_err, ddec_err, corr_arr)

    # Calculate the residual. ( O - C )
    rsd_ra, rsd_dec = residual_calc3_0(dRA, dDE, RA, DE, opt)

    if dra_err is None and ddec_err is None:
        # Calculate the a priori reduced Chi-square
        cov_arr = corr_arr * dra_err * ddec_err
        apr_chi2 = calc_chi2_2d(
            dra, dra_err, ddec, ddec_err, cov_arr, reduced=True)

        # Calculate the post-fitting reduced Chi-square
        M = opt.size
        pos_chi2_rdc = calc_chi2_2d(rsd_ra, dra_err, rsd_ddec, ddec_err,
                                    cov_arr, reduced=True,
                                    num_fdm=2*rsd_ra.size-1-M)

        print("# apriori reduced Chi-square for: %10.3f\n"
              "# posteriori reduced Chi-square for: %10.3f" %
              (apr_chi2, pos_chi2_rdc), file=flog)

        # Calculate the goodness-of-fit
        pos_chi2 = calc_chi2_2d(rsd_ra, dra_err, rsd_ddec, ddec_err, cov_arr)
        print("# goodness-of-fit is %10.3f" %
              calc_gof(2*rsd_ra.size-1-M, pos_chi2), file=flog)

        # Rescale the formal errors
        sig = sig * np.sqrt(pos_chi2_rdc)

    return opt, sig, corr_mat


##################  IESR transformation version 03 ##################
def tran_func3(ra, dec, r1, r2, r3, d1, d2, b2):
    """Coordinate transformation function.

    The transformation equation considers a rigid rotation together with
    two declination-dependent slopes and one declination bias, which could
    be given by the following
    d_RA^* = -r_x*sin(dec)*cos(ra) - r_y*sin(dec)*sin(ra) + r_z*cos(dec)
             + D_1*(dec-DE0)
    d_DE   = +r_x*sin(ra)         - r_y*cos(ra)
             + D_2*(dec-DE0) + B_2
    where the reference declination is choosen as DE0 = 0.0.
    The difference between v03 and v03-00 is the defination of the slope
    in the right ascension.

    Parameters
    ----------
    ra/dec : array of float
        right ascension/declination in radian
    r1/r2/r3 : float
        rotational angles around X-, Y-, and Z-axis
    d1/d2 : float
        two declination-dependent slopes in right ascension/declination
    b2 : float
        one bias in declination

    Returns
    -------
    dra/ddec : array of float
        R.A.(*cos(Dec.))/Dec. differences
    """

    dec0 = 0
    delta_dec = dec - dec0

    dra = [-r1 * cos(ra) * sin(dec) - r2 *
           sin(ra) * sin(dec) + r3 * cos(dec)
           + d1 * delta_dec][0]
    ddec = [r1 * sin(ra) - r2 * cos(ra)
            + d2 * delta_dec + b2][0]

    return dra, ddec


def jac_mat3(ra, dec):
    """Calculate the Jacobian matrix.

    The transformation equation considers a rigid rotation together with
    two declination-dependent slopes and one declination bias, which could
    be given by the following
    d_RA^* = -r_x*sin(dec)*cos(ra) - r_y*sin(dec)*sin(ra) + r_z*cos(dec)
             + D_1*(dec-DE0)
    d_DE   = +r_x*sin(ra)         - r_y*cos(ra)
             + D_2*(dec-DE0) + B_2
    where the reference declination is choosen as DE0 = 0.0.
    The difference between v03 and v03-00 is the defination of the slope
    in the right ascension.

    Parameters
    ----------
    ra : array of float
        right ascension in radian
    dec : array of float
        declination in radian

    Returns
    -------
    jac_mat : Jacobian matrix
    """

    # Partial deviation of dra and ddec.
    par11 = -sin(dec) * cos(ra)
    par12 = sin(ra)
    par21 = -sin(dec) * sin(ra)
    par22 = -cos(ra)
    par31 = cos(dec)
    par32 = np.zeros_like(dec)
    par41 = np.rad2deg(dec)  # unit: deg
    par42 = np.zeros_like(dec)
    par51 = np.zeros_like(dec)
    par52 = np.rad2deg(dec)  # unit: deg
    par61 = np.zeros_like(dec)
    par62 = np.ones_like(dec)

    par1 = concatenate((par11, par12))
    par2 = concatenate((par21, par22))
    par3 = concatenate((par31, par32))
    par4 = concatenate((par41, par42))
    par5 = concatenate((par51, par52))
    par6 = concatenate((par61, par62))

    # Jacobian matrix.
    jac_mat = np.stack((par1, par2, par3, par4, par5, par6), axis=-1)

    return jac_mat


def solve_tran_neq3(dra, ddec, ra, dec,
                    dra_err=None, ddec_err=None, corr=None):
    """Solve normal equation.

    The transformation equation considers a rigid rotation together with
    two declination-dependent slopes and one declination bias, which could
    be given by the following
    d_RA^* = -r_x*sin(dec)*cos(ra) - r_y*sin(dec)*sin(ra) + r_z*cos(dec)
             + D_1*(dec-DE0)
    d_DE   = +r_x*sin(ra)         - r_y*cos(ra)
             + D_2*(dec-DE0) + B_2
    where the reference declination is choosen as DE0 = 0.0.
    The difference between v03 and v03-00 is the defination of the slope
    in the right ascension.

    Parameters
    ----------
    dra/ddec : array of float
        offset in right ascension/declination
    ra/dec : array of float
        right ascension/declination in radian
    dra_err/ddec_err : array of float
        formal uncertainties of dra/ddec, default value is None
    corr : array of float
        correlation between dra and ddec, default value is None

    Returns
    -------
    opt : array of (6,)
        estimation of three rotational angles, two slopes, and one bias
    sig : array of (6,)
        formal uncertainty of estimations
    corr_mat : array of (6,6)
        correlation coefficients between estimations
    """

    # Jacobian matrix and its transpose.
    jac_mat = jac_mat3(ra, dec)
    jac_mat_t = np.transpose(jac_mat)

    # Weighted matrix.
    wgt_mat = calc_wgt_mat(dra_err, ddec_err, corr)

    # Calculate matrix A and b of matrix equation:
    # A * w = b.
    mat_tmp = np.dot(jac_mat_t, wgt_mat)
    A = np.dot(mat_tmp, jac_mat)
    dpos = concatenate((dra, ddec))
    b = np.dot(mat_tmp,  dpos)

    # Solve the equations.
    #  w = (r1, r2, r3, d1, d2, b2)^T
    w = np.linalg.solve(A, b)

    # Covariance.
    cov_mat = np.linalg.inv(A)
    sig, corr_mat = read_cov_mat(cov_mat)

    # Return the result.
    return w, sig, corr_mat


def residual_calc3(dRA, dDE, RA, DE, param):
    """Calculate the residuals of RA/Dec

    The transformation equation considers a rigid rotation together with
    two declination-dependent slopes and one declination bias, which could
    be given by the following
    d_RA^* = -r_x*sin(dec)*cos(ra) - r_y*sin(dec)*sin(ra) + r_z*cos(dec)
             + D_1*(dec-DE0)*cos(dec)
    d_DE   = +r_x*sin(ra)         - r_y*cos(ra)
             + D_2*(dec-DE0) + B_2
    where the reference declination is choosen as DE0 = 0.0.
    The difference between v03 and v03-00 is the defination of the slope
    in the right ascension.

    Parameters
    ----------
    dRA/dDE : array of float
        differences in R.A.(*cos(Dec.))/Dec.
    RA/DE : array of float
        Right ascension/Declination in radian
    param : array of float
        estimation of three rotation angles, two slopes, and one bias

    Returns
    ----------
    ResRA/ResDE : array of float
        residual array of dRA(*cos(Dec))/dDec in uas.
    """

    # Theoritical value
    dra, ddec = tran_func3(RA, DE, *param)

    # Calculate the residual. ( O - C )
    ResRA, ResDE = dRA - dra, dDE - ddec

    return ResRA, ResDE


def tran_fitting3(dra, ddec, ra, dec,
                  dra_err=None, ddec_err=None, corr_arr=None, flog=sys.stdout):
    """Least square fitting of transformation equation.

    The transformation equation considers a rigid rotation together with
    two declination-dependent slopes and one declination bias, which could
    be given by the following
    d_RA^* = -r_x*sin(dec)*cos(ra) - r_y*sin(dec)*sin(ra) + r_z*cos(dec)
             + D_1*(dec-DE0)*cos(dec)
    d_DE   = +r_x*sin(ra)         - r_y*cos(ra)
             + D_2*(dec-DE0) + B_2
    where the reference declination is choosen as DE0 = 0.0.
    The difference between v03 and v03-00 is the defination of the slope
    in the right ascension.

    Parameters
    ----------
    dra/ddec : array of float
        offset in right ascension/declination
    ra/dec : array of float
        right ascension/declination in radian
    dra_err/ddec_err : array of float
        formal uncertainties of dra/ddec, default value is None
    corr_arr : array of float
        correlation between dra and ddec, default value is None

    Returns
    -------
    opt : array of (6,)
        estimation of three rotational angles, two slopes, and one bias
    sig : array of (6,)
        formal uncertainty of estimations
    corr_mat : array of (6,6)
        correlation coefficients between estimations
    """

    opt, sig, corr_mat = solve_tran_neq3(
        dra, ddec, ra, dec, dra_err, ddec_err, corr_arr)

    # Calculate the residual. ( O - C )
    rsd_ra, rsd_dec = residual_calc3(dRA, dDE, RA, DE, opt)

    if dra_err is None and ddec_err is None:
        # Calculate the a priori reduced Chi-square
        cov_arr = corr_arr * dra_err * ddec_err
        apr_chi2 = calc_chi2_2d(
            dra, dra_err, ddec, ddec_err, cov_arr, reduced=True)

        # Calculate the post-fitting reduced Chi-square
        M = opt.size
        pos_chi2_rdc = calc_chi2_2d(rsd_ra, dra_err, rsd_ddec, ddec_err,
                                    cov_arr, reduced=True,
                                    num_fdm=2*rsd_ra.size-1-M)

        print("# apriori reduced Chi-square for: %10.3f\n"
              "# posteriori reduced Chi-square for: %10.3f" %
              (apr_chi2, pos_chi2_rdc), file=flog)

        # Calculate the goodness-of-fit
        pos_chi2 = calc_chi2_2d(rsd_ra, dra_err, rsd_ddec, ddec_err, cov_arr)
        print("# goodness-of-fit is %10.3f" %
              calc_gof(2*rsd_ra.size-1-M, pos_chi2), file=flog)

        # Rescale the formal errors
        sig = sig * np.sqrt(pos_chi2_rdc)

    return opt, sig, corr_mat


def main():
    pass


if __name__ == "__main__":
    main()

# -------------------- END -----------------------------------
