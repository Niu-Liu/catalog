# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 16:00:08 2017

Positional transformation.

@author: Neo

Oct 29, 2018: re-write all the codes
Apr 19, 2021: use 'curve_fit' to do the lsq fitting
"""

from scipy.optimize import curve_fit
import numpy as np
from numpy import sin, cos, pi, concatenate
# My modules
# from .cov_mat import calc_wgt_mat, read_cov_mat
from .stats_calc import calc_chi2_2d, calc_gof


__all__ = ["tran_fitting1", "tran_fitting2",
           "tran_fitting3", "tran_fitting3_0"]


# ###################### IERS Transformation Functions ######################
def rotation1(X, **kwargs):
    """A sample code

    Parameters
    ----------
    x : 1-D array
        (ra, dec) in radian
    """

    ra, dec = X
    dra = -locals["r1"] * cos(ra) * sin(dec) - locals["r2"] * sin(ra) * sin(dec) + \
        locals["r3"] * cos(dec)
    ddec = + locals["r1"] * sin(ra) - locals["r2"] * cos(ra)

    return np.concatenate((dra, ddec))


def tran_func1(pos, rx, ry, rz):
    """IERS Coordinate transformation function version 01.

    The transformation function considering onl the rigid rotation
    is given by
    d_RA^* = -r_x*sin(dec)*cos(ra) - r_y*sin(dec)*sin(ra) + r_z*cos(dec)
    d_DE   = +r_x*sin(ra) - r_y*cos(ra)

    Parameters
    ----------
    ra/dec : array of float
        right ascension/declination in radian
    rx/ry/rz : float
        rotational angles around X-, Y-, and Z-axis

    Returns
    -------
    dra/ddec : array of float
        R.A.(*cos(Dec.))/Dec. differences
    """

    ra, dec = pos

    dra = -rx * cos(ra) * sin(dec) - ry * sin(ra) * sin(dec) + rz * cos(dec)
    ddec = rx * sin(ra) - ry * cos(ra)
    dpos = np.concatenate((dra, ddec))

    return dpos


def tran_func2(pos, rx, ry, rz, dz):
    """IERS Coordinate transformation function version 02.

    The transformation equation considers a rigid rotation together with
    one declination bias, which could be given by the following
    d_RA^* = -r_x*sin(dec)*cos(ra) - r_y*sin(dec)*sin(ra) + r_z*cos(dec)
    d_DE   = +r_x*sin(ra)          - r_y*cos(ra) + dz

    Parameters
    ----------
    ra/dec : array of float
        right ascension/declination in radian
    rx/ry/rz : float
        rotational angles around X-, Y-, and Z-axis
    dz : float
        bias in declination

    Returns
    -------
    dra/ddec : array of float
        R.A.(*cos(Dec.))/Dec. differences
    """

    ra, dec = pos

    dra = [-rx * cos(ra) * sin(dec) - ry *
           sin(ra) * sin(dec) + rz * cos(dec)][0]
    ddec = [rx * sin(ra) - ry * cos(ra) + dz][0]
    dpos = np.concatenate((dra, ddec))

    return dpos


def tran_func3a(pos, rx, ry, rz, d1, d2, b2):
    """IERS Coordinate transformation function version 03(a).

    The transformation equation considers a rigid rotation together with
    two declination-dependent slopes and one declination bias, which could
    be given by the following
    d_RA^* = -r_x*sin(dec)*cos(ra) - r_y*sin(dec)*sin(ra) + r_z*cos(dec)
             + D_1*(dec-DE0)*cos(dec)
    d_DE   = +r_x*sin(ra)         - r_y*cos(ra)
             + D_2*(dec-DE0) + B_2
    where the reference declination is choosen as DE0 = 0.0.

    Parameters ----------
    ra/dec : array of float right ascension/declination in radian
    rx/ry/rz : float rotational angles around X-, Y-, and Z-axis
    d1/d2 : float
        two declination-dependent slopes in right ascension/declination
    b2 : float
        one bias in declination

    Returns
    -------
    dra/ddec : array of float
        R.A.(*cos(Dec.))/Dec. differences
    """

    ra, dec = pos
    dec0 = 0
    delta_dec = dec - dec0

    dra = [-rx * cos(ra) * sin(dec) - ry *
           sin(ra) * sin(dec) + rz * cos(dec)
           + d1 * delta_dec * cos(dec)][0]
    ddec = [rx * sin(ra) - ry * cos(ra)
            + d2 * delta_dec + b2][0]
    dpos = np.concatenate((dra, ddec))

    return dpos


def tran_func3b(pos, rx, ry, rz, d1, d2, b2):
    """IERS Coordinate transformation function version 03(b).

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
    rx/ry/rz : float
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

    ra, dec = pos
    dec0 = 0
    delta_dec = dec - dec0

    dra = [-rx * cos(ra) * sin(dec) - ry *
           sin(ra) * sin(dec) + rz * cos(dec)
           + d1 * delta_dec][0]
    ddec = [rx * sin(ra) - ry * cos(ra)
            + d2 * delta_dec + b2][0]
    dpos = np.concatenate((dra, ddec))

    return dpos


# ############################ Resolve Results ############################
def resolve_result(popt, pcov, tran_type):
    """Resolve the result

    """

    res = {}

    if tran_type == "1":
        para_name = ["rx", "ry", "rz"]
    elif tran_type == "2":
        para_name = ["rx", "ry", "rz", "dz"]
    elif tran_type in ["3", "3a", "3b"]:
        para_name = ["rx", "ry", "rz", "d1", "d2", "dz"]
    else:
        print("Undefined tran_type")
        os.system(1)

    for i, par in enumerate(para_name):
        res[par] = popt[i]
        res[par+"_err"] = np.sqrt(pcov[i, i])

    for i, pari in enumerate(para_name):
        for j, parj in enumerate(para_name[i+1:]):
            res[pari+parj+"_cor"] = pcov[i, j] / \
                res[pari+"_err"] / res[parj+"_err"]

    return res


# ################################ LSQ fitting ################################
def iers_tran_fitting(dra, ddec, ra, dec,
                      dra_err=None, ddec_err=None, corr_arr=None, flog=sys.stdout,
                      unit_deg=True, tran_type="1"):
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

    # Degree -> radian
    if unit_deg:
        ra = np.deg2rad(ra)
        dec = np.deg2rad(dec)

    # Position vector
    pos = np.concatenate((ra, dec))
    dpos = np.concatenate((dra, ddec))
    N = len(dra)

    # Create covariance matrix
    if dra_err is not None:
        cov_mat = np.diag(np.concatenate((dra_err ** 2, ddec_err**2)))

        if dra_ddec_cor is not None:
            # Consider the correlation between dra and ddec
            dra_ddec_cov = dra_err * ddec_err * dra_ddec_cor
            for i, dra_ddec_covi in enumerate(dra_ddec_cov):
                cov_mat[i, i+N] = dra_ddec_covi
                cov_mat[i+N, i] = dra_ddec_covi

    # Do the LSQ fit
    if tran_type == "1":
        trans_func = tran_func1
    elif tran_type == "2":
        trans_func = tran_func2
    elif tran_type == "3a":
        trans_func = tran_func3a
    elif tran_type in ["3", "3b"]:
        trans_func = tran_func3b
    else:
        print("Undefined tran_type!")
        os.system(1)

    if dra_err is None:
        popt, pcov = curve_fit(trans_func, pos, dpos)
    else:
        popt, pcov = curve_fit(trans_func, pos, dpos,
                               sigma=cov_mat, absolute_sigma=False)

    M = len(popt)
    res = resolve_result(popt, pcov, tran_type)

    # Prediction
    predict = trans_func(pos, *popt)
    res["predict_ra"] = predict[:N]
    res["predict_dec"] = predict[N:]

    # Residual
    res["residual_ra"] = dra - res["predict_ra"]
    res["residual_dec"] = ddec - res["predict_dec"]
    rsd_ra, rsd_ddec = res["residual_ra"], res["residual_dec"]

    # Chi squared per dof
    if dra_err is None:
        # Calculate chi2/ndof
        res["chi2_dof_ra"] = np.sum(res["residual_ra"]) / (N - M)
        res["chi2_dof_dec"] = np.sum(res["residual_dec"]) / (N - M)

        unit_wgt = np.ones_like(dra)
        zero_cov = np.zeros_like(dra)
        apr_chi2_rdc = calc_chi2_2d(dra, unit_wgt, ddec, unit_wgt, zero_cov,
                                    reduced=True, num_fdm=2*N-1-M)
        pos_chi2_rdc = calc_chi2_2d(dra, unit_wgt, ddec, unit_wgt, zero_cov,
                                    reduced=True, num_fdm=2*N-1-M)
        pos_chi2 = calc_chi2_2d(dra, unit_wgt, ddec, unit_wgt, zero_cov)

    else:
        # Calculate chi2/ndof
        res["chi2_dof_ra"] = np.sum(res["residual_ra"] / dra_err) / (N - M)
        res["chi2_dof_dec"] = np.sum(res["residual_dec"] / ddec_err) / (N - M)

        apr_chi2_rdc = calc_chi2_2d(dra, dra_err, ddec, ddec_err, dra_ddec_cov,
                                    reduced=True, num_fdm=2*N-1-M)
        pos_chi2_rdc = calc_chi2_2d(rsd_ra, dra_err, rsd_ddec, ddec_err,
                                    dra_ddec_cov, reduced=True, num_fdm=2*N-1-M)

        pos_chi2 = calc_chi2_2d(rsd_ra, dra_err, rsd_ddec, ddec_err, dra_ddec_cov)

    print("# apriori reduced Chi-square for: %10.3f\n"
          "# posteriori reduced Chi-square for: %10.3f" %
          (apr_chi2_rdc, pos_chi2_rdc), file=flog)

    # Calculate the goodness-of-fit
    print("# goodness-of-fit is %10.3f" %
          calc_gof(2*N-1-M, pos_chi2), file=flog)

    return res


def main():
    pass


if __name__ == "__main__":
    main()

# -------------------- END -----------------------------------
