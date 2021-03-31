# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# File name: VSH_analysis.py
"""
Created on Thu Jan  4 12:17:39 2018

@author: Neo(liuniu@smail.nju.edu.cn)

History
N. Liu, 10 Feb 2018: change the input parameters of function
                     'vsh_analysis', replacing variable 'datafile'
                     with 'pos_offset'.
N. Liu, 12 Feb 2018: add three parameters 'X_a', 'X_d', 'X' to the input
                     variables of functions 'catalog_comparison_VSH',
                     'VSH_analysis', 'apply_condition';
                     modified function 'print_outlier' to print the
                     normalized seperation information.

"""

import numpy as np
from numpy import sin, cos, pi, concatenate
import time
from os import path
# My modules
# from .vsh_deg1_cor import VSHdeg01_fitting
# from vsh_deg2_cor import VSHdeg02_fitting
# from tex_table import write_result_deg1, write_result_deg2
# from vector_direction import vec6_calc
# from glide_calc import GA_glide_decomposed
# sin = np.sin
# cos = np.cos

from .vsh_deg1_cor import jac_mat_deg01, wgt_mat
from .vsh_deg2_cor import jac_mat_deg02

#
# __all__ = ["write_residual", "print_outlier", "VSH_analysis",
#            "apply_condition", "catalog_comparison_VSH",
#            "vsh_analysis"]


# -----------------------------  FUNCTIONS -----------------------------
# def write_residual(
#         RA, DE, ERRdRA, ERRdDE, RdRA, RdDE, flg, fname):
#
#     DATA = np.transpose((RA, DE, ERRdRA, ERRdDE, RdRA, RdDE, flg))
#     np.savetxt(
#         fname, DATA, fmt="%5.1f " * 2 + "%9.1f " * 4 + "%d",
#         delimiter=",",
#         header="Residual after the VSH fitting\n"
#         "RdRAig  RdDcig\n"
#         "suffix: 'ig' for ICRF2 - gsf2016g\n"
#         "%s" % time.strftime('##%Y-%m-%d %H:%M:%S Begins!',
#                              time.localtime(time.time())))
#
#
# def print_outlier(source, ang_sep, X_a, X_d, X, ind_outl, flog):
#     '''
#     '''
#
#     outliers = np.extract(ind_outl, source)
#     X_a1 = np.extract(ind_outl, X_a)
#     X_d1 = np.extract(ind_outl, X_d)
#     X1 = np.extract(ind_outl, X)
#
#     print('## %d   Outliers: \n'
#           '##  Source     ang_sep    X_a    X_d    X' % outliers.size,
#           file=flog)
#
#     for (outlier, ang_sepi, X_ai, X_di, Xi) in zip(
#             outliers, ang_sep, X_a1, X_d1, X1):
#         print('## %10s  %+8.1f  %+8.3f  %+8.3f  %7.3f' %
#               (outlier, ang_sepi, X_ai, X_di, Xi), file=flog)
#
#
# def outlier_elim_separation(pos_offset, X_max=4.1, ang_sep_max=10,
#                             file_handle=None):
#     """
#     """
#
#     [sou, RAdeg, DEdeg, d_RA, e_dRA, d_DE, e_dDE, dra_ddec_cov,
#      ang_sep, X_a, X_d, X, flg] = pos_offset
#
#     # Outlier elimination
#     if X_max is None:
#         con = ang_sep <= ang_sep_max
#     elif ang_sep is None:
#         con = X <= X_max
#     else:
#         con = (X <= X_max) & (ang_sep <= ang_sep_max)
#
#     # Extract good observations
#     sous = sou[con]
#     RAdegs = RAdeg[con]
#     DEdegs = DEdeg[con]
#     d_RAs = d_RA[con]
#     d_DEs = d_DE[con]
#     e_dRAs = e_dRA[con]
#     e_dDEs = e_dDE[con]
#     dra_ddec_covs = dra_ddec_cov[con]
#     ang_seps = ang_sep[con]
#     X_as = X_a[con]
#     X_ds = X_d[con]
#     Xs = X[con]
#     flgs = flg[con]
#
#     # pack them
#     pos_offset1 = [sous, RAdegs, DEdegs,
#                    d_RAs, e_dRAs, d_DEs, e_dDEs, dra_ddec_covs,
#                    ang_seps, X_as, X_ds, Xs, flgs]
#
#     # write outliers informationipython
#     if file_handle is not None:
#         con_out = (X > X_max) | (ang_sep > ang_sep_max)
#         outliers = sou[con_out]
#         ang_sep1 = ang_sep[con_out]
#         X_a1 = X_a[con_out]
#         X_d1 = X_d[con_out]
#         X1 = X[con_out]
#
#         # print
#         print('## %d   Outliers: \n'
#               '##  Source     ang_sep    X_a    X_d    X' % outliers.size,
#               file=file_handle)
#         for (outlier, ang_sepi, X_ai, X_di, Xi) in zip(
#                 outliers, ang_sep1, X_a1, X_d1, X1):
#             print('## %10s  %8.3f  %+8.3f  %+8.3f  %7.3f' %
#                   (outlier, ang_sepi, X_ai, X_di, Xi), file=file_handle)
#
#     return pos_offset1
#
#
# # def VSH_analysis(sou, d_RA, d_DE, e_dRA, e_dDE, dra_ddec_cov, RArad, DErad,
# #                  flog, ftex, ang_sep, X_a, X_d, X):
# def VSH_analysis(sou, d_RA, d_DE, e_dRA, e_dDE, dra_ddec_cov, RArad, DErad,
#                  flog, ang_sep, X_a, X_d, X):
#     '''
#     '''
#
#     # Name of estimated parameters.
#     x1name = ['G_1', 'G_2', 'G_3', 'R_1', 'R_2', 'R_3']
#     x2name = ['E_{2,2}^{\\rm Re}', 'E_{2,2}^{\\rm Im}',
#               'E_{2,1}^{\\rm Re}', 'E_{2,1}^{\\rm Im}', 'E_{2,0}',
#               'M_{2,2}^{\\rm Re}', 'M_{2,2}^{\\rm Im}',
#               'M_{2,1}^{\\rm Re}', 'M_{2,1}^{\\rm Im}', 'M_{2,0}']
#
#     x1, sig1, corr1, ind_outl1, RdRA1, RdDE1 = VSHdeg01_fitting(
#         d_RA, d_DE, e_dRA, e_dDE, RArad, DErad, flog,
#         cov=dra_ddec_cov, elim_flag="none")
#
#     # unpack the results
#     gx,  gy,  gz,  wx,  wy,  wz = x1
#     egx, egy, egz, ewx, ewy, ewz = sig1
#     (r1, alr1, der1, errr1, erralr1, errder1,
#         g1, alg1, deg1, errg1, erralg1, errdeg1) = vec6_calc(x1, sig1)
#
#     # Print the result
#     # write_result_deg1(x1name, x1, sig1, corr1, flog)
#
#     # For log file.
#     print('#### for degree 1:\n',
#           '## Rotation component:\n',
#           ' %+4d +/- %3d |' * 3 % (wx, ewx, wy, ewy, wz, ewz),
#           '=> %4d +/- %3d' % (r1, errr1), file=flog)
#     print('##       apex: (%.1f +/- %.1f, %.1f +/- %.1f)' %
#           (alr1, erralr1, der1, errder1), file=flog)
#     print('## Glide component:\n',
#           ' %+4d +/- %3d |' * 3 % (gx, egx, gy, egy, gz, egz),
#           '=> %4d +/- %3d' % (g1, errg1), file=flog)
#     print('##       apex: (%.1f +/- %.1f, %.1f +/- %.1f)' %
#           (alg1, erralg1, deg1, errdeg1), file=flog)
#     print('##   correlation coefficients are:\n', corr1, file=flog)
#
#     # For grep the result in log file
#     print("#used_for_grep_vsh01  rx rx_err ry ry_err rz rz_err "
#           "r r_err gx gx_err gy gy_err gz gz_err g g_err\n"
#           "used_for_grep_vsh01",
#           ("  %.0f  %.0f" * 8) %
#           (wx, ewx, wy, ewy, wz, ewz, r1, errr1,
#            gx, egx, gy, egy, gz, egz, g1, errg1), file=flog)
#
#     # Print the outliers
#     # print_outlier(sou, ang_sep, X_a, X_d, X, ind_outl1, flog)
#
#     # # # For tex file.
#     # print('## for degree 1:\n',
#     #       '## Rotation component:\n',
#     #       ' $%+4d \\pm$ %3d &' * 3 % (wx, ewx, wy, ewy, wz, ewz),
#     #       ' $%4d \\pm$ %3d &(%.0f $\\pm$ %.0f, $%+.0f \\pm$ %.0f)' %
#     #       (r1, errr1, alr1, erralr1, der1, errder1), file=ftex)
#     # print('## Glide component:\n',
#     #       ' $%+4d \\pm$ %3d &' * 3 % (gx, egx, gy, egy, gz, egz),
#     #       ' $%4d \\pm$ %3d &(%.0f $\\pm$ %.0f, $%+.0f \\pm$ %.0f)' %
#     #       (g1, errg1, alg1, erralg1, deg1, errdeg1), file=ftex)
#     # print("## for degree 1:\n",
#     #       # "&"
#     #       " &$%+4d \\pm$ %3d " * 3 % (wx, ewx, wy, ewy, wz, ewz),
#     #       " &$%4d \\pm$ %3d " % (r1, errr1),
#     #       " &$%+4d \\pm$ %3d " * 3 % (gx, egx, gy, egy, gz, egz),
#     #       " &$%4d \\pm$ %3d &(%.0f $\\pm$ %.0f, $%+.0f \\pm$ %.0f) \\\\" %
#     #       (g1, errg1, alg1, erralg1, deg1, errdeg1), file=ftex)
#
#     # write_result_deg1(x1name, x1, sig1, corr1, ftex)
#
#     # Component towards Galactic-center and non-GA component
#     [G_GA, err_GA, G_NonGA, RA_NonGA, DC_NonGA,
#      errG_NonGA, errRA_NonGA, errDC_NonGA] = GA_glide_decomposed(
#         x1[:3], sig1[:3])
#
#     # For grep the result in log file
#     print("#used_for_grep_vsh01GA  g_ga  g_ga_err  g_noga  g_noga_err  "
#           "ra_noga  ra_noga_err  dec_noga  dec_noga_err\n"
#           "used_for_grep_vsh01GA",
#           ("  %.0f  %.0f" * 4) %
#           (G_GA, err_GA, G_NonGA, errG_NonGA,
#            RA_NonGA, errRA_NonGA, DC_NonGA, errDC_NonGA), file=flog)
#
#     # print("## GA &non-GA component:\n"
#     #       "# GA  non-GA  apex:\n"
#     #       "&$%+4d \\pm$ %3d &$%+4d \\pm$ %3d "
#     #       "&(%.0f $\\pm$ %.0f, $%+.0f \\pm$ %.0f)" %
#     #       (G_GA, err_GA, G_NonGA, errG_NonGA,
#     #        RA_NonGA, errRA_NonGA, DC_NonGA, errDC_NonGA), file=ftex)
#     # print("## for degree 1:\n",
#     #       " &$%+4d \\pm$ %3d " * 3 % (gx, egx, gy, egy, gz, egz),
#     #       " &$%4d \\pm$ %3d &$%4d \\pm$ %3d &$%4d \\pm$ %3d" %
#     #       (g1, errg1, G_GA, err_GA, G_NonGA, errG_NonGA),
#     #       "&(%.0f $\\pm$ %.0f, $%+.0f \\pm$ %.0f) \\\\" %
#     #       (RA_NonGA, errRA_NonGA, DC_NonGA, errDC_NonGA), file=ftex)
#
#     x2, sig2, corr2, ind_outl2, RdRA2, RdDE2 = VSHdeg02_fitting(
#         d_RA, d_DE, e_dRA, e_dDE, RArad, DErad, flog,
#         cov=dra_ddec_cov, elim_flag="none")
#     [gx,  gy,  gz,  wx,  wy,  wz] = x2[:6]
#     [egx, egy, egz, ewx, ewy, ewz] = sig2[:6]
#     (r2, alr2, der2, errr2, erralr2, errder2,
#         g2, alg2, deg2, errg2, erralg2, errdeg2) = vec6_calc(
#         x2[:6], sig2[:6])
#
# # Print the result
#     # For log file.
#     print('#### for degree 2:\n',
#           '## Rotation component:\n',
#           ' %+4d +/- %3d |' * 3 % (wx, ewx, wy, ewy, wz, ewz),
#           '=> %4d +/- %3d' % (r2, errr2), file=flog)
#     print('##       apex: (%.1f +/- %.1f, %.1f +/- %.1f)' %
#           (alr2, erralr2, der2, errder2), file=flog)
#     print('## Glide component:\n',
#           ' %+4d +/- %3d |' * 3 % (gx, egx, gy, egy, gz, egz),
#           '=> %4d +/- %3d' % (g2, errg2), file=flog)
#     print('##       apex: (%.1f +/- %.1f, %.1f +/- %.1f)' %
#           (alg2, erralg2, deg2, errdeg2), file=flog)
#     print('## quadrupole component:\n',   x2[6:], file=flog)
#     print('## formal uncertainties:\n', sig2[6:], file=flog)
#     print('##   correlation coefficients are:\n', corr2, file=flog)
#
#     # Print the outliers
#     # print_outlier(sou, ang_sep, X_a, X_d, X, ind_outl2, flog)
#
#     # For grep the result in log file
#     print("#used_for_grep_vsh02_1  rx rx_err ry ry_err rz rz_err "
#           "r r_err gx gx_err gy gy_err gz gz_err g g_err\n"
#           "used_for_grep_vsh02_1",
#           ("  %.0f  %.0f" * 8) %
#           (wx, ewx, wy, ewy, wz, ewz, r2, errr2,
#            gx, egx, gy, egy, gz, egz, g2, errg2), file=flog)
#
#     # Component towards Galactic-center and non-GA component
#     [G_GA, err_GA, G_NonGA, RA_NonGA, DC_NonGA,
#      errG_NonGA, errRA_NonGA, errDC_NonGA] = GA_glide_decomposed(
#         x2[:3], sig2[:3])
#
#     # For grep the result in log file
#     print("#used_for_grep_vsh02GA  g_ga  g_ga_err  g_noga  g_noga_err  "
#           "ra_noga  ra_noga_err  dec_noga  dec_noga_err\n"
#           "used_for_grep_vsh02GA",
#           ("  %.0f  %.0f" * 4) %
#           (G_GA, err_GA, G_NonGA, errG_NonGA,
#            RA_NonGA, errRA_NonGA, DC_NonGA, errDC_NonGA), file=flog)
#
#     # For grep the result in log file
#     (ER_22, EI_22, ER_21, EI_21, E_20,
#      MR_22, MI_22, MR_21, MI_21, M_20) = x2[6:]
#     (ER_22_err, EI_22_err, ER_21_err, EI_21_err, E_20_err,
#      MR_22_err, MI_22_err, MR_21_err, MI_21_err, M_20_err) = sig2[6:]
#     print("#used_for_grep_vsh02_2E  "
#           "R22  I22  R21  I21  20  "
#           "R22_err  I22_err  R21_err  I21_err  20_err\n"
#           "used_for_grep_vsh02_2E",
#           ("  %.0f" * 16) %
#           (ER_22, EI_22, ER_21, EI_21, E_20,
#            ER_22_err, EI_22_err, ER_21_err, EI_21_err, E_20_err), file=flog)
#     print("#used_for_grep_vsh02_2M  "
#           "R22  I22  R21  I21  20  "
#           "R22_err  I22_err  R21_err  I21_err  20_err\n"
#           "used_for_grep_vsh02_2M",
#           ("  %.0f" * 16) %
#           (MR_22, MI_22, MR_21, MI_21, M_20,
#            MR_22_err, MI_22_err, MR_21_err, MI_21_err, M_20_err), file=flog)
#
# #     # For tex file.
# #     print('## for degree 2:\n',
# #           '## Rotation component:\n',
# #           ' $%+4d \pm$ %3d &' * 3 % (wx, ewx, wy, ewy, wz, ewz),
# #           ' $%4d \pm$ %3d &(%.0f $\pm$ %.0f, $%+.0f \pm$ %.0f)' %
# #           (r2, errr2, alr2, erralr2, der2, errder2), file=ftex)
# #     print('## Glide component:\n',
# #           ' $%+4d \pm$ %3d &' * 3 % (gx, egx, gy, egy, gz, egz),
# #           ' $%4d \pm$ %3d &(%.0f $\pm$ %.0f, $%+.0f \pm$ %.0f)' %
# #           (g2, errg2, alg2, erralg2, deg2, errdeg2), file=ftex)
# #     write_result_deg2(x1name, x2name, x2, sig2, corr2, ftex)
#
#     # Return the residual
#     # return RdRA1, RdDE1, RdRA1, RdDE1
#     # return RdRA1, RdDE1, RdRA2, RdDE2
#     # return RdRA1, RdDE1
#
#
# def vsh_analysis(pos_offset, datafile,
#                  # vlbi2
#                  # main_dir="/home/nliu/solutions/GalacticAberration",
#                  # My MacOS
#                  main_dir="/Users/Neo/Astronomy/Works/201711_GDR2_ICRF3",
#                  label=''):
#     '''
#     '''
#
#     # main_dir = "/home/nliu/solutions/GalacticAberration"
#
#     # Log file.
#     FLOG = open("%s/logs/%s_vsh.log" % (main_dir, label), "w")
#     print('## LOG FILE\n'
#           '## Data: %s \n%s' %
#           (datafile, time.strftime('##%Y-%m-%d %H:%M:%S Begins!',
#                                    time.localtime(time.time()))),
#           file=FLOG)
#
#     # # Log file of tex table.
#     # FTEX = open("%s/logs/%s_vsh.tex" % (main_dir, label), "w")
#     # print('## LOG FILE\n'
#     #       '## The result of different kinds of transformation\n',
#     #       '## Data: %s \n%s' %
#     #       (datafile, time.strftime('## %Y-%m-%d %H:%M:%S Begins!',
#     #                                time.localtime(time.time()))),
#     #       file=FTEX)
#
#     # eject some outliers
#     # pos_offset = outlier_elim_separation(pos_offset, X_max=5.0,
#     # ang_sep_max=10, file_handle=FLOG)
#     N = pos_offset[0].size
#     pos_offset = outlier_elim_separation(
#         pos_offset, X_max=4.1, ang_sep_max=10)
#     # pos_offset = outlier_elim_separation(pos_offset, X_max=None,
#     #                                      ang_sep_max=10)
#
#     [sou, RAdeg, DEdeg, d_RA, e_dRA, d_DE, e_dDE, dra_ddec_cov,
#      ang_sep, X_a, X_d, X, flg] = pos_offset
#
#     RArad, DErad = np.deg2rad(RAdeg), np.deg2rad(DEdeg)  # Unit: rad
#
#     print('###########################################', file=FLOG)
#     print("# Sample size: ", sou.size,
#           "| Outlier: ", N - sou.size, file=FLOG)
#
#     # print('###########################################', file=FTEX)
#     # print("# Sample size:", sou.size,
#     #       "| Outlier: ", N - sou.size, file=FTEX)
#
#     # mas -> uas
#     d_RA, d_DE = d_RA * 1.e3, d_DE * 1.e3
#     e_dRA, e_dDE = e_dRA * 1.e3, e_dDE * 1.e3
#     dra_ddec_cov = dra_ddec_cov * 1.e6
#
#     # VSH_analysis(
#     #     sou, d_RA, d_DE, e_dRA, e_dDE, dra_ddec_cov,
#     #     RArad, DErad, FLOG, FTEX,
#     #     ang_sep, X_a, X_d, X)
#     VSH_analysis(
#         sou, d_RA, d_DE, e_dRA, e_dDE, dra_ddec_cov,
#         RArad, DErad, FLOG,
#         ang_sep, X_a, X_d, X)
#

# ---------------------------------------------------
def cov_matrix_calc(e_dRA, e_dDE, RA, DE, cov=None, deg=1, fit_type="full"):
    """Cacluate the covariance matrix for analysis.

    Parameters
    ----------
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
    """

    # Jacobian matrix and its transpose.
    if deg == 1:
        JacMat, JacMatT = jac_mat_deg01(RA, DE, fit_type)
    elif deg == 2:
        # JacMat, JacMatT = jac_mat_deg02(RA, DE, fit_type)
        JacMat, JacMatT = jac_mat_deg02(RA, DE)

    # Weighted matrix.
    WgtMat = wgt_mat(e_dRA, e_dDE, cov)

    # Calculate matrix A:
    A = np.dot(np.dot(JacMatT, WgtMat), JacMat)

    return A


# ---------------------------------------------------
def cov_matrix_deconv(A):
    """Deconvolve the covariance matrix for analysis.

    Parameters
    ----------
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
    """

    # Covariance.
    pcov = np.linalg.inv(A)
    sig = np.sqrt(pcov.diagonal())
    N = len(sig)

    # Correlation coefficient.
    corr_mat = np.array([pcov[i, j] / sig[i] / sig[j]
                         for j in range(N) for i in range(N)])
    corr_mat.resize((N, N))

    return sig, corr_mat


def covariance_analysis(e_dRA, e_dDE, RA, DE, cov=None, deg=1, fit_type="full"):
    """Estimate the uncertainty of VSH parameters

    Parameters
    ----------
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
    -------
    sigma: array
        formal uncertainty
    corr_mat : matrix
        corre coefficients
    """

    A = cov_matrix_calc(e_dRA, e_dDE, RA, DE, cov, deg, fit_type)

    sig, corr_mat = cov_matrix_deconv(A)

    return sig, corr_mat
# --------------------------------- END --------------------------------
