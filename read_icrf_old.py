#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File name: read_icrf2.py
"""
Created on Mon Apr  9 11:45:31 2018

@author: Neo(liuniu@smail.nju.edu.cn)
"""

import numpy as np
from numpy import genfromtxt, cos, deg2rad


__all__ = ["read_icrf1", "read_icrf1_cat", "read_icrf2"]


# -----------------------------  FUNCTIONS -----------------------------
def read_icrf1(icrf1_file="/Users/Neo/Astronomy/Data/catalogs/icrf/"
               "rsc95r01.dat", unit_pos_as=False, arc_err_flag=True):
    '''Read ICRF1 catalog.

    Parameters
    ----------
    icrf1_file : string
        ICRF1 data file
    unit_pos_as : Boolean
        flag to determine if the unit of RA./Dec. is arc-sec.
    arc_err_flag : Boolean
        flag to determine if returning the formal uncertainty of RA or
        RA*cos(Dec.). True for RA*cos(Dec.) while False for RA.


    Returns
    ----------
    icrf_name : array of string
        ICRF designation of source name;
    iers_name : array of string
        IERS designation of source name;
    RA/Dec : array of float
        Right Ascension/Declination in degree;
    e_RA/e_DE : array of float
        formal error of RA/Dec in milliarcsecond;
    corr : array of float between [-1, +1]
        correlation coefficient between RA and Dec in ICRF2 solution;
    X : array of int
        X band structure index
    S : array of int
        S band structure index
    Ep : array of float
        mean epoch of source position in Modified Julian day;
    Of : array of float
        First date of observation (MJD)
    Ol : array of float
        Last Julian date of observation (MJD)
    Nsess : array of float
        Number of observing sessions
    Ndel : array of float
        Number of delay, delay rate obs. pairs
    c : array of character, [DCO] Defining / Candidate / Other
        category of source
    '''

    # empty lists to store data
    icrf_name = []
    iers_name = []
    RAh = []
    RAm = []
    RAs = []
    Dec_sign = []
    DEd = []
    DEm = []
    DEs = []
    e_RAs = []
    e_DEs = []
    corr = []
    X = []
    S = []
    Ep = []
    Of = []
    Ol = []
    Nsess = []
    Ndel = []
    c = []

    for line in open(icrf1_file, "r"):
        icrf_name.append(line[6:22])
        iers_name.append(line[25:33])
        c.append(line[34])

        # Sturcture index
        if line[35] is ' ':
            X.append(0)
        else:
            X.append(int(line[35]))

        if line[37] is ' ':
            S.append(0)
        else:
            S.append(int(line[37]))

        # Right ascension
        RAh.append(int(line[42:44]))
        RAm.append(int(line[45:47]))
        RAs.append(float(line[48:57]))
        # Declination
        if line[59] is " ":
            Dec_sign.append(1)
        else:
            Dec_sign.append(-1)
        DEd.append(int(line[60:62]))
        DEm.append(int(line[63:65]))
        DEs.append(float(line[66:74]))

        e_RAs.append(float(line[77:85]))
        e_DEs.append(float(line[87:94]))

        if line[96:101] == " " * 5:
            corr.append(0.)
        else:
            corr.append(float(line[96:101]))

        # Epoch of observations
        Ep.append(float(line[102:111]))
        Of.append(float(line[112:121]))
        Ol.append(float(line[122:131]))

        # Number of observations
        Nsess.append(int(line[133:137]))
        Ndel.append(int(line[138:144]))

    # list -> array
    icrf_name = np.asarray(icrf_name)
    iers_name = np.asarray(iers_name)
    RAh = np.asarray(RAh)
    RAm = np.asarray(RAm)
    RAs = np.asarray(RAs)
    Dec_sign = np.asarray(Dec_sign)
    DEd = np.asarray(DEd)
    DEm = np.asarray(DEm)
    DEs = np.asarray(DEs)
    e_RAs = np.asarray(e_RAs)
    e_DEs = np.asarray(e_DEs)
    corr = np.asarray(corr)
    X = np.asarray(X)
    S = np.asarray(S)
    Ep = np.asarray(Ep)
    Of = np.asarray(Of)
    Ol = np.asarray(Ol)
    Nsess = np.asarray(Nsess)
    Ndel = np.asarray(Ndel)
    c = np.asarray(c)

    # calculate the position
    RA = (RAh + RAm / 60.0 + RAs / 3600) * 15  # degree
    Dec = Dec_sign * (DEd + DEm / 60.0 + DEs / 3600)  # degree

    if unit_pos_as:
        deg2as = 3.6e3
        RA = RA * deg2as
        Dec = Dec * deg2as

    # unit: as -> mas
    if arc_err_flag:
        e_RA = e_RAs * 15e3 * cos(np.deg2rad(Dec))
    else:
        e_RA = e_RAs * 15e3
    e_DE = e_DEs * 1.e3

    # JD -> MJD
    Ep = Ep - 2400000.5
    Of = Of - 2400000.5
    Ol = Ol - 2400000.5

    return [icrf_name, iers_name, RA, Dec, e_RA, e_DE, corr, c,
            X, S, Ep, Of, Ol, Nsess, Ndel]


def read_icrf1_pos(icrf1_file="/Users/Neo/Astronomy/Data/catalogs/icrf/"
                   "rsc95r01.dat", unit_pos_as=False, arc_err_flag=True):
    '''Read ICRF2 catalog.

    Parameters
    ----------
    icrf1_file : string
        ICRF1 data file
    unit_pos_as : Boolean
        flag to determine if the unit of RA./Dec. is arc-sec.
    arc_err_flag : Boolean
        flag to determine if returning the formal uncertainty of RA or
        RA*cos(Dec.). True for RA*cos(Dec.) while False for RA.


    Returns
    ----------
    icrf_name : array of string
        ICRF designation of source name;
    iers_name : array of string
        IERS designation of source name;
    RA/Dec : array of float
        Right Ascension/Declination in degree;
    e_RA/e_DE : array of float
        formal error of RA/Dec in milliarcsecond;
    corr : array of float between [-1, +1]
        correlation coefficient between RA and Dec in ICRF2 solution;
    c : array of character, [DCO] Defining / Candidate / Other
        category of source
    '''

    [icrf_name, iers_name, RA, Dec, e_RA, e_DE, corr, c,
     X, S, Ep, Of, Ol, Nsess, Ndel] = read_icrf1(icrf1_file)

    return [icrf_name, iers_name, RA, Dec, e_RA, e_DE, corr, c]


def rewrite_icrf1(ifile="/Users/Neo/Astronomy/Data/catalogs/icrf/rsc95r01.dat",
                  ofile="/Users/Neo/Astronomy/Data/catalogs/icrf/icrf1.cat"):
    """Rewrite the icrf1 catalog.

    Parameters
    ----------
    ifile : string
        file name with full path of icrf1 catalog
    ofile : string


    Returns
    ----------
    None
    """

    [icrf_name, iers_name, RA, Dec, e_RA, e_DE, corr, c,
     X, S, Ep, Of, Ol, Nsess, Ndel] = read_icrf1()

    fout = open(ofile, "w")
    print("# VLBI Celestial Reference Frame Solution ICRF1\n"
          "#\n"
          "# Columns  Units   Meaning\n"
          "#    1     --      ICRF designation\n"
          "#    2     --      IERS designation\n"
          "#    3     deg     Right ascension\n"
          "#    4     deg     Declination\n"
          "#    5     mas     Formal uncertainty of the right ascension "
          "(*cos(Dec))\n"
          "#    6     mas     Formal uncertainty of the declination\n"
          "#    7     --      Correlation between right ascension and "
          "declination\n"
          "#    8     days    Average epoch of observation(MJD)\n"
          "#    9     days    First epoch of observation(MJD)\n"
          "#   10     days    Last epoch of observation(MJD)\n"
          "#   11     --      Number of sessions\n"
          "#   12     --      Number of delays\n"
          "#   13     --      Number of delay rates\n"
          "#   14     --      Source Catagory ([DCO] Defining / Candidate "
          "/ Other)\n"
          "#   15     --      Structure index at X band\n"
          "#   16     --      Structure index at S band\n"
          "# Created date: %s\n#"
          % time.strftime("%d/%m/%Y", time.localtime()), file=fout)

    for (ICRFi, IERSi, RAi, Deci, e_RAi, e_DEi, corri, ci,
         Xi, Si, Epi, Ofi, Oli, Nsessi, Ndeli) in zip(
            icrf_name, iers_name, RA, Dec, e_RA, e_DE, corr, c,
            X, S, Ep, Of, Ol, Nsess, Ndel):

        print("%-8s  %8s  %14.10f  %+14.10f  %10.4f  %10.4f  %+10.4f  "
              "%7.2f  %7.2f  %7.2f  %5d  %10d  0  %s  %5.3f  %5.3f"
              % (ICRFi, IERSi, RAi, Deci, e_RAi, e_DEi, corri,
                 Epi, Ofi, Oli, Nsessi, Ndeli, ci, Xi, Si), file=fout)

    fout.close()


def read_icrf1_cat(icrf1_file="/Users/Neo/Astronomy/Data/catalogs/icrf/"
                   "icrf1.cat", unit_pos_as=False, arc_err_flag=True):
    '''Read ICRF1 catalog.

    Parameters
    ----------
    icrf1_file : string
        ICRF1 data file
    unit_pos_as : Boolean
        flag to determine if the unit of RA./Dec. is arc-sec.
    arc_err_flag : Boolean
        flag to determine if returning the formal uncertainty of RA or
        RA*cos(Dec.). True for RA*cos(Dec.) while False for RA.


    Returns
    ----------
    icrf_name : array of string
        ICRF designation of source name;
    iers_name : array of string
        IERS designation of source name;
    RA/Dec : array of float
        Right Ascension/Declination in degree;
    e_RA/e_DE : array of float
        formal error of RA/Dec in milliarcsecond;
    corr : array of float between [-1, +1]
        correlation coefficient between RA and Dec in ICRF2 solution;
    c : array of character, [DCO] Defining / Candidate / Other
        category of source
    '''

    icrf_name, iers_name, c = np.genfromtxt(icrf1_file, dtype=str,
                                            usecols=(0, 1, 13), unpack=True)
    RA, Dec, e_RA, e_DE, corr, c, Ep, Of, Ol = np.genfromtxt(
        icrf1_file, usecols=range(3, 10), unpack=True)

    Nsess, Ndel = np.genfromtxt(icrf1_file,  dtype=int,
                                usecols=(10, 11), unpack=True)

    return [icrf_name, iers_name, RA, Dec, e_RA, e_DE, corr, c,
            X, S, Ep, Of, Ol, Nsess, Ndel]


def read_icrf2(icrf2_file="/Users/Neo/Astronomy/Data/catalogs/"
               "icrf/icrf2.dat", unit_pos_as=False, arc_err_flag=True):
    '''Read ICRF2 catalog.

    Parameters
    ----------
    icrf2_file : string
        ICRF2 data file
    unit_pos_as : Boolean
        flag to determine if the unit of RA./Dec. is arc-sec.
    arc_err_flag : Boolean
        flag to determine if returning the formal uncertainty of RA or
        RA*cos(Dec.). True for RA*cos(Dec.) while False for RA.


    Returns
    ----------
    icrfn : array of string
        ICRF designation of source name;
    ivsn : array of string
        IVS designation of source name;
    iersn : array of string
        IERS designation of source name;
    RA/Dec : array of float
        Right Ascension/Declination in degree;
    e_RA/e_DE : array of float
        formal error of RA/Dec in milliarcsecond;
    cor : array of float between [-1, +1]
        correlation coefficient between RA and Dec in ICRF2 solution;
    mepo : array of float
        mean epoch of source position in Modified Julian day;
    Flag : array of character
        flag of source classification in ICRF2 catalog.
    '''

    icrfn, ivsn, iersn, Flag = genfromtxt(
        icrf2_file, usecols=(0, 1, 2, 3), dtype=str, unpack=True)
    RAh, RAm, RAs, Decd, Decm, Decs, e_RAs, e_DEas, cor = genfromtxt(
        icrf2_file, usecols=range(4, 13), unpack=True)

    mepo = genfromtxt(icrf2_file, usecols=(13,))

# determine the sign of Declination
    strDecs = genfromtxt(icrf2_file, usecols=(7,), dtype=str)
    signstr = [x[0] for x in strDecs]
    Dec_sign = np.where(np.array(signstr) == '-', -1, 1)

# calculate the position
    RA = (RAh + RAm / 60.0 + RAs / 3600) * 15  # degree
    Dec = Decd + Dec_sign * (Decm / 60.0 + Decs / 3600)  # degree

# unit: as -> mas
    if arc_err_flag:
        e_RA = e_RAs * 15e3 * cos(np.deg2rad(Dec))
    else:
        e_RA = e_RAs * 15e3
    e_DE = e_DEas * 1.0e3

    if unit_pos_as:
        deg2as = 3.6e3
        RA = RA * deg2as
        Dec = Dec * deg2as

    return icrfn, ivsn, iersn, RA, Dec, e_RA, e_DE, cor, mepo, Flag

# --------------------------------- END --------------------------------
