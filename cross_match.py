#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File name: list_crossmatch.py
"""
Created on Fri Apr 27 00:24:29 2018

@author: Neo(liuniu@smail.nju.edu.cn)
"""

import numpy as np


__all__ = ["list_crossmatch", "pos_Xmatch"]


# -----------------------------  FUNCTIONS -----------------------------
def list_crossmatch(X1, X2):
    '''Corssmatch between two list.

    Parameters
    ----------
    X1 : array_like
        first dataset, shape(N1, D)
    X2 : array_like
        second dataset, shape(N2, D)

    Returns
    -------
    '''

    # =======
    # Add some codes here to eliminate duplicate elemnets
    # =======

    com_list = []
    index1 = []
    index2 = []

    for i, x1 in enumerate(X1):
        indarr = np.where(X2 == x1)[0]

        if indarr:
            com_list.append(x1)
            index1.append(i)
            j = indarr[0]
            index2.append(j)

    com_list = np.asarray(com_list, dtype=str)
    index1 = np.asarray(index1, dtype=int)
    index2 = np.asarray(index2, dtype=int)

    return com_list, index1, index2


def position_taken(index, RA, RAc_err, Dec, Dec_err, cor):
    '''Extract the elements from array at specific index.

    Parameters
    ----------
    index :
        the indice corresponding to common sources
    RA / Dec :
        Right Ascension / Declination for all sources, degreees
    RAc_err / Dec_err :
        formal uncertainty of RA*cos(Dec) / Dec for all sources, micro-as.
    cor :
        correlation coeffient between RA and Dec for all sources.

    Returns
    ----------
    RAn / Decn :
        Right Ascension / Declination for common sources, degreees
    RAc_errn / Dec_err :
        formal uncertainty of RA*cos(Dec) / Dec for common sources, micro-as.
    corn :
        correlation coeffient between RA and Dec for common sources.
    '''

    RAn = np.take(RA, index)
    RAc_errn = np.take(RAc_err, index)
    Decn = np.take(Dec, index)
    Dec_errn = np.take(Dec_err, index)
    corn = np.take(cor, index)

    return RAn, RAc_errn, Decn, Dec_errn, corn


def pos_Xmatch(sou1, RA1, RAc_err1, Dec1, Dec_err1, cor1,
               sou2, RA2, RAc_err2, Dec2, Dec_err2, cor2):
    '''Crossmatch between Gaia and VLBI catalogs.

    Parameters
    ----------
    sou :
        source name (ICRF designation)
    RA / Dec :
        Right Ascension / Declination, degreees
    RAc_err / DC_err :
        formal uncertainty of RA*cos(Dec) / Dec, micro-as.
    cor :
        correlation coeffient between RA and Dec.

    Returns
    ----------
    soucom :
        name (ICRF designation) of common sources
    RAn / Decn :
        Right Ascension / Declination for common sources, degreees
    RAc_errn / Dec_err :
        formal uncertainty of RA*cos(Dec) / Dec for common sources, micro-as.
    cor :
        correlation coeffient between RA and Dec for common sources.
    '''

    soucom = []
    index1 = []
    index2 = []

    for i, soui in enumerate(sou1):
        indarr = np.where(sou2 == soui)[0]

        if indarr:
            soucom.append(soui)
            index1.append(i)
            j = indarr[0]
            index2.append(j)

    RA1n, RAc_err1n, Dec1n, Dec_err1n, cor1n = position_taken(
        index1, RA1, RAc_err1, Dec1, Dec_err1, cor1)
    RA2n, RAc_err2n, Dec2n, Dec_err2n, cor2n = position_taken(
        index2, RA2, RAc_err2, Dec2, Dec_err2, cor2)

    # list -> array
    soucom = np.asarray(soucom)

    return [soucom,
            RA1n, RAc_err1n, Dec1n, Dec_err1n, cor1n,
            RA2n, RAc_err2n, Dec2n, Dec_err2n, cor2n]


# --------------------------------- MAIN -------------------------------
def main():
    """Main function.
    """

    print("This code is just a module of functions.")


if __name__ == "__main__":
    main()
# --------------------------------- END --------------------------------
