#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File name: read_gaia.py
"""
Created on Thu Oct  4 16:00:02 2018

@author: Neo(liuniu@smail.nju.edu.cn)
"""

from astropy.table import Table, Column
from astropy import units as u
# from astropy.coordinates import SkyCoord
import numpy as np

# My modules
from .pos_err import error_ellipse_calc
from .get_dir import get_data_dir
from .pos_diff import nor_sep_calc
from .ga_func import gaia_ga_corr

__all__ = ["read_dr1_qso", "read_dr2_iers", "read_dr2_allwise",
           "read_edr3_agn", "read_edr3_crf", "read_edr3_icrf_sou"]


# -----------------------------  FUNCTIONS -----------------------------
def read_dr1_qso(dr1_qso_file=None):
    """Read the position information of Gaia DR1 quasar auxiliary solution.

    Parameter
    ---------
    dr1_qso_file : string
        file name and path of the Gaia DR1 quasar auxiliary solution

    Return
    ------
    gdr1 : an astropy.Table object
        data in the catalog
    """

    if dr1_qso_file is None:
        data_dir = get_data_dir()
        dr1_qso_file = "{}/gaia/dr1/qso.dat".format(data_dir)

    gdr1 = Table.read(dr1_qso_file, format="ascii.fixed_width_no_header",
                      names=["solution_id", "source_id", "ref_epoch",
                             "ra", "ra_err", "dec", "dec_err",
                             "ra_dec_corr", "phot_g_mean_mag",
                             "astrometric_priors_used", "icrf2_match",
                             "rot_flag"])

    # Add unit information
    gdr1["ra_err"].unit = u.mas
    gdr1["dec_err"].unit = u.mas

    # Calculate the semi-major axis of error ellipse
    pos_err, pos_err_min, pa = error_ellipse_calc(
        gdr1["ra_err"], gdr1["dec_err"], gdr1["ra_dec_corr"])
    del pos_err_min

    # Add the semi-major axis of error ellipse to the table
    pos_err = Column(pos_err, name="pos_err", unit=u.mas)
    pa = Column(pa, name="eepa", unit=u.deg)
    gdr1.add_columns([pos_err, pa], indexes=[9, 9])

    return gdr1


def modify_dr2_iers(dr2_qso_file=None):
    """Correct the wrong name in the Gaia-CRF2 subset.

    Parameter
    ---------
    dr2_qso_file : string
        file name and path of the Gaia DR2 auxiliary IERS catalog

    Return
    ------
    gdr2 : an astropy.Table object
        data in the catalog
    """

    if dr2_qso_file is None:
        data_dir = get_data_dir()
        dr2_qso_file = "{}/gaia/dr2/gaiadr2_iers0.fits".format(data_dir)

    # Read Gaia DR2 IERS quasar data
    gdr2 = Table.read(dr2_qso_file)

    # There are two small errore in the colomn iers_name in this sample
    # 0548+37A --> 0548+377
    # 1954+188 --> 1954+187

    oldnames = ["0548+37A", "1954+188"]
    newnames = ["0548+377", "1954+187"]

    for oldname, newname in zip(oldnames, newnames):
        idx = np.where(gdr2["iers_name"] == oldname)[0][0]
        gdr2[idx]["iers_name"] = newname

    gdr2.write(dr2_qso_file, overwrite=True)


def read_dr2_iers(dr2_qso_file=None, err_scaling=False):
    """Read the positional information of Gaia DR2 auxiliary IERS catalog.

    Parameter
    ---------
    dr2_qso_file : string
        file name and path of the Gaia DR2 auxiliary IERS catalog
    err_scaling : boolean
        if true, scale gaia position error by chi2/ndf

    Return
    ------
    gdr2 : an astropy.Table object
        data in the catalog
    """

    if dr2_qso_file is None:
        data_dir = get_data_dir()
        dr2_qso_file = "{}/gaia/dr2/gaiadr2_iers.fits".format(data_dir)

    # Read Gaia DR2 IERS quasar data
    gdr2 = Table.read(dr2_qso_file)

    # There are two small errore in the colomn iers_name in this sample
    # 0548+37A --> 0548+377
    # 1954+188 --> 1954+187

    # oldnames = ["0548+37A", "1954+188"]
    # newnames = ["0548+377", "1954+187"]
    #
    # for oldname, newname in zip(oldnames, newnames):
    #     idx = np.where(gdr2["iers_name"] == oldname)[0][0]
    #     gdr2[idx]["iers_name"] = newname

    # Only the positional information are kept.
    gdr2.keep_columns(["iers_name", "source_id",
                       "ra", "ra_error", "dec", "dec_error",
                       "parallax", "parallax_error",
                       "pmra", "pmra_error", "pmdec", "pmdec_error",
                       "ra_dec_corr", "ra_parallax_corr", "ra_pmra_corr",
                       "ra_pmdec_corr", "dec_parallax_corr", "dec_pmra_corr",
                       "dec_pmdec_corr", "parallax_pmra_corr",
                       "parallax_pmdec_corr", "pmra_pmdec_corr",
                       "phot_g_mean_mag", "phot_bp_mean_mag", "phot_rp_mean_mag",
                       "bp_rp", "astrometric_n_obs_al", "astrometric_n_good_obs_al",
                       "astrometric_matched_observations", "astrometric_chi2_al",
                       "astrometric_params_solved"])

    # Rename the column names
    gdr2.rename_column("ra_error", "ra_err")
    gdr2.rename_column("dec_error", "dec_err")
    gdr2.rename_column("parallax_error", "parallax_err")
    gdr2.rename_column("pmra_error", "pmra_err")
    gdr2.rename_column("pmdec_error", "pmdec_err")

    # Determine whether to scale the position error
    if err_scaling:
        # Degree of freedom = #observation - #parameter
        dof = gdr2["astrometric_n_good_obs_al"] - \
            gdr2["astrometric_params_solved"]
        scale = np.sqrt(gdr2["astrometric_chi2_al"] / dof)

        gdr2["ra_err"] = gdr2["ra_err"] * scale
        gdr2["dec_err"] = gdr2["dec_err"] * scale

    # Calculate the semi-major axis of error ellipse
    pos_err_max, pos_err_min, pa = error_ellipse_calc(
        gdr2["ra_err"], gdr2["dec_err"], gdr2["ra_dec_corr"])

    # Add the semi-major axis of error ellipse to the table
    pos_err_max = Column(pos_err_max, name="pos_err_max", unit=u.mas)
    pos_err_min = Column(pos_err_min, name="pos_err_min", unit=u.mas)
    pa = Column(pa, name="eepa", unit=u.deg)
    gdr2.add_columns([pos_err_max, pos_err_min, pa], indexes=[9, 9, 9])

    return gdr2


def read_dr2_allwise(dr2_qso_file=None):
    """Read the positional information of Gaia DR2 auxiliary AllWISE catalog.

    Parameter
    ---------
    dr2_qso_file : string
        file name and path of the Gaia DR2 auxiliary AllWISE catalog

    Return
    ------
    gdr2 : an astropy.Table object
        data in the catalog
    """

    if dr2_qso_file is None:
        data_dir = get_data_dir()
        dr2_qso_file = "{}/gaia/dr2/gaiadr2_qso_all.fits".format(data_dir)

    # Read Gaia DR2 quasar data
    gdr2 = Table.read(dr2_qso_file)

    # Rename the column names
    gdr2.rename_column("ra_error", "ra_err")
    gdr2.rename_column("dec_error", "dec_err")
    gdr2.rename_column("parallax_error", "parallax_err")
    gdr2.rename_column("pmra_error", "pmra_err")
    gdr2.rename_column("pmdec_error", "pmdec_err")

    # Calculate the semi-major axis of error ellipse
    pos_err_max, pos_err_min, pa = error_ellipse_calc(
        gdr2["ra_err"], gdr2["dec_err"], gdr2["ra_dec_corr"])

    # Add the semi-major axis of error ellipse to the table
    pos_err_max = Column(pos_err_max, name="pos_err_max", unit=u.mas)
    pos_err_min = Column(pos_err_min, name="pos_err_min", unit=u.mas)
    pa = Column(pa, name="eepa", unit=u.deg)
    gdr2.add_columns([pos_err_max, pos_err_min, pa])

    return gdr2


def read_edr3_agn(edr3_qso_file=None, calc_nor_pm=False):
    """Read positional information for AGN in Gaia edr3 catalog.

    Parameter
    ---------
    edr3_qso_file : string
        file name and path
    calc_nor_pm: boolean
        flag to tell if calculate the normalized proper motion

    Return
    ------
    gedr3 : an astropy.Table object
        data in the catalog
    """

    if edr3_qso_file is None:
        data_dir = get_data_dir()
        edr3_qso_file = "{}/gaia/edr3/gedr3_agn.fits".format(data_dir)

    # Read Gaia edr3 quasar data
    gedr3 = Table.read(edr3_qso_file)

    # Rename the column names
    gedr3.rename_column("ra_error", "ra_err")
    gedr3.rename_column("dec_error", "dec_err")
    gedr3.rename_column("parallax_error", "parallax_err")
    gedr3.rename_column("pmra_error", "pmra_err")
    gedr3.rename_column("pmdec_error", "pmdec_err")

    # Calculate the semi-major axis of error ellipse
    pos_err_max, pos_err_min, pa = error_ellipse_calc(
        gedr3["ra_err"], gedr3["dec_err"], gedr3["ra_dec_corr"])

    # Add the semi-major axis of error ellipse to the table
    pos_err_max = Column(pos_err_max, name="pos_err_max", unit=u.mas)
    pos_err_min = Column(pos_err_min, name="pos_err_min", unit=u.mas)
    pa = Column(pa, name="eepa", unit=u.deg)
    gedr3.add_columns([pos_err_max, pos_err_min, pa])

    # Calculate the normalized proper motion
    # This is a quantity similar to normalized separation
    if calc_nor_pm:
        data = nor_sep_calc(gedr3["pmra"], gedr3["pmra_err"],
                            gedr3["pmdec"], gedr3["pmdec_err"], gedr3["pmra_pmdec_corr"])
        nor_pm = Column(data[3], name="nor_pm", unit=None)
        gedr3.add_column(nor_pm)

        # GA-effect correction
        gedr3 = gaia_ga_corr(gedr3)

    return gedr3


def read_edr3_crf(edr3_qso_file=None, table_type="all", calc_pos_err=False,
                  calc_nor_pm=False, only_used_sou=False):
    """Read data for Gaia-CRF3 sources in Gaia edr3 catalog.

    Parameter
    ---------
    edr3_qso_file : string
        file name and path
    table_type : string
        flag to tell if return subset, could be "all", "orientation", "spin"
    calc_pos_err: Boolean
        flag to tell if calculate the position error ellipse parameters
    calc_nor_pm: boolean
        flag to tell if calculate the normalized proper motion
    only_used_sou : boolean
        flag to decide whether to keep only those sources used for calculating
        the orientation and spin parameters for Gaia-CRF3

    Return
    ------
    gedr3 : an astropy.Table object
        data in the catalog
    """

    if edr3_qso_file is None:
        data_dir = get_data_dir()
        edr3_qso_file = "{}/gaia/edr3/gedr3_frame_rotator_source.fits".format(
            data_dir)

    # Read Gaia edr3 quasar data
    gedr3 = Table.read(edr3_qso_file)

    # Rename the column names
    gedr3.rename_column("ra_error", "ra_err")
    gedr3.rename_column("dec_error", "dec_err")
    gedr3.rename_column("parallax_error", "parallax_err")
    gedr3.rename_column("pmra_error", "pmra_err")
    gedr3.rename_column("pmdec_error", "pmdec_err")

    # Check which subset should be used
    if table_type == "orientation":
        if only_used_sou:
            mask = gedr3["used_for_reference_frame_orientation"]
        else:
            mask = gedr3["considered_for_reference_frame_orientation"]

    elif table_type == "spin":
        if only_used_sou:
            mask = gedr3["used_for_reference_frame_spin"]
        else:
            mask = gedr3["considered_for_reference_frame_spin"]

    else:
        if only_used_sou:
            mask1 = gedr3["used_for_reference_frame_orientation"]
            mask2 = gedr3["used_for_reference_frame_spin"]
        else:
            mask1 = gedr3["considered_for_reference_frame_orientation"]
            mask2 = gedr3["considered_for_reference_frame_spin"]
        mask = (mask1 | mask2)

    gedr3 = gedr3[mask]

    # Calculate the normalized proper motion
    # This is a quantity similar to normalized separation
    if calc_nor_pm:
        data = nor_sep_calc(gedr3["pmra"], gedr3["pmra_err"],
                            gedr3["pmdec"], gedr3["pmdec_err"],
                            gedr3["pmra_pmdec_corr"])
        nor_pm = Column(data[3], name="nor_pm", unit=None)
        gedr3.add_column(nor_pm)

        # GA-effect correction
        gedr3 = gaia_ga_corr(gedr3)

    # Calculate the semi-major axis of error ellipse for position
    if calc_pos_err:
        pos_err_max, pos_err_min, pa = error_ellipse_calc(
            gedr3["ra_err"], gedr3["dec_err"], gedr3["ra_dec_corr"])

        # Add the semi-major axis of error ellipse to the table
        pos_err_max = Column(pos_err_max, name="pos_err_max", unit=u.mas)
        pos_err_min = Column(pos_err_min, name="pos_err_min", unit=u.mas)
        pa = Column(pa, name="eepa", unit=u.deg)
        gedr3.add_columns([pos_err_max, pos_err_min, pa])

    return gedr3


def read_edr3_icrf_sou(calc_pos_err=False, calc_nor_pm=False):
    """Read data for ICRF3 sources in Gaia edr3 catalog.

    Parameter
    ---------
    calc_pos_err: Boolean
        flag to tell if calculate the position error ellipse parameters
    calc_nor_pm: boolean
        flag to tell if calculate the normalized proper motion

    Return
    ------
    gedr3 : an astropy.Table object
        data in the catalog
    """

    data_dir = get_data_dir()
    edr3_qso_file = "{}/gaia/edr3/gaia_edr3_icrf3_source.fits".format(
        data_dir)

    # Read Gaia edr3 quasar data
    gedr3 = Table.read(edr3_qso_file)

    # Rename the column names
    gedr3.rename_column("ra_error", "ra_err")
    gedr3.rename_column("dec_error", "dec_err")
    gedr3.rename_column("parallax_error", "parallax_err")
    gedr3.rename_column("pmra_error", "pmra_err")
    gedr3.rename_column("pmdec_error", "pmdec_err")

    if calc_pos_err:
        # Calculate the semi-major axis of error ellipse for position
        pos_err_max, pos_err_min, pa = error_ellipse_calc(
            gedr3["ra_err"], gedr3["dec_err"], gedr3["ra_dec_corr"])

        # Add the semi-major axis of error ellipse to the table
        pos_err_max = Column(pos_err_max, name="pos_err_max", unit=u.mas)
        pos_err_min = Column(pos_err_min, name="pos_err_min", unit=u.mas)
        pa = Column(pa, name="eepa", unit=u.deg)
        gedr3.add_columns([pos_err_max, pos_err_min, pa])

    # Calculate the normalized proper motion
    # This is a quantity similar to normalized separation
    if calc_nor_pm:
        data = nor_sep_calc(gedr3["pmra"], gedr3["pmra_err"],
                            gedr3["pmdec"], gedr3["pmdec_err"],
                            gedr3["pmra_pmdec_corr"])
        nor_pm = Column(data[3], name="nor_pm", unit=None)
        gedr3.add_column(nor_pm)

        # GA-effect correction
        gedr3 = gaia_ga_corr(gedr3)

    return gedr3


if __name__ == "__main__":
    modify_dr2_iers()
# --------------------------------- END --------------------------------
