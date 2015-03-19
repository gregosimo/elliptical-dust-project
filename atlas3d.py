import os

from astropy.table import Table
import matplotlib.pyplot as plt
import numpy as np

import photometry as phot
import WISE_conversions as conv

# Maybe I want to subclass figure later on. But now... meh.
class SED(object):
    '''An object designed to provide an SED plot using internally stored points
    and spectra.'''
    
    def __init__(self, title):
        self.title = title
        self.plotgroups = []

    def add_point(self, pointobject):
        '''Adds a point to the SED.

        A point is a single wavelength/flux value pair. This adds the point to
        an internal list of points so that it can be recalled. Labels cannot be
        used for a single point.
        '''
        self.add_points([pointobject])

    def add_points(self, pointobjects, label=""):
        '''Adds a list of points to the SED.

        This list should normally be associated in some way, such as a group of 
        observations from a given survey. This is mainly for grouping things
        together with meaningful colors. In order to have the group show up in a
        legend, the label should be given.
        '''
        pointobject = SEDEntry(pointobjects, label)
        self.plotgroups.append(pointobject)
        
    def add_spectrum(spec, yspec_no_units=None, label="", xunit=None, 
            yunit=None):
        '''Adds a spectrum to the SED.
        
        A spectrum should ideally be a spectrum object as defined in this 
        module. If a spectrum is not created, add_spectrum should be called with
        the signature add_spectrum(xspec, yspec, xunit=unit, yunit=unit). 
        
        Labels are also allowed for legends.
        '''
        # defspec is an object which we know is definitely a spectrum!
        if not isInstance(spec, Spectrum):
            if xunit is None or yunit is None:
                # Have this inherit from astropy.units.UnitError
                # Used to indicate that you NEED a unit.
                raise UnitMissing
            defspec = Spectrum(spec, yspec_no_units, xunit=xunit, yunit=yunit)
        else:
            defspec=spec
        
        
        

# SED(title)
# SED.add_point(pointobject)
# SED.add_points(pointobjects)
# SED.add_points(pointobjects, label="Survey")
# SED.add_spectrum(spec, label="Spectrum")
# SED.add_spectrum(xspec_no_units, yspec_no_units, xunit=u.Jy, yunit=u.Jy)
# SED.plot()

# W1, W2, W3, W4
WISE_wavelengths = np.array([3.4e-6, 4.6e-6, 12e-6, 22e-6])*1e6
# NUV, FUV
GALEX_wavelengths = np.array([2267e-10, 1516e-10])*1e6
# J, H, Ks
TWOMASS_wavelengths = np.array([1.24e-6, 1.66e-6, 2.16e-6])*1e6

WISE_frequencies = 3e10 / (WISE_wavelengths * 1e-4)
GALEX_frequencies = 3e10 / (GALEX_wavelengths * 1e-4)
TWOMASS_frequencies = 3e10 / (TWOMASS_wavelengths * 1e-4)
TWOMASS_ZP = np.array([1594, 1024, 666.7])

def plot_Conroy_SED(ATLAS3DBASE, atlas3d_sample_row):
    '''Plots an SED against the Conroy model for ATLAS3D targets.'''
    # Generate photometry.
    try:
        W1, W1_err = phot.galaxy_photometry(ATLAS3DBASE, 
            atlas3d_sample_row["objstr_01"], "W1", brightness="flux")
        W2, W2_err = phot.galaxy_photometry(ATLAS3DBASE, 
            atlas3d_sample_row["objstr_01"], "W2", brightness="flux")
        W3, W3_err = phot.galaxy_photometry(ATLAS3DBASE, 
            atlas3d_sample_row["objstr_01"], "W3", brightness="flux")
        W4, W4_err = phot.galaxy_photometry(ATLAS3DBASE, 
            atlas3d_sample_row["objstr_01"], "W4", brightness="flux")
        NUV, NUV_err = phot.galaxy_photometry(ATLAS3DBASE, 
            atlas3d_sample_row["objstr_01"], "NUV", brightness="flux")
        FUV, FUV_err = phot.galaxy_photometry(ATLAS3DBASE, 
            atlas3d_sample_row["objstr_01"], "FUV", brightness="flux")
    except BaseException:
        return

    WISE_flux = np.array([W1, W2, W3, W4])
    WISE_flux_err = np.array([W1_err, W2_err, W3_err, W4_err])
    GALEX_flux = np.array([NUV, FUV])
    GALEX_flux_err = np.array([NUV_err, FUV_err])
    TWOMASS = np.array([atlas3d_sample_row["j_m_k20fe"], 
        atlas3d_sample_row["h_m_k20fe"], atlas3d_sample_row["k_m_k20fe"]])
    TWOMASS_err = np.array([atlas3d_sample_row["j_msig_k20fe"], 
        atlas3d_sample_row["h_msig_k20fe"], atlas3d_sample_row["k_msig_k20fe"]])
    TWOMASS_flux = conv.mag2flux(TWOMASS, 0, TWOMASS_ZP)
    TWOMASS_flux_err = conv.magerr2fluxerr(TWOMASS, TWOMASS_err, 0, 0, 
    TWOMASS_ZP, 0)

    # This is normalized to H-band
    fsps = Table.read(os.path.join(ATLAS3DBASE, "fsps-egals.txt"), format="ascii", 
                      names=["Wave", "F(0.2Gyr)", "F(2Gyr)", "F(5Gyr)", "F(10Gyr)"])
    fsps_02 = fsps["F(0.2Gyr)"] * TWOMASS_flux[1] / fsps["F(0.2Gyr)"][802]
    fsps_2 = fsps["F(2Gyr)"] * TWOMASS_flux[1] / fsps["F(2Gyr)"][802]
    fsps_5 = fsps["F(5Gyr)"] * TWOMASS_flux[1] / fsps["F(5Gyr)"][802]
    fsps_10 = fsps["F(10Gyr)"] * TWOMASS_flux[1] / fsps["F(10Gyr)"][802]
    fsps_frequencies = 3e10 / (fsps["Wave"] * 1e-4)

    plt.plot(fsps["Wave"], fsps_frequencies * fsps_02, 'b-', label="0.2 Gyr")
    plt.plot(fsps["Wave"], fsps_frequencies * fsps_2, 'g-', label="2 Gyr")
    plt.plot(fsps["Wave"], fsps_frequencies * fsps_5, 'r-', label="5 Gyr")
    plt.plot(fsps["Wave"], fsps_frequencies * fsps_10, 'c-', label="10 Gyr")

    plt.errorbar(GALEX_wavelengths, GALEX_flux * GALEX_frequencies, 
        GALEX_flux_err, fmt='ro')
    plt.errorbar(WISE_wavelengths, WISE_flux * WISE_frequencies, WISE_flux_err, 
        fmt='ro')
    plt.errorbar(TWOMASS_wavelengths, TWOMASS_flux * TWOMASS_frequencies, 
        TWOMASS_flux_err, fmt='ro')
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.xlabel("Wavelength (um)")
    plt.ylabel("vFv (Jansky Hz)")
    objname = atlas3d_sample_row["objstr_01"]
    plt.title("SED for {0}; SSP Age: {1} Gyr".format(objname, 
        atlas3d_sample_row["Age_SSP"]))

def read_Krajnovic_Table_D1(
        URL=("/home/regulus/simonian/year1/wise/ATLAS3D_DB/"
             "Krajnovic2011_Atlas3D_Paper2_TableD1.txt")):
    '''Reads the table from Kajnovich 2011

    In particular, this table contains information about dust.'''
    krajnovic_table = Table.read(
        URL, format="ascii.commented_header", guess=False, header_start=-5, 
        data_start=0)
    return krajnovic_table

def read_McDermid_Table_3(
        URL=("/home/regulus/simonian/year1/wise/ATLAS3D_DB/"
             "McDermid2015_Atlas3D_Paper30_Table3.txt")):
    '''Reads in the table from McDermid 2015

    This table contains all of the early-type galaxies in the ATLAS3D sample,
    as well as their properties as measured by the Re aperture.'''
    mcdermid_raw_table = Table.read(
        URL, format="ascii.basic", data_start=0, header_start=None, 
        fill_values=("--", "0"))
    mcdermid_table = Table(
        mcdermid_raw_table[[
            "col1", "col2", "col4", "col5", "col7", "col8", "col10", "col11",
            "col13", "col14", "col16", "col17", "col19", "col20", "col22",
            "col23"]], 
        names=(
            "Name", "Hbeta", "Hbeta_err", "Fe5015", "Fe5015_err", "Mgb",
            "Mgb_err", "Fe5270", "Fe5270_err", "Age_SSP", "Age_SSP_err",
            "[Z/H]_SSP", "[Z/H]_SSP_er", "[a/Fe]_SSP", "[a/Fe]_SSP_err",
            "Quality"))
    # The plus/minus symbols really ruin this command...
    # mcdermid_table = Table.read(
    #    URL, format="ascii.commented_header", guess=False, header_start=-4, 
    #    data_start=0)
    return mcdermid_table


def get_dustless_galaxies(krajnovic_table=None):
    '''Gets dustless galaxies in ATLAS3D. 

    If the krajnovic table is provided, it will filter it. Otherwise, it will
    return its own table.'''
    if krajnovic_table is None:
        krajnovic_table = read_Krajnovic_Table_D1()
    dustless = phot.astropy_table_row(krajnovic_table, "dust", ["N"])
    return dustless

def filter_ATLAS3D_table_for_dustless_galaxies(atlas3d_table):
    '''Picks out dustless galaxies from the atlas3d_table.

    Takes the atlas3d table and uses the Krajnovic et al 2011 table D1 to pick
    out the galaxies without signs of diffuse dust.'''
    dustless_galaxies = get_dustless_galaxies()

    filteredtable = phot.extract_subtable_from_column(
        atlas3d_table, "objstr_01", dustless_galaxies["name"])
    return filteredtable

def filter_out_bad_targets(atlas3d_table):
    '''Objects which cause errors for some reason or another.

    I should find the root cause of these problems, but this will help pick
    them out and skip them for now.
    '''
    newtable = atlas3d_table.copy()
    ###########################################################################
    # Insert exclusion rules here #

    ###########################################################################
    return newtable

