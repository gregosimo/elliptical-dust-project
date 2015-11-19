import os
from itertools import izip

from astropy.table import Table, vstack
import matplotlib.pyplot as plt
import numpy as np
import scipy.special

import photometry as phot
import band_conversions as conv
import statop as stat

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

def plot_dustless_galaxy_histogram(
        ages, title="Age Histogram", age_label="t_SSP (Gyr)", label=""):
    '''Creates a bar plot for ages of galaxies.
    '''
    plt.hist(ages, bins=5, range=(0.1, 14), rwidth=0.95, label=label)
    plt.xlabel(age_label)
    plt.ylabel("N")
    plt.title(title)

def atlas3d_ml_to_wise_ml(log_atlas3d_ml, log_atlas3d_lum, w1_mag, distance,
                          log_atlas3d_ml_err, log_atlas3d_lum_err, w1_mag_err,
                          distance_err):
    '''Converts the (M/L) in r-band for ATLAS3D to W1 with K20fe.
    
    Takes the mass-to-light ratio given in the ATLAS3D papers and performs a
    transformation to W1 as well as an aperture transformation. This function
    assumes that the mass-to-light ratio is constant over the entire galaxy,
    which is not an unreasonable assumption. The ATLAS3D luminosity provided is
    that in Paper XV in solar luminosities. The WISE magnitude should be that
    measured in the ATLAS3D aperture, NOT the K20 aperture, and should be given
    in AB magnitudes. Finally, in order to transition between 
    luminosity and magnitudes, the distance to the object (in Mpc) is needed.

    Handing errors is not as straightforward as it might seem. The
    uncertainties that go into the mass-to-light ratio should be the modeling
    uncertainties, the distance uncertainties, and the flux uncertainties for
    both the r-band and W1 fluxes. Typical errors for modeling as given in
    ATLAS3D are 6%. Errors for the photometry are 10%
    '''
    new_ml = (log_atlas3d_ml + 0.4 * (
        w1_mag - 5 * np.log10(distance*1e5) - 
        conv.SOLAR_ABSOLUTE_MAGNITUDES_AB["W1"]) + log_atlas3d_lum)
    new_ml_err = np.sqrt(
        log_atlas3d_ml_err**2 + (2 * distance_err / distance / np.log(10))**2 +
        (0.4 * w1_mag_err)**2 + log_atlas3d_lum_err**2)
    return new_ml, new_ml_err


def read_Krajnovic_Table_D1(
        URL=("/home/regulus/simonian/year1/wise/ATLAS3D_DB/"
             "Krajnovic2011_Atlas3D_Paper2_TableD1.txt")):
    '''Reads the table from Krajnovich 2011

    In particular, this table contains information about dust.'''
    krajnovic_table = Table.read(
        URL, format="ascii.commented_header", guess=False, header_start=-5, 
        data_start=0)
    return krajnovic_table

def read_MGE_model(modelfolder, galname, galcol="Galaxy"):
    '''Reads an MGE model file from Scott et al 2013.

    The structure of this file can be found in Table 2. This function will
    return the table, with the galaxy name given under the column of "galcol".
    '''
    modelpath = os.path.join(
        modelfolder, "mge_{0}.txt".format(phot.object_name_to_dir(galname)))
    modeltable = Table.read(
        modelpath, format="ascii.no_header", names=("Ij", "sigj", "qj"),
        data_start=1, guess=False)
    modeltable[galcol] = galname
    ordered_mt = modeltable[galcol, "Ij", "sigj", "qj"]
    return ordered_mt

def read_MGE_models(BASEDIR, galnames, galcol="Galaxy", 
                    modelfolder="mge_parameters_atlas3d"):
    '''Reads in the MGE models for the objects in galnames.

    This function will return an astropy table with the model parameters, and a
    column labeled "galcol", which will have the name of the galaxy
    corresponding to each model parameter. The utility of this approach lies in
    the astropy.table.Table.group_by() method, where subtables corresponding to
    each galaxy can be separated.
    '''
    modelfolderpath = os.path.join(BASEDIR, modelfolder)
    # Since vstack accepts a sequece of tables, we'll just make our own list.
    # We're not using a list comprehension for the sake of exception handling. 
    modellist = []
    for galname in galnames:
        try:
            mgemodel = read_MGE_model(
                    modelfolderpath, galname, galcol)
        except IOError:
            # This means that there isn't an MGE model for this galaxy. So
            # ignore it.
            continue
        except Exception:
            if not phot.check_if_column(galnames):
                raise ValueError("Galnames is not an Astropy Column")
            else:
                raise
        modellist.append(mgemodel)
    fullmodeltable = vstack(modellist)
    return fullmodeltable

def integrate_MGE_gaussians(mgetable, D=None, Derr=None, aperturesize=np.inf):
    '''Takes a table with MGE parameters and calculates fluxes.

    This involves using Equation (1) from Scott et al. (2013). Due to the
    ambiguity of the equation, this function will either return a flux or a
    luminosity. 
    
    In order to get a luminosity, distances will be required.
    Specify the distance in the D argument in Mpc. Derr will be used to
    calculate the error in the luminosity. The luminosity will be returned in
    terms of the log of the luminosity in solar luminosity units. Fluxes will
    be returned in terms of the log of the luminosity in solar luminosities per
    square parsec.

    The aperturesize parameter allows an elliptical aperture to be used of the
    same shape as the galaxy with a given semimajor axis. The default value is
    the numpy floating-point value for Infinity, in which case all flux will be
    included. The value of aperturesize should be given in arcseconds.
    '''
    # The sqrt(2) takes into account the fact that erf is defined as the
    # integral of e**t**2 rather than e**(t**2/2).
    apweights = scipy.special.erf(aperturesize/mgetable["sigj"]/np.sqrt(2))
    gauss_sum = np.sum(2 * np.pi * mgetable["Ij"] * 
            (mgetable["sigj"]/206265)**2 * mgetable["qj"] * apweights)
    logflux = np.log10(gauss_sum/4/np.pi)
    logfluxerr = 0.1 / np.log(10) # Flux errors are around 10 percent.
    if D is not None and Derr is not None:
        loglum = logflux + np.log10(4*np.pi) + 2 * np.log10(D*1e6)
        loglumerr = np.sqrt(logfluxerr**2 + (2*Derr/D/np.log(10))**2)
        return loglum, loglumerr
    elif D is None and Derr is None:
        return logflux, logfluxerr
    else:
        raise ValueError("D and Derr need to either both be specified, or "
                         "not.")

def integrate_MGE_gaussian_table(
        mgetable, distancetable, galnames=("Galaxy", "Galaxy"),
        lastgaussianscale=np.inf):
    '''Integrates the MGE gaussians for all given objects.

    MGEtable should be a large table containing all of the MGE expansion
    parameters. There should also be a column containing the galaxy name
    corresponding to each of the gaussians, so you know which one goes with
    which.

    Distancetable should be a table containing the galaxy name and the distance
    and distance errors under "D" and "D_err".

    Galnames should be a tuple containing the label for the galaxy column for
    both the mgetable and the distancetable, respectively.


    Lastgaussianscale is the factor which determines the aperture size for each
    object. The aperture size will be lastgaussianscale*max(sigj). In other
    words, the size of the aperture will be lastgaussianscale sigma of the
    largest Gaussian making up the MGE model. The default value is to have an
    infinitely large aperture.
    '''
    mgelums = []
    mgelumerrs = []
    mgegalname = []
    mge_grouped = mgetable.group_by(galnames[0])
    # Iterate over all of the galaxy tables.
    for key, group in izip(mge_grouped.groups.keys, mge_grouped.groups):
        distancerow = distancetable[np.where(
            distancetable[galnames[1]] == key[galnames[0]])]
        d, derr = distancerow["D"][0], distancerow["D_err"][0]
        largest_gaussian = max(group["sigj"])
        lum, lumerr = integrate_MGE_gaussians(
                group, d, derr, aperturesize=lastgaussianscale*largest_gaussian)
        mgelums.append(lum)
        mgelumerrs.append(lumerr)
        mgegalname.append(key[galnames[0]])
    lumtable = Table([mgegalname, mgelums, mgelumerrs],
                     names=(galnames[0], "logL", "logL_err"))
    return lumtable

def get_largest_gaussian(
        mgetable, galname="Galaxy", signame="sigj"):
    '''Gets the width of the largest gaussian for each galaxy in mgetable.

    MGEtable should be a large table containing all of the MGE expansion
    parameters. There should also be a column containing hte galaxy name
    corresponding to each of the Gaussians, so you know which one goes with
    which.

    This function will output a table containing the galaxy name along with the
    width of the largest gaussian.
    '''
    mgegals = []
    mgesigs = []
    mge_grouped = mgetable.group_by(galname)
    for key, group in izip(mge_grouped.groups.keys, mge_grouped.groups):
        largest_row = group[-1]
        mgegals.append(largest_row[galname])
        mgesigs.append(largest_row[signame])
    sigtable = Table([mgegals, mgesigs], names=(galname, signame))
    return sigtable

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

