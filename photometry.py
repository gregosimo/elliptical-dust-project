#! /usr/bin/env python
"""Performs elliptical aperture photometry by calling IRAF routines."""
import os
import math
import glob

from pyraf import iraf
from astropy import wcs
from astropy.io import fits
from astropy.table import Table, Column
from astroquery.ned import Ned
import numpy as np
import aplpy
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats.mstats

import queries as query
import synthetic_photometry as synphot
import WISE_conversions as conv
import masks

bands=["W1", "W2", "W3", "W4", "NUV", "FUV"]
IRBANDS = bands[:4]
UVBANDS = bands[4:]
MIR_Symbols = {0: {"marker": 'o', "markerfacecolor": 'white', "ls": ' ', 
                   "markeredgewidth": 1.5},
               1: {"marker": '^', "markerfacecolor": 'orange', "ls": ' '},
               2: {"marker": 'o', "markerfacecolor": 'green', "ls": ' '},
               3: {"marker": '*', "markerfacecolor": 'blue', "ls": ' '},
               4: {"marker": 'D', "markerfacecolor": 'white', "ls": ' ',
                   "markeredgecolor": 'red', "markeredgewidth": 1.5}}
LARGE_APERTURE_CORRECTION = {"W1": -0.034, "W2": -0.041, "W3": 0.03, "W4": 
        -0.029}


###############################################################################
# Aperture Photometry Routines                                                #
###############################################################################
# 
# There's a lot to remember to get the aperture photometry routine done. First
# make sure the BASEDIR is set up correctly. It should have the identifications
# of the galaxies as folders, with the images having names which follow the
# rules in match_filter(). 
#
# The best way to get BASEDIR set up correctly is to take the object coordinates
# and use the batch_image_download function to set up the BASEDIR folder.
#
# The way to set up BASEDIR for the photometric pipeline is to run
# build_pipeline().

def calc_DNflux(galaxydir, band, baseobjectfile="ellipse_aperture", 
        useskybase="sky_level", apertureCorrection=True):
    '''Calculates the flux of a galaxy in Data Numbers.

    This function uses the output from the ellipse package to calculate the
    background-subtracted flux of the galaxy. The total flux is calculated from
    the ellipse package and is stored in the table named with baseobjectfile.
    The sky values are determined from the file named with useskybase.
    '''
    ellipsetable = STSDAS_to_Astropy_Table(galaxydir,
            format_band_dependence(baseobjectfile, band, "tab"))
    DNflux = ellipsetable[0]["TFLUX_E"]
    aperture_area = ellipsetable[0]["NPIX_E"]
    # Right now we will only support sky backgrounds done through the 
    # pipeline.
    background = estimate_background(galaxydir, band)
    if apertureCorrection:
        fapcor = aperture_correction_factor(band)
    else:
        fapcor = 1
    objectflux = fapcor * (DNflux - background * aperture_area)
    if objectflux < 0:
        raise ValueError("Measured negative flux for object.")
    return objectflux
    
def galaxy_photometry(BASEDIR, name, band, baseobjectfile="ellipse_aperture", 
        useskybase="sky_level", uncertaintybase="uncertainty", flux=False, 
        errors=True, apertureCorrection=True, colorIndex=-2):
    '''Returns the elliptical aperture photometry-determined magnitude.

    This function requires that the adequate pipeline be constructed, where
    there is a folder tree under BASEDIR where each object maps to a folder
    labeled as the object name without spaces. For example, "NGC 1111" would be
    under the folder "NGC1111".

    Under each folder, there should be two sets of files outputted by the
    ellipse package. One should be called ellipse_aperture.{band}.tab, which
    contains the calculated total flux of the object, and the other should be
    named sky_level.{band}.tab. These should have information about the sky
    background; This should either be the output of the fitsky routine for
    WISE bands, or the output of the ellipse routine for UV bands (still under
    construction).

    Features which are under construction are on-the-fly aperture photometry and
    sky calculation without needing the sky_aperture and ellipse_aperture files,
    along with masking. If you lave the useskybase parameter alone, it will
    perform regular sky estimation.
    '''
    galaxydir = os.path.join(BASEDIR, object_name_to_dir(name))
    try:
        DNflux = calc_DNflux(galaxydir, band, baseobjectfile, useskybase)
    except ValueError:
        print "\nGot negative flux for {0}.\n".format(name)
        DNflux *= -1
    
    if errors:
        objectError = calc_DNerr(galaxydir, band)
        if flux:
            photvalue = conv.DN_flux_to_Jy(band, DNflux, colorIndex)
            err = conv.DN_err_to_Jansky_err(galaxydir, band, objectError,
                    DNflux=DNflux, colorIndex=colorIndex)
        else:
            photvalue = conv.DNflux2WISEmag(band, DNflux)
            err = conv.DN_err_to_mag_err(galaxydir, band, objectError,
                    DNflux=DNflux)
        return (photvalue, err)
    else:
        if flux:
            photvalue = conv.DN_flux_to_Jy(band, DNflux, colorIndex)
        else:
            photvalue = conv.DNflux2WISEmag(band, DNflux)
        return flux

def estimate_background(galaxydir, band, customskybase=""):
    '''Background estimator for all bands.

    There are many diferent ways of estimating backgrounds based on different
    surveys. This function will sort through all the different ways and
    transparently return the corresponding background value without the user
    having to iterate through all of the cases.

    The sky base is assumed to be "sky_level" for infrared bands. And
    "sky_ellipse" for UV bands. If a different sky base is to be used, the
    customskybase keyword should be specified.
    '''
    if not customskybase:
        skylevel = "sky_level"
        skyaperture = "sky_ellipse"
    else:
        skylevel = skyaperture = customskybase
    if band in IRBANDS:
        background = estimate_WISE_background(galaxydir, band,
                baseskyfile=skylevel)
    else:
        background = estimate_UV_background(galaxydir, band,
                baseellipsefile=skyaperture)

    return background


def calculate_correlated_pixel_noise(band):
    '''Calculated Fcorr for a particular band.

    Values taken from:
    http://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec2_3f.html#corrnoise
    for s_in/s_out
    and
    http://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec4_6ci.html
    for N_p
    '''
    EFFECTIVE_NOISE_PIXELS = {"W1": 13.772, "W2": 17.636, "W3": 35.476, "W4":
            24.462, "NUV": 1.0, "FUV": 1.0}
    INPUT_TO_OUTPUT_PIXEL_RATIO = {"W1": 2, "W2": 2, "W3": 2, "W4": 4, "NUV":
            1.0, "FUV": 1.0}
    return  (EFFECTIVE_NOISE_PIXELS[band] * 
            (INPUT_TO_OUTPUT_PIXEL_RATIO[band])**2)

def calc_DNerr(galaxydir, band, ellipsebase="ellipse_aperture",
        skybase="sky_level", uncertainty_base="uncertainty"):
    '''Calculates the uncertainty of a Data Number flux.

    This function requires bases for the ellipse routine, sky routine, and
    uncertainty routine. It will use values from these files to calculate the
    uncertainty of the flux based on the description given in the WISE All-sky
    explanatory Supplement:
    http://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec2_3f.html
    '''
    ellipseParams = STSDAS_to_Astropy_Table(galaxydir, 
            format_band_dependence(ellipsebase, band, "tab"))[0]
    imageUncertainty = STSDAS_to_Astropy_Table(galaxydir,
            format_band_dependence(uncertainty_base, band, "tab"))[0]

    fapcor = aperture_correction_factor(band)
    NA = ellipseParams["NPIX_E"]
    if band in UVBANDS:
        NB = NA
        sig_B = sky_file_background(galaxydir, band, skybase)
    else:
        skyParams = Table.read(os.path.join(galaxydir,
            format_band_dependence(skybase, band, "txt")), 
            format="ascii.daophot")
        NB = skyParams["NSKY"][0]
        sig_B = skyParams["STDEV"][0]
    total_sigi = imageUncertainty["TFLUX_E"]
    Fcorr = calculate_correlated_pixel_noise(band)
    # We get the background level from centroiding, which seems like a
    # mean-related measure.
    k = 1   
    # We can try to measure this and compare it to other errors later, but right
    # now this is not easily measurable in an automated way. I believe that this
    # should be minimal because of the large size of the aperture.
    sig_conf = 0

    sourceerr = np.sqrt(fapcor**2 * Fcorr * (total_sigi + k * NA**2 / NB * 
        sig_B**2) + sig_conf**2)
    return sourceerr

def object_name_to_dir(objectname):
    '''Converts the object name with spaces to the directory name.'''
    return objectname.replace(' ', "")

def fullphotometry(BASEDIR, WISE_Table):
    '''Performs the pipeline  building and photometry calculation of a table.

    This function is good for when the ultimate goal of an image set is just to
    get photometry out. The build_pipeline() and aperturePhotometryTable()
    functions get called as a unit.

    This function can also be used to generate unique prefixes for the objects.
    '''
    pass

###############################################################################
# Aperture Photometry Routines                                                #
###############################################################################
# 
# There's a lot to remember to get the aperture photometry routine done. First
# make sure the BASEDIR is set up correctly. It should have the identifications
# of the galaxies as folders, with the images having names which follow the
# rules in match_filter(). 
#
# The best way to get BASEDIR set up correctly is to take the object coordinates
# and use the batch_image_download function to set up the BASEDIR folder.
#
# The way to set up BASEDIR for the photometric pipeline is to run
# build_pipeline().

def calc_DNflux(galaxydir, band, baseobjectfile="ellipse_aperture",
        mask="", useskybase="sky_aperture"):
    '''Calculates the flux of a galaxy in Data Numbers.

    This function uses the output from the ellipse package to calculate the
    background-subtracted flux of the galaxy. The total flux is calculated from
    the ellipse package and is stored in the table named with baseobjectfile.
    The sky values are determined from the file named with useskybase.

    Masking is not implemented yet.
    '''
    ellipsetable = STSDAS_to_Astropy_Table(galaxydir,
            format_band_dependence(baseobjectfile, band, "tab"))
    DNflux = ellipsetable[0]["TFLUX_E"]
    aperture_area = ellipsetable[0]["NPIX_E"]
    # Right now we will only support sky backgrounds done through the pipeline.
    # No support for on-the-fly calculations unless there is a use case for
    # them.
    # TODO: Make sure UV sky background hasn't been broken.
    if band in IRBANDS:
        background = estimate_WISE_background(galaxydir, band, 
                aperture_area)
    else:
        background = estimate_UV_background(galaxydir, band,
                baseellipsefile=baseobjectfile)
    objectflux = DNflux - background
    if objectflux < 0:
        raise ValueError("Measured negative flux for object.")
    return objectflux


def galaxy_photometry(BASEDIR, name, band, baseobjectfile="ellipse_aperture", 
        mask="", useskybase="sky_aperture", uncertaintybase="uncertainty", 
        flux=False, errors=True):
    '''Returns the elliptical aperture photometry-determined magnitude.

    This function requires that the adequate pipeline be constructed, where
    there is a folder tree under BASEDIR where each object maps to a folder
    labeled as the object name without spaces. For example, "NGC 1111" would be
    under the folder "NGC1111".

    Under each folder, there should be two sets of files outputted by the
    ellipse package. One should be called ellipse_aperture.{band}.tab, which
    contains the calculated total flux of the object, and the other should be
    named sky_level.{band}.tab. These should have information about the sky
    background; This should either be the output of the fitsky routine for
    WISE bands, or the output of the ellipse routine for UV bands (still under
    construction).

    Features which are under construction are on-the-fly aperture photometry and
    sky calculation without needing the sky_aperture and ellipse_aperture files,
    along with masking. If you lave the useskybase parameter alone, it will
    perform regular sky estimation.
    '''
    galaxydir = os.path.join(BASEDIR, object_name_to_dir(name))
    try:
        objectflux = calc_DNflux(galaxydir, band, baseobjectfile, mask,
                useskybase)
    except ValueError:
        print "\nGot negative flux for {0}.\n".format(name)
        objectflux *= -1

    if errors:
        objectError = calc_DNerr(galaxydir, band)
        print flux
        if flux:
            photvalue = DN_flux_to_Jy(band, objectflux)
            err = DN_err_to_Jansky_err(galaxydir, band, objectError,
                    DNflux=objectflux)
        else:
            photvalue = DNflux2WISEmag(band, objectflux)
            err = DN_err_to_mag_err(galaxydir, band, objectError,
                    DNflux=objectflux)
        return (photvalue, err)
    else:
        if flux:
            photvalue = DN_flux_to_Jy(band, objectflux)
        else:
            photvalue = DNflux2WISEmag(band, objectflux)
        return flux

def DN_to_Jy_conversion(band):
    '''Returns the conversion factor between Data Numbers and Janskys for band.

    Taken from:
    http://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec2_3f.html#tbl1
    '''
    DN_to_Jy = {"W1": 1.9350e-6, "W2": 2.7048e-06, "W3": 1.8326e-06, "W4":
            5.2269e-05}
    return DN_to_Jy[band]


def DN_flux_to_Jy(band, objectflux):
    '''Converts a flux from Data Numbers to Janskys.
    '''
    return objectflux * DN_to_Jy_conversion(band)

def calculate_correlated_pixel_noise(band):
    '''Calculated Fcorr for a particular band.

    Values taken from:
    http://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec2_3f.html#corrnoise
    for s_in/s_out
    and
    http://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec4_6ci.html
    for N_p
    '''
    EFFECTIVE_NOISE_PIXELS = {"W1": 13.772, "W2": 17.636, "W3": 35.476, "W4":
        24.462}
    INPUT_TO_OUTPUT_PIXEL_RATIO = {"W1": 2, "W2": 2, "W3": 2, "W4": 4}
    return  (EFFECTIVE_NOISE_PIXELS[band] * 
            (INPUT_TO_OUTPUT_PIXEL_RATIO[band])**2)




def calc_DNerr(galaxydir, band, ellipsebase="ellipse_aperture",
        skybase="sky_level", uncertainty_base="uncertainty", mask=""):
    '''Calculates the uncertainty of a Data Number flux.

    This function requires bases for the ellipse routine, sky routine, and
    uncertainty routine. It will use values from these files to calculate the
    uncertainty of the flux based on the description given in the WISE All-sky
    explanatory Supplement:
    http://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec2_3f.html
    '''
    ellipseParams = STSDAS_to_Astropy_Table(galaxydir, 
            format_band_dependence(ellipsebase, band, "tab"))[0]
    skyParams = Table.read(os.path.join(galaxydir,
        format_band_dependence(skybase, band, "txt")), format="ascii.daophot")
    imageUncertainty = STSDAS_to_Astropy_Table(galaxydir,
            format_band_dependence(uncertainty_base, band, "tab"))[0]

    fapcor = 1
    NA = ellipseParams["NPIX_E"]
    NB = skyParams["NSKY"][0]
    total_sigi = imageUncertainty["TFLUX_E"]
    Fcorr = calculate_correlated_pixel_noise(band)
    # We get the background level from centroiding, which seems like a
    # mean-related measure.
    k = 1   
    sig_B = skyParams["STDEV"][0]**2
    # We can try to measure this and compare it to other errors later, but right
    # now this is not easily measurable in an automated way. I believe that this
    # should be minimal because of the large size of the aperture.
    sig_conf = 0

    sourceerr = (fapcor**2 * Fcorr * (total_sigi + k * NA**2 / NB * sig_B) +
            sig_conf)**(0.5)
    return sourceerr

def DN_err_to_mag_err(galaxydir, band, DNerr, baseobjectfile="ellipse_aperture",
        mask="", useskybase="sky_level", DNflux=0):
    '''Converts an error in Data Number to an error in magnitudes.

    If DNflux is given, this function will use it as the value for the object's
    flux in data numbers. If it isn't, then it will calculate it on its own.
    '''

    if not DNflux:
        DNflux = calc_DNflux(galaxydir, band, baseobjectfile, mask, useskybase)
    sigma_mag = (get_zero_point_magnitude_uncertainty(band)**2 + 1.179 *
            (DNerr**2 / DNflux **2))**0.5
    return sigma_mag

def DN_err_to_Jansky_err(galaxydir, band, DNerr, 
        baseobjectfile="ellipse_aperture", mask="", useskybase="sky_level", 
        DNflux=0):
    '''Converts an error in Data Numbers to an error in Janskys.

    If DNflux is given, this function will use it as the value for the object's
    flux in data numbers. If it isn't, then it will calculate it on its own.
    '''

    if not DNflux:
        DNflux = calc_DNflux(galaxydir, band, baseobjectfile, mask, useskybase)
    sigma_Jy = DN_to_Jy_conversion(band) * (DNflux**2 *
            (get_zero_point_flux_uncertainty(band)**2 / 
            get_zero_point_flux_level(band)**2 + 
            0.8483 * get_zero_point_magnitude_uncertainty(band)**2) +
            DNerr**2)**(0.5)
    return sigma_Jy

def get_zero_point_magnitude_uncertainty(band):
    '''Returns the zero-point magnitude uncertainty in a given band.

    The uncertainties in the zero-point magnitudes are taken from:
    http://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec2_3f.html#tbl1
    '''
    MAGZPUNC = {"W1": 0.006, "W2": 0.007, "W3": 0.015, "W4": 0.012}
    return MAGZPUNC[band]

def object_name_to_dir(objectname):
    '''Converts the object name with spaces to the directory name.'''
    return objectname.replace(' ', "")

def photometric_error(BASEDIR, name, band, ellipsebase="ellipse_aperture",
        skybase="sky_level", uncertainty_base="uncertainty", mask="", 
        flux=False):
    '''Calculates the photometric error of a magnitude calculation.

    This measurement uses the photometric pipeline to look up values of the
    quantities needed for a reliable error estimate. Note that this calculation
    requires the uncertainty Atlas images. Those will need to be downloaded as
    part of the pipeline as well.
    '''
    galaxydir = change_to_galaxy_dir(BASEDIR, name)

    object_flux = galaxy_photometry(BASEDIR, name, band,
            baseobjectfile=ellipsebase, mask=mask, useskybase=skybase,
            DNflux=True)

    if flux:
        return fluxerr
    else: 
        magerr = (get_zero_point_magnitude_uncertainty(band)**2 + 1.179 * 
            fluxerr**2 / object_flux**2)**(0.5)
        return magerr
###############################################################################
# Astropy Utilities                                                           #
###############################################################################

def combine_WISE_aperture_tables(apertureTable, wisetable, MIR_column):
    '''Combines the Aperture Photometry table with a WISE photometry table.
    '''
    return make_relevant_table(apertureTable["objstr_01"],
            apertureTable["w1apmag"], wisetable["w1gmag"],
            apertureTable["w2apmag"], wisetable["w2gmag"],
            apertureTable["w3apmag"], wisetable["w3gmag"],
            apertureTable["NUVapmags"], apertureTable["FUVapmags"], MIR_column)

def make_relevant_table(objstr, w1ap, w1wise, w2ap, w2wise, w3ap, w3wise, NUVap,
        FUVap, MIR):
    '''Extracts relevant columns from the raw WISE and GALEX tables.

    Relevant information includes elliptical aperture parameters,
    measured magnitudes, and exposure times.'''
    w1w2ap = w1ap - w2ap
    w2w3ap = w2ap - w3ap

    w1w2wise = w1wise - w2wise
    w2w3wise = w2wise - w3wise

    return Table((objstr, w1ap, w2ap, w3ap, w1wise, w2wise, w3wise, w1w2ap,
        w2w3ap, w1w2wise, w2w3wise, NUVap, FUVap, MIR), names=("objstr_01", 
        "w1apmag", "w2apmag", "w3apmag", "w1gmag", "w2gmag", "w3gmag", 
        "w1w2apcol", "w2w3apcol", "w1w2gcol", "w2w3gcol", "NUVapmag", 
        "FUVapmag", "MIR"))

def astropy_table_index(table, column, value):
    '''Returns the row index of the table which has the value in column.

    There are often times when you want to know the index of the row
    where a certain column has a value. This function will return a 
    list of row indices that match the value in the column.'''
    return np.where(table[column] == value)

def build_pipeline(BASEDIR, WISETable, maskthresh=150, inputband="W1",
        maskoutput="foreground.fits", ellipsepars="ellipsepars",
        maskconfigbase="default", ellipseoutput="ellipse_aperture",
        skycoord="fitsky", skybase="sky_level", uncertaintybase="uncertainty", 
        forceSkyAnnulus=False, runbands=bands):
    '''Basically runs all the commands necessary to build the ellipse aperture
    and sky measurement pipeline. It consists of running:
    allApertureTables
    allEllipseTables
    allSkyValues

    If you want a table of photometry, you'll have to run
    aperturePhotometryTable yourself.

    Masking is not trivial. To completely disable masking, set maskthresh to 0.
    The masking pipeline will be completely bypassed. If you want to use
    masking, you need to set the mask threshold, as well as the input image to
    generate the mask and the output filename of mask.
    ''' 
    print "Making Aperture Tables..."
    allApertureTables(BASEDIR, WISETable, runbands=runbands,
            outputbase=ellipsepars)
    if maskthresh != 0:
        print "Making Masks..."
        allMasks(BASEDIR, WISETable, threshold=maskthresh, maskband=inputband,
                output=maskoutput, ellipsebase=ellipsepars,
                maskconfigbase="default")
    else:
        maskoutput=""
    print "Making Ellipse Tables..."
    allEllipseTables(BASEDIR, WISETable, runbands=runbands, mask=maskoutput,
            baseoutput=ellipseoutput, baseparamname=ellipsepars)
    print "Making Sky Tables..."
    allSkyValues(BASEDIR, WISETable, runbands=runbands, coordbase=skycoord, 
            baseskyfile=skybase, ellipsebase=ellipsepars,
            forceAnnulus=forceSkyAnnulus)
    print "Making Uncertainty Tables..."
    allUncertaintyTables(BASEDIR, WISETable, runbands=runbands,
            ellipsebase=ellipsepars, baseuncertainty=uncertaintybase,
            skybase=skybase)
    write_pipeline_file("{0}.par".format(ellipsepars), 
            mask_threshold=maskthresh, mask_band=input_band, 
            mask_output=maskoutput, ellipse_parameters=ellipsepars, 
            mask_config_base=maskconfigbase, aperture_file=ellipseoutput, 
            sky_coordinates=skycoords, sky_base=sky_base, 
            uncertainty_base=uncertaintybase, bands_written=runbands)

def astropy_table_row(table, column, value):
    '''Returns the row of the table which has the value in column.

    If you want to know the row in an astropy table where a value in a
    column corresponds to a given value, this function will return that
    row. If there are multiple rows which match the value in the 
    column, you will get all of them. If no rows match the value, this
    function will throw a ValueError.'''
    return table[astropy_table_index(table, column, value)]

def extract_subtable_from_column(table, column, selections):
    '''Returns a table which only contains values in selections.

    This function will create a Table whose values in column are only
    those found in selections.
    '''
    indices = []
    for object in selections:
        indices.append(astropy_table_index(table, column, object)[0][0])
    return table[indices]



def change_to_galaxy_dir(BASEDIR, objectname):
    '''Returns the path of a galaxy's directory.

    Given the name of an object, it will return the full path to the
    directory containing all of the data concerning that object.
    '''
    return os.path.join(BASEDIR, object_name_to_dir(objectname))

def format_band_dependence(basename, band, extension="tab", pathto=''):
    '''Generates a table file which is dependent on a band name.

    The returned filename will have a format of 
    "/pathto/{basename}.{band}.{extension}".
    '''
    return os.path.join(pathto, "{0}.{1}.{2}".format(basename, band, extension))
    
def match_filter(directory, filter, fullpath=True, uncertainty=False, 
        sky=False):
    '''Finds the image which corresponds to the filter.

    For WISE images, this will require searching for "w?" in the
    strings.'''
    filtermap = {"W1": "w1-int", "W2": "w2-int", "W3": "w3-int", "W4": "w4-int",
            "FUV": "fd-int", "NUV": "nd-int"}
    filterstring = filtermap[filter]
    if uncertainty:
        filterstring = filterstring.replace("int", "unc")
    elif sky:
        filterstring = filterstring.replace("int", "skybg")
    filelist = glob.glob(os.path.join(directory, 
            "*{0}*.fits".format(filterstring)))
    if len(filelist) > 1:
        raise RuntimeError("Image conflict for {0}.".format(directory))
    elif len(filelist) == 0:
        raise RuntimeError("Could not find {1} file in {0}.".format(directory,
            filterstring))
    else:
        imagefile = filelist[0]
        if not fullpath:
            imagefile = os.path.basename(imagefile)
        return imagefile

def complete_for_bands(BASEDIR, objname, checkbands=bands):
    '''Determines if an object has full WISE and UV observations.

    It does this by checking whether all band patterns in checkbands are matched.
    If they are, then it returns true. If any are missing, then it returns false.
    Checkbands defaults to the WISE bands and UV bands as shown in the bands
    variable.
    '''
    galaxydir = change_to_galaxy_dir(BASEDIR, objname)
    for band in checkbands:
        # I don't like how this is done... If necessary, I'll try to
        # use another method to filter for complete objects without
        # utilizing the exception stack.
        try:
            match_filter(galaxydir, band)
        except RuntimeError:
            return False
    return True

def run_fitsky(image, annulus, coords, output, dannulus=10,
        algorithm="centroid", scale=1, fwhmpsf=6):
    '''Runs the fitsky procedure in IRAF in order to measure the sky background.

    This function measures the sky pixels in an annulus with inner edge at
    annulus pixels, and width dannulus. It will get the coordinates of objects
    from the coords argument. Each line in the coords argument will correspond
    to a line in output.
    '''
    #Fitskypar parameters.
    iraf.apphot()
    iraf.fitskypars.setParam("salgorithm", algorithm)
    iraf.fitskypars.setParam("annulus", annulus)
    iraf.fitskypars.setParam("dannulus", dannulus)
    # Datapars
    iraf.datapars.setParam("scale", scale)
    iraf.datapars.setParam("fwhmpsf", fwhmpsf)
    # Fitsky parameters.
    iraf.fitsky.setParam("coords", coords)
    iraf.fitsky.setParam("output", output)
    iraf.fitsky.setParam("interactive", "No")
    iraf.fitsky.setParam("verify", "No")
    iraf.fitsky.setParam("update", "No")
    iraf.fitsky.setParam("radplots", "No")

    iraf.fitsky(image)


def run_ellipse(image, ellipsepars, output, mask=""):
    '''Generates an ellipse table on the image from given parameters.

    The table of aperture parameters should be in the form of an STSDAS
    table. The necessary values are ellipticity, semimajor axis,
    position angle, X0, and Y0 (in pixels).
    
    A mask file can be specified for the ellipse routine. If a mask
    file is specified, this function will throw an error if the mask
    file isn't found. Therefore, if you wish to ignore masking, the
    mask parameter should be the empty string.
    
    NOTE: All filenames should contain full paths to the files.'''
    # IRAF will throw a cryptic error, or simply ignore the fact that
    # the mask doesn't exist. I want to enforce it to avoid silently
    # ignoring masking when I intend to mask.
    # NOTE: This breaks compability with Windows. Boo hoo.
    if not os.path.exists(os.path.join("/", mask)):
        raise ValueError("Mask file does not exist.")
    iraf.stsdas()
    iraf.stsdas.analysis()
    iraf.stsdas.analysis.isophote()
    #iraf.unlearn("ellipse")
    iraf.ellipse.setParam("inellip", ellipsepars)
    iraf.ellipse.setParam("dqf", mask)
    iraf.ellipse(image, output)

def generate_elliptical_aperture(inputfile, outputfile):
    '''Generates a polygonal aperture from ellipse table.

    The ellipse table should be given in inputfile, and the file to be
    output should be outputfile. Both inputfile and outputfile should
    have the path of the file specified. The elapert task also allows 
    you to specify a "coords" argument which contains the polygon 
    centers. I don't think this is needed for our purposes.
    '''
    iraf.stsdas()
    iraf.stsdas.analysis()
    iraf.stsdas.analysis.isophote()
    iraf.ellipse(inputfile, outputfile)

def elliptical_aperture(galaxydir, image, band,
                        outputname="ellipse.tbl"):
    '''Reads out the ellipse flux from the ellipse routine.

    This function takes an object which already has output from the 
    ellipse routine. '''
    outputtbl = os.path.join(galaxydir, outputname)
    output = STSDAS_to_Astropy_Table(galaxydir, outputtbl)
    DNflux = output[0]["TFLUX_E"]
    return DNflux 

def readSkyTable(galaxydir, band, area, baseellipsefile="sky_aperture"):
    '''Generates a background flux from an existing table.

    Since much of the calculations being done for objects is through a
    standard aperture size, it is easier to simply read the values from
    a pre-generated STSDAS table. This function is just for reading.
    '''
    ellipsetable = STSDAS_to_Astropy_Table(galaxydir,
            format_band_dependence(baseellipsefile, band, "tab"))
    if band in IRBANDS:
        return float(ellipsetable[0]["INTENS"]) * area
    else:
        return float(ellipsetable[0]["TFLUX_E"])

def writeregion(BASEDIR, WISErow, parambase="ellipsepars",
        outputbase="ellipseregion", runbands=bands):
    '''Writes a shitty DS9 region file.

    This writes a file with only one line, taking the physical coordinates of
    the object. Making a good region writer probably won't be that hard, but I
    don't feel like it.'''
    galaxydir = change_to_galaxy_dir(BASEDIR, WISErow["objstr_01"])
    for band in runbands:
        ellipsetable = STSDAS_to_Astropy_Table(galaxydir,
                format_band_dependence(parambase, band, "tab"))[0]
        xcoord, ycoord = ellipsetable["X0"], ellipsetable["Y0"]
        semimajor = ellipsetable["SMA"]
        semiminor = semimajor * (1 - ellipsetable["ELLIP"])
        pa = ellipsetable["PA"]
        ellipsestring="ellipse({0}, {1}, {2}, {3}, {4})".format(xcoord, ycoord,
                semiminor, semimajor, pa)
        f = open(format_band_dependence(outputbase, band, "reg", galaxydir), 
                "w")
        f.write("image\n")
        f.write(ellipsestring)
        f.close()

def getObjectFlux(galaxydir, band, baseobjectfile="ellipse_aperture"):
    '''Returns the flux of an object in Data Numbers'''
    ellipsetable = STSDAS_to_Astropy_Table(galaxydir,
            format_band_dependence(baseobjectfile, band, "tab"))
    return ellipsetable[0]["TFLUX_E"]

def estimate_WISE_background_old(galaxydir, band, area, 
        baseellipsefile="sky_aperture"):
    '''Returns the estimated background for a particular galaxy.

    This function gets the intensity from a predefined sky_aperture
    STSDAS table which should be the output of the ellipse task. This
    is a surface brightness. Getting the total sky flux requires
    providing the area of the object.
    '''
    # To be parallel with estimate_UV_background, this function should
    # provide an independent way of measuring a background. I don't
    # have a decent way of doing that yet.
    ellipsetable = STSDAS_to_Astropy_Table(galaxydir, 
            format_band_dependence(baseellipsefile, band, "tab"))
    return ellipsetable[0]["INTENS"] * area

def estimate_WISE_background(galaxydir, band, baseskyfile="sky_level"):
    '''Returns the estimated background for a particular galaxy. 

    This function utilizes the fitsky routine from IRAF.apphot to determine the
    total background flux in the galaxy. The fitsky routine returns a flux per
    pixel, so in order to determine the total background flux in the aperture,
    it needs the area of the aperture. It will then multiply the two and return
    them.
    
    The output of fitsky should be in the object's folder with base name given
    in baseskyfile, and a '.txt' extension.'''
    skypath = format_band_dependence(baseskyfile, band, "txt", pathto=galaxydir)
    skydata = Table.read(skypath, format="ascii.daophot")
    skylevel = skydata["MSKY"][0]
    return skylevel

def estimate_UV_background(galaxydir, band, baseellipsefile="sky_ellipse"):
    '''Returns the estimated background for a galax in GALEX bands.

    The background for UV bands is estimated by looking for files whose
    names contain the string given in skymarker.'''
    ellipsetable = STSDAS_to_Astropy_Table(galaxydir,
            format_band_dependence(baseellipsefile, band, "tab"))
    return ellipsetable[0]["TFLUX_E"] / ellipsetable[0]["NPIX_E"]


def objectHasImage(BASEDIR, objname):
    '''Checks if an object has a folder containing its images.'''
    return os.path.exists(change_to_galaxy_dir(BASEDIR, objname))

def filterTableforExistingObjects(BASEDIR, fulltable):
    '''Creates another table that only has the objects with images.'''
    return filterTable(BASEDIR, fulltable, objectHasImage)

def filterTableforCompleteBands(BASEDIR, fulltable):
    '''Returns a table that only has objects with complete observations'''
    return filterTable(BASEDIR, fulltable, complete_for_bands)

def filterContaminatedObjects(BASEDIR, fulltable):
    '''Returns a table which doesn't have contaminated objects'''
    return filterTable(BASEDIR, fulltable, (lambda BASEDIR, objname: not
        isObjectContaminated(BASEDIR, objname)))

def filterTable(BASEDIR, fulltable, isTrue):
    '''Filters a table based on a boolean method isTrue.'''
    filteredTable = Table(fulltable, copy=True, masked=False)
    for i, object in enumerate(fulltable["objstr_01"]):
        if not isTrue(BASEDIR, object):
            filteredTable.remove_row(np.argwhere(filteredTable["objstr_01"] ==
                    object)[0][0])
    return filteredTable

def isObjectContaminated(BASEDIR, objname, contfile="Nearby_Stars.txt"):
    '''Determines if an object is on a list containing contaminated
    galaxies.

    The file listing contaminated galaxies should be given in the
    contfile keyword.'''
    contfileobj = open(os.path.join(BASEDIR, contfile))
    contobjects = contfileobj.readlines()
    return (objname+"\n") in contobjects

def runOnImages(BASEDIR, fulltable, func, **kwargs):
    '''Goes through a table of objects and runs a function on them.

    The function must be able to accept the BASEDIR as well as an
    Astropy row. The function can also accept keyword arguments via
    kwargs.'''
    for row in fulltable:
        galaxydir = change_to_galaxy_dir(BASEDIR, row["objstr_01"])
        try:
            func(BASEDIR, row, **kwargs)
        except RuntimeError, e:
            print e

def allMasks(BASEDIR, fulltable, threshold=100, maskband="W1",
        output="foreground.fits", ellipsebase="ellipsepars", 
        maskconfigbase="default"):
    '''Goes through BASEDIR and generates all of the foreground masks.'''

    runOnImages(BASEDIR, fulltable, masks.mask_algorithm, threshold=threshold,
            maskband=maskband, output=output, ellipsebase=ellipsebase,
            maskconfigbase=maskconfigbase)

def allApertureTables(BASEDIR, fulltable, runbands=bands,
        outputbase="ellipsepars"):
    '''Goes through BASEDIR and generates all the aperture tables.

    The full WISE table will be necessary.'''
    runOnImages(BASEDIR, fulltable, genApertureTable, runbands=runbands,
            outputbase=outputbase)

def allSkyTables(BASEDIR, fulltable, runbands=bands):
    '''Goes through BASEDIR and generates all sky tables.

    This function also allows for single-object corrections to be made.
    '''
    runOnImages(BASEDIR, fulltable, genSkytables, runbands=runbands)

def allUncertaintyTables(BASEDIR, fulltable, baseuncertainty="uncertainty", 
        ellipsebase="ellipsepars", skybase="sky_level", runbands=bands):
    '''Goes through BASEDIR and generates all uncertainty tables.'''
    runOnImages(BASEDIR, fulltable, genImageUncertainty,
            baseuncertainty=baseuncertainty, ellipsebase=ellipsebase,
            skybase=skybase, runbands=runbands)

def allSkyValues(BASEDIR, fulltable, runbands=bands, coordbase="fitsky", 
        baseskyfile="sky_level", ellipsebase="ellipsepars", forceAnnulus=False):
    '''Goes through BASEDIR and generates all sky tables.

    This function also allows for single-object corrections to be made.
    '''
    runOnImages(BASEDIR, fulltable, genSkyValues, runbands=runbands,
            coordbase=coordbase, baseskyfile=baseskyfile, 
            ellipsebase=ellipsebase, forceAnnulus=forceAnnulus)

def allEllipseTables(BASEDIR, fulltable, runbands=bands, 
        mask="foregroundmask.fits", baseoutput="ellipse_aperture",
        baseparamname="ellipsepars"):
    '''Goes through BASEDIR and generates all object tables.

    This function also allows for single-object corrections to be made.
    '''
    runOnImages(BASEDIR, fulltable, genEllipsetables,
            baseparamname=baseparamname, runbands=runbands,
            mask=mask, baseoutput=baseoutput)

def allSkyParams(BASEDIR, fulltable, runbands=bands):
    '''Goes through BASEDIR and generates all sky parameter files.'''
    runOnImages(BASEDIR, fulltable, genSkyParam, runbands=runbands)

def getPixelScale(band):
    '''Returns the pixel scale for an image in a given band. Scale is
    given as arcsec/pixel.'''
    bands = {"W1": 1.37, "W2": 1.37, "W3": 1.37, "W4": 1.37, "NUV": 1.5, 
             "FUV": 1.5}
    return bands[band]

def ellipseOnBands(BASEDIR, WISErow, baseparamname, output, mask=""):
    '''Runs ellipse on all bands for a given galaxy.

    This function will take a WISE row corresponding to a particular galaxy and
    then run the ellipse package for all bands in that galaxy folder. 
    '''

def genImageUncertainty(BASEDIR, WISErow, baseuncertainty="uncertainty",
        ellipsebase="ellipse_aperture", skybase="sky_level", runbands=bands):
    '''Sums the variance of uncertainty pixels over an aperture.

    This function requires uncertainty files to be located within the galaxy
    folder. These files will be discovered by running match_filter, and then
    replacing the "int" with "unc".

    The uncertainty file will then be squared, and placed in a file with the
    same name, but with "unc" replaced by "var".

    The table which will be output by the ellipse package will have a base
    filename given by baseuncertainty.
    '''
    galaxydir = change_to_galaxy_dir(BASEDIR, WISErow["objstr_01"])
    for band in runbands:
        if band in IRBANDS:
            source_uncertainty_from_uncertainty_file(galaxydir, band,
                    rreplace(match_filter(galaxydir, band), "int", "unc", 1),
                    ellipsebase, baseuncertainty)
        elif band in UVBANDS:
            image = match_filter(galaxydir, band)
            source_uncertainty_from_image(galaxydir, band, image, ellipsebase,
                    baseuncertainty, skybase)

def source_uncertainty_from_image(galaxydir, band, image, 
        ellipsebase="ellipsepars", outputbase="uncertainty", 
        skybase="sky_level"):
    '''Calculates the uncertainty of a GALEX image from the image itself.

    This uncertainty estimation uses the form of:
    sig_i^2 = \frac{S_i - \bar{B}}/t

    where t is the exposure time in seconds, since S_i and \bar{B} are in counts
    per second. This function will also result in an STSDAS table from ellipse,
    so it will be functionally the same as the STSDAS table from the WISE
    calculation.
    '''
    # We're going to create an image which is scaled correctly, and then run the
    # ellipse command on it. That way we end up with a properly-formatted STSDAS
    # table.
    calc_template = "(im1 - {sky}) / {exptime}"
    scaled = rreplace(image, "int", "sca", 1)
    ellipsepars = format_band_dependence(ellipsebase, band, "tab", galaxydir)
    output = format_band_dependence(outputbase, band, "tab", galaxydir)

    skybackground = estimate_background(galaxydir, band)

    header = fits.getheader(image)
    exposuretime = header["EXPTIME"]

    calc_command = calc_template.format(sky=skybackground, exptime=exposuretime)
    masks.run_imcalc(image, scaled, calc_command)

    run_ellipse(scaled, ellipsepars, output)

def source_uncertainty_from_uncertainty_file(galaxydir, band, uncfile,
        ellipsebase="ellipsepars", outputbase="uncertainty"):
    '''Calculates the source uncertainty from an uncertainty file.

    The uncertainty file itself should be in uncfile. The ellipse parameters
    should be named as usual with the base of ellipsebase. And the function will
    output a table file as usual with the complete uncertainty given as the
    TFLUX_E parameter.
    '''
    varfile = rreplace(uncfile, "unc", "var", 1)


    ellipse_file = format_band_dependence(ellipsebase, band, "tab",
            galaxydir)
    output = format_band_dependence(baseuncertainty,
        band, "tab", galaxydir)
    run_ellipse(varfile, ellipse_file, output)



def run_imfunc(infile, outfile, func):
    '''Runs imfunc on the given image.

    All possible functions can be viewed in the imfunc documentation. The
    currently relevant ones are:

    square - Square the image.
    '''
    # There's a really shitty IRAF "feature" where if imfunc acts on a file
    # which already exists, it will simply add on another layer, which
    # confuses the hell out of ellipse. So if a previous file exists, I'll
    # delete it manually.
    if os.path.isfile(varfile):
        os.remove(varfile)
    run_imfunc(uncfile, varfile, "square")

    iraf.images()
    iraf.imutil()
    iraf.imfunc(infile, outfile, func)



def genEllipsetables(BASEDIR, WISErow, baseparamname="ellipsepars",
        baseoutput="ellipse_aperture", mask="foreground.fits", runbands=bands):
    '''Generates a table on the object for each band.'''
    # There should be a better way of joining this and genSkyTables, but that's
    # taking too much effort, and I want to just have this part done.
    galaxydir = change_to_galaxy_dir(BASEDIR, WISErow["objstr_01"])
    for band in runbands:
        objimage = match_filter(galaxydir, band)
        run_ellipse(objimage, format_band_dependence(baseparamname, band, 'tab',
            galaxydir), format_band_dependence(baseoutput, band, 'tab', 
            galaxydir), mask=os.path.join(galaxydir, mask))

def genSkyValues(BASEDIR, WISErow, coordbase="fitsky",
        ellipsebase="ellipsepars", baseskyfile="sky_level", skyratio=2.5,
        annulus=0, dannulus=30, runbands=bands, forceAnnulus=False):
    '''Generates sky values for each galaxy.
    
    The sky values are generated via the IRAF fitsky routine. The output of
    fitsky will be located at baseskyfile.{band}.txt files within the galaxy
    folder. 
    
    The inner edge of the annulus to be used in fitsky is determined from the
    aperture size. The routine will multiply the aperture by skyratio in order
    to get the sky aperture size. Maybe there could be a way to customize the
    radius for different objects without necessarily following a hard-and-fast
    rule like this, but I think we'll have enough leeway with our object to make
    this approximation.'''
    galaxydir = change_to_galaxy_dir(BASEDIR, WISErow["objstr_01"])


    # If annulus is 0, that means we want to scale the annulus off of the
    # aperture. If we don't make a separate annulus_override variable, 
    # setting annulus for W1 is disable resetting it for W2-4.
    for band in runbands:
        if band in IRBANDS or forceAnnulus:
            measure_sky_from_annulus(galaxydir, band, coordbase, 
                    baseskyfile, ellipsebase, annulus, skyratio, dannulus)
        else:
            measure_sky_from_skyfile(galaxydir, band, baseskyfile, 
                    ellipsebase)

def measure_sky_from_annulus(galaxydir, band, coordbase="fitsky",
        baseskyfile="sky_level", ellipsebase="ellipsepars", annulus=0, 
        skyratio=2.0, dannulus=10):
    '''Uses an annulus to measure the sky level from an image.

    This function uses fitsky to measure the sky level with an annulus. The
    inner edge of the annulus will be determined either by aperture size, or
    through the annulus keyword. If the annulus keyword is zero, the inner
    annulus size will be the aperture semimajoraxis times skyratio. Otherwise,
    the inner edge of the annulus will be overridden to the value given in
    annulus. The outer edge of the annulus will simply be given by annulus +
    dannulus.'''
    annulus_override = annulus
    coordpath = format_band_dependence(coordbase, band, "coo", galaxydir)
    skypath = format_band_dependence(baseskyfile, band, "txt", galaxydir)
    image = match_filter(galaxydir, band)
    ellipsepars = STSDAS_to_Astropy_Table(galaxydir,
            format_band_dependence(ellipsebase, band, "tab"))

    if not annulus_override:
        annulus = skyratio * ellipsepars["SMA"]
    
    with open(coordpath, 'w') as f:
        f.write("{0} {1}".format(ellipsepars["X0"][0], ellipsepars["Y0"][0]))

    run_fitsky(image, annulus, coordpath, skypath, dannulus=dannulus)

def measure_sky_from_skyfile(galaxydir, band, baseskyfile="sky_level",
        ellipsebase="ellipsepars"):
    '''Measures the sky level from a separate sky file.

    The sky level will be calculated from a separate file with only background
    counts. A file similar to the output of fitsky will be created.'''
    image_path = match_filter(galaxydir, band, sky=True)
    ellipse_param_path = format_band_dependence(ellipsebase, band, "tab",
            galaxydir)
    ellipse_output_path = format_band_dependence(baseskyfile, band, "tab",
            galaxydir)

    run_ellipse(image_path, ellipse_param_path, ellipse_output_path)
    

def sky_file_background(galaxydir, band, ellipse_output_base="sky_ellipse"):
    '''Returns sky uncertainty from sky file.

    This function calculates the variance of the sky by using quantiles. It is
    robust against outliers as well as local to the object. The output base
    should be the output of the ellipse routine to determine the sky level.
    '''
    ellipse_output = format_band_dependence(ellipse_output_base, band, "tab")
    skylevel = STSDAS_to_Astropy_Table(galaxydir, ellipse_output)
        
    image = match_filter(galaxydir, band, sky=True)
    imageval = fits.getdata(image, view=np.ma.MaskedArray)
    imageval.mask = ~mask_ellipse(imageval, skylevel["X0"], skylevel["Y0"],
        skylevel["SMA"] * (1 - skylevel["ELLIP"]), skylevel["SMA"],
        skylevel["PA"])

    skyquants = scipy.stats.mstats.mquantiles(imageval, [0.16, 0.5])
    skystd = skyquants[1] - skyquants[0]
    return skystd

def test_if_in_ellipse(x, y, xcenter, ycenter, a, b, pa):
    '''Tests if the point x,y lies within the described ellipse.

    All of the arguments should make sense except for pa. PA should be given in
    degees E of N.
    '''
    alpha = (pa + 90) * math.pi / 180.0
    xcen = x - xcenter
    ycen = y - ycenter
    return (xcen * math.cos(alpha) + ycen * math.sin(alpha))**2 / b**2 + (xcen *
            math.sin(alpha) - ycen * math.cos(alpha))**2 / a**2 < 1

def mask_ellipse(image, xcenter, ycenter, a, b, pa):
    '''Creates a mask on the image which is shaped like an ellipse.
    
    Image should be a fits image. And xcenter and ycenter should be in physical
    pixels, not numpy indices. The conversion will take place in this function.
    '''
    image_coords = np.indices(image.shape)
    mask = test_if_in_ellipse(image_coords[1], image_coords[0], xcenter-1,
            ycenter-1, a, b, pa)
    return mask
    
    


def generateRegions(BASEDIR, WISEtable, outputbase="ellipseregion", 
        parambase="ellipsepars", runbands=bands):
    '''Runs through all objects and make DS9 regions.
    '''
    runOnImages(BASEDIR, WISEtable, writeregion, outputbase=outputbase, 
            parambase=parambase, runbands=runbands)

def generateEllipseCutouts(BASEDIR, WISEtable, runbands=IRBANDS, sky=True):
    '''Runs through all objects and creates cutouts in their folder.
    '''
    current_backend = matplotlib.get_backend()
    matplotlib.use("Agg")
    runOnImages(BASEDIR, WISEtable, createEllipseCutouts, runbands=runbands,
            sky=sky)
    matplotlib.use(current_backend)

def createEllipseCutouts(BASEDIR, WISErow, runbands=IRBANDS, sky=True):
    '''Creates a set of four cutouts with the aperture and sky ellipses

    A cutout for each band will be created that contains the aperture
    photometry ellipse as well as the ellipse which samples the sky.
    '''
    galaxydir = change_to_galaxy_dir(BASEDIR, WISErow["objstr_01"])
    for band in runbands:
        # We can either get the photometry from the WISErow, or we can
        # get it directly from the STSDAS tables. The latter seems to 
        # be more direct, since those are actually used for photometry
        # and sky.
        aperturepars = STSDAS_to_Astropy_Table(galaxydir, 
                format_band_dependence("ellipse_aperture", band, "tab"))
        gc = aplpy.FITSFigure(match_filter(galaxydir, band))
        gc.show_grayscale()

        px = getPixelScale(band)
        # Make the ellipse indicating the aperture:
        Xval, Yval = gc.pixel2world(aperturepars["X0"][0], aperturepars["Y0"][0])
        height = 2 * px * aperturepars["SMA"] / 3600.0
        width = height * (1.0 - float(aperturepars["ELLIP"]))
        angle = float(aperturepars["PA"])
        gc.show_ellipses(Xval, Yval, width, height, angle=angle,
            edgecolor="yellow")
        # Now make the sky annulus:
        if sky:
            skypars = Table.read(os.path.join(galaxydir,
                format_band_dependence("sky_level", band, "txt")),
                format="ascii.daophot")
            Xval, Yval = gc.pixel2world(skypars["XINIT"][0], 
                    skypars["YINIT"][0])
            radius_in = (float(skypars.meta["keywords"]["ANNULUS"]["value"]) * 
                    px / 3600.0)
            radius_out = (radius_in +
                float(skypars.meta["keywords"]["DANNULUS"]["value"]) * px / 
                3600.0)
            gc.show_circles([Xval]*2, [Yval]*2, [radius_in, radius_out],
                    edgecolor="cyan")

        gc.save(format_band_dependence(object_name_to_dir(WISErow["objstr_01"]),
            band, "png", galaxydir))
        plt.close("all")

def generatePixelMasks(galaxydir, masterfile="foreground.reg",
        maskbasename="foreground", execbands=bands):
    '''Creates a set of pixel masks in a directory.

    If a set of pixel masks are set to be created, there should be a DS9 region
    file located in the galaxy directory with the filename given by masterfile.
    Then, this function will generate pixel maps appropriate for each image at
    the correct locations.

    If there are no objects to be masked out, the absence of the masterfile will
    generate blank pixel maps.'''
    # This stuff is gonna be done through iraf. I think that's the best way.
    regionfile = os.path.join(galaxydir, masterfile)
    iraf.proto()
    for band in execbands:
        outputfile = format_band_dependence(maskbasename, band, "pl", galaxydir)
        imagefile = match_filter(galaxydir, band)
    # On second thought, this won't work. Never mind.

# Maybe have a function that generates aperture tables given just the values we
# want to place in the table. Nothing more, nothing less. And then this function
# can be called to do that with a WISErow.
def genApertureTable(BASEDIR, WISErow, outputbase="ellipsepars", runbands=bands):
    '''Creates a table of elliptical aperture parameters from WISE.

    The directory which contains all of the objects should be given as
    BASEDIR. The row corresponding to the WISE object should be given
    in WISErow. This will creat a separate table file for each band.
    This is to prevent issues with inellip interpreting the different 
    entries as different semimajor axes to try out.
    
    The output should currently not be specified. It may be changed
    with some code refactoring to allow for custom specification of 
    band names. Otherwise, don't fiddle with it.'''

    objectdir = change_to_galaxy_dir(BASEDIR, WISErow["objstr_01"])
    columnfile = os.path.join(BASEDIR, "../ellipse_columns.txt")
    for band in runbands:
        output = format_band_dependence(outputbase, band)
        tableForEllipseRoutine = extractEllipseParamsfromWISE(BASEDIR, WISErow,
                band)
        createEllipseParamTable(objectdir, tableForEllipseRoutine, output, 
                columnfile)


def createEllipseParamTable(galaxydir, params, outputfile,
            colfile="../../ellipse_columns.txt"):
    '''Creates a parameter table for the ellipse routine.

    The table will essentially have the X0, Y0, SMA, PA, and ELLIP
    parameters in it, and will be a viable input to the inellip 
    parameter for the ellipse routine. This method assumes that those
    variables will be passed through the params argument as an astropy
    table.
    '''
    tempfile = os.path.join(galaxydir, "ellipsedata.txt")
    params.write(tempfile, format="ascii.no_header")
    iraf.tables()
    iraf.tables.ttools()
    # I call os.path.normpath here because I expect the column to be in
    # a parent directory. I guess to be absolutely safe, I should call
    # it on all input to IRAF tasks.
    iraf.tcreate(os.path.join(galaxydir, outputfile),
            os.path.normpath(os.path.join(galaxydir, colfile)), tempfile)

def extractEllipseParamsfromWISE(BASEDIR, WISErow, band):
    '''Converts WISE constraints to ellipse task constraintsu

    The directory information is needed to correct WCS
    information from the corresponding FITS file. This function will
    also only give parameters for one band. It will then return a table
    which contains the values appropriate for the ellipse task.
    
    This can be used in conjunction with the createEllipseParamTable
    function to create inellip apertures for the ellipse task.'''
    objectdir = change_to_galaxy_dir(BASEDIR, WISErow["objstr_01"])
    pixelscale = getPixelScale(band)
    # We first want the ellipticity:
    # There's a minimum value to the ellipticity, so we can't have it be less
    # than 0.05.
    ellipticity = max(0.05, 1 - WISErow["{0}ba".format(band.lower())])
    # Now for the semimajor axis
    sma = WISErow["{0}rsemi".format(band.lower())] / pixelscale
    # Now the position angle
    # Adding a small value in order to prevent convergence problems with PA=0
    pa = WISErow["{0}pa".format(band.lower())]+0.001
    if pa > 90:
        pa -= 180
    # The next two items are the X center and Y center.
    FITS_image = match_filter(objectdir, band)
    hdulist = fits.open(FITS_image)
    w = wcs.WCS(hdulist[0].header)
    coord = np.array([[WISErow["ra"], WISErow["dec"]]])
    x, y = w.wcs_world2pix(coord, 1)[0]
    values = [[ellipticity], [sma], [pa], [x], [y]]
    data_table = Table(values, names=("ELLIP", "SMA", "PA", "X0", "Y0"))
    return data_table


def STSDAS_to_Astropy_Table(workdir, filename, outputfile=None):
    '''Converts data in STSDAS table to an Astropy table.

    All that's needed is a filename. If an output file is desired, then 
    it can be specified as well.'''
    colfile = os.path.join(workdir, "magcolumns.txt")
    datfile = os.path.join(workdir, "magdata.txt")
    iraf.tables()
    iraf.tables.ttools()
    iraf.tdump.setParam("cdfile", colfile)
    iraf.tdump.setParam("datafile", datfile)
    iraf.tdump(os.path.join(workdir, filename))
    columns = Table.read(colfile, format="ascii.no_header")
    data = Table.read(datfile, format="ascii")
    fulldata = Table(data, names=columns['col1'])
    if outputfile:
        fulldata.write(outputfile)
    return fulldata

def download_WISE_images(BASEDIR, objstr, ra, dec):
    '''Downloads the WISE Atlas image and places it into the pipeline.

    For simplicity, RA/DEC resolution for the object shoul be done already using
    the IPAC table validator.
    '''
    coaddID = query.query_metadata(ra, dec)
    query.query_image(BASEDIR, objstr, coaddID)

# This can be fixed pretty easily by making runbands a mandatory argument, and
# then constructing Columns while iterating. I'm pretty sure those can be added 
# to a Table more easily than Rows.
def aperturePhotometryTable(BASEDIR, objectnames, runbands=bands,
        ellipseoutput="ellipse_aperture", skybase="sky_level", 
        uncertaintybase="uncertainty", flux=False, apertureCorrection=True, 
        colorIndices=None):
    '''Creates a table with generated aperture photometry.

    The magnitudes will be located in columns labeled "w?apmag". All magnitudes
    will be given in the AB system.
    '''
    fulltable = Table([objectnames], names=["objstr_01"])
    for band in runbands:
        bandmags, magerrs = photometryOnBand(BASEDIR, objectnames, band, 
                ellipseoutput, skybase, uncertaintybase, flux=flux, 
                errors=True, apertureCorrection=apertureCorrection,
                colorIndices=colorIndices)

        # How to keep the column name within our standard. Though I suppose we
        # could just change the standard. Look into that. See if the current
        # standard is hard-coded somewhere, or if that's just what I've been
        # doing to stay consistent with the WISE values.
        # We want w1apmag and NUVapmag.
        magtemplate = "{0}apmag"
        errtemplate = "{0}aperr"
        if band in IRBANDS:
            magname, errname = tuple([template.format(band.lower()) for template
                in [magtemplate, errtemplate]])
        else:
            magname, errname = tuple([template.format(band) for template
                in [magtemplate, errtemplate]])

        fulltable[magname] = bandmags
        fulltable[errname] = magerrs
    
    return fulltable

def photometryOnBand(BASEDIR, objectnames, band,
        baseobjectfile="ellipse_aperture", skybase="sky_level", 
        uncertaintybase="uncertainty", flux=False, errors=False, 
        apertureCorrection=True, colorIndices=None):
    '''Performs photometry on an array of objects in a given band.
    
    If flux is given as true, the flux of the object will be given in Janskys
    rather than the default magnitudes.
    
    If errors is true, then instead of simply returning an array of values, this
    function will return a 2-tuple with the flux/mag value in the first
    position, and the error in the second position.'''
    # photOutput can either be a list, or a list of 2-tuples if error was
    # specified.
    if colorIndices is not None:
        if len(colorIndices) is not len(objectnames):
            raise ValueError("Need same number of color indices and objects.")
        photOutput = [galaxy_photometry(BASEDIR, galname, band, baseobjectfile, 
            skybase, flux=flux, errors=errors,
            apertureCorrection=apertureCorrection, colorIndex=colorIndex) for 
            galname, colorIndex in zip(objectnames, colorIndices)]
    else:
        photOutput = [galaxy_photometry(BASEDIR, galname, band, baseobjectfile, 
            skybase, flux=flux, errors=errors,
            apertureCorrection=apertureCorrection, colorIndex=-2) for galname 
            in objectnames]
    if errors:
        magsAndErrs = zip(*photOutput)
        return np.array(magsAndErrs[0]), np.array(magsAndErrs[1])
    else: 
        return np.array(photOutput)

def createDifferencePlot(xval, valtocompare, xerror, valerror, xlabel, ylabel,
        title, label=''):
    '''Plots the difference between two values against the value.

    This plot is used for illustrating how consistent two datasets are
    from each other.'''
    difference, errors = calc_statistical_difference(valtocompare, xval, valerr,
            xerror)
    plt.errorbar(xval, difference, errors, fmt="o", label=label)
    plt.plot([min(xval)+0.01, max(xval)-0.01], [0, 0], 'k-')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

def doubleDifferencePlot(xfirst, xsecond, yfirst, ysecond, xfirsterr,
        xseconderr, yfirsterr, yseconderr, xlabel, ylabel, title, label=""):
    '''Makes a plot of one difference of quantities vs another difference.

    This plot can be used to add even more information about data consistency
    than a single difference plot.
    '''
    xdiff, xerrs = calc_statistical_difference(xfirst, xsecond, xfirsterr,
            xseconderr)
    ydiff, yerrs = calc_statistical_difference(yfirst, ysecond, yfirsterr,
            yseconderr)
    maxdiff = max(np.absolute(xdiff).max(), np.absolute(ydiff).max())
    plt.errorbar(xdiff, ydiff, yerrs, xerrs, fmt=".", label=label)
    plt.plot([-maxdiff - 0.2, maxdiff + 0.2], [0, 0], 'k-')
    plt.plot([0, 0], [-maxdiff - 0.2, maxdiff + 0.2], 'k-')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

def calc_statistical_difference(minuend, subtrahend, minuerr, subtraerr):
    '''Returns statistically subtracted value of two arrays.

    This function takes two arrays involving two measurements with errors. It
    then returns a 2-tuple. The first is simply the difference of the mean. The
    second is the errors of the differences.
    '''
    means = minuend - subtrahend
    meanerrs = np.sqrt(minuerr**2 + subtraerr**2)
    return means, meanerrs

def createFractionalDifferencePlot(xval, valtocompare, xerror, valerror, 
        xlabel, ylabel, title, label=""):
    '''Plots the fractional difference between two values against one value.

    The valtocompare is the minuend while the x value is the subtrahend. The
    errors in the plot are determined using standard propagation of errors.'''
    fracdiff = (valtocompare - xval) / xval
    errors = np.sqrt((valerror / xval)**2 + (xerror * valtocompare / 
        xval**2)**2)
    plt.errorbar(xval, fracdiff, errors, fmt="o", label=label)
    plt.plot([10**np.floor(np.log10(min(xval)) + 0.01),
        10**np.ceil(np.log10(max(xval))- 0.01)], [0, 0], 'k-')
    plt.xscale("log")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

def ColorHistogramByClass(band1, band2, groups, xlabel, title, bins, xrange=(-4,
    4)):
    '''Creates a histogam for colors for different classes.
    '''
    color = band1 - band2
    colorgroup = color.group_by(groups)
    plt.hist(colorgroup.groups, bins, range=xrange, label=["Class {0}".format(i)
        for i in range(5)], color=["black", "yellow", "green", "blue", "red"], histtype="bar")
    plt.xlabel(xlabel)
    plt.ylabel("N")
    plt.title(title)
    plt.legend()

def makePlots(BASEDIR, w1mags, w2mags, w3mags, w1apmags, w2apmags, w3apmags):
    '''Plots the WISE photometry versus aperture photometry.
    '''
    plt.plot(w1mags, w1apmags - w1mags, "rx", label="W1")
    plt.plot(w2mags, w2apmags - w2mags, "g*", label="W2")
    plt.plot(w3mags, w3apmags - w3mags, "b+", label="W3")

    plt.xlabel("WISE magnitudes")
    plt.ylabel("Magnitude Difference (Ap-WISE)")
    plt.title("Magnitude matches")
    plt.legend(loc="upper left")

def MIRplot(x, y, groupkey, xlabel='', ylabel='', title='', loc='upper right'):
    '''Makes a plot that automatically differentiates between MIR classes.

    The x and y data need to be columns which have the same length as 
    groupkey. Groupkey should be the list of MIR classes which are in the same
    order as x and y. Labels can also be added as desired.
    '''
    xgroup = x.group_by(groupkey)
    ygroup = y.group_by(groupkey)

    for MIRclass in xgroup.groups.keys:
        plt.plot(xgroup.groups[MIRclass], ygroup.groups[MIRclass], 
        label="Class {0}".format(MIRclass), **MIR_Symbols[MIRclass])

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc=loc)

def generateCMDs(magtable):
    '''Generates permutations of Color-Magnitude Diagrams.

    The diagrams that are generated should be considered "sensible",
    which means a color between UV and IR, with a magnitude that's 
    either UV or IR.'''
    MIRclass = magtable["MIR class"]
    W1 = magtable["w1apmag"]
    W2 = magtable["w2apmag"]
    W3 = magtable["w3apmag"]
    FUV = magtable["FUVapmags"]
    NUV = magtable["NUVapmags"]
    F1color = FUV - W1
    F2color = FUV - W2
    F3color = FUV - W3
    N1color = NUV - W1
    N2color = NUV - W2
    N3color = NUV - W3

    title = "WISE CMD"
    plt.figure()
    MIRplot(W1, F1color, MIRclass, "W1", "FUV-W1", title, loc="lower left")
    plt.figure()
    MIRplot(W2, F2color, MIRclass, "W2", "FUV-W2", title, loc="lower left")
    plt.figure()
    MIRplot(W3, F3color, MIRclass, "W3", "FUV-W3", title)
    plt.figure()
    MIRplot(W1, N1color, MIRclass, "W1", "NUV-W1", title, loc="lower left")
    plt.figure()
    MIRplot(W2, N2color, MIRclass, "W2", "NUV-W2", title, loc="lower left")
    plt.figure()
    MIRplot(W3, N3color, MIRclass, "W3", "NUV-W3", title)

def generateColorColors2MASS(magtable):
    '''Generates permutations of Color-Color Diagrams.
  
    The produced diagrams considered sensible involve a cross-band
    with a intra-band color. For example, a UV-IR vs. an IR-IR color.
    '''
    MIRclass = magtable["MIR class"]
    W1 = magtable["j_m_k20fe"]
    W2 = magtable["h_m_k20fe"]
    W3 = magtable["k_m_k20fe"]
    FUV = magtable["FUVapmags"]
    NUV = magtable["NUVapmags"]
    F1color = FUV - W1
    F2color = FUV - W2
    F3color = FUV - W3
    N1color = NUV - W1
    N2color = NUV - W2
    N3color = NUV - W3
    W12color = W1-W2
    W23color = W2-W3
    FNcolor = FUV - NUV

    title = "2MASS CMD"
    plt.figure()
    MIRplot(F1color, W12color, MIRclass, "FUV-J", "J-H", title,
            loc="lower left")
    plt.figure()
    MIRplot(F1color, W23color, MIRclass, "FUV-J", "H-K", title,
            loc="lower right")
    plt.figure()
    MIRplot(F1color, FNcolor, MIRclass, "FUV-J", "FUV-NUV", title,
            loc="left")
    plt.figure()
    MIRplot(F2color, W12color, MIRclass, "FUV-H", "J-H", title,
            loc="lower left")
    plt.figure()
    MIRplot(F2color, W23color, MIRclass, "FUV-H", "H-K", title,
            loc="lower right")
    plt.figure()
    MIRplot(F2color, FNcolor, MIRclass, "FUV-H", "FUV-NUV", title,
            loc="lower right")
    plt.figure()
    MIRplot(F3color, W12color, MIRclass, "FUV-K", "J-H", title,
            loc="lower left")
    plt.figure()
    MIRplot(F3color, W23color, MIRclass, "FUV-K", "H-K", title,
            loc="lower right")
    plt.figure()
    MIRplot(F3color, FNcolor, MIRclass, "FUV-K", "FUV-NUV", title,
            loc="upper left")
    plt.figure()
    MIRplot(N1color, W12color, MIRclass, "NUV-J", "J-H", title,
            loc="lower right")
    plt.figure()
    MIRplot(N1color, W23color, MIRclass, "NUV-J", "H-K", title,
            loc="lower right")
    plt.figure()
    MIRplot(N1color, FNcolor, MIRclass, "NUV-J", "FUV-NUV", title,
            loc="upper left")
    plt.figure()
    MIRplot(N2color, W12color, MIRclass, "NUV-H", "J-H", title,
            loc="lower right")
    plt.figure()
    MIRplot(N2color, W23color, MIRclass, "NUV-H", "H-K", title,
            loc="lower right")
    plt.figure()
    MIRplot(N2color, FNcolor, MIRclass, "NUV-H", "FUV-NUV", title,
            loc="upper right")
    plt.figure()
    MIRplot(N3color, W12color, MIRclass, "NUV-K", "J-H", title,
            loc="lower right")
    plt.figure()
    MIRplot(N3color, W23color, MIRclass, "NUV-K", "H-K", title, 
            loc="upper left")
    plt.figure()
    MIRplot(N3color, FNcolor, MIRclass, "NUV-K", "FUV-NUV", title, 
    loc="lower left")

def generateColorColors(magtable):
    '''Generates permutations of Color-Color Diagrams.
  
    The produced diagrams considered sensible involve a cross-band
    with a intra-band color. For example, a UV-IR vs. an IR-IR color.
    '''
    MIRclass = magtable["MIR class"]
    W1 = magtable["w1apmag"]
    W2 = magtable["w2apmag"]
    W3 = magtable["w3apmag"]
    FUV = magtable["FUVapmags"]
    NUV = magtable["NUVapmags"]
    F1color = FUV - W1
    F2color = FUV - W2
    F3color = FUV - W3
    N1color = NUV - W1
    N2color = NUV - W2
    N3color = NUV - W3
    W12color = W1-W2
    W23color = W2-W3
    FNcolor = FUV - NUV

    title = "WISE CMD"
    plt.figure()
    MIRplot(F1color, W12color, MIRclass, "FUV-W1", "W1-W2", title)
    plt.figure()
    MIRplot(F1color, W23color, MIRclass, "FUV-W1", "W2-W3", title,
            loc="lower left")
    plt.figure()
    MIRplot(F1color, FNcolor, MIRclass, "FUV-W1", "FUV-NUV", title,
            loc="upper left")
    plt.figure()
    MIRplot(F2color, W12color, MIRclass, "FUV-W2", "W1-W2", title,
            loc="upper left")
    plt.figure()
    MIRplot(F2color, W23color, MIRclass, "FUV-W2", "W2-W3", title,
            loc="lower left")
    plt.figure()
    MIRplot(F2color, FNcolor, MIRclass, "FUV-W2", "FUV-NUV", title,
            loc="upper left")
    plt.figure()
    MIRplot(F3color, W12color, MIRclass, "FUV-W3", "W1-W2", title,
            loc="upper left")
    plt.figure()
    MIRplot(F3color, W23color, MIRclass, "FUV-W3", "W2-W3", title,
            loc="lower right")
    plt.figure()
    MIRplot(F3color, FNcolor, MIRclass, "FUV-W3", "FUV-NUV", title)
    plt.figure()
    MIRplot(N1color, W12color, MIRclass, "NUV-W1", "W1-W2", title,
            loc="upper left")
    plt.figure()
    MIRplot(N1color, W23color, MIRclass, "NUV-W1", "W2-W3", title,
            loc="lower left")
    plt.figure()
    MIRplot(N1color, FNcolor, MIRclass, "NUV-W1", "FUV-NUV", title,
            loc="upper left")
    plt.figure()
    MIRplot(N2color, W12color, MIRclass, "NUV-W2", "W1-W2", title,
            loc="upper left")
    plt.figure()
    MIRplot(N2color, W23color, MIRclass, "NUV-W2", "W2-W3", title,
            loc="lower left")
    plt.figure()
    MIRplot(N2color, FNcolor, MIRclass, "NUV-W2", "FUV-NUV", title,
            loc="upper left")
    plt.figure()
    MIRplot(N3color, W12color, MIRclass, "NUV-W3", "W1-W2", title,
            loc="upper left")
    plt.figure()
    MIRplot(N3color, W23color, MIRclass, "NUV-W3", "W2-W3", title, 
            loc="lower right")
    plt.figure()
    MIRplot(N3color, FNcolor, MIRclass, "NUV-W3", "FUV-NUV", title)
    
def plotWithVerticalLines(xvalues, yvalues, specialx, xlabel="", ylabel="",
        title=""):
    '''Makes a plot with vertical lines

    This will make a plo with points. The x and y values of the points should be
    given as xvalues and yvalues. The specialx should be a list of floats which
    specify where on the x axis the vertical lines should be.'''

    plt.plot(xvalues, yvalues, 'r*')
    for x in specialx:
        ymin, ymax = plt.ylim()
        plt.plot([x, x], [ymin, ymax], 'k-')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

def makeSkyProfile(image, center, startradius, npoints, dannulus, skyname):
    '''Makes a profile of the sky level.

    The profile begins at startradius pixels, and then continues for npoints
    taking steps of dannulus. The rings contain no overlap.
    '''
    radii = np.linspace(startradius, startradius + npoints * dannulus, dannulus)
    skyprofile = []
    skyerrs = []
    for radius in radii:
        run_fitsky(image, radius, center, skyname)
        skyvalues = Table.read(skyname, format="ascii.daophot")
        skyprofile.append(skyvalues["MSKY"])
        skyerrs.append(skyvalues["STDEV"])

    plt.errorbar(radii, skyprofile, skyerrs)
    plt.xlabel("Radius (pixels)")
    plt.ylabel("Sky Level (DN)")
    plt.title("Sky Profile")

def plotSkyResults(BASEDIR, name, band, basename, limit, colorIndex=2,
        flux=False):
    '''Uses sky files generated by generate_sky_profile_files to plot
    magnitudes.

    This function will see how the background annulus affects the magnitude
    of an object.
    '''
    radii = []
    mags = []
    errors = []
    for i in xrange(limit):
        skybase = "{0}.{1}".format(basename, i)
        mag, err = galaxy_photometry(BASEDIR, name, band, useskybase=skybase,
                flux=flux)
        print mag, err
        skyinfo = Table.read(format_band_dependence(skybase, band, "txt",
                change_to_galaxy_dir(BASEDIR, name)), format="ascii.daophot")
        radius = (float(skyinfo.meta["keywords"]["ANNULUS"]["value"]) * 
                float(skyinfo.meta["keywords"]["SCALE"]["value"]))
        radii.append(radius)
        mags.append(mag)
        errors.append(err)
    plt.errorbar(radii, mags, errors)
    plt.xlabel("Annulus Radius (pixel)")
    plt.ylabel("Source {0}".format(band))
    plt.title("Magnitude dependence on background annulus for {0}".format(name))

    
def generate_sky_profile_files(galaxydir, band, startradius, npoints, dannulus, 
        basename, coordbase="fitsky"):
    '''Creates a series of sky files which sample from a radial profile.

    The profile begins at startradius pixels, and then continues for npoints,
    taking steps of dannulus. The rings contain no overlap. The output will be
    ordered sequentially starting with basename and ending with band.txt.'''
    image = match_filter(galaxydir, band)
    radii = np.arange(startradius, startradius + npoints * dannulus, dannulus)
    coordpath = format_band_dependence(coordbase, band, "coo", galaxydir)
    for i, radius in enumerate(radii):
        run_fitsky(image, radius, coordpath, 
                format_band_dependence("{0}.{1}".format(basename, i), band,
                    "txt", galaxydir), dannulus)


def classifyAgeColor(color, age, boundaries):
    '''Makes a plot with objects on a W2-W3 vs age plane, and class boundaries.

    The boundaries should be a 4-tuple of 2-tuples with low and high boundaries,
    respectively.
    '''
    plt.plot(color, age, 'r*')
    ylims = plt.ylim()
    plt.plot([boundaries[0][0]]*2, ylims, 'k-', label="Class 0")
    plt.plot([boundaries[0][1]]*2, ylims, 'k-')
    plt.plot([boundaries[1][0]]*2, ylims, 'y-', label="Class 1")
    plt.plot([boundaries[1][1]]*2, ylims, 'y-')
    plt.plot([boundaries[2][0]]*2, ylims, 'g-', label="Class 2")
    plt.plot([boundaries[2][1]]*2, ylims, 'g-')
    plt.plot([boundaries[3][0]]*2, ylims, 'b-', label="Class 3")
    plt.plot([boundaries[3][1]]*2, ylims, 'b-')
    plt.plot([boundaries[4][0]]*2, ylims, 'r-', label="Class 4")
    plt.plot([boundaries[4][1]]*2, ylims, 'r-')

    plt.xlabel("W2-W3")
    plt.ylabel("Age (Gyr)")
    plt.legend()
    plt.title("Plot to Classify Galaxies")


def elliptical_fit_with_WISE(BASEDIR, WISErow, fixedParams=True):
    '''Performs elliptical photometry on an image given WISE info.

    The image should be a path to the image. The WISE row should come
    from the WISE table and have the axis ratio, semimajor axis,
    coordinates, and position angle.
    
    To hold ellipse parameters other than semimajor axis fixed, set 
    fixedParams equal to True. Otherwise, they'll be allowed to vary as
    well.'''
    galaxydir = change_to_galaxy_dir(BASEDIR, WISErow["objstr_01"])
    # We only want backgrounds for IR images. UV images don't converge.
    for band in bands[:4]:
        pixelscale = getPixelScale(band)
        FITS_Image = match_filter(galaxydir, band)
        hdulist = fits.open(FITS_Image)
        w = wcs.WCS(hdulist[0].header)
        coord = np.array([[WISErow["ra"], WISErow["dec"]]])
        center = tuple(w.wcs_world2pix(coord, 1)[0])
        ellipticity = 1 - WISErow["{0}ba".format("w1")]
        sma = WISErow["{0}rsemi".format("w1")] / pixelscale
        pa = WISErow["{0}pa".format("w1")]

        elliptical_fit(galaxydir, FITS_Image, center, ellipticity, pa, sma,
                outputname=format_band_dependence("elliptical_fits", band),
                holdParamsFixed=fixedParams)


def elliptical_fit(galaxydir, image, center, ellipticity, position_angle,
        semimajor_axis, outputname="output.tbl", holdParamsFixed=True):
    '''Generates a set of ellipse fit files.
    
    This function uses the given ellipse parameters as starting values
    for the next fit, but also allows them to vary as chosen by the
    ellipse algorithm. The end result is a STSDAS table with a name
    given by the outputname keyword.
    
    If you would like to hold the given parameters fixed, only letting
    the semimajor axis vary, set holdParamsFixed equal to true.
    Otherwise, all parameters will be fitted.'''
    if holdParamsFixed:
        fix = "Yes"
    else:
        fix = "No"
    if position_angle > 90:
        position_angle -= 180
    iraf.stsdas()
    iraf.stsdas.analysis()
    iraf.stsdas.analysis.isophote()
    iraf.unlearn("ellipse", "geompar", "controlpar")
    iraf.geompar.setParam("x0", center[0])
    iraf.geompar.setParam("y0", center[1])
    iraf.geompar.setParam("ellip0", max(0.05, ellipticity))
    iraf.geompar.setParam("pa0", position_angle)
    iraf.geompar.setParam("sma0", semimajor_axis)
    iraf.controlpar.setParam("hcenter", fix)
    iraf.controlpar.setParam("hellip", fix)
    iraf.controlpar.setParam("hpa", fix)
    iraf.ellipse(os.path.join(galaxydir, image), os.path.join(galaxydir,
            outputname))






###############################################################################
# Miscellaneous Photometry Routines
###############################################################################

def Marino_Table_1_to_WISE_table(Marino_Table1, hyperledatable, refrac=8):
    '''Converts Table 1 from Marino et al. into a WISE table.

    The columns will be renamed: the apertures given as nuvrsemi and fuvrsemi
    will be the re/4 values. Although the D25 values capture the entire galaxy,
    they're much more difficult to get since they are not in the paper itself,
    but have to be retrieved from HYPERLEDA. I don't feel like doing that for a
    test of UV photometry.
    '''
    mt1 = Marino_Table1
    hlt = hyperledatable
    objstr = mt1["Ident."]
    ra, dec = hlt["al2000"]/24*360, hlt["de2000"]
    nuvrsemi = fuvrsemi = mt1["re"] / refrac

    nuvpa = fuvpa = hlt["pa"].filled(0.05)
    nuvba = fuvba = 10**(-hlt["logr25"])
    converted_table = Convert_to_UV_Table(objstr, ra, dec, nuvrsemi, fuvrsemi,
            nuvpa, fuvpa, nuvba, fuvba)
    return converted_table

def Hyperleda_Table_to_WISE_Table(hyperledatable):
    '''Converts the output of HYPERLEDA to a WISE table.

    This table uses the D25 aperture as the aperture size. For other aperture
    sizes, make a different function that uses both the table from HYPERLEDA as
    well as Table 1.
    '''
    hlt = hyperledatable
    objstr = hlt["name"]
    ra, dec = hlt["al2000"]/24*360, hlt["de2000"]
    nuvrsemi = fuvrsemi = 0.1 * 60 * 10**(hlt["logd25"]) / 2
    # For objects which are nearly circular, it appears that the pa column is
    # left out. As a result, we should be able to fill it with whatever value we
    # want.
    nuvpa = fuvpa = hlt["pa"].filled(0.05)
    nuvba = fuvba = 10**(-hlt["logr25"])
    converted_table = Convert_to_UV_Table(objstr, ra, dec, nuvrsemi, fuvrsemi,
            nuvpa, fuvpa, nuvba, fuvba)
    return converted_table

def Jarrett_Table_2_to_WISE_table(Jarrett_table2):
    '''Converts Table 2 from Jarrett et al. into a WISE table.

    The main change is that columns are renamed. And the isophotal radii are
    given by R1 and R4. There will be one trick: the isophotal radii for W3 in
    elliptical galaxies will need to be inserted manually.'''
    # An alias to abbreviate these references.
    jt2 = Jarrett_table2
    # We'll first create the WISE table, and then change the W3 values as
    # needed. This is to prevent any weird aliasing issues.
    objstr = jt2["Name"]
    ra, dec = jt2["R.A."], jt2["Decl."]
    w1rsemi = w2rsemi = w3rsemi = jt2["R1_iso"]
    w4rsemi = jt2["R4_iso"]
    w1pa = w2pa = w3pa = w4pa = jt2["P.A."]
    w1ba = w2ba = w3ba = w4ba = jt2["Axis"]
    converted_table = Convert_to_WISE_Table(objstr, ra, dec, w1rsemi, w2rsemi, 
            w3rsemi, w4rsemi, w1pa, w2pa, w3pa, w4pa, w1ba, w2ba, w3ba, w4ba)
    # We now apply the W3 corrections to the elliptical galaxies, given by:
    # NGC 584:  52".9
    # NGC 777:  64".0
    # NGC 4486: 154".7 
    ## ASTROPY_BUG: We should have the commands:
    ## converted_table["w3rsemi"][astropy_table_index(converted_table,
    ## "objstr_01", "NGC 584")[0][0]] = 52.9
    ## and
    ## converted_table[astropy_table_index(converted_table,
    ## "objstr_01", "NGC 584")[0][0]]["w3rsemi"] = 52.9
    ## be equivalent. But they don't appear to be. Look into that!
    converted_table["w3rsemi"][astropy_table_index(converted_table, "objstr_01", 
            "NGC 584")[0][0]] = 52.9
    converted_table["w3rsemi"][astropy_table_index(converted_table, "objstr_01", 
            "NGC 777")[0][0]] = 64.0
    converted_table["w3rsemi"][astropy_table_index(converted_table, "objstr_01", 
            "NGC 4486")[0][0]] = 154.7
    return converted_table

def Convert_to_UV_Table(objstr, ra, dec, nuvrsemi, fuvrsemi, nuvpa, fuvpa,
        nuvba, fuvba):
    '''Creates a WISE Table of UV parameters from given arrays of objects.'''
    fulltable = [objstr, ra, dec, nuvrsemi, fuvrsemi, nuvpa, fuvpa, nuvba,
            fuvba]
    names = ("objstr_01", "ra", "dec", "nuvrsemi", "fuvrsemi", "nuvpa", "fuvpa",
            "nuvba", "fuvba")
    return Table(fulltable, names=names)

def Convert_to_WISE_Table(objstr, ra, dec, w1rsemi, w2rsemi, w3rsemi, w4rsemi, 
        w1pa, w2pa, w3pa, w4pa, w1ba, w2ba, w3ba, w4ba):
    '''Creates a WISE table from given arrays of objects.

    The table should be able to be found at
    $WISE/Jarrett_DB/WISE_Isophotal-Aperture_Photometry.txt.
    '''
    fulltable = [objstr, ra, dec, w1rsemi, w2rsemi, w3rsemi, w4rsemi, w1pa, w2pa,
            w3pa, w4pa, w1ba, w2ba, w3ba, w4ba]
    names = ("objstr_01", "ra", "dec", "w1rsemi", "w2rsemi", "w3rsemi", 
            "w4rsemi", "w1pa", "w2pa", "w3pa", "w4pa", "w1ba", "w2ba", "w3ba", 
            "w4ba")
    return Table(fulltable , names=names)

def write_pipeline_file(filename, **kwargs):
    '''Writes keyword arguments to a file.'''
    f = open(filename)
    for k,v in kwargs.iteritems:
        f.write("{0}: {1}\n".format(k, v))
    f.close()

def rreplace(s, old, new, occurrence):
    '''Behaves like string.replace(), except replaces from the right rather than
    from the left. This code taken from:

    http://stackoverflow.com/questions/2556108/how-to-replace-the-last-occurence-of-an-expression-in-a-string
    '''
    li = s.rsplit(old, occurrence)
    return new.join(li)
