#! /usr/bin/env python
"""Performs elliptical aperture photometry by calling IRAF routines."""
import os
import math
import glob

from pyraf import iraf
from astropy import wcs
from astropy.io import fits
from astropy.table import Table, Column
import numpy as np
import aplpy
import matplotlib
import matplotlib.pyplot as plt

import queries as query
import synthetic_photometry as synphot

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
        mask="", useskybase="sky_aperture", apertureCorrection=True):
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
        background = estimate_WISE_background(galaxydir, band)
    else:
        background = estimate_UV_background(galaxydir, band,
                baseellipsefile=baseobjectfile)

    fapcor = aperture_correction_factor(band)
    objectflux = fapcor * (DNflux - background * aperture_area)
    if objectflux < 0:
        raise ValueError("Measured negative flux for object.")
    return objectflux
    
def galaxy_photometry(BASEDIR, name, band, baseobjectfile="ellipse_aperture", 
        mask="", useskybase="sky_aperture", uncertaintybase="uncertainty", 
        flux=False, errors=True, apertureCorrection=True, colorIndex=None):
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
        DNflux = calc_DNflux(galaxydir, band, baseobjectfile, mask,
                useskybase)
    except ValueError:
        print "\nGot negative flux for {0}.\n".format(name)
        DNflux *= -1
    
    if errors:
        objectError = calc_DNerr(galaxydir, band)
        if flux:
            photvalue = DN_flux_to_Jy(band, DNflux, colorIndex)
            err = DN_err_to_Jansky_err(galaxydir, band, objectError,
                    DNflux=DNflux, colorIndex=colorIndex)
        else:
            photvalue = DNflux2WISEmag(band, DNflux)
            err = DN_err_to_mag_err(galaxydir, band, objectError,
                    DNflux=DNflux)
        return (photvalue, err)
    else:
        if flux:
            photvalue = DN_flux_to_Jy(band, DNflux, colorIndex)
        else:
            photvalue = DNflux2WISEmag(band, DNflux)
        return flux

def color_correction(band, index):
    '''Returns the color correction appropriate for a power law.

    This function will take a power law index and select the correct flux
    correction factor for the band. To use the correction factor, divide the
    uncorrected zero-point by the factor to obtain the corrected zero-point.

    Note that now index can be a numpy array!
    '''
    fluxcorrection = {"W1": np.array([1.0283, 1.0084, 0.9961, 0.9907, 0.9921, 
        1.0000, 1.0142, 1.0347]),
        "W2": np.array([1.0206, 1.0066, 0.9976, 0.9935, 0.9943, 1.0000, 1.0107,
            1.0265]),
        "W3": np.array([1.1344, 1.0088, 0.9393, 0.9169, 0.9373, 1.0000, 1.0181,
            1.2687]),
        "W4": np.array([1.0142, 1.0013, 0.9934, 0.9905, 0.9926, 1.0000, 1.0130,
            1.0319])}
    return fluxcorrection[band][3-index]

def DN_to_Jy_conversion(band, colorIndex=-2):
    '''Returns the conversion factor between Data Numbers and Janskys for band.

    Taken from:
    http://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec2_3f.html#tbl1
    '''
    DN_to_Jy = {"W1": 1.9350e-6, "W2": 2.7048e-06, "W3": 1.8326e-06, "W4":
            5.2269e-05}
    return DN_to_Jy[band] / color_correct(band, colorIndex)


def DN_flux_to_Jy(band, objectflux, colorIndex=-2):
    '''Converts a flux from Data Numbers to Janskys.
    '''
    return objectflux * DN_to_Jy_conversion(band, colorIndex)

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

    fapcor = aperture_correction_factor(band)
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
        DNflux=0, colorIndex=-2):
    '''Converts an error in Data Numbers to an error in Janskys.

    If DNflux is given, this function will use it as the value for the object's
    flux in data numbers. If it isn't, then it will calculate it on its own.
    '''

    if not DNflux:
        DNflux = calc_DNflux(galaxydir, band, baseobjectfile, mask, useskybase)
    sigma_Jy = DN_to_Jy_conversion(band, colorIndex) * (DNflux**2 *
            (get_zero_point_flux_uncertainty(band)**2 / 
            get_zero_point_flux_level(band, colorIndex)**2 + 
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

def build_pipeline(BASEDIR, WISETable, runbands=bands):
    '''Basically runs all the commands necessary to build the ellipse aperture
    and sky measurement pipeline. It consists of running:
    allApertureTables
    allEllipseTables
    allSkyValues

    If you want a table of photometry, you'll have to run
    aperturePhotometryTable yourself.
    ''' 
    allApertureTables(BASEDIR, WISETable, runbands=runbands)
    allEllipseTables(BASEDIR, WISETable, runbands=runbands)
    allSkyValues(BASEDIR, WISETable, runbands=runbands)
    allUncertaintyTables(BASEDIR, WISETable, runbands=runbands)

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

def get_zero_point_flux_level(band, colorIndex=-2):
    '''Returns the zero-point flux level for a band in Janskys.

    According to the WISE Explanatory supplement, the zero-point flux level is
    the quantity which depends on the SED of the object, not actually something
    to do with the flux or magnitude of objects. So, I'm hoping it will be most
    appropriate to install the color corrections into this function.

    Taken from:
    http://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec2_3f.html#tbl1
    '''
    f0 = {"W1": 306.682, "W2": 170.663, "W3": 29.0448, "W4": 8.2839}
    corrected_zero_point = f0[band] / color_correction(band, colorIndex)
    return corrected_zero_point

def get_zero_point_flux_uncertainty(band):
    '''Returns the zero-point flux uncertainty for a band in Janskys.

    Taken from:
    http://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec2_3f.html#tbl1
    '''
    sig_f0 = {"W1": 4.6, "W2": 2.6, "W3": 0.436, "W4": 0.124}
    return sig_f0[band]


def aperture_correction_factor(band):
    '''Returns the aperture correction factor for a given band.

    This factor multiplies the background-subtracted flux of an object in order
    to correct for the light lost from the creation of the ATLAS images.
    '''
    LARGE_APERTURE_CORRECTION = {"W1": -0.034, "W2": -0.041, "W3": 0.03, "W4": 
            -0.029}
    return 10**(LARGE_APERTURE_CORRECTION[band]/2.5)

def change_to_galaxy_dir(BASEDIR, objectname):
    '''Returns the path of a galaxy's directory.

    Given the name of an object, it will return the full path to the
    directory containing all of the data concerning that object.
    '''
    return os.path.join(BASEDIR, object_name_to_dir(objectname))

def format_band_dependence(basename, band, extension="tab", pathto=''):
    '''Generates a table file which is dependent on a band name.

    The returned filename will have a format of 
    "{basename}.{band}.{extension}".
    '''
    return os.path.join(pathto, "{0}.{1}.{2}".format(basename, band, extension))
    
def match_filter(directory, filter, fullpath=True, uncertainty=False):
    '''Finds the image which corresponds to the filter.

    For WISE images, this will require searching for "w?" in the
    strings.'''
    filtermap = {"W1": "w1-int", "W2": "w2-int", "W3": "w3-int", "W4": "w4-int", 
            "FUV": "fd-int", "NUV": "nd-int"}
    filterstring = filtermap[filter]
    if uncertainty:
        filterstring.replace("int", "unc")
    filelist = glob.glob(os.path.join(directory, 
            "*{0}*.fits".format(filterstring)))
    if len(filelist) > 1:
        raise RuntimeError("Image conflict for {0}.".format(directory))
    elif len(filelist) == 0:
        raise RuntimeError("Could not find file in {0}.".format(directory))
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
        algorithm="centroid"):
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

def estimate_UV_background(galaxydir, band, baseellipsefile="sky_aperture"):
    '''Returns the estimated background for a galax in GALEX bands.

    The background for UV bands is estimated by looking for files whose
    names contain the string given in skymarker.'''
    ellipsetable = STSDAS_to_Astropy_Table(galaxydir,
            format_band_dependence(baseobjectfile, band, "tab"))
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

def allApertureTables(BASEDIR, fulltable, runbands=bands):
    '''Goes through BASEDIR and generates all the aperture tables.

    The full WISE table will be necessary.'''
    runOnImages(BASEDIR, fulltable, genApertureTable, runbands=runbands)

def allSkyTables(BASEDIR, fulltable, runbands=bands):
    '''Goes through BASEDIR and generates all sky tables.

    This function also allows for single-object corrections to be made.
    '''
    runOnImages(BASEDIR, fulltable, genSkytables, runbands=runbands)

def allUncertaintyTables(BASEDIR, fulltable, runbands=bands):
    '''Goes through BASEDIR and generates all uncertainty tables.'''
    runOnImages(BASEDIR, fulltable, genImageUncertainty, runbands=runbands)

def allSkyValues(BASEDIR, fulltable, runbands=bands):
    '''Goes through BASEDIR and generates all sky tables.

    This function also allows for single-object corrections to be made.
    '''
    runOnImages(BASEDIR, fulltable, genSkyValues, runbands=runbands)

def allEllipseTables(BASEDIR, fulltable, runbands=bands):
    '''Goes through BASEDIR and generates all object tables.

    This function also allows for single-object corrections to be made.
    '''
    runOnImages(BASEDIR, fulltable, genEllipsetables, runbands=runbands)

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
        ellipsebase="ellipse_aperture", runbands=bands):
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
        intfile = match_filter(galaxydir, band)
        uncfile = rreplace(intfile, "int", "unc", 1)
        varfile = rreplace(uncfile, "unc", "var", 1)

        # There's a really shitty IRAF "feature" where if imfunc acts on a file
        # which already exists, it will simply add on another layer, which
        # confuses the hell out of ellipse. So if a previous file exists, I'll
        # delete it manually.
        if os.path.isfile(varfile):
            os.remove(varfile)
        run_imfunc(uncfile, varfile, "square")

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
    iraf.images()
    iraf.imutil()
    iraf.imfunc(infile, outfile, func)



def genEllipsetables(BASEDIR, WISErow, baseparamname="ellipsepars",
        baseoutput="ellipse_aperture", runbands=bands):
    '''Generates a table on the object for each band.'''
    # There should be a better way of joining this and genSkyTables, but that's
    # taking too much effort, and I want to just have this part done.
    galaxydir = change_to_galaxy_dir(BASEDIR, WISErow["objstr_01"])
    for band in runbands:
        objimage = match_filter(galaxydir, band)
        run_ellipse(objimage, format_band_dependence(baseparamname, band, 'tab',
            galaxydir), format_band_dependence(baseoutput, band, 'tab', 
            galaxydir))

def genSkyValues(BASEDIR, WISErow, coordbase="fitsky",
        ellipsebase="ellipse_aperture", baseskyfile="sky_level", skyratio=2.0,
        annulus=0, dannulus=10, runbands=bands):
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
    # aperture. If we don't make a separate annulus_override variable, setting
    # annulus for W1 is disable resetting it for W2-4.
    annulus_override = annulus
    for band in runbands:
        coordpath = format_band_dependence(coordbase, band, "coo", galaxydir)
        skypath = format_band_dependence(baseskyfile, band, "txt", galaxydir)
        image = match_filter(galaxydir, band)
        ellipsepars = STSDAS_to_Astropy_Table(galaxydir,
                format_band_dependence(ellipsebase, band, "tab"))

        if not annulus_override:
            annulus = skyratio * ellipsepars["SMA"]
        
        with open(coordpath, 'w') as f:
            f.write("{0} {1}".format(ellipsepars["X0"][0], ellipsepars["Y0"][0]))

        run_fitsky(image, annulus, coordpath, skypath, 
                dannulus=dannulus)

def generateEllipseCutouts(BASEDIR, WISEtable, runbands=IRBANDS):
    '''Runs through all objects and creates cutouts in their folder.
    '''
    current_backend = matplotlib.get_backend()
    matplotlib.use("Agg")
    runOnImages(BASEDIR, WISEtable, createEllipseCutouts, runbands=runbands)
    matplotlib.use(current_backend)

def createEllipseCutouts(BASEDIR, WISErow, runbands=IRBANDS):
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
        skypars = Table.read(os.path.join(galaxydir,
            format_band_dependence("sky_level", band, "txt")),
            format="ascii.daophot")
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
        Xval, Yval = gc.pixel2world(skypars["XINIT"][0], skypars["YINIT"][0])
        radius_in = (float(skypars.meta["keywords"]["ANNULUS"]["value"]) * px /
            3600.0)
        radius_out = (radius_in +
            float(skypars.meta["keywords"]["DANNULUS"]["value"]) * px / 3600.0)
        gc.show_circles([Xval]*2, [Yval]*2, [radius_in, radius_out],
                edgecolor="cyan")

        gc.save(format_band_dependence(object_name_to_dir(WISErow["objstr_01"]), 
            band, "png", galaxydir))

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
    ellipticity = max(0.05, 1 - WISErow["{0}ba".format("w1")])
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
        baseobjectfile="ellipse_aperture", mask="foreground.pl",
        skybase="sky_level", uncertaintybase="uncertainty",  
        ellipsebase="ellipsepars", flux=False, apertureCorrection=True,
        colorIndices=None):
    '''Creates a table with generated aperture photometry.

    The magnitudes will be located in columns labeled "w?apmag". All magnitudes
    will be given in the AB system.
    '''
    fulltable = Table([objectnames], names=["objstr_01"])
    for band in runbands:
        bandmags, magerrs = photometryOnBand(BASEDIR, objectnames, band, 
                baseobjectfile, mask, skybase, uncertaintybase, flux=flux, 
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
        baseobjectfile="ellipse_aperture", mask="foreground.pl",
        skybase="sky_aperture", uncertaintybase="uncertainty", flux=False, 
        errors=False, apertureCorrection=True, colorIndices=None):
    '''Performs photometry on an array of objects in a given band.
    
    If flux is given as true, the flux of the object will be given in Janskys
    rather than the default magnitudes.
    
    If errors is true, then instead of simply returning an array of values, this
    function will return a 2-tuple with the flux/mag value in the first
    position, and the error in the second position.'''
    # photOutput can either be a list, or a list of 2-tuples if error was
    # specified.
    if colorIndices is not None:
        photOutput = [galaxy_photometry(BASEDIR, galname, band, baseobjectfile, 
            mask, skybase, flux=flux, errors=errors,
            apertureCorrection=apertureCorrection, colorIndex=colorIndex) for 
            galname, colorIndex in zip(objectnames, colorIndices)]
    else:
        photOutput = [galaxy_photometry(BASEDIR, galname, band, baseobjectfile, 
            mask, skybase, flux=flux, errors=errors,
            apertureCorrection=apertureCorrection, colorIndex=None) for galname 
            in objectnames]
    if errors:
        magsAndErrs = zip(*photOutput)
        return np.array(magsAndErrs[0]), np.array(magsAndErrs[1])
    else: 
        return np.array(photOutput)

def createDifferencePlot(xval, valtocompare, errors, xlabel, ylabel, title):
    '''Plots the difference between two values against the value.

    This plot is used for illustrating how consistent two datasets are
    from each other.'''
    difference = valtocompare - xval
    plt.errorbar(xval, difference, errors, fmt="o")
    plt.plot([min(xval)+0.01, max(xval)-0.01], [0, 0], 'k-')
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

def flux2mag(flux, zeropoint):
    '''Converts from a flux to a magnitude given a zero-point.'''
    return zeropoint - 2.5 * math.log10(flux)

def Vega2ABmag(band, vegamag):
    '''Converts Vega magnitudes to AB magnitudes.

    Conversions to AB magnitudes given by the WISE Explanatory
    Supplement. Section IV.4.h.3.'''
    offsets = {"W1": 2.699, "W2": 3.339, "W3": 5.174, "W4": 6.620}
    return vegamag + offsets[band]

def DNflux2WISEmag(band, flux):
    '''Converts the flux from a WISE Atlas image to a WISE magnitude.

    The flux needs to be given in units of data numbers. The infrared
    fluxes will be returned in the Vega system while the UV fluxes will
    be returned in the AB system.
    '''
    zeropoints = {"W1": 20.5, "W2": 19.5, "W3": 18.0, "W4": 13.0, "NUV": 20.08,
                  "FUV": 18.82}
    return flux2mag(flux, zeropoints[band])

def Jansky2WISEmag(band, flux, colorIndex=-2):
    '''Converts a flux in Janskys to a WISE magnitude.'''
    # There may be a unifying way of doing this. But I'm not feeling it right
    # now.
    return -2.5 * np.log10(flux/get_zero_point_flux_level(band, colorIndex))

def Jansky_err_to_WISE_mag_err(band, flux, fluxerr):
    '''Converts an error in Janskys to an error in WISE magnitude.'''
    mag_err = np.sqrt(1.179 * ((fluxerr / flux)**2 +
        (get_zero_point_flux_uncertainty(band) / 
            get_zero_point_flux_level(band))**2))
    return mag_err


###############################################################################
# Miscellaneous Photometry Routines
###############################################################################
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

def rreplace(s, old, new, occurrence):
    '''Behaves like string.replace(), except replaces from the right rather than
    from the left. This code taken from:

    http://stackoverflow.com/questions/2556108/how-to-replace-the-last-occurence-of-an-expression-in-a-string
    '''
    li = s.rsplit(old, occurrence)
    return new.join(li)
