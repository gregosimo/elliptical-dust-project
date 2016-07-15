#! /usr/bin/env python
"""Performs elliptical aperture photometry by calling IRAF routines."""
import os
import math
import glob
import warnings

from pyraf import iraf
from astropy import wcs
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.stats import sigma_clip
from astropy.table import Table, Column, join, vstack
from astroquery.ned import Ned
import numpy as np
import aplpy
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats.mstats

import masks
import queries as query
import synthetic_photometry as synphot
import band_conversions as conv
import statop as stat
import astropy_util as au

bands=["W1", "W2", "W3", "W4", "NUV", "FUV"]
IRBANDS = bands[:4]
UVBANDS = bands[4:]
TWOMASSBANDS = ["J", "H", "Ks"]
MASKBANDS = ["W1", "NUV"]
#STSDAS_COLUMN = "/home/gregory/work/ellipse_columns.txt"
STSDAS_COLUMN = "/home/regulus/simonian/year1/wise/ellipse_columns.txt"


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
        useskybase="sky_level", skymethod="adaptive", apertureCorrection=True):
    '''Calculates the flux of a galaxy in Data Numbers.

    This function uses the output from the ellipse package to calculate the
    background-subtracted flux of the galaxy. The total flux is calculated from
    the ellipse package and is stored in the table named with baseobjectfile.
    The sky values are determined from the file named with useskybase. The
    method for determining sky values should be specified in skymethod; it can
    be "adaptive", "annulus", or "skyfile".
    '''
    ellipsetable = STSDAS_to_Astropy_Table(
        format_band_dependence(baseobjectfile, band, "tab", galaxydir))
    DNflux = ellipsetable[0]["TFLUX_E"]
    aperture_area = ellipsetable[0]["NPIX_E"]
    # Right now we will only support sky backgrounds done through the 
    # pipeline.
    background = read_background(galaxydir, band, useskybase, skymethod)
    if np.ma.is_masked(background):
        raise ValueError("Background not found!")
    if apertureCorrection:
        fapcor = aperture_correction_factor(band)
    else:
        fapcor = 1
    objectflux = fapcor * (DNflux - background * aperture_area)
    if objectflux < 0:
        raise ValueError("Measured negative flux for object.")
    return objectflux
    
def galaxy_photometry(BASEDIR, name, band, baseobjectfile="ellipse_aperture", 
        useskybase="sky_level", skymethod="adaptive", 
        uncertaintybase="uncertainty", ZPuncertainty=True, brightness="AB",
        errors=True, apertureCorrection=True, colorIndex=-2, max_mag_err=0.3):
    '''Returns the elliptical aperture photometry-determined magnitude.

    This function requires that the adequate pipeline be constructed, where
    there is a folder tree under BASEDIR where each object maps to a folder
    labeled as the object name without spaces. For example, "NGC 1111" would be
    under the folder "NGC1111".

    Under each folder, there should be three sets of files. The file containing
    the total flux outputted by the ellipse package should be labeled as 
    "baseobjectfile.{band}.tab". The file with the information about the sky 
    should be labeled with the base of "useskybase.{band}.{suf}" where suf
    depends on the value passed to skymethod. If skymethod is "annulus", the
    suffix should be "txt" since the file should be the output of fitsky. If
    skymethod is "skyfile", the suffix should be "tab" since the file should be
    the output of ellipse. "Adaptive" should adapt to the necessary sky
    measurement methods.

    The max_mag_err keyword indicates how large the magnitude error should be
    before the object is flagged as a non-detection. The way non-detections are
    marked is by having a blank field for the magnitude, and putting the upper
    magnitude limit in the magnitude error field. Although this is somewhat
    counterintuitive, it will make sure that upper limits are not accidentally
    plotted as actual values.
    '''
    galaxydir = os.path.join(BASEDIR, object_name_to_dir(name))
    DNflux = calc_DNflux(galaxydir, band, baseobjectfile, useskybase,
            skymethod, apertureCorrection=apertureCorrection)
    
    if errors:
        objectError = calc_DNerr(galaxydir, band, ellipsebase=baseobjectfile,
                skybase=useskybase, skymethod=skymethod, 
                uncertainty_base=uncertaintybase)
        if brightness is "flux":
            photvalue = conv.DN_flux_to_Jy(band, DNflux, colorIndex)
            err = conv.DN_err_to_Jansky_err(galaxydir, band, objectError,
                    DNflux, ZPunc=ZPuncertainty, colorIndex=colorIndex)
        else:
            if brightness is "AB":
                photvalue = conv.DNflux2ABmag(band, DNflux)
            else:
                photvalue = conv.DNflux2Vegamag(band, DNflux)
            err = conv.DN_err_to_mag_err(galaxydir, band, objectError,
                    DNflux, ZPunc=ZPuncertainty)
            if err >= max_mag_err:
                fluxupperlimit = 3.0 * objectError
                err = np.nan
                # I don't want to duplicate this, but I don't feel like making
                # a better logic. I think this entire function should be
                # trimmed down to not return anything other than (photvalue,
                # err) tuples.
                if brightness is "AB":
                    photvalue = conv.DNflux2ABmag(band, fluxupperlimit)
                else: 
                    photvalue = conv.DNflux2Vegamag(band, fluxupperlimit)
        return (photvalue, err)
    else:
        if brightness is "flux":
            photvalue = conv.DN_flux_to_Jy(band, DNflux, colorIndex)
        elif brightness is "Vega":
            photvalue = conv.DNflux2Vegamag(band, DNflux)
        else:
            photvalue = conv.DNflux2ABmag(band, DNflux)
            
        return photvalue

def build_pipeline(
        BASEDIR, WISETable, maskthresh=0, foregroundbase="foreground",
        pixelmaskbase="bad_pixels", maskbase="mask", maskconfig="", 
        skipmask=False, overwritemask=False, ellipsepars="ellipsepars", 
        ellipseoutput="ellipse_aperture", regionbase="ellipseregions", 
        skycoord="fitsky", clip_background=False, skybase="sky_level", 
        skygens="adaptive", skyratio=1.5, uncertaintybase="uncertainty", 
        runbands=bands, ignore_exceptions=False):
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
    generate the mask and the output filename of mask. If masks have aleady been
    generated and you'd simply like to skip the creation, enable the skipmasks
    keyword.
    ''' 
    print "Making Aperture Tables..."
    allApertureTables(BASEDIR, WISETable, runbands=runbands, 
                      outputbase=ellipsepars,
                      ignore_exception=ignore_exceptions)
    print "Generating Regions..."
    generateRegions(BASEDIR, WISETable, outputbase=regionbase,
                    parambase=ellipsepars, runbands=runbands,
                    ignore_exception=ignore_exceptions)
    if not skipmask:
        print "Making Masks..."
        masks.build_masks(BASEDIR, WISETable, threshold=maskthresh,
                          runbands=runbands, foregroundbase=foregroundbase,
                          pixelmaskbase=pixelmaskbase, outputbase=maskbase, 
                          maskconfig=maskconfig, overwrite=overwritemask,
                          ignore_exception=ignore_exceptions,
                          regionbase=regionbase)
    print "Making Ellipse Tables..."
    allEllipseTables(BASEDIR, WISETable, runbands=runbands, maskbase=maskbase, 
                     baseoutput=ellipseoutput, baseparamname=ellipsepars,
                     ignore_exception=ignore_exceptions)
    print "Making Sky Tables..."
    allSkyValues(BASEDIR, WISETable, runbands=runbands, coordbase=skycoord, 
                 baseskyfile=skybase, skygens=skygens, ellipsebase=ellipsepars, 
                 maskbase=maskbase, skyratio=skyratio, clip=clip_background,
                 ignore_exception=ignore_exceptions)
    print "Making Uncertainty Tables..."
    allUncertaintyTables(BASEDIR, WISETable, runbands=runbands, 
                         ellipsebase=ellipsepars, 
                         baseuncertainty=uncertaintybase, skybase=skybase, 
                         skymethod=skygens, ignore_exception=ignore_exceptions)
    write_pipeline_file("{0}.par".format(ellipsepars), 
                        mask_threshold=maskthresh, mask_output=maskbase, 
                        ellipse_parameters=ellipsepars, 
                        mask_config_base=maskconfig, 
                        aperture_file=ellipseoutput, sky_coordinates=skycoord, 
                        sky_base=skybase, uncertainty_base=uncertaintybase, 
                        bands_written=runbands)
    print "Done!"


def fullphotometry(BASEDIR, WISE_Table):
    '''Performs the pipeline  building and photometry calculation of a table.

    This function is good for when the ultimate goal of an image set is just to
    get photometry out. The build_pipeline() and aperturePhotometryTable()
    functions get called as a unit.

    This function can also be used to generate unique prefixes for the objects.
    '''
    pass

def aperture_correction_factor(band):
    '''Returns the aperture correction factor for a given band.

    This factor multiplies the background-subtracted flux of an object in order
    to correct for the light lost from the creation of the ATLAS images.
    '''
    LARGE_APERTURE_CORRECTION = {"W1": -0.034, "W2": -0.041, "W3": 0.03, "W4": 
            -0.029, "NUV": 0.0, "FUV": 0.0}
    return 10**(LARGE_APERTURE_CORRECTION[band]/2.5)

##############################################################################
# Background Routines #
##############################################################################

def sky_from_fitsky_file(galaxydir, band, baseskyfile="sky_level"):
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

def sky_from_patch_file(galaxydir, band, baseskyfile="sky_level"):
    '''Returns the estimated background for a particular galaxy.

    This function uses the routine calculated from patches on elliptical annuli
    in order to get an estimate of the background level.
    '''
    skypath = format_band_dependence(baseskyfile, band, "txt", pathto=galaxydir)
    skydata = Table.read(skypath, format="ascii.basic")
    skylevel = skydata["background"][0]
    return skylevel

def read_background(galaxydir, band, skybase="sky_level",
        skymethod="adaptive"):
    '''Background estimator for all bands.

    There are many diferent ways of estimating backgrounds based on different
    surveys. This function will sort through all the different ways and
    transparently return the corresponding background value without the user
    having to iterate through all of the cases.

    The sky base is the base file for sky files. The skymethod keyword
    determines which sky method will be used to estimate the background. The
    currently valid values are "annulus", "skyfile", "patch" and "adaptive".
    Annulus and skyfile both force that respective method to be used to estimate
    the sky while adaptive uses an annulus for WISE images and a skyfile for
    GALEX images. The type of estimation will also determine which suffixed will
    be affixed to the skybase, so that sky file will need to be built in the
    pipeline.
    '''
    if skymethod.lower() == "adaptive":
        skymethod = adaptive_background[band]

    if skymethod.lower() == "aperture":
        background = sky_from_ellipse_table(galaxydir, band,
                baseellipsefile=skybase)
    elif skymethod.lower() == "annulus":
        background = sky_from_fitsky_file(galaxydir, band,
                baseskyfile=skybase)
    elif skymethod.lower() == "patch":
        background = sky_from_patch_file(galaxydir, band, baseskyfile=skybase)

    # Background values should *not* be NaN.
    if np.isnan(background):
        galname = extract_name_from_galaxy_dir(galaxydir)
        raise ValueError("Sky background value is nan for {0}".format(galname))
    return background


def readSkyTable(galaxydir, band, area, baseellipsefile="sky_aperture"):
    '''Generates a background flux from an existing table.

    Since much of the calculations being done for objects is through a
    standard aperture size, it is easier to simply read the values from
    a pre-generated STSDAS table. This function is just for reading.
    '''
    ellipsetable = STSDAS_to_Astropy_Table(
        format_band_dependence(baseellipsefile, band, "tab", galaxydir))
    if band in IRBANDS:
        return float(ellipsetable[0]["INTENS"]) * area
    else:
        return float(ellipsetable[0]["TFLUX_E"])

def get_sky_pixels(galaxydir, band, skybase="sky_level", method="adaptive"):
    '''Extracts the number of pixels used to determine the sky value.

    This function is meant to retrieve the sky pixels based on the method used
    to determine the background.
    '''
    if method.lower() == "adaptive":
        method = adaptive_background[band]
    if method.lower() == "patch":
        skyParams=Table.read(os.path.join(galaxydir,
            format_band_dependence(skybase, band, "txt")), format="ascii.basic")
        pixels = skyParams["patches"][0]
    elif method.lower() == "annulus":
        skyParams = Table.read(os.path.join(galaxydir,
            format_band_dependence(skybase, band, "txt")),
            format="ascii.daophot")
        pixels = skyParams["NSKY"][0]
    return pixels


def sky_from_ellipse_table(galaxydir, band, baseellipsefile="sky_level"):
    '''Returns the estimated background for a galax in GALEX bands.

    The background for UV bands is estimated by looking for files whose
    names contain the string given in skymarker.'''
    ellipsetable = STSDAS_to_Astropy_Table(
        format_band_dependence(baseellipsefile, band, "tab", galaxydir))
    return ellipsetable[0]["TFLUX_E"] / ellipsetable[0]["NPIX_E"]


##############################################################################
# Noise Routines #
##############################################################################

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
        skybase="sky_level", skymethod="adaptive", 
        uncertainty_base="uncertainty"):
    '''Calculates the uncertainty of a Data Number flux.

    This function requires bases for the ellipse routine, sky routine, and
    uncertainty routine. It will use values from these files to calculate the
    uncertainty of the flux based on the description given in the WISE All-sky
    explanatory Supplement:
    http://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec2_3f.html
    '''
    ellipseParams = STSDAS_to_Astropy_Table(
        format_band_dependence(ellipsebase, band, "tab", galaxydir))[0]
    imageUncertainty = STSDAS_to_Astropy_Table(
        format_band_dependence(uncertainty_base, band, "tab", galaxydir))[0]

    fapcor = aperture_correction_factor(band)
    NA = ellipseParams["NPIX_E"]
    # TODO if time remains, make this robust to the sky method.
    NB = get_sky_pixels(galaxydir, band, skybase, skymethod)
    sig_B = get_sky_error(galaxydir, band, skybase, skymethod)
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

def get_sky_error(galaxydir, band, skybase="sky_level", method="adaptive"):
    '''Extracts the error in the sky measurement from a method.

    This function is meant to retrieve sky errors based on which method was used
    to pick them out.
    '''
    if method.lower() == "adaptive":
        method = adaptive_background[band]
    if method.lower() == "patch":
        skyParams=Table.read(os.path.join(galaxydir,
            format_band_dependence(skybase, band, "txt")), format="ascii.basic")
        error = skyParams["error"][0]
    elif method.lower() == "annulus":
        skyParams=Table.read(os.path.join(galaxydir,
            format_band_dependence(skybase, band, "txt")), 
            format="ascii.daophot")
        error = skyParams["STDEV"][0]
    return error
    
##############################################################################
# Path Routines #
##############################################################################

def object_name_to_dir(objectname):
    '''Converts the object name with spaces to the directory name.'''
    if isinstance(objectname, np.ndarray):
        newobj = np.core.defchararray.replace(objectname, " ", "")
    elif isinstance(objectname, str):
        newobj = objectname.replace(' ', "")
    else:
        raise TypeError("Incorrect type passed to convert to directory.")
    return newobj

def change_to_galaxy_dir(BASEDIR, objectname):
    '''Returns the path of a galaxy's directory.

    Given the name of an object, it will return the full path to the
    directory containing all of the data concerning that object.
    '''
    return os.path.join(BASEDIR, object_name_to_dir(objectname), "")

def split_galaxy_dir(galaxydir):
    '''Splits a galaxydir into the basename and galaxy name.

    A galaxydir should be given in the form of "$BASEDIR/NGCXXXX/". This
    function will split the galaxydir into a tuple of "$BASEDIR" and "NGCXXXX". 
    It essentially functions as the inverse of change_to_galaxy_dir.
    '''
    return os.path.split(os.path.dirname(galaxydir))
    

def extract_name_from_galaxy_dir(galaxydir):
    '''Takes a galaxydir and extracts the name of the galaxy from it.

    A galaxydir should be given in the form of "$BASEDIR/NGCXXXX/". This
    function will prune off the "NGCXXXX" portion of the galaxydir. 
    '''
    return os.path.basename(os.path.dirname(galaxydir))

def format_band_dependence(basename, band, extension="tab", pathto=''):
    '''Generates a table file which is dependent on a band name.

    The returned filename will have a format of 
    "/pathto/{basename}.{band}.{extension}".
    '''
    return os.path.join(pathto, "{0}.{1}.{2}".format(basename, band, 
                                                     extension))
    
def objectHasImage(BASEDIR, objname):
    '''Checks if an object has a folder containing its images.'''
    return os.path.exists(change_to_galaxy_dir(BASEDIR, objname))

def filterTableforExistingObjects(BASEDIR, fulltable):
    '''Creates another table that only has the objects with images.'''
    return filterTable(BASEDIR, fulltable, objectHasImage)

def filter_table_for_complete_bands(
        BASEDIR, fulltable, copy=True, completebands=bands, galcol="objstr_01"):
    '''Returns a table that only has objects with complete observations'''
    return filterTableforCompleteBands(BASEDIR, fulltable, copy, completebands,
                                       galcol)

def filterTableforCompleteBands(
        BASEDIR, fulltable, copy=True, galcol="objstr_01", completebands=bands):
    '''Returns a table that only has objects with complete observations'''
    return filterTable(BASEDIR, fulltable, complete_for_bands, copy=copy,
                       checkbands=completebands, galcol=galcol)

def match_filter(directory, band, fullpath=True, uncertainty=False, 
        sky=False):
    '''Finds the image which corresponds to the filter.

    This is a great way of determining what the filename is for an image for a
    given galaxy for a given band. The only two type currently installed are
    WISE images as well as GALEX images. These images need to be present in the
    directory in order for this function to work. They also need to have
    retained their original names. When the fullpath option is disabled, only
    the image name will be returned, not the full path of the image.
    
    There are a number of options to match_filter to change what type of image
    you would like. The uncertainty option returns the path to the uncertainty
    image (whether or not one exists). The sky option returns the path to the
    sky background image (whether or not that exists).'''
    filtermap = {"W1": "w1-int", "W2": "w2-int", "W3": "w3-int", "W4": "w4-int",
            "FUV": "fd-int", "NUV": "nd-int"}
    filterstring = filtermap[band]
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

def load_image(galaxydir, band, mask="", uncertainty=False, sky=False):
    '''Loads a FITS image of a galaxy.

    This is an extension of match_filter which not just gets the image filename,
    but rather loads the entire FITS image.
    '''
    imagepath = match_filter(galaxydir, band, uncertainty=uncertainty, sky=sky)
    maskpath = os.path.join(galaxydir, mask)

    try:
        mask = fits.getdata(maskpath)
    except IOError:
        mask = np.ma.nomask
    image = np.ma.array(fits.getdata(imagepath), mask=mask)
    # image = mask_invalid_areas(image, band)
    return image

# Convert this into the bad aperture detection function. In order to do that
# we'll need to:
def detect_aperture_out_of_bounds(image, ):
    '''Masks out parts of the image which weren't exposed to the sky.

    In particular, the GALEX image only has a circular region which contains sky
    information. The rest is just zero, and is really annoying to work around
    for objects that are close to the edge. In order to get around this, those
    parts will just be masked out.
    '''
    xcenter, ycenter = (1913, 1941)
    radius = 1451
    if band in UVBANDS:
        return np.ma.array(image, mask=np.logical_not(mask_circle(image,
            xcenter, ycenter, radius)))

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

def runOnImages(BASEDIR, fulltable, func, **kwargs):
    '''Goes through a table of objects and runs a function on them.

    The function must be able to accept the BASEDIR as well as an
    Astropy row. The function can also accept keyword arguments via
    kwargs.'''
    try:
        ignore_exception = kwargs.pop("ignore_exception")
    except KeyError:
        ignore_exception = False
    for row in fulltable:
        galaxydir = change_to_galaxy_dir(BASEDIR, row["objstr_01"])
        try:
            func(BASEDIR, row, **kwargs)
        except (Exception, iraf.IrafError) as e:
            if ignore_exception:
                print e
            else:
                raise

def get_ellipse_output_tables(
        BASEDIR, galaxies, band, aperturebase="ellipse_aperture"):
    '''Returns a table containing all ellipse outputs for objects in BASEDIR.

    This routine basically loops through all of the given objects in BASEDIR,
    and returns a concatenated table of all of the outputs of the ellipse
    routine. This relies on the assumption that all of the STSDAS tables of
    aperturebase have only one entry.
    '''
    tablelist = []
    for gal in galaxies:
        galaxydir = change_to_galaxy_dir(BASEDIR, gal)
        aperturepath = format_band_dependence(
            aperturebase, band, "tab", galaxydir)
        ellipsetable = STSDAS_to_Astropy_Table(aperturepath)
        tablelist.append(ellipsetable)
    fulltable = vstack(tablelist)
    try:
        fulltable["objstr_01"] = galaxies
    except ValueError:
        print "Ellipse output tables have more than one row."
        raise
    return fulltable

def get_masked_fractions(
    BASEDIR, galaxies, band, aperturebase="ellipse_aperture"):
    '''Returns an array containing the fraction of masked pixels for galaxies.

    The output will probably not be exact, since the total number of pixels in
    the ellipse, valid and invalid, is not included in the output of ellipse.
    Therefore, the total area will be calculated from the area of the ellipse.
    It's not guaranteed that this will be the same number. If needed,
    additional investigations can occur.'''
    aperture_output = get_ellipse_output_tables(BASEDIR, galaxies, band,
                                                aperturebase)
    fullareas = (math.pi * aperture_output["SMA"]**2 *
        (1-aperture_output["ELLIP"]))
    frac_areas = 1 - aperture_output["NPIX_E"] / fullareas
    return frac_areas

##############################################################################
# Deprecated functions? #
##############################################################################

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

##############################################################################
# IRAF Wrappers #
##############################################################################

def run_fitsky(image, annulus, coords, output, dannulus=10,
        algorithm="centroid", scale=1, fwhmpsf=6, sighighclip=3.0,
        siglowclip=3.0, rejectiter=10):
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
    iraf.fitskypars.setParam("shireject", sighighclip)
    iraf.fitskypars.setParam("sloreject", siglowclip)
    iraf.fitskypars.setParam("snreject", rejectiter)
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
    iraf.ellipse.setParam("interactive", False)
    iraf.ellipse(image, output)

    

##############################################################################
# Region routines #
#############################################################################

def writeregion(BASEDIR, WISErow, parambase="ellipsepars",
        outputbase="ellipseregion", runbands=bands):
    '''Writes a shitty DS9 region file.

    This writes a file with only one line, taking the physical coordinates of
    the object. Making a good region writer probably won't be that hard, but I
    don't feel like it.'''
    galaxydir = change_to_galaxy_dir(BASEDIR, WISErow["objstr_01"])
    for band in runbands:
        ellipsetable = STSDAS_to_Astropy_Table(
            format_band_dependence(parambase, band, "tab", galaxydir))[0]
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


##############################################################################
# Contamination Routines #
##############################################################################

def filterContaminatedObjects(BASEDIR, fulltable):
    '''Returns a table which doesn't have contaminated objects'''
    return filterTable(BASEDIR, fulltable, (lambda BASEDIR, objname: not
        isObjectContaminated(BASEDIR, objname)))

def isObjectContaminated(BASEDIR, objname, contfile="Nearby_Stars.txt"):
    '''Determines if an object is on a list containing contaminated
    galaxies.

    The file listing contaminated galaxies should be given in the
    contfile keyword.'''
    contfileobj = open(os.path.join(BASEDIR, contfile))
    contobjects = contfileobj.readlines()
    return (objname+"\n") in contobjects

##############################################################################
# Pipeline Functions #
##############################################################################

def allMasks(
        BASEDIR, fulltable, maskband, threshold=5, foregroundbase="foreground",
        pixelmaskbase="bad_pixels", outputbase="mask", maskconfig="", 
        ignore_exception=False, overwrite=False, regionbase="ellipseregion"):
    '''Goes through BASEDIR and generates all of the foreground masks.'''

    runOnImages(BASEDIR, fulltable, masks.mask_algorithm, threshold=threshold,
            maskband=maskband, outputbase=outputbase, maskconfig=maskconfig,
            ignore_exception=ignore_exception, overwrite=overwrite,
            regionbase=regionbase, pixelmaskbase=pixelmaskbase)

def allApertureTables(BASEDIR, fulltable, runbands=bands,
        outputbase="ellipsepars", ignore_exception=False):
    '''Goes through BASEDIR and generates all the aperture tables.

    The full WISE table will be necessary.'''
    runOnImages(BASEDIR, fulltable, genApertureTable, runbands=runbands,
            outputbase=outputbase, ignore_exception=ignore_exception)

def allUncertaintyTables(BASEDIR, fulltable, baseuncertainty="uncertainty", 
        ellipsebase="ellipsepars", skybase="sky_level", skymethod="adaptive",
        runbands=bands, ignore_exception=False):
    '''Goes through BASEDIR and generates all uncertainty tables.'''
    runOnImages(BASEDIR, fulltable, genImageUncertainty,
            baseuncertainty=baseuncertainty, ellipsebase=ellipsebase,
            skybase=skybase, skymethod=skymethod, runbands=runbands,
            ignore_exception=ignore_exception)

def allSkyValues(BASEDIR, fulltable, runbands=bands, coordbase="fitsky", 
        baseskyfile="sky_level", skygens="adaptive", ellipsebase="ellipsepars",
        maskbase="foreground", skyratio=2.5, clip=False, 
        ignore_exception=False):
    '''Goes through BASEDIR and generates all sky tables.

    This function also allows for single-object corrections to be made.
    '''
    runOnImages(BASEDIR, fulltable, genSkyValues, runbands=runbands,
            coordbase=coordbase, baseskyfile=baseskyfile, skygens=skygens,
            ellipsebase=ellipsebase, skyratio=skyratio, clip=clip,
            ignore_exception=ignore_exception, maskbase=maskbase)

def allEllipseTables(BASEDIR, fulltable, runbands=bands, 
        maskbase="mask", baseoutput="ellipse_aperture",
        baseparamname="ellipsepars", ignore_exception=False):
    '''Goes through BASEDIR and generates all object tables.

    This function also allows for single-object corrections to be made.
    '''
    runOnImages(BASEDIR, fulltable, genEllipsetables,
            baseparamname=baseparamname, runbands=runbands,
            maskbase=maskbase, baseoutput=baseoutput,
            ignore_exception=ignore_exception)

def allSkyParams(BASEDIR, fulltable, runbands=bands, ignore_exception=False):
    '''Goes through BASEDIR and generates all sky parameter files.'''
    runOnImages(BASEDIR, fulltable, genSkyParam, runbands=runbands,
            ignore_exception=ignore_exception)

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
    if os.path.isfile(outfile):
        os.remove(outfile)

    iraf.images()
    iraf.imutil()
    iraf.imfunc(infile, outfile, func)


def genImageUncertainty(BASEDIR, WISErow, baseuncertainty="uncertainty",
        ellipsebase="ellipse_aperture", skybase="sky_level",
        skymethod="adaptive", runbands=bands):
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
                    baseuncertainty, skybase, skymethod)

def source_uncertainty_from_image(galaxydir, band, image, 
        ellipsebase="ellipsepars", outputbase="uncertainty", 
        skybase="sky_level", skymethod="adaptive"):
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

    skybackground = read_background(galaxydir, band, skybase=skybase,
            skymethod=skymethod)

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
    run_imfunc(uncfile, varfile, "square")


    ellipse_file = format_band_dependence(ellipsebase, band, "tab",
            galaxydir)
    output = format_band_dependence(outputbase, band, "tab", galaxydir)
    run_ellipse(varfile, ellipse_file, output)

def genEllipsetables(BASEDIR, WISErow, baseparamname="ellipsepars",
        baseoutput="ellipse_aperture", maskbase="mask", runbands=bands):
    '''Generates a table on the object for each band.
    
    It uses parameters provided in ellipsepars, and outputs the table into
    baseoutput.
    
    Masks may be provided in two ways. First priority is given to a custom mask
    provided in alt_mask. If alt_mask doesn't exist, then the regular mask
    passed to the mask keyword will work. This sounds like a ridiculous system,
    but it's the easiest thing to do with the code structured the way it
    currently is.'''
    # There should be a better way of joining this and genSkyTables, but that's
    # taking too much effort, and I want to just have this part done.
    galaxydir = change_to_galaxy_dir(BASEDIR, WISErow["objstr_01"])
    for band in runbands:
        objimage = match_filter(galaxydir, band)
        run_ellipse(objimage, format_band_dependence(baseparamname, band, 'tab',
            galaxydir), format_band_dependence(baseoutput, band, 'tab', 
            galaxydir), mask=format_band_dependence(maskbase, band, "fits",
            galaxydir))

def genSkyValues(BASEDIR, WISErow, coordbase="fitsky",
        ellipsebase="ellipsepars", baseskyfile="sky_level", skygens="adaptive", 
        skyratio=2.5, annulus=0, dannulus=30, runbands=bands, patch_area=4000,
        num_patches=90, maskbase="foreground", alt_mask="foreground_alt.fits",
        clip=False):
    '''Generates sky values for each galaxy.
    
    The sky values can be generated in two ways: through an annulus or through a
    given sky file. Which method is used depends on the value of skygens. Usable
    values are "all", "annulus", "skyfile", and "adaptive". "Annulus" and
    "skyfile" force all bands able to carry out that method to do so. "Adaptive"
    will only generate those files which are meant to be used for sky values.
    "All" generates all sky values which can be generated so that they will all
    be options for aperturePhotometryTable.
    
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
        if skygens.lower() == "adaptive":
            background_method = adaptive_background[band]
        else:
            background_method = skygens.lower()

        if background_method == "aperture":
            measure_sky_from_skyfile(galaxydir, band, baseskyfile, ellipsebase)
        elif background_method == "annulus":
            measure_sky_from_annulus(galaxydir, band, coordbase, baseskyfile, 
                                     ellipsebase, annulus, skyratio, dannulus)
        elif background_method == "patch":
            measure_sky_from_patches(
                galaxydir, band, patch_area, num_patches, baseskyfile, 
                ellipsebase, skyratio, 
                mask=format_band_dependence(maskbase, band, "fits"), clip=clip)

def measure_sky_from_patches(galaxydir, band, area=4000, numpatches=90,
        baseskyfile="sky_level", ellipsebase="ellipsepars", scale=1.5,
        mask="", backgroundmapbase="background_map", clip=False):
    '''Uses elliptical patches to measure the sky level from an image.

    This function measures the sky in two concentric elliptical apertures
    centered around the object, with the innermost boundary having size scale *
    the photometric aperture. The other aperture sizes will be chosen such that
    they will yield a total number of regions equal to numpatches, each
    containing area pixels.

    This function will then write to {baseskyfile}.{band}.txt with the
    background level as well as statistics about the method used to calculate
    the background.

    If backgroundmapbase is given, this function will also output a map
    illustrating the locations of the patches along with the value in each patch
    at {backgroundmapbase}.{band}.fits.
    '''
    ellipsepars = STSDAS_to_Astropy_Table(
        format_band_dependence(ellipsebase, band, "tab", galaxydir))
    backgroundmapfile = format_band_dependence(backgroundmapbase, band, "fits",
            galaxydir)
    imagename = match_filter(galaxydir, band, fullpath=False)
    image = load_image(galaxydir, band, mask=mask)
    xcenter, ycenter = ellipsepars["X0"][0], ellipsepars["Y0"][0]
    semimajor = scale * ellipsepars["SMA"][0]
    semiminor = semimajor * (1 - ellipsepars["ELLIP"][0])
    pa = ellipsepars["PA"][0]
    patchbackgrounds, patchstandards, scales = background_from_patches(image, 
            xcenter, ycenter, semimajor, semiminor, pa, area, numpatches,
            backgroundmapfile=backgroundmapfile)
    if clip:
        patchbackgrounds = sigma_clip(patchbackgrounds, 3, 5)
        patchstandards = np.ma.array(patchstandards, mask=patchbackgrounds.mask)
    skyfile = format_band_dependence(baseskyfile, band, "txt", galaxydir)
    write_background_from_patches(
        skyfile, imagename, xcenter, ycenter, scales["ainit"], scales["binit"], 
        scales["amid"], scales["bmid"], scales["aout"], scales["bout"], pa, 
        patchbackgrounds.mean(), patchbackgrounds.std(), patchstandards.mean(), 
        np.ma.count(patchbackgrounds), clip)

def write_background_from_patches(
        filename, imagename, x0, y0, a0, b0, a1, b1, a2, b2, pa, background, 
        error, dispersion, patches, clipped):
    '''Writes the file representing a background value written from patches.
    
    The first argument is the name of the file, with full path.
    
    It's then followed by:
    x0: x-coordinate of the center of the patch ellipse
    y0: y-coordinate of the center of the patch ellipse
    a0: Inner semimajor axis of the patch ellipse
    b0: Inner semiminor axis of the patch ellipse
    a1: Middle semimajor axis of the patch ellipse
    b1: Middle semiminor axis of the patch ellipse
    a2: Outer semimajor axis of the patch ellipse
    b2: Outer semiminor axis of the patch ellipse
    background: The mean background value of all of the patches
    error: The standard deviation of the patch means
    dispersion: The mean of the patch standard deviations
    patches: The number of patches used
    clippsed: Whether sigma-clipping was used on these patches.
    '''
    tableoutline = {
        "name": [imagename], "X0": [x0], "Y0": [y0], "A0": [a0], "B0": [b0], 
        "A1": [a1], "B1": [b1], "A2": [a2], "B2": [b2], "PA": [pa], 
        "background": [background], "error": [error], "dispersion": 
        [dispersion], "patches": [patches], "clipped":[clipped]} 
    backgroundtable = Table(tableoutline, names=["name", "X0", "Y0", 
            "A0", "B0", "A1", "B1", "A2", "B2", "PA", "background", "error", 
            "dispersion", "patches", "clipped"])
    backgroundtable.write(filename, format="ascii.basic")

def spoof_background_patch(filepath, skyvalue, dispersion, error):
    '''Makes a background which appears to come from patches, but is really
    just set by the values given above.'''
    write_background_from_patches(filepath, "dummy.fits", 1, 1, 1, 1, 2, 2, 3,
                                  3, 0, skyvalue, error, dispersion, 90,
                                  False)

def spoof_background_patches(BASEDIR, objects, skyvalues, dispersions, errors,
                             filename):
    '''Spoofs backgrounds for a list of objects.'''
    for galname, skyval, skydisp, skyerr in zip(objects, skyvalues,
                                                dispersions, errors):
        galaxydir = change_to_galaxy_dir(BASEDIR, galname)
        filepath = os.path.join(galaxydir, filename)
        spoof_background_patch(filepath, skyval, skydisp, skyerr)


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
    ellipsepars = STSDAS_to_Astropy_Table(
        format_band_dependence(ellipsebase, band, "tab", galaxydir))

    # The centroid algorithm doesn't converge for GALEX images because there are
    # too few counts.
    if band in UVBANDS:
        algorithm="mean"
        sighiclip=1.0
    else:
        algorithm="centroid"
        sighiclip=0.0

    if not annulus_override:
        annulus = skyratio * ellipsepars["SMA"]
    
    with open(coordpath, 'w') as f:
        f.write("{0} {1}".format(ellipsepars["X0"][0], ellipsepars["Y0"][0]))

    run_fitsky(image, annulus, coordpath, skypath, 
            dannulus=dannulus, fwhmpsf=getPSFFWHM(band, pixel=True),
            algorithm=algorithm)

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
    

def sky_file_background(galaxydir, band, ellipse_output_base="sky_level"):
    '''Returns sky uncertainty from sky file.

    This function calculates the variance of the sky by using quantiles. It is
    robust against outliers as well as local to the object. The output base
    should be the output of the ellipse routine to determine the sky level.
    '''
    ellipse_output = format_band_dependence(ellipse_output_base, band, "tab",
                                            galaxydir)
    skylevel = STSDAS_to_Astropy_Table(ellipse_output)
        
    image = match_filter(galaxydir, band, sky=True)
    imageval = fits.getdata(image, view=np.ma.MaskedArray)
    imageval.mask = ~mask_ellipse(imageval, skylevel["X0"], skylevel["Y0"],
        skylevel["SMA"] * (1 - skylevel["ELLIP"]), skylevel["SMA"],
        skylevel["PA"])

    skyquants = scipy.stats.mstats.mquantiles(imageval, [0.16, 0.5])
    skystd = skyquants[1] - skyquants[0]
    return skystd

def sky_annulus_histogram(galaxydir, band, fitsky_output_base="sky_level",
        bins=20, range=None):
    '''Plots a histogram of pixel values within a sky annulus.'''
    fitsky_output = format_band_dependence(fitsky_output_base, band, "txt",
            galaxydir)
    skypars = Table.read(fitsky_output, format="ascii.daophot")
    Xval, Yval = skypars["XINIT"][0], skypars["YINIT"][0]
    rin = float(skypars.meta["keywords"]["ANNULUS"]["value"])
    rout = rin + float(skypars.meta["keywords"]["DANNULUS"]["value"])

    image = match_filter(galaxydir, band)
    imageval = fits.getdata(image, view=np.ma.MaskedArray)
    imageval.mask = ~mask_annulus(imageval, Xval, Yval, rin, rout)
    annuluspixels = imageval.compressed()

    plt.hist(annuluspixels, bins=bins, range=range)
    plt.xlabel("Pixel value (count/s)")
    plt.ylabel("N")
    plt.title("{2} pixels centered at ({0:.0f}, {1:.0f}) ".format(Xval, Yval, 
        band) + "between {0:.0f} and {1:.0f} pixels".format(rin, rout))
    return imageval

def test_if_in_elliptical_shell_portion(
        x, y, xcenter, ycenter, ainner, binner, scale, pa, angle1, angle2):
    '''Tests if the point x,y lies within the portion of the elliptical shell.

    All of these arguments should be in terms of pixels, except for the angles
    and scale.  The angles should be given in units of degrees. Angle1 and 
    Angle2 should also be in the range of -180 to 180. A scale of 1 would make
    the outer ellipse identical to the inner ellipse.
    
    The angles are measured from the centers of the ellipses, not the foci.'''
    xcen = x - xcenter
    ycen = y - ycenter
    withininner = test_if_in_ellipse(x, y, xcenter, ycenter, ainner, binner, pa)
    withinouter = test_if_in_ellipse(
        x, y, xcenter, ycenter, ainner*scale, binner*scale, pa)
    inshell = np.logical_and(np.logical_not(withininner), withinouter)
    # This moves the angle from being 0 at (1, 0) to being 0 at the P.A.
    # i.e. this angle is zero on the major axis.
    angles = np.arctan2(ycen, xcen)*180 / math.pi 
    # Read this as angle1 <= angles < angle2
    inportion = np.logical_and(angle1 <= angles, angles < angle2)
    return np.logical_and(inshell, inportion)

def mask_elliptical_shell_portion(
        image, xcenter, ycenter, ainner, binner, scale, pa, angle1, angle2):
    image_coords = np.indices(image.shape)
    mask = np.logical_not(
        test_if_in_elliptical_shell_portion(
            image_coords[1], image_coords[0], xcenter-1, ycenter-1, ainner, 
            binner, scale, pa, angle1, angle2))
    return mask

def patch_background(
        image, xcenter, ycenter, ainner, binner, scale, pa, angle1, angle2, 
        bgmap=None):
    '''Calculates the background and error in an elliptical segment
    
    Takes an image as a numpy array, and then finds the background at the
    elliptical segment by taking the mean value. It also returns the standard
    deviation within that patch.

    This function also has the option of filling in a background map with the
    same size as the image. The region used to calculate the background will be
    set to the background value. This may be useful in determining which areas
    are particularly deviant in background value.
    '''
    ellipsewindow = mask_elliptical_shell_portion(
        image, xcenter, ycenter, ainner, binner, scale, pa, angle1, angle2)
    segmentimage = np.ma.array(image, mask=ellipsewindow)
    background = np.ma.mean(segmentimage)
    std = segmentimage.std()
    if bgmap is not None:
        bgmap[~segmentimage.mask] = background
    return (background, std)

def calculate_sky_ellipses(ainit, binit, totalarea):
    '''Calculates ellipse sizes for the Gil de Paz sky algorithm.

    This function starts with an initial ellipse, and then returns two
    elliptical annuli which each contain half of totalarea. This function will
    return a 4-tuple with amid, bmid, aout, bout.'''

    initarea = math.pi * ainit * binit
    annulusarea = totalarea / 2.0

    amid = ainit * np.sqrt(annulusarea / initarea + 1)
    midscale = amid / ainit
    bmid = binit * midscale
    midarea = math.pi * amid * bmid

    aout = amid * np.sqrt(annulusarea / midarea + 1)
    outscale = aout / amid
    bout = bmid * outscale

    return amid, bmid, aout, bout

def background_from_patches(
        fullimage, xcenter, ycenter, ainit, binit, pa, area, numpatches, 
        backgroundmapfile=''):
    '''Calculates background from a series of elliptical patches.

    There needs to be an initial specification of an ellipse, which is given by
    xcenter, ycenter, ainit, binit, and pa. From that initial ellipse, patches
    will be calculated to be between that ellipse and a larger ellipse chosen to
    make the patch areas close to area. Once all patches from the initial
    elliptical annulus are used, another annulus is made using the same
    algorithm with the previously larger annulus used as the smaller one. This
    process continues until the total number of patches is greater than
    minpatch.
    '''
    bgsample = np.ma.zeros(numpatches)
    stdsample = np.ma.zeros(numpatches)

    amid, bmid, aout, bout = calculate_sky_ellipses(
        ainit, binit, numpatches*area)

    # In order to move the numbers in and out of this function conveniently,
    # we'll put them in the scales dictionary.
    scales = {"ainit": ainit, "binit": binit, "amid": amid, "bmid": bmid,
            "aout": aout, "bout": bout}

    image = extract_from_image_with_height_width(
        fullimage, xcenter, ycenter, aout, aout)
    if backgroundmapfile:
        fullbgimage = np.ma.array(fullimage, copy=True)
        fullbgimage.fill(0)
        bgimage = extract_from_image_with_height_width(
            fullbgimage, xcenter, ycenter, aout, aout)
    else:
        bgimage=None

    newxcenter = newycenter = aout+5

    numsections = numpatches / 2
    print fullimage.shape

    angles = np.linspace(-180, 180, numsections+1)
    for (i, (angle1, angle2)) in enumerate(zip(angles[:-1], angles[1:])):
        midscale = amid / ainit
        # Because we're using a view centered on the actual image, we'll set 
        # the center bits to 0.
        bg, std = patch_background(
            image, newxcenter, newycenter, ainit, binit, midscale, pa, angle1, 
            angle2, bgmap=bgimage)
        bgsample[i] = bg
        stdsample[i] = std
    for (i, (angle1, angle2)) in enumerate(zip(angles[:-1], angles[1:])):
        outscale = aout / amid
        bg, std = patch_background(
            image, newxcenter, newycenter, amid, bmid, outscale, pa, angle1, 
            angle2, bgmap=bgimage)
        bgsample[i+numsections] = bg
        stdsample[i+numsections] = std
    if bgimage is not None:
        hdu = fits.PrimaryHDU(fullbgimage.data)
        hdulist = fits.HDUList([hdu])
        hdulist.writeto(backgroundmapfile, clobber=True)
    return bgsample, stdsample, scales

def test_if_in_ellipse(x, y, xcenter, ycenter, a, b, pa):
    '''Tests if the point x,y lies within the described ellipse.

    All of the arguments should make sense except for pa. PA should be given in
    degees E of N.
    '''
    alpha = (pa+90) * math.pi / 180.0
    xcen = x - xcenter
    ycen = y - ycenter
    return ((xcen * math.cos(alpha) + ycen * math.sin(alpha))**2 / a**2 + 
        (xcen * math.sin(alpha) - ycen * math.cos(alpha))**2 / b**2 < 1)

def test_if_in_annulus(x, y, xcenter, ycenter, rin, rout):
    '''Tests if the point x,y lies within the described annulus.

    Tests whether the object is within an annulus between rin and rout.
    '''
    xcen = x - xcenter
    ycen = y - ycenter
    return np.logical_and(
        (xcen**2 + ycen**2 <= rout**2), 
        (xcen**2 + ycen**2 >= rin**2))

def mask_ellipse(image, xcenter, ycenter, a, b, pa):
    '''Creates a mask on the image which is shaped like an ellipse.
    
    Image should be a fits image. And xcenter and ycenter should be in physical
    pixels, not numpy indices. The conversion will take place in this function.
    '''
    image_coords = np.indices(image.shape)
    mask = test_if_in_ellipse(
        image_coords[1], image_coords[0], xcenter-1, ycenter-1, a, b, pa)
    return mask

def mask_circle(image, xcenter, ycenter, radius):
    '''Creates a mask on the image which is shaped like a circle.

    Image should be a numpy array, and xcenter and center should be in physical
    pixels, not numpy indices; the conversion will take place in this function.
    '''
    return mask_ellipse(image, xcenter, ycenter, radius, radius, 0)
    
def mask_annulus(image, xcenter, ycenter, rin, rout): 
    '''Masks out a circular annulus on an image.

    Image should be a fits image. And xcenter and ycenter should be in physical
    pixels, not numpy indices. The conversion will take place in this
    function.'''
    image_coords = np.indices(image.shape)
    mask = test_if_in_annulus(
        image_coords[1], image_coords[0], xcenter-1, ycenter-1, rin, rout)
    return mask

def getPSFFWHM(band, pixel=False):
    '''Returns the effective size of a point source in an Atlas Image.

    Size taken from:
    http://wise2.ipac.caltech.edu/docs/release/allwise/expsup/sec4_4.html#coadbeam

    The numbers are returned in arcseconds by default. If you would prefer
    pixels, then set the pixel keyword to True.
    '''
    widths = {"W1": 8.3, "W2": 9.1, "W3": 9.5, "W4": 16.8, "NUV": 5.3, "FUV":
            4.2}
    chosenwidth = widths[band]
    if pixel:
        chosenwidth /= getPixelScale(band)
    return chosenwidth

def generateRegions(BASEDIR, WISEtable, outputbase="ellipseregion", 
        parambase="ellipsepars", runbands=bands, ignore_exception=False):
    '''Runs through all objects and make DS9 regions.
    '''
    runOnImages(
        BASEDIR, WISEtable, writeregion, outputbase=outputbase, 
        parambase=parambase, runbands=runbands, ignore_exception=ignore_exception)

def generateEllipseCutouts(BASEDIR, WISEtable, runbands=bands, 
        skyAperture=True, skyimage=False, skyprefix="sky_level",
        aperturefile="ellipse_aperture", ignore_exception=False,
        maskbase="mask", suffix="", sizescale=1.5):
    '''Runs through all objects and creates cutouts in their folder.
    '''
    current_backend = matplotlib.get_backend()
    matplotlib.use("Agg")
    reload(aplpy)
    runOnImages(
        BASEDIR, WISEtable, createEllipseCutouts, runbands=runbands, 
        skyAperture=skyAperture, skyimage=skyimage, skyprefix=skyprefix, 
        aperturefile=aperturefile, ignore_exception=ignore_exception,
        maskbase=maskbase, suffix=suffix, sizescale=sizescale)
    matplotlib.use(current_backend)

def ellipse_cutout_grid(
        BASEDIR, WISEtable, runbands=bands, skyAperture=True, skyimage=False, 
        skyprefix="sky_level", aperturefile="ellipse_aperture", 
        skymethod="adaptive", maskbase="mask", scalesize=1.5, 
        scale="ellipse", scalebarlength=30, scalebarband="FUV", 
        ignore_exception=False):
    '''Returns a figure with a grid of cutout figures.

    The horizontal grid tracks are different bands in runbands. The vertical
    grid tracks are different images in WISEtable.

    To display the sky aperture, enable the skyAperture flag. To show the sky
    image rather than the main exposure, enable the skyimage flag. In order to
    show the sky annulus, the skyprefix keyword is needed, which when
    band-expanded will point to the sky file for that band. This is assisted by
    skymethod, which will correctly extract the aperture size and shape from
    the correct filename.

    The aperturefile will, when band-expanded, point to the file which contains
    the elliptical aperture size for that band.

    The mask to be overlaid will be a file at the band-expanded maskbase.

    The size of the cutout will be given by scalelength and sizescale. The
    possibilities for scalelength are "ellipse", "sky", "W1", and "manual".
    "Ellipse" will set the size of the cutout to be sizescale * major axis of
    ellipse aperture. "Sky" will do the same, but with the sky aperture. "W1"
    will set all the cutouts to be sizescale * major axis of W1 ellipse
    aperture. A "manual" scalelength will set the size to sizescale in
    arcseconds.

    Lastly, ignore_exception will create the table while ignoring any
    exceptions which come up. This is not recommended for the final run.

    WARNING: DO NOT SUPPLY THE ENTIRE TABLE TO THIS FUNCTION!
    '''
    if len(WISEtable) > 10:
        raise ValueError("Woah! Too many objects to render, buddy!")
    f = plt.figure(figsize=(4*5, 4*len(WISEtable)))
    # Without these margins, the axes are impossible to see.
    top_margin = 0.1
    bottom_margin = 0.1
    left_margin = 0.1
    right_margin = 0.1
    gridheight = (1.0 - top_margin - bottom_margin)/len(WISEtable)
    gridwidth = (1.0 - left_margin - right_margin)/len(runbands)
    for i, WISErow in enumerate(WISEtable):
        for j, band in enumerate(runbands):
            draw_coords = [top_margin + j * gridwidth, 1.0 - left_margin -
                           (i+1) * gridheight, gridwidth, gridheight]
            if j==0:
                galname = WISErow["objstr_01"]
            else:
                galname=""

            # If we want W1, we're going to set it to be w1 manually.
            if scale == "W1":
                tilescale = "manual"
                tilesize = scalesize * WISErow["w1rsemi"]
            else:
                tilescale = scale
                tilesize = scalesize
            # Also, let's only set the scale bar on the last image.
            if band == scalebarband:
                scalebar=True
            else:
                scalebar=False
            try:
                draw_ellipse_cutout(
                    BASEDIR, WISErow, band, f, skyAperture=skyAperture,
                    skyimage=skyimage, skyprefix=skyprefix,
                    aperturefile=aperturefile, skymethod=skymethod,
                    maskbase=maskbase, sizescale=tilesize,
                    coords=draw_coords, hide_x_labels=True, hide_y_labels=True,
                    galname=galname, galcoord=(0.4, 0.9),
                    scalelength=tilescale, scalebar=scalebar,
                    scalebarlength=scalebarlength)
            except iraf.IrafError as e:
                continue
    return f


def createEllipseCutouts(
        BASEDIR, WISErow, runbands=bands, skyAperture=True, skyimage=False, 
        skyprefix="sky_level", aperturefile="ellipse_aperture",
        skymethod="adaptive", maskbase="mask", suffix="", sizescale=1.5,
        scalelength="sky"):
    '''Creates a set of four cutouts with the aperture and sky ellipses

    A cutout for each band will be created that contains the aperture
    photometry ellipse as well as the ellipse which samples the sky.
    '''
    print "Creating Cutout for {0}".format(WISErow["objstr_01"])
    galaxydir = change_to_galaxy_dir(BASEDIR, WISErow["objstr_01"])
    cutoutfig = plt.figure()
    for band in runbands:
        draw_ellipse_cutout(
            BASEDIR, WISErow, band, cutoutfig, skyAperture=skyAperture, 
            skyimage=skyimage, skyprefix=skyprefix, aperturefile=aperturefile,
            skymethod=skymethod, maskbase=maskbase,
            sizescale=sizescale, scalelength=scalelength)

        outputbase = object_name_to_dir(WISErow["objstr_01"])
        if suffix:
            outputbase += "_" + suffix
        if skyimage:
            filename = format_band_dependence(
                    outputbase + "_sky", band, "png", galaxydir)
        else:
            filename = format_band_dependence(
                    outputbase, band, "png", galaxydir)

        cutoutfig.savefig(filename)
        plt.close(cutoutfig)


def draw_ellipse_cutout(
        BASEDIR, WISErow, band, figure, skyAperture=True, skyimage=False, 
        skyprefix="sky_level", aperturefile="ellipse_aperture",
        skymethod="adaptive", maskbase="mask", sizescale=1.5, 
        coords=[1, 1, 1, 1], hide_x_labels=False, hide_y_labels=False, 
        galname="", galcoord=(0.1, 0.9), scalelength="ellipse", scalebar=True,
        scalebarlength=5, show_scalebarlength=False):
    '''Draws a cutout given for a particular band into a figure instance.

    A cutout for each band will be created that contains the aperture
    photometry ellipse as well as the ellipse which samples the sky.
    '''
    galaxydir = change_to_galaxy_dir(BASEDIR, WISErow["objstr_01"])
    # We can either get the photometry from the WISErow, or we can
    # get it directly from the STSDAS tables. The latter seems to 
    # be more direct, since those are actually used for photometry
    # and sky.
    aperturepars = STSDAS_to_Astropy_Table(
        format_band_dependence(aperturefile, band, "tab", galaxydir))
    # This is to show masked values in the cutout.
    imagehdulist = fits.open(match_filter(galaxydir, band, sky=skyimage))
    imagehdu = imagehdulist[0]
    maskhdulist = fits.open(format_band_dependence(maskbase, band, "fits",
        galaxydir))
    maskhdu = maskhdulist[0]
    # Since maskhdu is binary 0/1, this should yield the desired outcome.
    imagehdu.data = np.ma.MaskedArray(
        imagehdu.data, mask=maskhdu.data).filled(np.nan)
    gc = aplpy.FITSFigure(imagehdu, figure=figure, subplot=coords)
    gc.show_grayscale(invert=True)
    gc.set_nan_color("1.0")
    gc.refresh()

    px = getPixelScale(band)
    # Make the ellipse indicating the aperture:
    Xval, Yval = gc.pixel2world(aperturepars["X0"][0], 
                                aperturepars["Y0"][0])
    height = 2 * px * aperturepars["SMA"] / 3600.0
    width = height * (1.0 - float(aperturepars["ELLIP"]))
    angle = float(aperturepars["PA"])
    gc.show_ellipses(Xval, Yval, width, height, angle=angle,
        edgecolor="red")
    # Now make the sky annulus:
    if skyAperture:
        drawSkyParams(galaxydir, band, gc, skyprefix=skyprefix,
                method=skymethod)
    # Now resize the image.
    if scalelength is "ellipse":
        sl = WISErow["w1rsemi"]
    elif scalelength is "sky":
        sl = get_outer_sky_length(galaxydir, band)
    elif scalelength is "manual":
        sl = 1
    else:
        raise ValueError("Don't understand scalelength"
                         "{0}".format(scalelength))
    cutoutsize = sizescale * sl / 3600.0
    gc.recenter(Xval, Yval, radius=cutoutsize)
    print "Cutout radius for {0} is {1}.".format(galname, cutoutsize*3600)
    # Now we want to put the name of the galaxy on the image.
    # Since the coordinates are in percentile units of the image, they need to
    # be transformed to the units of the figure.
    imgwidth = coords[2]
    imgheight = coords[3]
    xcoord = coords[0] + galcoord[0] * imgwidth
    ycoord = coords[1] + galcoord[1] * imgheight
    gc.add_label(galcoord[0], galcoord[1], galname, relative=True, 
                 size="x-large")
    if hide_x_labels:
        gc.hide_xtick_labels()
        gc.hide_xaxis_label()
    if hide_y_labels:
        gc.hide_ytick_labels()
        gc.hide_yaxis_label()
    if scalebar:
        gc.add_scalebar(scalebarlength / 3600.0)
        gc.scalebar.set_linewidth(3)
        if show_scalebarlength:
            gc.scalebar.set_label('{0:d}"'.format(scalebarlength))
            gc.scalebar.set_font_size("large")
        else:
            gc.scalebar.set_label("")
    gc.refresh()

def drawSkyParams(galaxydir, band, gc, skyprefix="sky_level", method="adaptive"):
    '''Draws shapes used for estimating the background values.

    There are different methods used to measure the sky level, so this method
    organizes which one should be used in order to get the correct answer.
    '''
    px = getPixelScale(band)

    if method.lower() == "adaptive":
        method = adaptive_background[band]
    if method.lower() == "annulus":
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
                edgecolor="blue")
    elif method.lower() == "patch":
        skypars = Table.read(os.path.join(galaxydir,
            format_band_dependence("sky_level", band, "txt")),
            format="ascii.basic")
        Xval, Yval = gc.pixel2world(skypars["X0"][0], skypars["Y0"][0])
        major_in = 2 * skypars["A0"] * px / 3600.0
        minor_in = 2 * skypars["B0"] * px / 3600.0
        major_mid = 2 * skypars["A1"] * px / 3600.0
        minor_mid = 2 * skypars["B1"] * px / 3600.0
        major_out = 2 * skypars["A2"] * px / 3600.0
        minor_out = 2 * skypars["B2"] * px / 3600.0
        angle=skypars["PA"]

        gc.show_ellipses([Xval]*3, [Yval]*3, [minor_in, minor_mid, minor_out], 
                         [major_in, major_mid, major_out], angle=[angle]*3, 
                         edgecolor="blue")

def get_outer_sky_length(galaxydir, band, skyprefix="sky_level",
                         method="adaptive"):
    '''Retrieves the outmost length of the sky measurement in pixels.'''
    px = getPixelScale(band)

    if method.lower() == "adaptive":
        method = adaptive_background[band]
    if method.lower() == "annulus":
        skypars = Table.read(
            format_band_dependence(skyprefix, band, "txt", galaxydir), 
            format="ascii.daophot")
        outer_dim = (float(skypars.meta["keywords"]["ANNULUS"]["value"]) +
                       float(skypars.meta["keywords"]["DANNULUS"]["value"]))
    elif method.lower() == "patch":
        skypars = Table.read(
            format_band_dependence(skyprefix, band, "txt", galaxydir), 
            format="ascii.basic")
        outer_dim = skypars["A2"]
    return (outer_dim * px / 3600.0)


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
    for band in runbands:
        output = format_band_dependence(outputbase, band)
        tableForEllipseRoutine = extractEllipseParamsfromWISE(objectdir, 
                WISErow, band)
        createEllipseParamTable(objectdir, tableForEllipseRoutine, output)


def createEllipseParamTable(galaxydir, params, outputfile):
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
    iraf.tcreate(os.path.join(galaxydir, outputfile), STSDAS_COLUMN, tempfile)

def extractEllipseParamsfromWISE(objectdir, WISErow, band):
    '''Converts WISE constraints to ellipse task constraintsu

    The directory information is needed to correct WCS
    information from the corresponding FITS file. This function will
    also only give parameters for one band. It will then return a table
    which contains the values appropriate for the ellipse task.
    
    This can be used in conjunction with the createEllipseParamTable
    function to create inellip apertures for the ellipse task.'''
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
    x, y = getpixelcoords(FITS_image, WISErow["ra"], WISErow["dec"])
    values = [[ellipticity], [sma], [pa], [x], [y]]
    data_table = Table(values, names=("ELLIP", "SMA", "PA", "X0", "Y0"))
    return data_table

def getpixelcoords(imagepath, ra, dec):
    '''Gets the coordinates of the RA and Dec from an image.'''
    # We are using SkyCoords to ensure that the coordinates will be passed to
    # WCS in decimal form.
    hdulist = fits.open(imagepath)
    w = wcs.WCS(hdulist[0].header)
    coord = np.array([[ra, dec]])
    try:
        x, y = w.wcs_world2pix(coord, 1)[0]
    except TypeError:
        coords = SkyCoord(ra=ra, dec=dec)
        coord = np.array([[coords.ra.degree, coords.dec.degree]])
        x, y = w.wcs_world2pix(coord, 1)[0]

    return x, y

def STSDAS_to_Astropy_Table(filepath, outputpath=None):
    '''Converts data in STSDAS table to an Astropy table.

    All that's needed is a filename. If an output file is desired, then 
    it can be specified as well.'''
    workdir = os.path.dirname(filepath)
    colfile = os.path.join(workdir, "magcolumns.txt")
    datfile = os.path.join(workdir, "magdata.txt")
    iraf.tables()
    iraf.tables.ttools()
    iraf.tdump.setParam("cdfile", colfile)
    iraf.tdump.setParam("datafile", datfile)
    iraf.tdump(filepath)
    columns = Table.read(colfile, format="ascii.no_header")
    data = Table.read(datfile, format="ascii")
    fulldata = Table(data, names=columns['col1'])
    if outputpath:
        fulldata.write(outputpath)
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
def aperturePhotometryTable(
        BASEDIR, objectnames, runbands=bands, ellipseoutput="ellipse_aperture", skybase="sky_level",
        skymethod="adaptive", uncertaintybase="uncertainty", ZPuncertainty=True,
        brightness="AB", apertureCorrection=True, colorIndices=None, 
        ignore_exception=False):
    '''Creates a table with generated aperture photometry.

    The magnitudes will be located in columns labeled "w?apmag". All magnitudes
    will be given in the AB system.

    THIS FUNCTION IS DEPRECATED!!!
    '''
    warnings.warn("aperturePhotometryTable is deprecated. Use "
                  "aperture_photometry_table instead.",
                  warnings.DeprecationWarning)
    fulltable = Table([objectnames], names=["objstr_01"])
    for band in runbands:
        bandmags, magerrs = photometryOnBand(
            BASEDIR, objectnames, band, ellipseoutput, skybase, skymethod, 
            uncertaintybase, ZPuncertainty=ZPuncertainty, brightness=brightness, 
            errors=True, apertureCorrection=apertureCorrection, 
            colorIndices=colorIndices, ignore_exception=ignore_exception)

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

def aperture_photometry_table(
        BASEDIR, objectnames, runbands=bands, ellipseoutput="ellipse_aperture", 
        skybase="sky_level", skymethod="adaptive", 
        uncertaintybase="uncertainty", ZPuncertainty=True, brightness="AB", 
        apertureCorrection=True, colorIndices=None, ignore_exception=False,
        deextinction=""):
    '''Creates a table with generated aperture photometry.

    The magnitudes will be located in columns labeled "w?apmag". All magnitudes
    will be given in the AB system.

    De-extinction will be performed if a path to an E(B-V) file is provided in
    the deextinction parameter.
    '''
    # How to keep the column name within our standard. Though I suppose we
    # could just change the standard. Look into that. See if the current
    # standard is hard-coded somewhere, or if that's just what I've been
    # doing to stay consistent with the WISE values.
    # We want w1apmag and NUVapmag.
    #
    # This section is about setting up the table outline with a dictionary.
    photcolumns = {"objstr_01": []}
    photkeys = [name_photometry_column(band, category="ap") for band in runbands]
    errkeys = [name_photometry_column(band, category="ap", error=True) for band 
               in runbands]
    # This column was added so that upper limits could be kept track of.
    limkeys = [name_photometry_column(band, category="ap", limit=True) for band
               in runbands]
    for photkey, photerr, limkey in zip(photkeys, errkeys, limkeys):
        photcolumns[photkey] = []
        photcolumns[photerr] = []
        photcolumns[limkey] = []

    if colorIndices is None:
        colorIndices = [2] * len(objectnames)
    if len(colorIndices) != len(objectnames):
        raise ValueError("Need same number of color indices and objects.")

    for galname, colorIndex in zip(objectnames, colorIndices):
        measurements = []
        for band in runbands:
            try:
                # We bundle the magnitude and error into a tuple located in the
                # measurements list.
                measurements.append(galaxy_photometry(
                    BASEDIR, galname, band, ellipseoutput, skybase, skymethod, 
                    uncertaintybase=uncertaintybase, 
                    ZPuncertainty=ZPuncertainty, brightness=brightness, 
                    apertureCorrection=apertureCorrection, 
                    colorIndex=colorIndex))
            except ValueError as e:
                # This -99 value will only be used internally to this
                # function to specify where we need to mask the array.
                # Outside users of the API will not need to bother
                # themselves with this.
                print ("Likely encountered negative flux for "
                "{0}. Detection may be marginal. Masking".format(galname))
                measurements.append((-99.0, -99.0))
            except iraf.IrafError as e:
                galaxydir = change_to_galaxy_dir(BASEDIR, galname)
                try:
                    imagefiles = match_filter(galaxydir, band)
                except RuntimeError:
                    print ("{0} image not found for {1}").format(band, 
                                                                 galname)
                    measurements.append((-99.0, -99.0))
                else: 
                    if ignore_exception:
                        print ("Pipeline problem for {0}. "
                               "Masking.").format(galname)
                        measurements.append((-99.0, -99.0))
                    else:
                        raise e

            
        # Add the measurements to photcolumns.
        for (photkey, errkey, limkey, measurement) in zip(photkeys, errkeys,
                                                          limkeys, measurements):
            phot, err = measurement
            photcolumns[photkey].append(phot)
            photcolumns[errkey].append(err)
            # All of this is to ensure that our limits are taken care of.
            # Even doing just (phot is np.nan) shouldn't trigger on missing
            # images because those should be set to -99.0 instead. But, let's
            # be explicit and not have weird cases that weren't kept track of
            # from popping up.
            if not np.isnan(phot) and np.isnan(err):
                # A non-detection is an upper limit on flux.
                if brightness == "flux":
                    photcolumns[limkey].append(stat.UPPER)
                # A non-detection is a lower limit on magnitudes.
                elif (brightness == "AB") or (brightness == "Vega"):
                    photcolumns[limkey].append(stat.LOWER)
                else:
                    raise ValueError("Don't recognize {0}".format(brightness))
            elif (phot == -99.0) and (err == -99.0):
                photcolumns[limkey].append(stat.NA)
            else:
                photcolumns[limkey].append(stat.DETECTION)


        photcolumns["objstr_01"].append(galname)
    
    # We want the limits to be string arrays.
    for lim in limkeys:
        photcolumns[lim] = Column(photcolumns[lim], dtype="a1")
    photometry_table = Table(photcolumns, masked=True)
    # Once we make the table, we need to mask out the columns correctly. The
    # best way to do this for an astropy table is to iterate through the data
    # columns and set the mask equal to the where the values are negative.
    # Even though I set negative flux measurements to be -99.0, any negative
    # values for the flux or for magnitudes should be alarming.
    for (photkey, errkey) in zip(photkeys, errkeys):
        photcol = photometry_table[photkey]
        errcol = photometry_table[errkey]

        # Galaxy_photometry returns NaN values if the error on an object is too
        # large. When this is the case, we want to replace NaN values with
        # -99.0, so that they will be masked later on.
        photcol.mask = np.logical_or(photcol < 0, np.isnan(photcol))
        errcol.mask = np.logical_or(errcol < 0, np.isnan(errcol))

    # Now deal with extinction.
    photometry_table = deextinct_data(photometry_table,
                                      extinction=deextinction, 
                                      runbands=runbands)

    return photometry_table


def deextinct_data(photometry_table, extinction="", runbands=bands):
    '''Uses the IRSA dust map to de-extinct data.

    A photometry table with the usual photometric entries should be provided.
    This will result in bands that end with "unextmag".

    The extinction keyword should be the location of the IRSA extinction table.
    The extinction table should be in the IPAC table format from the IRSA dust
    extinction service.
    '''

    if extinction is not "":
        extinction_table = conv.get_extinction_table(extinction)
        extinction_table.rename_column("objname", "objstr_01")
        extincted_table = join_by_galaxy_name(photometry_table, 
                                              extinction_table)
        for band in runbands:
            # Get the names of the photometry columns.
            ap_mag = name_photometry_column(band, error=False, category="ap")
            ap_err = name_photometry_column(band, error=True, category="ap")
            ap_lim = name_photometry_column(band, limit=True, category="ap")
            # Get the names of the unextincted columns
            category = "unext"
            unext_mag = name_photometry_column(band, error=False,
                                             category=category)
            unext_err = name_photometry_column(band, error=True,
                                               category=category)
            unext_lim = name_photometry_column(band, limit=True,
                                               category=category)
            # Just in case both UVBANDS are not passed through at the same
            # time. When I move de-exinction to galaxy_photometry, this will be
            # a moot point!
            try:
                unextmags, unexterrs, unextlim = conv.extinction_correction(
                    band, extincted_table[ap_mag], extincted_table["E_B_V_SFD"],
                    extincted_table[ap_err], extincted_table[ap_lim], 
                    extincted_table["stdev_E_B_V_SFD"], deredden=True)
                extincted_table[unext_mag] = unextmags
                extincted_table[unext_err] = unexterrs
                extincted_table[unext_lim] = unextlim
            except KeyError as e:
                extincted_table[unext_mag] = extincted_table[ap_mag]
                extincted_table[unext_err] = extincted_table[ap_err]
                extincted_table[unext_lim] = extincted_table[ap_lim]
        return extincted_table
    else:
        return photometry_table



def unpack_bands_from_table(table, extractbands=bands, category="unext"):
    '''Returns a tuple containing extracted magnitudes and errors.

    This is a convenience function to automatically extract the magnitudes and
    errors that are stored in the table. It will be an N-tuple of 2-tuples,
    where N is the length of extractbands, which should be a list of bands we
    want to extract.
    '''
    pass

def flip_limits(limarray):
    '''Inverts limits on limarray.'''
    upindices = np.where(limarray == UPPER)
    lowindices = np.where(limarray == LOWER)

    limarray[upindices] = LOWER
    limarray[lowindices] = UPPER
    return limarray
            
def name_photometry_column(band, error=False, category="unext", limit=False):
    '''Generates the names of photometry columns in the photometry table.

    Photometry columns are the columns which will be returned in the aperture
    photometry table. Currently there are three main categories: each with a
    "mag", "err", and "lim" ending.

    The "ap" category is for magnitudes straight from aperture photometry.
    There may be slight instrumental corrections added on, such as aperture
    corrections, but no significant astronomical processing has been added on.

    The "unext" category is for magnitudes which have been corrected for
    extinction. 
    '''
    # Separate case for 2MASS colors because they are not done via photometry,
    # but ONLY through catalog entries.
    if band in TWOMASSBANDS:
        stringtemplate = "{0}_m{1}_k20fe"
        if error:
            errstring="sig"
        elif limit:
            errstring="lim"
        else:
            errstring=""
        # We take the first index of band because only the first character is
        # used in the column name.
        colname = stringtemplate.format(band[0].lower(), errstring)
    else:
        if error:
            suffix = "err"
        elif limit:
            suffix = "lim"
        else:
            suffix = "mag"

        if category not in ["ap", "unext"]:
            raise ValueError("Can not understand photometry category")

        if band in IRBANDS:
            prefix = band.lower()
        else:
            prefix = band

        colname = prefix + category + suffix

    return colname


def photometryOnBand(BASEDIR, objectnames, band,
        baseobjectfile="ellipse_aperture", skybase="sky_level",
        skymethod="adaptive", uncertaintybase="uncertainty", ZPuncertainty=True,
        brightness="AB", errors=False, apertureCorrection=True, 
        colorIndices=None, ignore_exception=False):
    '''Performs photometry on an array of objects in a given band.
    
    If flux is given as true, the flux of the object will be given in Janskys
    rather than the default magnitudes.
    
    If errors is true, then instead of simply returning an array of values, this
    function will return a 2-tuple with the flux/mag value in the first
    position, and the error in the second position.'''
    # photOutput can either be a list, or a list of 2-tuples if error was
    # specified.
    photOutput = []
    if colorIndices is None:
        colorIndices = [-2] * len(objectnames)
    if len(colorIndices) is not len(objectnames):
        raise ValueError("Need same number of color indices and objects.")
    for galname, colorIndex in zip(objectnames, colorIndices):
        try:
            photOutput.append(galaxy_photometry(
                BASEDIR, galname, band, baseobjectfile, skybase, skymethod, 
                uncertaintybase=uncertaintybase, ZPuncertainty=ZPuncertainty, 
                brightness=brightness, errors=errors, 
                apertureCorrection=apertureCorrection, colorIndex=colorIndex))
        except Exception as e:
            if ignore_exception:
                if type(e) is ValueError:
                    print "Likely got negative flux for {0}".format(galname)
                else:
                    print "Could not get photometry for {0}".format(galname)
    if errors:
        magsAndErrs = zip(*photOutput)
        return np.array(magsAndErrs[0]), np.array(magsAndErrs[1])
    else: 
        return np.array(photOutput)

def createDifferencePlot(xval, yval, valtocompare, yerror, valerror, xlabel, 
                         ylabel, title, label=''):
    '''Plots the difference between two values against the value.

    This plot is used for illustrating how consistent two datasets are
    from each other.'''
    difference, errors = stat.subtract(
        valtocompare, yval, valerror, yerror)
    plt.errorbar(xval, difference, errors, fmt="o", label=label)
    plt.plot([min(xval)+0.01, max(xval)-0.01], [0, 0], 'k-')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

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

def calc_statistical_elliptical_mass_to_light_ratio(
    W1, W2, W1err, W2err, retlog=True):
    '''Turns a W1-W2 color to a mass-to-light ratio.

    This function implements Equation 8 in Jarrett 2013. Note that it only
    applies to early-type galaxies.'''

    w1w2, w1w2err, _ = stat.subtract(
        conv.AB2Vegamag('W1', W1), conv.AB2Vegamag('W2', W2), W1err, W2err)

    # I'm adding two terms: one for conversion from [3.6] to W1, and another
    # for conversion from W1 Vega to r-band AB.
    masslightlog = 0.04 + 3.98 * w1w2
    masslightlogerr = 0
    
    if retlog:
        return masslightlog, masslightlogerr
    else:
        masslight = 10**(masslightlog)
        masslighterr = np.log10(10) * masslight * masslightlogerr

        return masslight, masslighterr

def calc_Calzetti_star_formation_rate(w4, w4err, w4lim, dist, disterr):
    '''Calculates a star formation rate from W4.

    The relation is given in Calzetti et al (2007), Equation 9.
    SFR (Msun/yr) = 1.27e-38 * L_24um (erg/s)^0.8850
    '''
    # These functions should return values in erg/s
    w4lum = conv.ABmag2specLum("W4", w4, dist)
    w4lumerr = conv.mag_err_to_spec_lum_err("W4", w4, w4err, dist, disterr)
    w4lumlim = stat.invert_limits(w4lim)

    sfr = 1.27e-38 * w4lum**0.8850
    sfr_err = 1.27e-38 * 0.8850 * w4lumerr / w4lum**(1-0.8850)
    sfr_lim = w4lumlim
    return (sfr, sfr_err, sfr_lim)


def check_if_column(seq):
    '''Verifies that the sequence is an Astropy Column.'''
    return isInstance(seq, Column)

def doubleDifferencePlot(xfirst, xsecond, yfirst, ysecond, xfirsterr, 
                         xseconderr, yfirsterr, yseconderr, xlabel, ylabel, 
                         title, label="", fmt="."):
    '''Makes a plot of one difference of quantities vs another difference.

    This plot can be used to add even more information about data consistency
    than a single difference plot.
    '''
    xdiff, xerrs, xlims = stat.subtract(xfirst, xsecond, xfirsterr,
            xseconderr)
    ydiff, yerrs, ylims = stat.subtract(yfirst, ysecond, yfirsterr,
            yseconderr)
    maxdiff = max(np.absolute(xdiff).max(), np.absolute(ydiff).max())
    plt.errorbar(xdiff, ydiff, yerrs, xerrs, fmt=fmt, label=label)
    plt.plot([-maxdiff - 0.2, maxdiff + 0.2], [0, 0], 'k-')
    plt.plot([0, 0], [-maxdiff - 0.2, maxdiff + 0.2], 'k-')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

def plotSED(bands, mags, errs, modelx=[], modely=[], modellabels=[],
        title="SED", xlabel="Wavelength (nm)", ylabel="AB Mag"):
    '''Plots an SED of the given bands.

    The bands should be a list of band names, their effective wavelengths will
    be looked-up internally. Mags should be the flux-magnitude corresponding to
    those bands, and errs should be the errors of those magnitudes.

    The optional keywords modelx and modely plot models in addition to the SED.
    They are meant to be lists of arrays, so an empty list means there are no
    model arrays.
    '''
    wavelengths = np.array([effective_wavelength(band) for band in bands])
    plt.errorbar(wavelengths, mags, errs)
    for x, y, label in zip(modelx, modely, modellabels):
        plt.plot(modelx, modely, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
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
        center = getpixelcoords(FITS_Image, WISErow["ra"], WISErow["dec"])
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

# This is probably not miscellaneous enough for this section, but I'm not
# exactly sure where it should go.
def getPixelScale(band):
    '''Returns the pixel scale for an image in a given band. Scale is
    given as arcsec/pixel.'''
    bands = {"W1": 1.37, "W2": 1.37, "W3": 1.37, "W4": 1.37, "NUV": 1.5, 
             "FUV": 1.5}
    return bands[band]


def copy_subimage_from_file_with_height_width(
        original, centerx, centery, height, width, destination, overwrite=True):
    '''Copies a part of an image to a destination file.

    The part will have the center given by centerx, centery, and will have a
    given height and width'''

    lx, ux, ly, uy = height_width_to_coordinates(
        centerx, centery, height, width)

    copy_subimage_from_file(original, lx, ux, ly, uy, destination,
                            overwrite=overwrite)

def copy_subimage_from_file(
        original, lowerx, upperx, lowery, uppery, destination, overwrite=True):
    '''Extracts a subimage from a file to another file.

    The original should be the path to the FITS file. And the bounds should be
    the bounds in the image. The file will be written to destination.
    '''
    hdulist = fits.open(original)
    full_image = hdulist[0].data
    extracted_image = extract_from_image_with_bounds(
        full_image, lowerx, upperx, lowery, uppery)
    hdulist[0].data = extracted_image
    hdulist.writeto(destination, clobber=overwrite)

def extract_from_image_with_height_width(
        original, centerx, centery, height, width):
    '''Copies a part of an image using a center and a height and width.

    A cutout of the original image centered at the given x and y coordinates,
    and with given height and width will be returned. If the coordinates given
    will not fit entirely in the original image, the returned image will be the
    subset of the image which does lie within the original image.

    Height and width along with the center should be given in image
    coordinates, not numpy coordinates.
    '''
    lx, ux, ly, uy = height_width_to_coordinates(
        centerx, centery, height, width)

    subimage = extract_from_image_with_bounds(original, lx, ux, ly, uy)
    return subimage

def extract_from_image_with_bounds(original, lowerx, upperx, lowery, uppery):
    '''Copies a part of an image with given bounds.

    A cutout of the original image with the given bounds will be returned. If
    the given bounds are seen to lie outside of the image, then they will be
    corrected to lie on the boundary of the image.

    Coordinates should be given in image coordinates, not numpy coordinates.
    This would mean transposing x and y.
    '''
    xsize, ysize = original.shape
    lowerx = fix_to_within_bounds(int(lowerx), xsize, 0)
    upperx = fix_to_within_bounds(int(upperx), xsize, 0)
    lowery = fix_to_within_bounds(int(lowery), ysize, 0)
    uppery = fix_to_within_bounds(int(uppery), ysize, 0)

    subimage = original[lowery:uppery,lowerx:upperx]
    return subimage

def height_width_to_coordinates(centerx, centery, height, width):
    '''Converts a height and a width to boundary limits.

    The results will be returned as lowerx, upperx, lowery, uppery.'''
    lowerx = centerx - width
    upperx = centerx + width
    lowery = centery - height
    uppery = centery + height

    return lowerx, upperx, lowery, uppery


def fix_to_within_bounds(val, valmax, valmin=0):
    '''Fixes bounds such that val lies between valmin and valmax.

    "Between" takes on the traditional Python definition of inclusive for min
    and exclusive for max. If val is outside of those bounds, it is reset to be
    one of the edge values.'''
    if val < valmin:
        val = valmin
    elif val >= valmax:
        val = valmax-1
    return val

def run_imcopy(original, destination):
    '''A raw layer on top of imcopy.

    Imcopy has a lot of flexibility, such as with copying multiple images, or
    copying sections of an image, or pattern-matching. However, utilizing these
    features shouldn't be done by working directly with this file, but rather by
    having layers on top of it which are more pythonic.

    For simple image copying, this function should be simple enough.
    '''
    # There's a logic tree here:
    # Destination is a file (ends with .fits):
    # - If destination exists, delete it, then proceed with the copy.
    # - If destination doesn't exist, proceed with the copy
    # - If the path to the file doesn't exist, make the path, and then copy.
    # Destination is a directory (does not end with fits):
    # - If destination exists, 
    if os.path.isfile(destination):
        os.remove(destination)
    elif os.path.isdir(destination):
        extension = ".fits"
        basefile = os.path.basename(original)
        filename = basefile[:basefile.index(extension)+len(extension)]
        destination_path = os.path.join(destination, filename)
        # In the corner case where the directory exists, but we haven't put a
        # file in there yet, this will prevent failures to remove from causing
        # major problems.
        # If this doesn't work, simply do a os.path.isfile(destination_path)
        # before removing.
        try:
            os.remove(destination_path)
        except OSError:
            pass
    else:
        # This spaghettifies the code logic a bit. But I don't want to spend
        # too much time on redoing this code. If weird bugs come up, that time
        # might be a little better spent.
        # I basically want to create the path to destination if it doesn't 
        # already exist.
        if destination.endswith(".fits"):
            parent = os.path.dirname(destination)
            try:
                os.makedirs(parent)
            except OSError:
                # This means that the parent directory exists already.
                pass
        else:
            extension = ".fits"
            basefile = os.path.basename(original)
            filename = basefile[:basefile.index(extension)+len(extension)]
            destination_path = os.path.join(destination, filename)
            # In the corner case where the directory exists, but we haven't put a
            # file in there yet, this will prevent failures to remove from causing
            # major problems.
            # If this doesn't work, simply do a os.path.isfile(destination_path)
            # before removing.
            try:
                os.remove(destination_path)
            except OSError:
                pass
            try:
                os.makedirs(destination)
            except OSError:
                pass




    iraf.images()
    iraf.imutil()
    iraf.imcopy(original, destination)

def Gil_de_Paz_Table_1_to_WISE_table(GdP_Table1):
    gdp1 = GdP_Table1
    objstr = gdp1["Name"]
    decsigns = np.where(gdp1["DE-"] == "-", -1.0, 1.0)
    ra, dec = ((gdp1["RAh"].astype(float) + gdp1["RAm"].astype(float)/60.0 +
            gdp1["RAs"].astype(float)/60/60)*360/24,
            (gdp1["DEd"].astype(float) + gdp1["DEm"].astype(float)/60.0 +
             gdp1["DEs"].astype(float)/60.0/60.0) * decsigns)
    nuvrsemi = fuvrsemi = gdp1["MajAxis"] / 2 * 60
    # There's gonna be some aliasing going along here. Be wary.
    gdp1["PA"].fill_value = 0.05
    nuvpa = fuvpa = gdp1["PA"].filled()
    nuvba = fuvba = gdp1["MinAxis"] / gdp1["MajAxis"]
    converted_table = Convert_to_UV_Table(objstr, ra, dec, nuvrsemi, fuvrsemi,
            nuvpa, fuvpa, nuvba, fuvba)
    return converted_table

def Jeong_Table_1_to_WISE_table(Jeong_Table1, hyperledatable):
    '''Converts Table 1 from Jeong into a WISE table.

    The original paper allows the position angle to be freely fit by the ellipse
    profile in I-band. However, they don't include what those values are. They
    don't include the axis ratio either. As a result, I'll just use those from
    HYPERLEDA since it should be close enough to V-band observations.
    '''
    jt1 = Jeong_Table1
    hlt = hyperledatable
    objstr = jt1["Galaxy"]
    ra, dec = hlt["al2000"]/24*360, hlt["de2000"]
    #nuvrsemi, fuvrsemi = jt1["RNUV"]*5, jt1["RFUV"]*5
    nuvrsemi = fuvrsemi = 0.1 * 60 * 10**(hlt["logd25"]) / 2
    nuvpa = fuvpa = hlt["pa"].filled(0.05)
    nuvba = fuvba = 10**(-hlt["logr25"])
    converted_table = Convert_to_UV_Table(objstr, ra, dec, nuvrsemi, fuvrsemi,
            nuvpa, fuvpa, nuvba, fuvba)
    return converted_table

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
    ## converted_table["w3rsemi"][au.astropy_table_index(converted_table,
    ## "objstr_01", "NGC 584")[0][0]] = 52.9
    ## and
    ## converted_table[au.astropy_table_index(converted_table,
    ## "objstr_01", "NGC 584")[0][0]]["w3rsemi"] = 52.9
    ## be equivalent. But they don't appear to be. Look into that!
    converted_table["w3rsemi"][au.astropy_table_index(converted_table, "objstr_01", 
            "NGC 584")[0][0]] = 52.9
    converted_table["w3rsemi"][au.astropy_table_index(converted_table, "objstr_01", 
            "NGC 777")[0][0]] = 64.0
    converted_table["w3rsemi"][au.astropy_table_index(converted_table, "objstr_01", 
            "NGC 4486")[0][0]] = 154.7
    return converted_table

def filterTable(BASEDIR, fulltable, isTrue, **kwargs):
    '''Filters a table based on a boolean method isTrue.
    
    isTrue should have a call signature of isTrue(BASEDIR, WISErow, **kwargs)'''
    try:
        copy = kwargs.pop("copy")
        galcol = kwargs.pop("galcol")
    except KeyError:
        copy=True
    filteredTable = Table(fulltable, copy=copy, masked=False)
    for i, object in enumerate(fulltable[galcol]):
        if not isTrue(BASEDIR, object, **kwargs):
            filteredTable.remove_row(np.argwhere(filteredTable[galcol] ==
                    object)[0][0])
    return filteredTable

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


def join_by_galaxy_name(table1, table2, names=("objstr_01", "objstr_01"),
                        join_type="inner", conflict_suffixes=("_A", "_B"),
                        additional_keys=[]):
    '''Joins two tables by the provided name columns. 

    By default, both columns should be called "objstr_01", in which, if both
    columns are in folder form (without a space), it will behave like a regular
    join. If the columns are not in folder form, this function will reduce both
    columns to be in folder form before performing the join. It will also be
    capable of performing joins where the galaxy names are in differently-named
    columns. In this case, the galaxy name of the output column will be decided
    by whichever table is passed first to table1. In order to allow certain
    columns to keep their name, conflict suffixes should not be assumed to have
    an underscore, or any other joining character as done by the Astropy
    default. These must be added on their own.
    '''
    # Here are a list of corner cases that I can come up with:
    # 1) names are different and name2 does not have a different column with
    #   name1
    # 2) Names are the same, in which case a temporary copy of column 2 should
    #   be restored at the end of the operation.
    # 3) Names are different, but column 2 already has a column with name1. I
    # don't know how to deal with that off the top of my head.
    name1, name2 = names
    # Saving table columns in temporary variables. Make sure to put them back!
    tempcol1 = table1[name1]
    tempcol2 = table2[name2]
    # Now format them to be in folder form.
    # If they are strings, we want to do this. If they are not, forget about
    # it.
    try:
        table1[name1] = object_name_to_dir(table1[name1])
        table2[name1] = object_name_to_dir(table2[name2])
    except TypeError:
        table2[name1] = table2[name2]
    # Now join them.
    newtable = join(table1, table2, keys=[name1]+additional_keys, 
                    join_type=join_type, table_names=list(conflict_suffixes),
                    uniq_col_name="{col_name}{table_name}")
    if name1 != name2:
        try:
            del(newtable[name2])
        except KeyError:
            if name2 in table1.colnames:
                pass
            else:
                raise
    # Set columns back.
    table1[name1] = tempcol1
    return newtable

def multijoin_by_galaxy_name(*tables, **kwargs):
    '''Joins multiple tables by the provided name columns.

    This function joins an arbitrarily large number of tables together by a
    sequence of names provided in the names keyword argument. The length of the 
    names list should correspond to the number of tables. It will return one 
    large table.  I haven't dealt with collisions yet...

    If the joins should be left-hand joins so that the resulting table is the
    same size as the left-most table, pass the "left=True" keyword.
    '''
    # Maybe add in a mechanism to deal with multiple join types. But I don't
    # think it's worth the thought at this point.
    names = kwargs["names"]
    if len(names) != len(tables):
        raise ValueError("Names and Tables have different lengths")
    if "left" in kwargs:
        join_type = "left"
    else:
        join_type = "inner"
    temptable = tables[0]
    finalname = names[0]
    for (newtab, newname) in zip(tables[1:], names[1:]):
        temptable = join_by_galaxy_name(
            temptable, newtab, names=(finalname, newname), 
            join_type=join_type)
    return temptable

def write_pipeline_file(filename, **kwargs):
    '''Writes keyword arguments to a file.'''
    f = open(filename, 'w')
    for k,v in kwargs.iteritems():
        f.write("{0}: {1}\n".format(k, v))
    f.close()

def rreplace(s, old, new, occurrence):
    '''Behaves like string.replace(), except replaces from the right rather than
    from the left. This code taken from:

    http://stackoverflow.com/questions/2556108/how-to-replace-the-last-occurence-of-an-expression-in-a-string
    '''
    li = s.rsplit(old, occurrence)
    return new.join(li)

def band_dictionary(lookups, keys):
    '''Creates a dictionary where the keys are a list of band names, and the 
    values are in lookups.'''
    thedic = {}
    for i, band in enumerate(bands):
        thedic[band] = lookups[i]
    return thedic

adaptive_background = band_dictionary(["annulus"]*4 + ["patch"]*2, bands)
