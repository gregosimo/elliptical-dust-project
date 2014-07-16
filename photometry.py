#! /usr/bin/env python
"""Performs elliptical aperture photometry by calling IRAF routines."""
import os
import math
import glob

from pyraf import iraf
from astropy import wcs
from astropy.io import fits
from astropy.table import Table
import numpy as np
import aplpy
import matplotlib
import matplotlib.pyplot as plt

import queries as query

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

###############################################################################
# Aperture Photometry Routines                                                #
###############################################################################
# 
# There's a lot to remember to get the aperture photometry routine done. First
# make sure the BASEDIR is set up correctly. It should have the identifications
# of the galaxies as folders, with the images having names which follow the
# rules in match_filter(). 
#
# With a table of WISE catalog entries, you then use those to create ellipse
# parameter files. This can be done by running:
# >>> allApertureTables(BASEDIR, WISE_Table)
# This command will go through all the directories for objects in WISE_Table
# and then use the aperture photometry information from the Catalog to create
# ellipsepars.tab files. It will also do this for all bands which are located
# in the bands list at the top of this file. These files only have 5 columns, 
# the SMA, ELLIP, PA, X0, and Y0 values.
# 
# Once the parameter files are written, we then need to feed them to the
# ellipse package so that they will be run, and the full information about the
# photometry will be made. These will be located in ellipse_aperture.tab files.
# This can be done by running:
# >>> allEllipseTables(BASEDIR, WISE_Table)
#
# After generating ellipse tables, we now want to generate tables for sky
# measurements. We first do this by generating the parameter files. This can be
# done as before by running:
# >>> allSkyParams(BASEDIR, WISE_Table)
# And then generate the tables through the ellipse routine by running:
# >>> allSkyTables(BASEDIR, WISE_Table)
# 
# Now that the object and sky tables are set, we can now run aperture photometry
# by running:
# >>> galaxy_photometry(BASEDIR, objname, band)
# 
# If you want a table of magnitudes for all objects, use the command:
# >>> aperturePhotometryTable(BASEDIR, objnames)

def galaxy_photometry(BASEDIR, name, band, baseobjectfile="ellipse_aperture", 
        mask="foreground.pl", useskybase="sky_aperture", 
        ellipsebase="ellipsepars"):
    '''Returns the elliptical aperture photometry-determined magnitude.

    This function requires that the adequate pipeline be constructed, where
    there is a folder tree under BASEDIR where each object maps to a folder
    labeled as the object name without spaces. For example, "NGC 1111" would be
    under the folder "NGC1111".

    Under each folder, there should be two sets of files outputted by the
    ellipse package. One should be called ellipse_aperture.{band}.tab, which
    contains the calculated total flux of the object, and the other should be
    named sky_aperture.{band}.tab. These should have information about the sky
    background; either direct measurement from background images, or a mean
    background value which can be multiplied by the area and subtracted from the
    object's flux.

    Features which are under construction are on-the-fly aperture photometry and
    sky calculation without needing the sky_aperture and ellipse_aperture files,
    along with masking. If you lave the useskybase parameter alone, it will
    perform regular sky estimation.
    '''
    galaxyfolder = os.path.join(BASEDIR, object_name_to_dir(name))
    ellipsetable = STSDAS_to_Astropy_Table(galaxyfolder,
            format_band_dependence(baseobjectfile, band, "tab"))
    DNflux = ellipsetable[0]["TFLUX_E"]
    aperture_area = ellipsetable[0]["NPIX_E"]
    # IF we specify a sky table to use, use it. Otherwise, it will
    # measure the sky on-the-fly.
    if useskybase:
        background = readSkyTable(galaxyfolder, band, aperture_area, useskybase)
    elif band in IRBANDS:
        background = estimate_WISE_background(galaxyfolder, band)
    else:
        background = estimate_UV_background(galaxyfolder, band,
                baseellipsefile=ellipsebase)
    objectflux = DNflux - background
    if objectflux < 0:
        print "\nGot negative flux for {0}.\n".format(name)
        objectflux *= -1
    Vegamag = DNflux2WISEmag(band, objectflux)
    return Vegamag

def object_name_to_dir(objectname):
    '''Converts the object name with spaces to the directory name.'''
    return objectname.replace(' ', "")

def change_to_galaxy_dir(BASEDIR, objectname):
    '''Returns the path of a galaxy's directory.

    Given the name of an object, it will return the full path to the
    directory containing all of the data concerning that object.
    '''
    return os.path.join(BASEDIR, object_name_to_dir(objectname))

def format_band_dependence(basename, band, extension="tab"):
    '''Generates a table file which is dependent on a band name.

    The returned filename will have a format of 
    "{basename}.{band}.{extension}".
    '''
    return "{0}.{1}.{2}".format(basename, band, extension)
    
def match_filter(directory, filter):
    '''Finds the image which corresponds to the filter.

    For WISE images, this will require searching for "w?" in the
    strings.'''
    filtermap = {"W1": "w1", "W2": "w2", "W3": "w3", "W4": "w4", "FUV": 
            "fd-int", "NUV": "nd-int"}
    filelist = glob.glob(os.path.join(directory, 
            "*{0}*.fits".format(filtermap[filter])))
    if len(filelist) > 1:
        raise RuntimeError("Image conflict for {0}.".format(directory))
    elif len(filelist) == 0:
        raise RuntimeError("Could not find file in {0}.".format(directory))
    else:
        return filelist[0]

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

def run_ellipse(galaxydir, image, ellipsepars, mask="foreground.pl", 
        outputname="ellipse.tbl"):
    '''Generates an ellipse table on the image from given parameters.

    The table of aperture parameters should be in the form of an STSDAS
    table. The necessary values are ellipticity, semimajor axis,
    position angle, X0, and Y0 (in pixels). This function will create 
    an STSDAS table at galaxydir/outputname.
    
    A mask file can be specified for the ellipse routine. If a mask
    file is specified, this function will throw an error if the mask
    file isn't found. Therefore, if you wish to ignore masking, the
    mask parameter should be the empty string.'''
    maskpath = os.path.join(galaxydir, mask)
    tablepath = os.path.join(galaxydir, ellipsepars)
    outputtbl = os.path.join(galaxydir, outputname)
    # IRAF will throw a cryptic error, or simply ignore the fact that
    # the mask doesn't exist. I want to enforce it to avoid silently
    # ignoring masking when I intend to mask.
    if not os.path.exists(maskpath):
        raise ValueError("Mask file does not exist.")
    iraf.stsdas()
    iraf.stsdas.analysis()
    iraf.stsdas.analysis.isophote()
    #iraf.unlearn("ellipse")
    iraf.ellipse.setParam("inellip", tablepath)
    iraf.ellipse.setParam("dqf", maskpath)
    iraf.ellipse(image, outputtbl)

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

def estimate_WISE_background(galaxydir, band, area, 
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

def estimate_UV_background(galaxydir, band, baseellipsefile="sky_aperture"):
    '''Returns the estimated background for a galax in GALEX bands.

    The background for UV bands is estimated by looking for files whose
    names contain the string given in skymarker.'''
    ellipsetable = STSDAS_to_Astropy_Table(galaxydir,
            format_band_dependence(baseobjectfile, band, "tab"))
    return ellipsetable[0]["TFLUX_E"]


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

def genEllipsetables(BASEDIR, WISErow, baseparamname="ellipsepars",
        baseoutput="ellipse_aperture", runbands=bands):
    '''Generates a table on the object for each band.'''
    # There should be a better way of joining this and genSkyTables, but that's
    # taking too much effort, and I want to just have this part done.
    galaxydir = change_to_galaxy_dir(BASEDIR, WISErow["objstr_01"])
    for band in runbands:
        objimage = match_filter(galaxydir, band)
        run_ellipse(galaxydir, objimage, format_band_dependence(baseparamname,
            band), outputname=format_band_dependence(baseoutput, band), mask="")


def genSkytables(BASEDIR, WISErow, baseparamname="sky_params", 
        baseoutput="sky_aperture", runbands=bands):
    '''Generates table with sky at widest W1 isophote.

    A problem with generating sky at each band is that the "sky"
    isophote moves in at high wavelength, which doesn't make physical
    sense given that the galaxy should have the same extent. To
    compensate for this, we will generate our sky background by taking
    the largest elliptical isophote in W1, and evaluating the mean
    isophotal intensity of that ellipse in the other bands, regardless
    of how wide the object appears to be in those bands. That way, we
    measure the sky background in the same way we measure the flux
    within the W1 isophote.

    If the ratio between the sky isophote semimajor axis, and the
    photometric isophote semimajor axis is smaller than minsep, then it
    will be set so that it is minsep.

    For cases where the sky axis just isn't being chosen correctly,
    the axisOverride keyword will set the semimajor axis to the value
    given, in pixels.
    '''
    galaxydir = change_to_galaxy_dir(BASEDIR, WISErow["objstr_01"])
    for band in runbands:
        skyimage = match_filter(galaxydir, band)
        if band in UVBANDS:
            skyimage = skyimage.replace("-int", "-skybg")
        run_ellipse(galaxydir, skyimage, format_band_dependence(baseparamname,
            band), outputname=format_band_dependence(baseoutput, band), mask="")

def genSkyParam(BASEDIR, WISErow, baseoutput="sky_aperture", minsep=2.0,
        axisOverride=0, runbands=bands):
    '''Generates the parameter file for sky using the widest W1 isophote.
    
    We don't want to generate sky for each band separately because we
    want the same amount of light coming in from the galaxy. The
    objects tend to be brightest at W1, so we'll use that to determine
    where the object ends and the sky background begins.
    
    The output file is specified in baseoutput, and by default will be 
    "sky_aperture.{band}.tab".
    
    If minsep is provided, it specifies the minimum ratio between the 
    sky aperture semimajor axis and the photometry aperture semimajor
    axis. If the ratio is less than this, the sky aperture will be
    adjusted so that the ratio is minsep.'''
    # The sky is going to be measured by running the ellipse routine on the
    # W1 image with the aperture parameters as the initial condition. We want to
    # use the widest aperture which the ellipse routine can still count as an
    # isophote. If it turns out that aperture isn't much bigger, then we use the
    # minsep flag to set the sky to be at least that large.

    galaxydir = change_to_galaxy_dir(BASEDIR, WISErow["objstr_01"])
    photprops = STSDAS_to_Astropy_Table(galaxydir, "ellipse_aperture.W1.tab")[0]
    # Now we run the ellipse routine in a sampling mode.
    elliptical_fit(galaxydir, match_filter(galaxydir, "W1"), (photprops["X0"],
        photprops["Y0"]), photprops["ELLIP"], photprops["PA"], photprops["SMA"],
        outputname="sky_output.tab", holdParamsFixed=False)
    skyprops = STSDAS_to_Astropy_Table(galaxydir, 
            "sky_output.tab")[-1:]
    # We measure sky for infrared and UV differently. Therefore, we'll
    # have two different cases.
    for band in runbands:
        table_name = format_band_dependence("sky_params", band)
        skyimage = match_filter(galaxydir, band)
        if band in IRBANDS:
            if axisOverride:
                skyprops["SMA"] = axisOverride
            elif skyprops["SMA"]/ photprops["SMA"] < minsep:
                skyprops["SMA"] = photprops["SMA"]*minsep
        else:
            skyprops = STSDAS_to_Astropy_Table(galaxydir,
                    format_band_dependence("ellipsepars", band))
            skyimage = skyimage.replace("-int", "-skybg")
        # Using all of skyprops causes the columns to be mislabeled. In
        # order to bypass this, I will create a table that only has
        # columns relevant to fitting the ellipse routine.
        createEllipseParamTable(galaxydir, skyprops[["ELLIP", "SMA", "PA", "X0",
            "Y0"]], table_name)

def generateEllipseCutouts(BASEDIR, WISEtable):
    '''Runs through all objects and creates cutouts in their folder.
    '''
    current_backend = matplotlib.get_backend()
    matplotlib.use("Agg")
    runOnImages(BASEDIR, WISEtable, createEllipseCutouts)
    matplotlib.use(current_backend)

def createEllipseCutouts(BASEDIR, WISErow):
    '''Creates a set of four cutouts with the aperture and sky ellipses

    A cutout for each band will be created that contains the aperture
    photometry ellipse as well as the ellipse which samples the sky.
    '''
    galaxydir = change_to_galaxy_dir(BASEDIR, WISErow["objstr_01"])
    for band in IRbands:
        # We can either get the photometry from the WISErow, or we can
        # get it directly from the STSDAS tables. The latter seems to 
        # be more direct, since those are actually used for photometry
        # and sky.
        aperturepars = STSDAS_to_Astropy_Table(galaxydir, 
                format_band_dependence("ellipsepars", band))
        skypars = STSDAS_to_Astropy_Table(galaxydir, 
                format_band_dependence("sky_aperture", band))
        ellipsetables = [aperturepars, skypars]
        gc = aplpy.FITSFigure(match_filter(galaxydir, band))
        Xvals, Yvals = gc.pixel2world(np.array([el["X0"][0] for el in
                ellipsetables]), np.array([el["Y0"][0] for el in ellipsetables]))
        gc.show_grayscale()
        # These should work. However, aplpy is not behaving well with
        # these objects. So I'm just taking the easy way and doing it
        # with a for-loop.
#        widths = np.array([el["SMA"] * (1.0 - float(el["ELLIP"])) for el in 
#                  ellipsetables])
#        heights = np.array([el["SMA"] for el in ellipsetables])
#        angles = np.array([float(el["PA"]) for el in ellipsetables])
#        gc.show_ellipses(Xvals, Yvals, widths, heights, angle=angles)
        for el in ellipsetables:
            Xval, Yval = gc.pixel2world(el["X0"][0], el["Y0"][0])
            px = getPixelScale("W1")
            # Apparently there are undefined ellipticities, so there 
            # needs to be a way for Tables to handles this more 
            # graciously.
            width = 2 * px * el["SMA"] * (1.0 - float(el["ELLIP"])) / 3600.0
            height = 2 * px * el["SMA"] / 3600.0
            angle = float(el["PA"])
            gc.show_ellipses(Xval, Yval, width, height, angle=angle,
                             edgecolor="yellow")
        gc.save(os.path.join(galaxydir,
            format_band_dependence(object_name_to_dir(WISErow["objstr_01"]), band,
            "png")))

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
        outputfile = os.path.join(galaxydir,
                format_band_dependence(maskbasename, band, "pl"))
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
    sma = WISErow["{0}rsemi".format("w1")] / pixelscale
    # Now the position angle
    # Adding a small value in order to prevent convergence problems with PA=0
    if pa > 90:
        pa -= 180
    pa = WISErow["{0}pa".format("w1")]+0.001
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

def aperturePhotometryTable(BASEDIR, objectnames,
        baseobjectfile="ellipse_aperture", mask="foreground.pl",
        skybase="sky_aperture", ellipsebase="ellipsepars", runbands=bands):
    '''Creates a table with generated aperture photometry.

    The magnitudes will be located in columns labeled "w?apmag". All magnitudes
    will be given in the AB system.
    '''
    # This function should have a better way of specifying which bands should be
    # used to create the table.
    w1apmags = photometryOnBand(BASEDIR, objectnames, "W1", baseobjectfile, 
            mask, skybase, ellipsebase)
    w2apmags = photometryOnBand(BASEDIR, objectnames, "W2", baseobjectfile, 
            mask, skybase, ellipsebase)
    w3apmags = photometryOnBand(BASEDIR, objectnames, "W3", baseobjectfile, 
            mask, skybase, ellipsebase)
    #NUVapmags = photometryOnBand(BASEDIR, objectnames, "NUV", baseobjectfile, 
    #        mask, skybase, ellipsebase)
    #FUVapmags = photometryOnBand(BASEDIR, objectnames, "FUV", baseobjectfile, 
    #        mask, skybase, ellipsebase)
    #finalTable = Table([objectnames, w1apmags, w2apmags, w3apmags, NUVapmags, 
    #        FUVapmags], names=("objstr_01", "w1apmag", "w2apmag", "w3apmag",
    #        "NUVapmags", "FUVapmags"))
    
    finalTable = Table([objectnames, w1apmags, w2apmags, w3apmags], 
            names=("objstr_01", "w1apmag", "w2apmag", "w3apmag"))
    return finalTable

def photometryOnBand(BASEDIR, objectnames, band,
        baseobjectfile="ellipse_aperture", mask="foreground.pl",
        skybase="sky_aperture", ellipsebase="ellipsepars"):
    '''Creates an array of object magnitudes in a particular band.'''
    return np.array([galaxy_photometry(BASEDIR, galname, band, baseobjectfile, 
        mask, skybase, ellipsebase) for galname in objectnames])

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
    print iraf.lpar("controlpar")
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
