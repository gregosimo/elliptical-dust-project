import subprocess
import glob
import os
import itertools as it
import math

import numpy as np
import numpy.ma as ma
import matplotlib
import matplotlib.pyplot as plt
import astropy
from scipy import interpolate, integrate
from astropy import units as u
from astropy.table import Column, Table
from astropy import constants as const

import synthetic_photometry as synphot
DEBUG = True

GRASIL_PATH = "/home/regulus/simonian/year1/grasil/"
GRASIL_GRID_PATH = os.path.join(GRASIL_PATH, "grasil_grid_new_2/")
GRASIL_fluxunit = u.erg / u.s / u.AA
Zsol = 0.02
#ZP_inband = {'W1': 5.4188e-15,
#             'W2': 2.5172e-15,
def execute_age_range():
    '''This is not a method to be called. I'm just documenting ipython 
    commands I'm writing. If calling GRASIL from python is needed, it 
    can be expanded on.'''
    for i in range(14):
        metallicity=0.0004
        print "Starting {0:02g} Gyr".format(i)
        subprocess.call([os.path.join(GRASIL_PATH, "grasil"), "elliptical", 
                         "tgal={0:g}".format(i)])
        a = np.loadtxt("elliptical.spe")
        np.save("grasil_grid_new/elliptical_{0:02d}Gyr_{1:2g}Z.npy".format(i,
                metallicity), a)
        print "finished {0:02g} Gyr with metallicity {1:02g}".format(i,
                metallicity)

def grasil_batch(GRASIL_PATH, filestring, project, **arglists):
    '''Executes a batch of GRASIL calculations.
    
    GRASIL will use the ($project).par file to determine the parameters
    to use in the calculations. Parameters to be iterated over ought to
    be specified in arglists.

    The parameter name should be the keyword name and a list of the
    parameter values should be the keyword value.
    '''
    # Figure out a better way of doing this later for more parameters.
    #indices = [0]*len(arglists)
    for Z in arglists["Z"]:
        for t in arglists["ages"]:
            data = run_grasil(project, Z=Z, tgal=t)
            data.write(os.path.join(GRASIL_PATH, 
                    filestring.format(Z=Z, ages=t)), format='votable')



def run_grasil(basename, **options):
    '''Runs GRASIL and outputs the spectrum to a numpy file.

    This function runs GRASIL with command-line options given through
    the keywords. Note that options not specified through the command-
    line will be taken from the config file specified by basename.par.
    This function will return an astropy table with the spectrum data.
    NOTE that the metadata on the Table does not incorporate unit
    information. This will be included later.

    Arguments:
    basename: A keyword argument specifying the parameter file to be
        used.
    options: The list of options to be passed to GRASIL through the
        command-line. The only options which are not passed to GRASIL
        are the options specified below.
    GRASIL_PATH: A keyword argument that specifies the location of the 
    `   GRASIL executable. 
    '''
    if "GRASIL_PATH" in options:
        GRASIL_PATH = options["GRASIL_PATH"]
        del(options["GRASIL_PATH"])
        
    arglist = map(lambda pair: "{0}={1:.2g}".format(pair[0], pair[1]), options)
    arglist.insert(0, basename)
    arglist.insert(0, os.path.join(GRASIL_PATH, "grasil"))
    command = " ".join(arglist)
    subprocess.call(command)
    fullspecs = Table.read("{0}.spe".format(basename), format="ascii")
    GRASILmetadata(fullspecs)
    return fullspecs

def GRASILmetadata(grasiltable):
    '''Fills in metadata that is omitted from the table read routine.

    The additions are in-place, so this function will not return
    anything. Units have not been dealt with, so those have still not
    implemented.
    '''
    newnames = ['Wavelength', "Cirrus Emission", 
                "Starlight Extincted by Molecular Clouds Only", 
                "Starlight Extincted", "Molecular Cloud Emission", "Total",
                "Starlight Unextincted", "Unextincted Bulge", 
                "Extincted bulge"]

    columnunits = [GRASIL_fluxunit]*8
    columnunits.insert(0, u.AA)
    for oldname, newname, column, colunit in zip(grasiltable.columns.keys(), 
            newnames, grasiltable.columns.values(), columnunits):
        grasiltable.rename_column(oldname, newname)

def readGRASIL(output="elliptical.spe", columns=['col1', 'col6']):
    '''Reads the output file from a GRASIL run.
    '''
    # The other columns are the same luminosity units. I just don't
    # feel like filling them in right now.
    column_units = {'col1': u.AA, 'col6': u.erg / u.s / u.AA}
    fulldata = Table.read(output, format="ascii")
    outputcols = tuple([fulldata[colname].data * column_units[colname] for 
                        colname in columns])
    return outputcols

def selectFromGRASIL_GRID(age, metallicity, GRIDPATH=GRASIL_GRID_PATH):
    '''Selects an SED from the GRASIL grid located at GRIDPATH.
    
    Returns a 2-tuple containing the wavelengths and total specific
    luminosity as astropy quantities.'''
    filename = os.path.join(GRIDPATH,
            "elliptical_{0:02d}Gyr_{1:.2g}Z.npy".format(age, metallicity))
    data = np.load(filename)
    return data[:,0]*u.AA, data[:,5]*u.erg/u.s/u.AA

def loadGRASILSpecinWISEband(age, metallicity, GRIDPATH=GRASIL_GRID_PATH):
    '''Selects a GRASIL SED in just the region relevant to WISE.

    Returns a 2-tuple containing the wavelengths and total specific
    luminosity as astropy quantities.'''
    fullwaves, fullspec = selectFromGRASIL_GRID(age, metallicity, GRIDPATH)
    waves, spec = specinrange(fullwaves, [fullspec], 1*u.um, 30*u.um)
    return (waves, spec)


def specinrange(spec, values, lowval, highval):
    '''Excises the part of the spectrum contained in the given range.

    Spec should be either the wavelength or frequency array which is 
    used to find the index. Values are all other arrays which we want 
    to excise in the same range. Lowval and highval are the lowe and
    upper limits, respectively. They should be unit-aware.

    It returns a tuple containing the excised spectrum as the first
    entry, and all the other values in the order they were given.
    '''
    rangeindices = np.where(np.logical_and(spec >= lowval, spec <= highval))
    excised_spec = spec[rangeindices]
    excised_ranges = [val[rangeindices] for val in values]
    excised_ranges.insert(0, excised_spec)
    return tuple(excised_ranges)

def expandcode(filename):
    '''Converts a numpy array to 64-bit floats.

    If a particular array is unable to represent a ncessary range of
    floats using 32bits, this function will read in the numpy array
    from the given filename, convert it to use 64-bit floats, return it
    as well as overwriting the file with the 64-bit array so that
    the conversion doesn't have to happen anymore.
    '''
    smallarr = np.load(filename)
    largearr = np.array(smallarr, dtype=np.float64)
    np.save(filename, largearr)
    return largearr



def SEDpowerlaw(x, y, order=1, testplot=False):
    '''Determines the power-law dependence of a polynomial.
    
    The testplot parameter will display a plot demonstrating the
    goodness-of-fit of the power law.'''
    # NOTE: if F_nu propto nu^a and F_lamda propto lambda^b, then you
    # have the relationship: a = -b - 2
    # Also, I ran this for the WISE band, and got an index of 0.78
    logx = np.log(x.value)
    logy = np.log(y.value)
    coeffs = np.polyfit(logx, logy, deg=order)

    if testplot:
        plt.plot(logx, logy)
        ypts = logx * coeffs[0] + coeffs[1]
        plt.plot(logx, ypts)
        plt.xlabel("Log x")
        plt.ylabel("Log y")
        plt.title("Power Law Fit")
    return coeffs[0]

def collectWavelengthAndTotalspec(fulltable):
    '''Depending on the type of fulltable, it will attempt to
    successfully extract the wavelength and total spectrum from the
    GRASIL table.'''
    if isinstance(fulltable, np.ndarray):
        wavelengths = fulltable[:,0]
        spec = fulltable[:,5]
    elif isinstance(fulltable, astropy.table.table.Table):
        wavelengths = fulltable['Wavelength']
        spec = fulltable['Total']
    else:
        raise ValueError("Don't recognize table type.")
    return (wavelengths, spec)

def collectMetallicities(DATAPATH, Z):
    '''Returns the spectra of all datafiles found in a given
    metallicity. It will return a list of tuples, containing the
    wavelength and total spectrum of each age.
    '''
    Zsol = 0.02
    filelist = sorted(glob.glob(os.path.join(DATAPATH, 
            "elliptical_*Gyr_{0:.2g}Z.npy".format(Z))))[1:]
    datalist = map(np.load, filelist)
    wvspectuples = map(collectWavelengthAndTotalspec, datalist)
    return wvspectuples

def getColorEvolution(color, Z, correctionindex=-2):
    '''Returns a list of the evolution of color at a given metallicity.

    The colors should be the string representation of the color (e.g. 
    "W1-W2"). Metallicity should be a valid metallicity which we have a
    file for. The list will be color at 1Gyr intervals, excluding 0
    Gyr.
    '''
    datapath = GRASIL_GRID_PATH
    ages = np.arange(1, 14)
    wvspectuples = collectMetallicities(datapath, Z)
    evolution = [synphot.WISE_color(color[0:2], color[3:5], wv*u.AA, spec * 
            GRASIL_fluxunit, correctindex=correctionindex).value for wv, 
            spec in wvspectuples] * u.dimensionless_unscaled
    return evolution

def makeSeriesNonOverlapping(serieslist, minscale=1.1):
    '''Scales different series so that they do not cross.

    The first entry in the series list will be considered the minimum
    anchor. The next entries will be scaled by a constant to make them
    non-intersecting. If the curves are already nonintersecting, the
    minscale parameter will scale the series even if they already do
    not intersect. Note that the minscale parameter has to be greater
    than one.
    '''
    if minscale < 1:
        raise ValueError("Minscale has to be greater than one.")

    scaledseries = [serieslist[0]]
    for i, series in enumerate(serieslist[1:]):
        newseries = scaleSeriesNonOverlapping(scaledseries[i], series, 
                minscale)
        scaledseries.append(newseries)
    return scaledseries
    
def scaleSeriesNonOverlapping(anchorseries, testseries, scale):
    ''' Scales one series up so it doesn't overlap with another series.

    Anchorseries is the series which is fixed. The testseries is then
    scaled by an amount to make it non-overlapping. If the testseries
    is already non-overlapping, it is scaled by minimumscale.
    '''
    maxratio = np.amax(anchorseries/testseries)
    return testseries * scale * maxratio

#######################################################################
# FIGURE GENERATION CODE GOES UNDER HERE ##############################
#######################################################################

def W1W2objectswithmodel(model_color):
    datapath = '/home/regulus/simonian/year1/wise/WISE_data_Martini.npy'
    fulldata = np.load(datapath)
    w1w2 = fulldata[:,5]+0.2939
    distances = Column(data=[23.2, 10.5, 31.0, 24.2, 29.4, 16.8, 16.8, 16.8,
            16.8, 16.8 ,16.8, 16.8, 16.8, 32.4, 42.6, 26.4, 50.7])
    plt.plot(distances, w1w2, 'r*')
    plt.plot(np.mean(distances), model_color, 'b*')
    plt.xlabel("Distance (Mpc)")
    plt.ylabel("W1-W2")
    plt.title("Colors of galaxies")
    return

def colorMagGrid(MW1list, MW2list, MW3list, MW4list, legendnames=[]):
    '''Generates a 4x3 grid of color-magnitude diagrams.

    The absolute magnitudes should be astropy quantities, even though 
    they are dimensionless.

    In order to accomodate different datasets, the absolute magnitudes
    should also be given as a list of magnitudes. Data from different
    datasets should be different elements of the list. That way, their
    color on the plot will be different.
    '''

    W1W2list = [MW1 - MW2 for MW1, MW2 in zip(MW1list, MW2list)]
    W2W3list = [MW2 - MW3 for MW2, MW3 in zip(MW2list, MW3list)]
    W3W4list = [MW3 - MW4 for MW3, MW4 in zip(MW3list, MW4list)]

    Magtitles = ["MW1", "MW2", "MW3", "MW4"]
    Colortitles = ["W1-W2", "W2-W3", "W3-W4"]
    f, axarr = plt.subplots(4, 3, sharex='col', sharey='row')
    for i, maglist in enumerate([MW1list, MW2list, MW3list, MW4list]):
        axarr[i, 0].set_ylabel(Magtitles[i])
        for j, colorlist in enumerate([W1W2list, W2W3list, W3W4list]):
            axarr[-1, j].set_xlabel(Colortitles[j])
            for color, mag in zip(colorlist, maglist):
                axarr[i, j].plot(color.value, mag.value, 'o')
    f.suptitle("WISE Color-Magnitude Diagram")

def colorcolorEvolutionPlot(colorx, colory, correctionindex=-2):
    '''Plots the evolution of a galaxy on a color-color diagram.

    Plots colorx vs colory, and how objects evolve with accrding to
    color.
    '''
    plt.figure()
    metallicities = [0.05, 0.02, 0.008, 0.004, 0.0004]
    for Z in metallicities:
        xcolordata = getColorEvolution(colorx, Z,
                                       correctionindex=correctionindex)
        ycolordata = getColorEvolution(colory, Z,
                                       correctionindex=correctionindex)
        plt.plot(xcolordata[1:-1].value, ycolordata[1:-1].value, label="Z={0:.2g} Zsun".format(Z/Zsol))
#        plt.plot(xcolordata[0].value, ycolordata[0].value, 'g*')
    Martinidata = \
            np.load('/home/regulus/simonian/year1/wise/WISE_data_Martini.npy')
    datadic = {"W1-W2": Martinidata[:,5], "W2-W3": Martinidata[:,6], 
               "W3-W4": Martinidata[:,7]}
    plt.plot(datadic[colorx], datadic[colory], 'r.')
    plt.xlabel(colorx)
    plt.ylabel(colory)
    plt.title("Galaxy Evolution")
    plt.legend(loc="lower right")

def BressanComparison():
    '''Makes the plot corresponding to the bottom panel of Fig. 6 of
    the Bressan et al 1998 paper.'''

    wavelength, flux = selectFromGRASIL_GRID(5, 0.02)
    fluxjy = flux.to(u.Jy,
            equivalencies=u.equivalencies(spectral_density=wavelength))
    xlimits = (0.3*u.um, 100*u.um)
    ylimits = (3e-13*u.Jy, 2.1e-11*u.Jy)

def metallicityEvolutionPlot(color):
    plt.figure()
    metallicities = [0.05, 0.02, 0.008]
    for Z in metallicities:
        ages = np.arange(1, 14)
        colorevol = getColorEvolution(color, Z)
        plt.plot(ages, colorevol, label="Z={0:.2g} Zsun".format(Z/Zsol))
    plt.xlabel("Age (Gyr)")
    plt.ylabel(color)
    plt.title("Evolution of {0}".format(color))
    plt.legend()

def colorMagnitude(color, colordata, magnitude, magnitudedata, 
        title="Color Magnitude Diagram"):
    '''Creates a Color-Magnitude diagram from existing data.
    
    The objects should be Astropy Quantities (even though they'll be
    unitless).
    
    Arguments:
    color, magnitude: String representations of the color and magnitude.
    colordata, magnitudedata: Arrays of the colors and magnitudes.'''

    plt.plot(colordata.value, magnitudedata.value, 'b*')
    plt.xlabel(color)
    plt.ylabel(magnitude)
    plt.title(title)

def plotSpectrumEvolution(metallicity):
    '''Plots overlaid spectra to show how they evolve.
    '''
    waves, specs = zip(*[loadGRASILSpecinWISEband(age, metallicity) for age in
                         xrange(1, 13)])
    scaledspecs = makeSeriesNonOverlapping(specs, minscale=1.2)
    for wave, spec in zip(waves, scaledspecs):
        plt.semilogy(wave.to(u.um).value, spec.value)

def plotProgression(xvals, yvals, shades=None, cmap="Blues"):
    '''Plots the points following a shading convention.

    This function can get more sophisticated as needed.
    Can optionally specify the type of colormap or the progression
    that will be used.
    '''
    if not shades:
        normval = 4.0 *  (len(xvals)-1) / 3.0
        shades = np.arange(len(xvals)) / normval + 0.25
    mymap = matplotlib.cm.get_cmap(cmap)
    plt.scatter(xvals, yvals, c=shades, cmap=mymap, edgecolors="face",
               vmin=0.0, vmax=1.0)
    plt.plot(xvals, yvals, 'b:')


# ages = np.arange(1,13)
# colors = np.zeros_like(ages, dtype=np.float64)
# for i,age in enumerate(ages):
#     filename = "elliptical_{0:02g}Gyr_Zsol.npy".format(age)
#     specdata = np.load(filename)[109:186,:]
#     wavelengths = specdata[:,0]*1e-4
#     spectrum = specdata[:,5]
#     color = WISE_colors("W1", "W2", wavelengths, spectrum)
#     colors[i]=color
# 
# feh = np.linspace(-0.3, 0.6, 4)
# Zs = 0.01336*10**feh
# colors = np.zeros_like(Zs, dtype=np.float64)
# for i,Z in enumerate(Zs):
#     filename = "elliptical_8Gyr_{0:02g}Z.npy".format(Z)
#     specdata = np.load(filename)[109:186,:]
#     wavelengths = specdata[:,0]*1e-4
#     spectrum = specdata[:,5]
#     color = WISE_colors("W3", "W4", wavelengths, spectrum)
#     colors[i]=color


def execute_metallicity_range():
    '''This is not a method to be called. I'm just documenting ipython 
    commands I'm writing. If calling GRASIL from python is needed, it 
    can be expanded on.'''
    feh = np.linspace(-0.3, 0.6, 4)
    Zs = 0.01336*10**feh
    for Z in Zs:
        dgas = 1.0/110.0
        metalstring = "{0:02g}Z".format(Z) 
        print "Starting {0}".format(metalstring)
        subprocess.call(["./grasil", "elliptical", "dsug={0:g}".format(dgas/Z)])
        a = np.loadtxt("elliptical.spe")
        np.save("elliptical_8Gyr_{0}.npy".format(metalstring), a)
        print "finished {0}".format(metalstring)

def plotagerangecolors():

    ages = xrange(13)
    colors = ["red", "orange", "yellow", "green", "blue", "purple", "black"]
    for age in ages:
        label = "{0} Gyr".format(age)
        spectrum = \
                np.load("elliptical_{0:02g}Gyr_Zsol.npy".format(age))[109:186,:]
        plt.plot(spectrum[:,0]*1e-4, np.log10(spectrum[:,5]), label=label)
    plt.xlabel("Wavelength (um)")
    plt.ylabel("Log L/(10^30 erg/s/A)")
    plt.title("Galaxy spectra with different ages")
    plt.legend(loc="upper right")

    feh = np.linspace(-0.3, 0.6, 4)
    Zs = 0.01336*10**feh
    for i, Z in enumerate(Zs):
        label = "[Fe/H]={0:02g}".format(feh[i])
        spectrum = \
                np.load("elliptical_8Gyr_{0:02g}Z.npy".format(Z))[109:186,:]
        plt.plot(spectrum[:,0]*1e-4, np.log10(spectrum[:,5]), '*', label=label)
    plt.xlabel("Wavelength (um)")
    plt.ylabel("Log L/(10^30 erg/s/A)")
    plt.title("Galaxy spectra at different metallicities")
    plt.legend(loc="upper right")


