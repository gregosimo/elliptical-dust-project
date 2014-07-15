import subprocess
import glob
import os
import itertools as it
import math

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import astropy
from scipy import interpolate, integrate
from astropy import units as u
from astropy.table import Column, Table
from astropy import constants as const

import grasil as gr
import synthetic_photometry as synphot

IRS_PATH = "/home/regulus/simonian/year1/wise/enhanced_specs/"

def genSpitzerSpectrum(pathdir, name, testplot=False, truncate=None, 
                       flag=False, spectype="spitzer"):
    '''Takes all the spectra in a directory, and merges them.
    '''
    spectypes = {"spitzer": spitzerspec, "SINGS": singspec}
    objpath = os.path.join(pathdir, name.replace(" ", "_"))
    paths = os.listdir(objpath)
    specs = [spectypes[spectype](os.path.join(objpath, path)) for path in paths]
    spitzer_wavelengths, spitzer_flux, spitzer_err = concatspecs(*specs)
    spitzer_wavelengths, spitzer_flux, spitzer_err = refit_spectrum(
        spitzer_wavelengths, spitzer_flux, spitzer_err, 14.27*u.um)
    if truncate:
        spitzer_wavelengths, spitzer_flux, spitzer_err = gr.specinrange(
                spitzer_wavelengths, [spitzer_flux, spitzer_err], 3*u.um, 
                truncate)
    if testplot:
        plt.plot(spitzer_wavelengths.to(u.um).value, spitzer_flux.value)
        plt.xlabel("Wavelength (um)")
        plt.ylabel("Flux ({0})".format(spitzer_flux.unit))
        plt.title("Combined Spitzer Spectrum")
    return spitzer_wavelengths, spitzer_flux, spitzer_err

def genEnhancedSpitzerSpectrum(spitzerdir, objname):
    '''Turns an enhanced spectrum into a smooth, combined spectrum.
    
    This method will look in the directory given under spitzerdir
    corresponding to the object. If there are multiple entries, it will
    join them together. Since the spectra are generally discontinuous
    across SL and LL, this script will also normalize them across the 
    discontinuity.
    '''
    objpath = os.path.join(pathdir, name.replace(" ", "_"))
    paths = os.listdir(objpath)
    specs = [spitzerspec(os.path.join(objpath, path)) for path in paths]

def spitzerspec(filename, flag=False):
    '''Reads a Spitzer Spectrum from a file.

    It will return a tuple containing the wavelength, flux density, and
    the uncertainty in flux density.
    '''
    if flag:
        flag="bit-flag"
    return readIPACspec(filename, ["wavelength", "flux_density", "error"], flag)

def singspec(filename, flag=False):
    '''Reads a SINGS spectrum from a file.

    It will return a tuple containing the wavelength, surface
    brightness, and the uncertainty in the surface brightness.'''
    return readIPACspec(filename, ["WAVELENGTH", "FLUX", "FLUX_UNCERTAINTY"])

def readIPACspec(filename, columns, flag=""):
    '''Reads a spectrum from an IPAC table.

    The filename should specify the location of the file. The columns
    to be extracted should be supplied as a list in the columns 
    argument. If flagged values are to be used, then the flag keyword
    should be replaced with the name of the flag column.
    '''
    dat = Table.read(filename, format='ipac')
    if flag:
        masks = dat[flag]
    fields = []
    for col in columns:
        if flag:
            dat[col].mask = masks
        fields.append(extractQuantityfromColumn(dat[col]))
    return tuple(fields)


def splitspec(splitarr, otherarrs, splitpoint):
    '''Splits a spectrum into two around splitpoint.

    Splitarr is the array which you are using to determine the
    splitpoint. For example, when spliting a spectrum at a certain
    wavelength, the wavelength array should be splitarr.

    Otherarrs is a list of other arrays to truncate at the same point
    as splitarr.
    '''
    splitindex = np.where(splitarr >= splitpoint)[0][0]
    lowsplit, highsplit = splitarr[:splitindex], splitarr[splitindex:]
    fullarrs = zip(*[(arr[:splitindex], arr[splitindex:]) for arr in 
            otherarrs])
    lowarrs, higharrs = list(fullarrs[0]), list(fullarrs[1])
    lowarrs.insert(0, lowsplit)
    higharrs.insert(0, highsplit)
    return tuple(lowarrs), tuple(higharrs)

def refit_spectrum(wavelengths, flux, error, splitpoint):
    '''Takes a spectrum which has a discontinuity at splitpoint and 
    refits it to be continuous. The wavelengths are presumably the 
    array that will determine where the split will occur. The flux will
    be split in the same way as wavelength.'''
    (lowwv, lowflux, lowerr), (highwv, highflux, higherr) = splitspec(
            wavelengths, [flux, error], splitpoint)
    newhighflux, newhigherr = normspecs(lowwv, lowflux, highwv, highflux, 
            splitpoint, lowerr, higherr, interpolate=True)
    fullwv = np.hstack((lowwv.value, highwv.value)) * wavelengths.unit
    fullflux = np.hstack((lowflux.value, newhighflux.value)) * flux.unit
    fullerr = np.hstack((lowerr.value, newhigherr.value)) * flux.unit
    return fullwv, fullflux, fullerr

def normspecs(anchorx, anchory, tonormx, tonormy, normpoint, anchorerr=None,
             tonormerr=None, interpolate=False):
    '''Normalizes a spectrum to an anchor by scaling.
    
    The location where they should be anchored is determined by
    normpoint. The scaling will be done via interpolation of both
    spectra.
    
    If the error of the spectrum to be normalized is given, then a
    2-tuple will be returned that has the scaled flux as well as the
    scaled error. If the error is not given, then only the scaled flux
    will be returned (not in a 1-tuple).'''
    if interpolate:
        scale = genNormScale(anchorx, anchory, tonormx, tonormy, normpoint,
                anchorerr=anchorerr, tonormerr=tonormerr)
    else:
        scale = np.median(anchory[-10:]) / np.median(tonormy[:10])
    if tonormerr:
        return scale*tonormy, scale*tonormerr
    else:
        return scale*tonormy

def genNormScale(anchorx, anchory, tonormx, tonormy, normpoint, anchorerr=None,
                 tonormerr=None):
    '''Generates the scale factor needed to normalize a spectrum at one
    point to another at the same point.

    The scale factor should be used to multiply tonormy and tonormerr
    in order to make them continuous with anchorx and anchory.
    '''
    normval = np.array(normpoint)*normpoint.unit
    anchorval = synphot.interpolate_unitsafe(anchorx, anchory, normval,
            yerr=anchorerr, testplot=True) 
    tonormval = synphot.interpolate_unitsafe(tonormx, tonormy, normval, 
            yerr=tonormerr, testplot=True)
    scale = anchorval / tonormval
    return scale

def overlappingwavelengths(xbelow, xabove):
    '''Returns whether given arrays overlap.

    The requirement for xbelow and xabove is that the first value of
    xbelow has to be less than the first value of xabove. This function
    will throw an exception if that is not satisfied.
    '''
    if xbelow[0] > xabove[0]:
        raise ValueError("Order of spectra are not correct.")

    return xbelow[-1] > xabove[0]

def joinspecs(xless, yless, xmore, ymore, errless=None, errmore=None):
    '''Joins two spectra into a single normalized spectrum.

    This function requires overlap because it takes the overlapping
    region between the spectra and sets the means equal to each other.
    The high-wavelength region is anchored to the low-wavelength 
    region. It returns a tuple with the full wavelengths and full
    fluxes.
    '''
    overlap = overlappingwavelengths(xless, xmore)
    if overlap:
        # Find where the low-wavelength data start to overlap with the
        # high-wavelength data.
        lowoverlap = ma.where(xless >= xmore[0])
        # And now find where the high-wavelength data start to overlap 
        # with the low-wavelength data.
        highoverlap = ma.where(xmore <= xless[-1])

        # Now get the spectra for the regions.
        lowoverlapspecs = yless[lowoverlap]
        highoverlapspecs = ymore[highoverlap]
        if lowoverlapspecs.size == 0:
            raise ValueError("No overlap between spectra.")

        scalefactor = np.mean(lowoverlapspecs) / np.mean(highoverlapspecs)
    else:
        scalefactor=np.median(yless[-5:])/np.median(ymore[:5])

    fullx = np.hstack((xless, xmore)).value * xless.unit
    fully = np.hstack((yless, scalefactor * ymore)).value * yless.unit
    # Now put them in monotonically-increasing wavelength.
    sortargs = np.argsort(fullx)
    
    if (errless is not None) and (errmore is not None):
        fullerr = np.hstack((errless, errmore)).value * errless.unit
        return averagedups(fullx[sortargs], fully[sortargs], 
                           errs=fullerr[sortargs])
    else: 
        return averagedups(fullx[sortargs], fully[sortargs])

def averagedups(duparr, valuearr, errs=None):
    '''Averages the values for duplicate entries.

    This function will search for duplicates in duparr and then average
    the values of valuearr corresponding to the duplicate entries. The 
    errs keyword specifies an array of errors.

    This function will return a 2-tuple consisting of the unique
    wavelengths and the averaged flux for each wavelength. If errs was
    specified, it will return a 3-tuple also containing the error of
    the average.

    NOTE: This implementation is EXTREMELY inefficient.
    '''
    unduped = np.unique(duparr)
    averaged = np.array(unduped, dtype=valuearr.dtype) 
    averr = np.array(unduped, dtype=errs.dtype) 
    for i, uniq in enumerate(unduped):
        dupind = np.where(duparr == uniq)
        averaged[i] = np.mean(valuearr[dupind]).value
        if errs is not None:
            averr[i] = np.sqrt(np.sum(errs[dupind]**2)/dupind[0].size).value
    if errs is not None:
        return unduped, averaged*valuearr.unit, averr*errs.unit
    else:
        return unduped, averaged

def concatspecs(*specs):
    '''Concatenates an arbitrary number of spectra.

    This function requies that the spectra are in an order such that
    consecutive spectra overlap. The arguments should be tuples
    containing the x and y coordinates of the spectra.
    '''
    # NOTE: For completeness, this should also organize the errors.
    sortedspecs = sortbywavelengthstart(*specs)

    if len(specs[0]) == 2:
        fullx, fully = reduce(lambda x, y: joinspecs(x[0], x[1], y[0], y[1]),
                              sortedspecs)
        return fullx, fully
    elif len(specs[0]) == 3:
        fullx, fully, fullerr = reduce(lambda x, y: joinspecs(x[0], x[1], y[0],
                y[1], x[2], y[2]), sortedspecs)
        return fullx, fully, fullerr

def sortbywavelengthstart(*specs):
    '''Sorts the spectra by the initial wavelength.

    The arguments should be tuples where the first item is the 
    wavelengths array. 
    '''
    return sorted(specs, key=lambda x: x[0][0])

def extractQuantityfromColumn(col):
    '''Extracts the Column as a Quantity object.

    Columns and Quantity objects do not play well together. So this
    function extracts the data out of a column and converts it to a
    Quantity object.'''
    return col.data * col.unit
