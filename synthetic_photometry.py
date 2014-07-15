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

fluxunit_wv = u.W / u.cm ** 2 / u.um
fluxunit_freq = u.Jy
fluxunit_inband = u.W / u.cm**2
isophots = {"W1": 3.3526 * u.um,
            "W2": 4.6028 * u.um,
            "W3": 11.5608 * u.um,
            "W4": 22.0883 * u.um}
ZP_wv = {'W1': 8.180e-15 * fluxunit_wv,
         'W2': 2.415e-15 * fluxunit_wv,
         'W3': 6.515e-17 * fluxunit_wv,
         'W4': 5.090e-18 * fluxunit_wv}
ZP_freq = {'W1': 306.681 * fluxunit_freq,
           'W2': 170.663 * fluxunit_freq,
           'W3': 29.0448 * fluxunit_freq,
           'W4': 8.2839 * fluxunit_freq}
WISE_RESPONSE_PATH = "/home/regulus/simonian/year1/grasil/"
DEBUG_INTERPOLATION = False

def geninterpolation(specx, specy, weights=None, testplot=False):
    '''Generates a 1-d spline of a SED for interpolation.

    Requires the wavelength array as well as the SED array. The spline is fit
    without any smoothing because the model data should be perfectly smooth.

    If the wavelength range isn't the standard wavelength range between 2-30
    microns, then it can be specified with the wavelengthrange keyword. A plot
    to test the fit will also be generated if the testplot keyword is set to
    True.

    NOTE: When using astropy units, the units will be dropped when
    interfaced with the scipy interpolation routine. You will need to 
    manually check that the units are the correct ones, and replace
    them after the interpolation.
    '''
    tck = interpolate.splrep(specx, specy, w=weights)
    
    if testplot:
        xfine = np.linspace(specx[0], specx[-1], 10000)
        yfine = interpolate.splev(xfine, tck, der=0)
        plt.figure()
        plt.semilogy(specx, specy, 'x', xfine, yfine, 'b-')
        plt.title('SED Spline interpolation')
        plt.xlabel("Wavelength (um)")
        plt.ylabel("Luminosity")

    return tck

def interpolate_unitsafe(origx, origy, newx, yerr=None, testplot=False):
    '''Interpolates given data to a new domain while conserving units.

    This function takes data given as x and y arrays (which need to be
    the same size), and interpolates their relationship onto a new
    set of x-values. The interpolated y-values are now returned as an
    array which is the same size as the x-values.

    This function uses astropy units to manage the interpolation and 
    hides the non-unit-friendly behavior of the scipy interpolation 
    routines. As a result, the necessary unit conversions will be done
    automatically between x and y, and the returned y-value will have 
    the same units as the previous x-value. Using this function will
    ensure safe unit conversions when interpolating.
    '''
    if yerr:
        weights = 1.0/yerr.value
    else:
        weights=None

    spline = geninterpolation(origx.to(newx.unit).value, origy.value, 
                              testplot=testplot, weights=weights)
    newy = interpolate.splev(newx.value, spline, der=0)
    return newy * origy.unit



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
        averaged[i] = np.mean(valuearr[dupind])
        if errs is not None:
            averr[i] = np.sqrt(np.sum(errs[dupind]**2)/dupind[0].size)
    if errs is not None:
        return unduped, averaged, averr
    else:
        return unduped, averaged


def flux2WISE(filter, flux, correctindex=-2):
    '''Converts a flux in a total filter to a WISE magnitude.
    
    The filter should be a string of 'Wn' where n is the filter number.
    The flux should be energy flux, not photon flux.
    
    There is an uncertainty of 1.5% in the flux zeropoints calculated
    by this function. (Wright et al. 2010)'''
    # I'll add in more as I need them.
    fluxcorrection = {2: {"W1": 1.0084, "W2": 1.0066, "W3": 1.0088, 
                          "W4": 1.0013}, 
                      1: {"W1": 0.9961, "W2": 0.9976, "W3": 0.9393, 
                          "W4": 0.9934}, 
                      0: {"W1": 0.9907, "W2": 0.9935, "W3": 0.9169, 
                          "W4": 0.9905},
                      -1: {"W1": 0.9921, "W2": 0.9943, "W3": 0.9373,
                           "W4": 0.9926},
                      -2: {"W1": 1.0000, "W2": 1.0000, "W3": 1.0000, 
                          "W4": 1.0000}}
    if flux.unit.is_equivalent(fluxunit_wv):
        f_zp = ZP_wv[filter] 
    elif flux.unit.is_equivalent(fluxunit_freq):
        f_zp = ZP_freq[filter]
    else:
        raise ValueError("{0} is not a compatible flux unit.".format(flux.unit))
    fluxrat = flux / (f_zp / fluxcorrection[correctindex][filter])
    Wmag = -2.5 * np.log10(fluxrat.decompose())
    return Wmag

def WISE_flux(filter, wavelengths, spectrum, error=None):
    '''Simulates a WISE band observation of a spectrum.

    The filter to simulate should be given in the filter keyword. The
    wavelengths and the spectrum will then be convolved with the
    appropriate response curve to yield a total flux. The spectrum
    should be in units of flux, not luminosity. This is not a WISE 
    magnitude.
    
    Arguments:
    filter: A string that contains the name of the filter: Wn.
    wavelengths: An array of wavelengths that correspond to the indices
        of the spectrum.
    spectrum: The SED of the object we are measuring the flux for.
    '''
    filter_response = np.load(os.path.join(WISE_RESPONSE_PATH, 
            "RSR-{0}.EE.npy".format(filter)))
    normflux = filter_convolve(filter_response[:,0]*u.um, filter_response[:,1],
            wavelengths, spectrum, isophots[filter], error=error)
    return normflux 

def WISE_mag(filter, wavelengths, spectrum, distance, error=None, 
             correctindex=-2):
    '''Calculates the WISE band observation of a spectrum.

    The filter to simulate should be given in the filter keyword. The
    wavelengths and spectrum will then be convolved with the
    appropriate response curve. That flux will then be calibrated to
    the appropriate zero-point to yield a WISE magnitude.

    Arguments:
    filter: A string consisting of the name of the filter: Wn.
    wavelengths: An array of wavelengths that correspond to the entries
        of the spectrum.
    spectrum: SED of the object we're calculating the magnitude of in terms of
        power per unit wavelength.
    distance: The distance of the object from Earth.

    Keyword Arguments:
    error: An optional array specifying the errors of the spectrum.
    correctindex: Include a flux correction to simulate observed 
        data. If correctindex is none, no correction is applied. If
        it is an integer index between -4 and 3, inclusive, then it
        will apply a correction corresponding to a F_nu that increases
        with that index.
    '''
    flux = WISE_flux(filter, wavelengths, spectrum / 4 / math.pi / distance**2,
            error=error)
    # Do I want to multiply or divide? Let's see...
    # Can also return a WISE error if needed. However, right now I'm 
    # just concerned about errors due to relative flux.
    Wfilter = flux2WISE(filter, flux, correctindex=correctindex)
    return Wfilter

def WISE_color(filter1, filter2, wavelengths, spectrum, error=None, 
               correctindex=-2):
    '''Calculates the WISE color of an object.
    
    Arguments:
    filter1: The ID of the bluer filter.
    filter2: The ID of the redder filter.
    wavelengths: An array corresponding to the wavelengths measured in
        the spectrum.
    spectrum: The amplitudes of the spectrum.
    
    Keyword Arguments:
    error: An optional array that has the errors of the flux spectrum
    correctindex: Performs a color correction to make the model data
        better correspond to the measured data. If specified, the value
        should be the power law index of the spectrum.'''
    mag1 = WISE_mag(filter1, wavelengths, spectrum, u.Mpc, error=error,
                    correctindex=correctindex)
    mag2 = WISE_mag(filter2, wavelengths, spectrum, u.Mpc, error=error,
                    correctindex=correctindex)
    return (mag1 - mag2)

def filter_convolve(respwaves, response_curve, wavelengths, spectrum, isophot, 
                    error=None):
    '''Convolves a spectrum with a given WISE filter.

    Takes a given spectrum and convolves it with the given response
    curve. This function assumes that the spectrum is smooth compared
    to the response curve.

    Filter_convolve will handle subtleties resulting from units. As a
    result, the isophotal wavelength of the band is required to convert
    to and from alternative unit systems.
    '''
    # We are converting the spectrum to erg/s/cm**2/A because the
    # integral to get the signal assumes the specific flux is in terms
    # of wavelength and not frequency. We should be able to get it back
    # after integrating, though.
    #
    # I don't think its worth it to integrate in frequency space as
    # defined in Wright et al 2010, Eq (1).
    spectrumconv = spectrum.to(u.erg/u.s/u.cm**2/u.AA, 
            equivalencies=u.equivalencies.spectral_density(wavelengths))
    if error:
    # Something is not right here, and is causing the interpolation to
    # be off when converting. I'm going to look more fully into this. 
    # But for now, I don't think it's that important, and I will simply
    # use the case without smoothing until I can test what is causing
    # the smoothing to go haywire.
        errorconv=None
    #   errorconv = error.to(u.erg/u.s/u.cm**2/u.AA, 
    #           equivalencies=u.equivalencies.spectral_density(wavelengths))
    else:
        errorconv=None

    specrefined = interpolate_unitsafe(wavelengths,
            spectrumconv, respwaves, yerr=errorconv,
            testplot=DEBUG_INTERPOLATION)
    convol = specrefined * response_curve
    # The units of respwaves should cancel out, and response_curve is
    # dimensionless. Therefore, totflux should have the same units as
    # specrefined.
    totflux = (integrate.simps(convol.value, respwaves.value) /
            integrate.simps(response_curve, respwaves.value))
    return (totflux * specrefined.unit).to(spectrum.unit, 
            equivalencies=u.equivalencies.spectral_density(isophot))

def testreduction():
    '''Tests that the magnitudes and colors of Vega-like objects are zero.'''
    wvth = np.linspace(2, 30, 100000)*u.um
    #bbspec = blackbody_wv(wvth, 9602*u.K)
    bbspec = blackbody_freq(wvth.to(u.Hz,
            equivalencies=u.equivalencies.spectral()), 9602*u.K)
    bblum = bbspec * 4 * math.pi**2 * (2.5 * u.R_sun)**2
    testw1 = WISE_mag('W1', wvth, bblum, 7.68 * u.pc)
    testw2 = WISE_mag('W2', wvth, bblum, 7.68 * u.pc)
    testw3 = WISE_mag('W3', wvth, bblum, 7.68 * u.pc)
    testw4 = WISE_mag('W4', wvth, bblum, 7.68 * u.pc)
    testw1w2 = WISE_color('W1', 'W2', wvth, bblum)
    testw2w3 = WISE_color('W2', 'W3', wvth, bblum)
    testw3w4 = WISE_color('W3', 'W4', wvth, bblum)
    print "W1: {0:.4g}, W2: {1:.4g}, W3: {2:.4g}, W4: {3:.4g}, W1-W2: {4:.4g}, W2-W3: {5:.4g}, W3-W4: {6:.4g}".format(testw1, testw2, testw3, testw4, testw1w2, testw2w3, testw3w4)
    return

def blackbody_freq(freqs, temp):
    '''Generates a blackbody curve over frequencies.
    
    The frequencies should be an array over which the blackbody is to be
    evaluated. The temperature should be the temperature of the blackbody.'''
    arg = (const.h * freqs / const.k_B / temp)
    return 2 * const.h / const.c**2 * freqs**3 / (np.exp(arg.decompose()) - 1)

def blackbody_wv(wavelengths, temp):
    '''Generates a blackbody curve given wavelengths and a temperature.'''
    coeff = 2 * const.h * const.c**2
    arg = (const.h * const.c / wavelengths / const.k_B / temp)
    return coeff / wavelengths**5 / (np.exp(arg.decompose()) - 1)

