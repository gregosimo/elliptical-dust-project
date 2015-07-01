import numpy as np
from astropy.table import Table

WISE_bands = ["W1", "W2", "W3", "W4"]
TWOMASS_bands = ["J", "H", "Ks"]
GALEX_bands = ["NUV", "FUV"]

# Dictionaries of Zero-points for bands
ZERO_POINT_FLUXES = {"W1": 306.682, "W2": 170.663, "W3": 29.0448, "W4": 8.2839, 
                     "NUV": 3810, "FUV": 3620}
ZERO_POINT_FLUX_UNCERTAINTIES = {"W1": 4.6, "W2": 2.6, "W3": 0.436, "W4": 0.124, 
                                 "NUV": 0, "FUV": 0}
# Unfortunately, magnitudes are given in either Vega or AB. And WISE does
# both while GALEX only does AB. In my mind, it's more pure to include these
# zero-points, and then do the conversions as desired.
ZERO_POINT_VEGA_MAGS = {"W1": 20.5, "W2": 19.5, "W3": 18.0, "W4": 13.0} 

ZERO_POINT_AB_MAGS = {"NUV": 20.08, "FUV": 18.82}

ZERO_POINT_VEGA_MAG_UNC = {"W1": 0.006, "W2": 0.007, "W3": 0.015, "W4": 0.012}

ZERO_POINT_AB_MAG_UNC = {"FUV": 0.05, "NUV": 0.03}

# Multiply by a number in DN to get Janskys.
DN_TO_JANSKY_FACTOR = {"W1": 1.9350e-6, "W2": 2.7048e-06, "W3": 1.8326e-06, 
                       "W4": 5.2269e-05, "NUV": 3.53e-5, "FUV": 1.07e-4}

VEGA_TO_AB_CONVERSIONS = {"W1": 2.699, "W2": 3.339, "W3": 5.174, "W4": 6.620, 
                          "J": 0.91, "H": 1.39, "Ks": 1.85}

COLOR_CORRECTIONS = {"W1": np.array([1.0283, 1.0084, 0.9961, 0.9907, 0.9921, 
    1.0000, 1.0142, 1.0347]),
    "W2": np.array([1.0206, 1.0066, 0.9976, 0.9935, 0.9943, 1.0000, 1.0107,
        1.0265]),
    "W3": np.array([1.1344, 1.0088, 0.9393, 0.9169, 0.9373, 1.0000, 1.0181,
        1.2687]),
    "W4": np.array([1.0142, 1.0013, 0.9934, 0.9905, 0.9926, 1.0000, 1.0130,
        1.0319]),
    # The UV doesn't have tabulated color-correction tables.
    "NUV": np.ones(8),
    "FUV": np.ones(8)}

###############################################################################
# Generic conversion routines
###############################################################################

# There are fluxes and there are magnitudes, and these will convert between
# either of them generically.

# Actually, I think there should be a way to calibrate flux to an arbitrary
# magnitude. This will give us the most flexible method of converting between
# fluxes and magnitudes. This will also enable units to be used, but I don't
# think we're really ready to do that.
def flux2mag(flux, calibflux, calibmag):
    '''Converts from a flux to a magnitude given a calibration point for flux
    and magnitudes. The calibpoints should be defined such that:
    mag=calibmag corresponds to flux=calibflux.
    
    Most of the time, either calibflux will be one or calibmag will be zero since
    that's how most photometric systems are defined.'''
    return calibmag - 2.5 * np.log10(flux/calibflux)

def fluxerr2magerr(flux, fluxerr, calibflux, calibfluxerr, calibmag,
        calibmagerr):
    '''Converts an error in flux to an error in magnitude given calibration
    errors.

    Usually, only one of calibflux/calibfluxerr or calibmag/calibmagerr will be
    used.
    '''
    magerr = np.sqrt(calibmagerr**2 + 1.179 * ((fluxerr / flux)**2 +
            (calibfluxerr / calibflux)**2))
    return magerr

def mag2flux(mag, calibmag, calibflux):
    '''Converts a magnitude to a flux.

    This function requires both a calibration magnitude and calibration flux
    such that calibmag corresponds to calibflux. Oftentimes, either calibmag=0
    or calibflux=1 because photometric systems use zero-points. However, this
    does not necessarily have to be the case.
    '''
    flux = calibflux * 10**(-(mag-calibmag)/2.5)
    return flux

def magerr2fluxerr(mag, magerr, calibmag, calibmagerr, calibflux, calibfluxerr):
    '''Converts magnitude errors to flux errors.'''
    fluxerr = 10**(-(mag-calibmag)/2.5) * np.sqrt(calibfluxerr**2 + 0.8483 *
            calibflux**2 * (magerr**2 + calibmagerr**2))
    return fluxerr

###############################################################################
# Data Number Conversions
###############################################################################

def DN_flux_to_Jy(band, objectflux, colorIndex=-2):
    '''Converts a flux from Data Numbers to Janskys.
    '''
    return objectflux * DN_to_Jy_conversion_factor(band, colorIndex)

def DN_err_to_Jansky_err(galaxydir, band, DNerr, DNflux,
        baseobjectfile="ellipse_aperture", mask="", useskybase="sky_level", 
        colorIndex=-2, ZPunc=True):
    '''Converts an error in Data Numbers to an error in Janskys.

    If DNflux is given, this function will use it as the value for the object's
    flux in data numbers. If it isn't, then it will calculate it on its own.
    '''

    zpfluxlevel = get_zero_point_flux_level(band, colorIndex)
    if ZPunc:
        zpfluxunc =  get_zero_point_flux_uncertainty(band)
        zpmagunc = get_zero_point_magnitude_uncertainty(band)
    else:
        zpfluxunc = zpmagunc = 0.0
    sigma_Jy = DN_to_Jy_conversion_factor(band, colorIndex) * (DNflux**2 *
            (zpfluxunc**2 / zpfluxlevel**2 + 0.8483 * zpmagunc**2) + 
            DNerr**2)**(0.5)
    return sigma_Jy

def DN_to_Jy_conversion_factor(band, colorIndex=-2):
    '''Returns the conversion factor between Data Numbers and Janskys for band.

    Taken from:
    http://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec2_3f.html#tbl1
    '''
    return DN_TO_JANSKY_FACTOR[band] / color_correction(band, colorIndex)

def DNflux2Vegamag(band, flux):
    '''Converts the flux from a raw image to a Vega magnitude.

    The flux needs to be given in units of data numbers. The infrared
    fluxes will be returned in the Vega system while the UV fluxes will
    be returned in the AB system.
    '''
    # Since the natural zero-points are in different systems, this function
    # will be broken up into cases.
    if band in WISE_bands:
        mag = flux2mag(flux, 1, get_zero_point_magnitude_level(band))
    elif band in UV_bands:
        raise ValueError("Cannot convert GALEX measurements to Vega.")
    return 

def DNflux2ABmag(band, flux):
    '''Returns a flux in the AB system.

    This returns a flux that is given in the AB magnitude system. The AB
    magnitude system returns both WISE and GALEX fluxes. This differs from the
    Vega system, where UV fluxes cannot be converted.'''
    # Since the natural zero-points are in different systems, this function
    # will be broken up into cases.
    mag = flux2mag(flux, 1, get_zero_point_magnitude_level(band))
    if band in WISE_bands:
        mag = Vega2ABmag(band, mag)
    return mag


def DN_err_to_mag_err(galaxydir, band, DNerr, DNflux, 
                      baseobjectfile="ellipse_aperture", mask="", 
                      useskybase="sky_level", skymethod="adaptive", 
                      apertureCorrection=False, ZPunc=True):
    '''Converts an error in Data Number to an error in magnitudes.

    If DNflux is given, this function will use it as the value for the object's
    flux in data numbers. If it isn't, then it will calculate it on its own.
    '''

    zplevel = get_zero_point_magnitude_level(band)
    if ZPunc:
        zpunc = get_zero_point_magnitude_uncertainty(band)
    else:
        zpunc = 0.0
    sigma_mag = fluxerr2magerr(DNflux, DNerr, 1, 0, zplevel, zpunc)
    return sigma_mag

###############################################################################
# AB and Vega conversions
###############################################################################

def Vega2ABmag(band, vegamag):
    '''Converts Vega magnitudes to AB magnitudes.

    WISE Conversions to AB magnitudes given by the WISE Explanatory
    Supplement. Section IV.4.h.3.
    
    2MASS conversions are from Blanton et al (2007) AJ 133 734. (Thanks Paul!)'''
    try:
        return vegamag + VEGA_TO_AB_CONVERSIONS[band]
    except KeyError:
        raise ValueError("Could not convert GALEX band to Vega system.")

def AB2Vegamag(band, ABmag):
    '''Converts AB magnitudes to Vega magnitudes.

    WISE Conversions to AB magnitudes given by the WISE Explanatory
    Supplement. Section IV.4.h.3.
    
    2MASS conversions are from Blanton et al (2007) AJ 133 734. (Thanks Paul!)'''
    try:
        return ABmag - VEGA_TO_AB_CONVERSIONS[band]
    except KeyError:
        raise ValueError("Could not convert GALEX BAND to Vega system.")

###############################################################################
# Flux-Magnitude Conversions
###############################################################################


def Jansky2Vegamag(band, flux, colorIndex=-2):
    '''Converts a flux in Janskys to a Vega magnitude.
    
    Note that GALEX UV observations can't be expressed in Vega magnitudes, so
    attempting to convert a UV flux to Vega magnitudes will result in a
    ValueError.'''
    if band in WISE_bands:
        mag = flux2mag(flux, get_zero_point_flux_level(band, colorIndex), 0)
    elif band in GALEX_bands:
        raise ValueError("Could not convert GALEX band to Vega system.")
    return mag

def Jansky2ABmag(band, flux, colorIndex=-2):
    '''Converts a flux in Janskys to an AB magnitude.'''
    basemag = flux2mag(flux, get_zero_point_flux_level(band, colorIndex), 0)
    if band in WISE_bands:
        mag = Vega2ABmag(band, basemag)
    elif band in GALEX_bands:
        mag = basemag
    return mag

def Jansky_err_to_mag_err(band, flux, fluxerr):
    '''Converts an error in Janskys to an error in magnitudes.'''
    err = fluxerr2magerr(flux, fluxerr, get_zero_point_flux_level(band),
            get_zero_point_flux_uncertainty(band), 0, 0)
    return err

def Vegamag2Jansky(band, mag, colorIndex=-2):
    '''Converts a Vega magnitude into Janskys.'''
    if band in WISE_bands:
        flux = mag2flux(mag, 0, get_zero_point_flux_level(band))
    elif band in GALEX_bands:
        raise ValueError("Could not convert GALEX band to Vega system.")
    return flux

def ABmag2Jansky(band, mag, colorIndex=-2):
    '''Converts an AB magnitude into Janskys.'''
    if band in WISE_bands:
        mag = AB2Vegamag(band, mag)
    flux = mag2flux(mag, 0, get_zero_point_flux_level(band))
    return flux

def Mag_err_to_Jansky_err(band, mag, magerr):
    '''Converts an error in WISE magnitudes to an error in Janskys.'''
    err =  magerr2fluxerr(mag, magerr, 0, 0, get_zero_point_flux(band),
                          get_zero_point_uncertainty(band))
    return err



###############################################################################
# Zero-point Utilities
###############################################################################

def get_zero_point_flux_level(band, colorIndex=-2):
    '''Returns the zero-point flux level for a band in Janskys.

    According to the WISE Explanatory supplement, the zero-point flux level is
    the quantity which depends on the SED of the object, not actually something
    to do with the flux or magnitude of objects. So, I'm hoping it will be most
    appropriate to install the color corrections into this function.

    Taken from:
    http://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec2_3f.html#tbl1
    http://galexgi.gsfc.nasa.gov/docs/galex/FAQ/counts_background.html
    '''
    corrected_zero_point = ZERO_POINT_FLUXES[band] / color_correction(band, 
            colorIndex)
    return corrected_zero_point

def get_zero_point_flux_uncertainty(band):
    '''Returns the zero-point flux uncertainty for a band in Janskys.

    Taken from:
    http://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec2_3f.html#tbl1
    '''
    return ZERO_POINT_FLUX_UNCERTAINTIES[band]

def get_zero_point_magnitude_level(band):
    '''Returns the zero-point between DN and magnitudes.
    
    Since WISE is based on the Vega system, if the band is a WISE band, the
    zero-point will be a Vega zero-point.
    
    GALEX is based on the AB system, so a GALEX zero-point will be in AB.'''
    if band in WISE_bands:
        zp = ZERO_POINT_VEGA_MAGS[band]
    elif band in GALEX_bands:
        zp = ZERO_POINT_AB_MAGS[band]
    return zp

def get_zero_point_magnitude_uncertainty(band):
    '''Returns the zero-point magnitude uncertainty in a given band.

    The uncertainties in the zero-point magnitudes are taken from:
    http://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec2_3f.html#tbl1
    '''
    if band in WISE_bands:
        zpunc = ZERO_POINT_VEGA_MAG_UNC[band]
    elif band in GALEX_bands:
        zpunc = ZERO_POINT_AB_MAG_UNC[band]
    return zpunc

def color_correction(band, index):
    '''Returns the color correction appropriate for a power law.

    This function will take a power law index and select the correct flux
    correction factor for the band. To use the correction factor, divide the
    uncorrected zero-point by the factor to obtain the corrected zero-point.

    Note that now index can be a numpy array!
    '''
    if band in GALEX_bands:
        return 1.0
    return fluxcorrection[band][3-index]

def Flux_table_to_WISE_mag_Table(Flux_Table, color_indices, bands=WISE_bands):
    '''Takes a table and converts the flux measurements to magnitude
    measurements.

    It assumes that flux and flux errors are stored with "W?" and "W?_err"
    keys.'''
    Mag_Table = Table(Flux_Table, copy=True)
    for band in bands:
        Mag_Table[band] = Jansky2WISEmag(band, Flux_Table[band], color_indices)
        Mag_Table["{0}_err".format(band)] = Jansky_err_to_WISE_mag_err(band,
                Flux_Table[band], Flux_Table["{0}_err".format(band)])
    return Mag_Table
