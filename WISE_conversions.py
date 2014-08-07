import numpy


###############################################################################
# Data Number Conversions
###############################################################################


def DN_flux_to_Jy(band, objectflux, colorIndex=-2):
    '''Converts a flux from Data Numbers to Janskys.
    '''
    return objectflux * DN_to_Jy_conversion(band, colorIndex)

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

def DN_to_Jy_conversion(band, colorIndex=-2):
    '''Returns the conversion factor between Data Numbers and Janskys for band.

    Taken from:
    http://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec2_3f.html#tbl1
    '''
    DN_to_Jy = {"W1": 1.9350e-6, "W2": 2.7048e-06, "W3": 1.8326e-06, "W4":
            5.2269e-05}
    return DN_to_Jy[band] / color_correct(band, colorIndex)

###############################################################################
# Flux-Magnitude Conversions
###############################################################################

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

def WISEmag2Jansky(band, mag, colorIndex=-2):
    '''Converts a WISE magnitude into Janskys.'''
    return get_zero_point_flux_level(band, colorIndex) * 10**(-mag/2.5)

def WISE_mag_err_to_Jansky_err(band, mag, magerr):
    '''Converts an error in WISE magnitudes to an error in Janskys.'''
    jansky_err = (10**(-mag/2.5) *
        np.sqrt(get_zero_point_flux_uncertainty(band)**2 +
        get_zero_point_flux_level(band)**2 * 0.8483))
    return jansky_err


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

def get_zero_point_magnitude_uncertainty(band):
    '''Returns the zero-point magnitude uncertainty in a given band.

    The uncertainties in the zero-point magnitudes are taken from:
    http://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec2_3f.html#tbl1
    '''
    MAGZPUNC = {"W1": 0.006, "W2": 0.007, "W3": 0.015, "W4": 0.012}
    return MAGZPUNC[band]

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
