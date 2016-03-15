'''The statop module handles calculations where uncertainty has to be handle.

All uncertainties are currently handled through quadrature. So they take the
form of "\sigma_y = \sum_i \left(\frac{dy}{dx_i}\right) \sigma_{x_i}". The
functionality in this module would be AWESOME if it were transferred to a
class. However, that's a lot of scaffolding I don't want to deal with right
now. But subclassing Quantity may be worth it in the future, when I'm less
invested in the way things are now.

These functions take a set of three arguments for each variable: the value, the
error, and a code indicating whether it's a limit or not. The codes are given
in the exported constants: UPPER_LIMIT_SYMBOL, LOWER_LIMIT_SYMBOL,
DATA_POINT_SYMBOL, and NO_LIMIT_SYMBOL, abbreviated as UPPER, LOWER, DETECTION,
and NA, respectively. By referencing these constants, it should not be
necessary to use the character symbols themselves, except for debugging
purposes.

The syntax for most of these functions will take the form: func(*values,
*errors, *limits). When exceptions to this form occurs, check the docstring.
The function returns a 3-tuple containing the new value, error and limit. 
'''

import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Column

import band_conversions as conv

UPPER_LIMIT_SYMBOL = 'u'
LOWER_LIMIT_SYMBOL = 'l'
DATA_POINT_SYMBOL = 'd'
NO_LIMIT_SYMBOL = 'n'
UPPER = UPPER_LIMIT_SYMBOL
LOWER = LOWER_LIMIT_SYMBOL
NA = NO_LIMIT_SYMBOL
DETECTION = DATA_POINT_SYMBOL

def filter_limit_bands(runbands):
    '''Removes all bands which do not have limits calculated by this pipeline.

    This currently removes all bands which are not WISE or GALEX bands.
    '''
    filteredbands = filter(lambda x: x in (conv.WISE_bands + conv.GALEX_bands),
                           runbands)
    return filteredbands

def generate_limit(testlim, length):
    '''If testlim is None, generate an array of default limits with length.

    If testlim is valid, then it will be returned.
    '''
    if testlim is None:
        testlim = np.array([DETECTION]*length)
    return testlim

def invert_limits(limits):
    '''Toggles between upper and lower limits.

    UPPER and LOWER limits will switch, while valid/unconstrained values will
    remain as they were.
    '''
    newlimits = np.array(limits, subok=True)
    upperindices = np.where(limits == UPPER)
    lowerindices = np.where(limits == LOWER)
    # For the case when limits is not an array, but just a float.
    try:
        newlimits[upperindices] = LOWER
        newlimits[lowerindices] = UPPER
    except IndexError:
        if upperindices[0].shape[0] == 1:
            newlimits = LOWER
        elif lowerindices[0].shape[0] == 1:
            newlimits = UPPER
        elif limits == DETECTION or limits == NA:
            newlimits = limits
        else:
            raise ValueError("Limit is not recognizable")
    return newlimits

def combine_limits(lim1, lim2):
    '''Combines arrays of limits according to combine_limit.

    See combine_limit for the algebra
    '''
    limitlist = [combine_limit(v1, v2) for (v1, v2) in zip(lim1, lim2)]
    if isinstance(lim1, Column) and isinstance(lim2, Column):
        newlimits = Column(limitlist)
    else:
        newlimits = np.array(limitlist)
    return newlimits
        

def combine_inverted_limits(lim1, lim2):
    '''This is used for cases where one of the limits needs to be flipped.

    This is common in cases like subtraction or division. Basically if one of
    the operations is monotonic decreasing.
    '''
    return combine_limits(lim1, invert_limits(lim2))

def combine_limit(lim1, lim2):
    '''Combines limits in a logically valid way.

    The set of rules which govern limits are:
    u + u -> u
    u + l -> n
    u + 0 -> u
    u + n -> n
    l + u -> n
    l + l -> l
    l + 0 -> l
    l + n -> n
    0 + u -> u
    0 + l -> l
    0 + 0 -> 0
    0 + n -> n
    n + u -> n
    n + l -> n
    n + 0 -> n
    n + n -> n
    '''
    # Implementation details. 
    # Utilizing the symmetric property of these will only require cases for:
    ## u + u -> u
    ## u + l -> n
    ## u + 0 -> u
    ## u + n -> n
    ## l + l -> l
    ## l + 0 -> l
    ## l + n -> n
    ## 0 + 0 -> 0
    ## 0 + n -> n
    ## n + n -> n
    # This makes 10 relations
    # One easy thing to program is
    if lim2 == NA:
        return NA
    # 6 left
    elif lim1 == lim2:
        return lim1
    # 3 left
    elif lim2 == DETECTION:
        return lim1
    # 1 left
    elif lim1 == UPPER and lim2 == LOWER:
        return NA
    else:
        return combine_limit(lim2, lim1)
    

def subtract(minuend, subtrahend, minuerr, subtraerr,
                                minulim=None, subtralim=None):
    '''Returns statistically subtracted value of two arrays.

    This function takes two arrays involving two measurements with errors which
    may be upper or lower limits. It then returns a 3-tuple. The first element
    is simply the difference of the values. The second is the error of the 
    difference. And the third represents whether the differences are limits or
    not.

    If limits are not given, then the third element will simply be limits
    indicating all data points are valid.
    '''
    try:
        minulim = generate_limit(minulim, len(minuend))
    except TypeError:
        minulim = generate_limit(minulim, 1)[0]
    try:
        subtralim = generate_limit(subtralim, len(subtrahend))
    except TypeError:
        subtralim = generate_limit(subtralim, 1)[0]
    difference, differr, difflim = add(
        minuend, -subtrahend, minuerr, subtraerr, minulim,
        invert_limits(subtralim))
    return (difference, differr, difflim)

def add(augend, addend, augerr, adderr, auglim=None, addlim=None):
    '''Returns the statistically summed value of two arrays.

    This function takes two arrays involving two measurements with errors. It
    then returns a 2-tuple. The first value is simply the sum, and the second
    is the error on the sum.
    '''
    try:
        auglim = generate_limit(auglim, len(augend))
    except TypeError:
        auglim = generate_limit(auglim, 1)[0]
    try:
        addlim = generate_limit(addlim, len(addend))
    except TypeError:
        addlim = generate_limit(addlim, 1)[0]
    sums = augend + addend
    sumerr = np.sqrt(augerr**2 + adderr**2)
    sumlim = combine_limits(auglim, addlim)
    return (sums, sumerr, sumlim)


def divide(dividend, divisor, dividenderr, divisorerr, dividendlim=None, 
           divisorlim=None):
    '''Returns the statistically divided quotient of two arrays.

    This function takes two arrays involving two measurements with errors. It
    then returns a 2-tuple. The first is simply the ratio of the numbers. The
    second is the error of that ratio.'''
    try:
        dividendlim = generate_limit(dividendlim, len(dividend))
    except TypeError:
        dividendlim = generate_limit(dividendlim, 1)[0]
    try:
        divisorlim = generate_limit(divisorlim, len(divisor))
    except TypeError:
        divisorlim = generate_limit(divisorlim, 1)[0]
    quotient = dividend / divisor
    # This is more robust to the dividend being equal to zero. If the divisor
    # is equal to zero, we will still have problems.
    quoterrs = np.sqrt(
        (dividenderr / divisor)**2 + (dividend * divisorerr / divisor**2)**2)
    quotlims = combine_inverted_limits(dividendlim, divisorlim)
    return quotient, quoterrs, quotlims

def multiply(multiplicand, multiplier, multiplicerr, multiplierr, 
             multipliclim=None, multiplilim=None):
    '''Returns the statistically multiplied product of two arrays.

    This function takes two arrays involving two measurements with errors. It
    returns a 2-tuple. The first value of the tuple is the product; the second
    is the error of that product.
    '''
    try:
        multipliclim = generate_limit(multipliclim, len(multiplicand))
    except TypeError:
        multipliclim = generate_limit(multipliclim, 1)[0]
    try:
        multiplilim = generate_limit(multiplilim, len(multiplier))
    except TypeError:
        multiplilim = generate_limit(multiplilim, 1)[0]

    product = multiplicand * multiplier
    # I could do this the fancy way, but the fancy way fails if either of the
    # multiplicand or multiplier are zero. So let's not.
    producterr = np.sqrt(
        (multiplier * multiplicerr)**2 + (multiplicand * multipliererr)**2)
    productlim = combine_limits(multipliclim, multiplilim)
    return product, producterr, productlim

def fractional_difference(num, denom, numerr, denomerr, numlim=None, 
                          denomlim=None):
    '''Returns the fractional difference between num and denom.

    This equation takes the fractional difference (denom-num)/denom. It returns
    a 2-tuple. The first value of the tuple is the fraction, and the second is
    the error on that fraction.
    '''
    frac, fracerr, fraclim = divide(
        num, denom, numerr, denomerr, numlim, denomlim)
    fracdiff = 1 - frac
    return fracdiff, fracerr, invert_limits(fraclim)

def exponentiate(base, power, baserr, powerr, baselim=None, powlim=None):
    '''Returns the exponentiation of the given exponent.

    Base can be given as any base. It returns a 2-tuple. The first value of the
    tuple is the exponentiation, and the second is the error on that
    exponentiation.
    '''
    try:
        baselim = generate_limit(baselim, len(base))
    except TypeError:
        baselim = generate_limit(baselim, 1)[0]
    try:
        powlim = generate_limit(powlim, len(power))
    except TypeError:
        powlim = generate_limit(powlim, 1)[0]
    logarithm = base**power
    logerr = np.sqrt((logarithm * np.log(base) * powerr)**2 + (
        power * base**(power-1) * baserr)**2)
    loglim = combine_limits(baselim, powlim)
    return logarithm, logerr, loglim

def logarithm(num, numerr, base=10, numlim=None):
    '''Returns the logarithm of the given number.

    Base can be given as any base. It returns a 2-tuple. The first value of the
    tuple is the logarithm, and the second is the error on the logarithm.
    '''
    try:
        numlim = generate_limit(numlim, len(num))
    except TypeError:
        numlim = generate_limit(numlim, 1)[0]
    exponent = np.log(num) / np.log(base)
    experr = numerr / num / np.log(base)
    return exponent, experr, numlim

def fraction_of_sums(allvalues, allerrs, nummask, denommask, propagate=False):
    r'''Return statistically summed and divided quotient of many arrays.

    Allvalues and allerrs should be sequences of some given length. Nummask and
    denommask should be boolean (or boolean-like) arrays which indicate the
    values to be included in the numerator and the denominator.

    The form of this is:
    \sum_k \left[ \frac{\sum_i \left(a_k b_i - b_k a_i\right) x_i}
    {\left(\sum_i b_i x_i\right)^2\right]**2 \sigma_k**2
    '''
    valarray = np.ma.array(allvalues, dtype=np.float)
    errarray = np.ma.array(allerrs, dtype=np.float)
    if propagate:
        valarray = valarray.filled(np.nan)
        errarray = errarray.filled(np.nan)
    numindicator = np.array(nummask, 
                            dtype=np.int).reshape((valarray.shape[0], 1))
    denomindicator = np.array(denommask, 
                              dtype=np.int).reshape((valarray.shape[0],1))
    answer = np.sum(numindicator * valarray, axis=0) / np.sum(denomindicator * 
        valarray, axis=0)
    anserr = 0
    denomsum = np.sum(denomindicator * valarray, axis=0)
    for k in xrange(valarray.shape[0]):
        numindex = numindicator[k]
        denomindex = denomindicator[k]
        errindex = allerrs[k]

        # This represents the a_k b_i - b_k a_i
        indicatordiff = (numindex * denomindicator - denomindex * numindicator)
        # Finish off the part inside the brackets
        bracket = np.sum(indicatordiff * valarray, axis=0) / denomsum**2
        # Now multiply by sigma squared.
        errnum = bracket**2 * errindex**2

        anserr += errnum
    return answer, np.sqrt(anserr)

def get_lim(val, valerr=None, vallim=None, lim=DETECTION):
    '''Returns the subset of val and valerr which match the given limit.

    The default is to fetch all detections. If all three quantities are given,
    then this function will return a 2-tuple containing the entries of val and
    valerr which correspond to the given limit. If vallim is omitted, all values
    will be returned. If valerr is omitted, then the error entr in the 2-tuple
    will be None.
    '''
    if vallim is None:
        newval = val
        newvalerr = valerr
    else: 
        if len(val) != len(vallim):
            raise ValueError("Value and Limits need to be the same size.")
        limind = np.where(vallim == lim)
        newval = val[limind]
        if valerr is None:
            newvalerr = None
        else:
            newvalerr = valerr[limind]
    return newval, newvalerr

def errorbar(x, y, *args, **kwargs):
    '''Wraps the error bar and allows for separate plotting of limits.

    Requires x values and y values. It will require the keyword argument ylim
    to specify whether the y-values are limits or not. Additional keywords are
    ufmt and lfmt, which are format strings for the upper and lower limits,
    respectively. If they are omitted, the default will be triangles. All other
    keywords are passed to the matplotlib errorbar function.

    This function's matplotlib call signature is more like:
    errorbar(x, y, yerr=None, xerr=None, ylim=None, fmt=None, **kwargs)
    Therefore, yerr, xerr, ylim, and fmt can be given as positional arguments;
    they don't have to be specified as keyword arguments.
    '''
    if len(args) >= 1:
        kwargs["yerr"] = args[0]
    if len(args) >= 2:
        kwargs["xerr"] = args[1]
    if len(args) >= 3:
        kwargs["ylim"] = args[2]
    if len(args) >= 4:
        kwargs["fmt"] = args[3]
    if len(args) >= 5:
        raise TypeError("Can't handle positional arguments past fmt.")
    
    _errorbar(x, y, **kwargs)

    

def _errorbar(x, y, **kwargs):
    '''Wraps the error bar and allows for separate plotting of limits.

    Requires x values and y values. It will require the keyword argument ylim
    to specify whether the y-values are limits or not. Additional keywords are
    ufmt and lfmt, which are format strings for the upper and lower limits,
    respectively. If they are omitted, the default will be triangles. All other
    keywords are passed to the matplotlib errorbar function.
    '''
    ylim = kwargs.pop("ylim", None)
    fmt = kwargs.get("fmt")
    if fmt is None:
        fmtc=""
    else:
        fmtc=fmt[0]
    ufmt = kwargs.pop("ufmt", fmtc+"v")
    lfmt = kwargs.pop("lfmt", fmtc+"^")
    if ylim is None:
        plt.errorbar(x, y, **kwargs)
    else:
        xerr = kwargs.pop("xerr", None)
        yerr = kwargs.pop("yerr", None)

        # Take care of the data points.
        # This assumes that x-values will never have limits. It shouldn't be
        # difficult to add in later, but this requires giving more thought to
        # how they will be displayed. It will have to be more than simply
        # having a triangle. Probably an arrow of some sort.
        normx, normxerr = get_lim(x, xerr, ylim, DETECTION)
        normy, normyerr = get_lim(y, yerr, ylim, DETECTION)
        plt.errorbar(normx, normy, normyerr, normxerr, **kwargs)

        # We want the limits to be automatically unfilled. So if there is a
        # fill value for upper limits, we'll remove it.
        if "fillstyle" in kwargs:
            del(kwargs["fillstyle"])
        # Now upper limits
        ux, uxerr = get_lim(x, xerr, ylim, UPPER)
        uy, uyerr = get_lim(y, yerr, ylim, UPPER)
        for prop in ["fmt", "label"]:
            if prop in kwargs:
                del(kwargs[prop])
        plt.errorbar(ux, uy, fmt=ufmt, fillstyle="none", **kwargs)

        # Now lower limits
        lx, lxerr = get_lim(x, xerr, ylim, LOWER)
        ly, lyerr = get_lim(y, yerr, ylim, LOWER)
        plt.errorbar(lx, ly, fmt=lfmt, fillstyle="none", **kwargs)
