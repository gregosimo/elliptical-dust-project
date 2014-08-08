import os
import os.path
import subprocess

def run_sextractor(image, weights, config="wise.sex", **options):
    '''Runs SExtractor on an image.

    The configuration file to be used can be specified through the config
    keyword.

    Other keywords to overwrite the configuration file parameters can be
    specified as additional keyword arguments.
    '''
    command = ["sex", "-c", config, "-WEIGHT_IMAGE", weights]
    for key, value in options:
        command.append("-"+key)
        command.append(value)
    subprocess.call(command)

def sextractor_mask(image, weights, threshold, config="wise.sex", 
        thresh_type="RELATIVE", minimum_area=15, maskoutput="mask.fits"):
    '''
    Creates a mask for an image.

    Uses a threshold value to determine whether an object is a foreground
    objects or part of the galaxy.
    '''
    run_sextractor(image, weights, config, DETECT_THRESH=threshold,
            THRESH_TYPE=thresh_type, DETECT_MINAREA=minimum_area, 
            CHECK_TYPE='BACKGROUND', CHECK_IMAGE=maskoutput)

def mask_algorithm(BASEDIR, WISErow, lowfrac=0.5, r_high=17, highthresh=51,
        lowthresh=50, output="foreground.fits"):
    '''Creates a mask file for the object in WISErow.

    The general algorithm for the mask creation algorithm is to find bright
    stars outside of a radius given by r_high. It then finds stars outside of a
    radius given by the isophotal aperture times lowfrac.
    '''
    # This will be implemented once we decide which algorithm to use.
    pass

def w1w3simulmask(w1image, w1weights, w3image, w3weights, w1initthresh, 
        w3initthresh, interval, diffactor, w1endthresh, w3threshfloor, w1mask, 
        w3mask, accummask):
    '''Masking algorithm which runs SExtractor on the w3 image to mask out the
    galaxy, and then uses the mask generated from the w3 image on the w1 image
    in order to capture Rayleigh-Jeans foreground sources.

    The algorithm starts out with w1initthresh and w3initthresh being given to 
    DETECT_THRESH in SExtractor. With each iteration, the threshold for w1 will
    decrease by interval, while the threshold for w3 will decrease by diffactor
    * interval. Based on behavior which has been observed, diffactor should
    probably be >1. The iterating continues until the threshold for w1 drops
    below w1endthresh.

    In order to deal with cases where the threshold for w3 would drop below
    zero if diffactor is greater than 1, there will be a w3threshfloor argument
    which bottoms out the value of w3. This should probably be at around 5
    sigma.

    Each iteration will have the mask add on to accummask to preserve point
    sources which may be embedded in the galaxy. The location of the 
    intermediate w1 and w3 masks can be specified through the w1mask and w3mask
    arguments.
    '''
    w1thresh = w1initthesh
    w3thresh= w3initthresh
    while w1thresh >= w1endthresh:
        if w3thresh < w3threshfloor:
            w3thresh = w3threshfloor
        # Generate the W3 mask.
        sextractor_mask(


def mask_circle(image, center, radius, outputfile):
    '''Masks a circular region of the image.

    The effect of this function is to set a circular region in an image to zero,
    thus effectively masking it. All numbers should be in units of pixels. The
    masked image will then be outputted into outputfile.
    '''
    xcenter, ycenter = center
    command = build_imcalc_circle(xcenter, ycenter, radius)
    run_imcalc(image, outputfile, command)

def build_imcalc_circle(xcenter, ycenter, radius):
    '''Builds the command for creating a circle in imcalc.'''
    command = "if (x-{0})**2 + (y-{1})**2 < {2}**2 then 0 else im1".format(
            xcenter, ycenter, radius)
    return command
