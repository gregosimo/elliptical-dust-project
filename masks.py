import os
import os.path
import subprocess

from pyraf import iraf

def run_sextractor(image, weights, config="wise.sex", **options):
    '''Runs SExtractor on an image.

    The configuration file to be used can be specified through the config
    keyword.

    Other keywords to overwrite the configuration file parameters can be
    specified as additional keyword arguments.
    '''
    command = ["sex", "-c", config, "-WEIGHT_IMAGE", weights]
    for key, value in options.iteritems():
        command.append("-"+key)
        command.append(str(value))
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
        w3initthresh, interval, diffactor, w1endthresh, w3threshfloor,
        masked_image, w1mask, w3mask, accummask):
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
    arguments. The masked w1 image will be stored in masked_image.
    '''
    w1thresh = w1initthresh
    w3thresh= w3initthresh
    while w1thresh >= w1endthresh:
        if w3thresh < w3threshfloor:
            w3thresh = w3threshfloor
        # Generate the W3 mask.
        sextractor_mask(w3image, w3weights, w3thresh, maskoutput=w3mask)
        # Create the Masked W1 image.
        apply_mask(w1image, w3mask, masked_image)
        # Get the point sources from the masked W1 image.
        sextractor_mask(masked_image, w1weights, w1thresh, maskoutput=w1mask)
        # Add the current mask to the accumulated mask.
        # Note that this flies in the face of EAFP, but imarith doesn't throw an
        # exception in the case of a missing argument. It just prints out that
        # the argument isn't valid.
        if os.path.isfile(accummask):
            # This is not good because it will write tempmask.fits to our local
            # directory, which may in general be a bad idea.
            tempfile = "/home/regulus/simonian/year1/wise/sextractor_demo/NGC6946/tempmask.fits"
            run_imarith(accummask, '+', w1mask, tempfile)
            print os.path.exists(tempfile)
            os.rename("tempmask.fits", accummask)
        else:
            os.rename(w1mask, accummask)
        w1thresh -= interval
        w3thresh -= interval * diffactor
    print "Finished mask at {0}.".format(accummask)


def apply_mask(image, mask, outputfile):
    '''Applies a mask to an image.

    This will set all the pixels which are nonzero in the mask to be zero in the
    image. The masked file will then be located at outputfile.
    '''
    command = "if im2 then 0 else im1"
    run_imcalc([image, mask], outputfile, command)

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

def run_imcalc(image, output, command, overwrite=True):
    '''Runs the IRAF imcalc routine.

    This function takes either an image or a list of images, and then runs the 
    command given on them, resulting in an image at the location given by output.

    Imcalc normally avoids overwriting an image and instead places the new image
    as a data cube. This behavior is largely undesired for our purposes. As a
    result, the overwrite flag will delete the previous file and then will place
    the output of imcalc onto the next location. If overwrite is true and the
    output file does not initially exist, this function will behave as expected
    and simply write the file to the destination.
    '''
    try:
        os.remove(output)
    except OSError:
        pass
    iraf.stsdas()
    iraf.toolbox()
    iraf.imgtools()
    if type(image) is list:
        imagestring = ','.join(image)
    else:
        imagestring = image
    iraf.imcalc(imagestring, output, command)

def run_imarith(arg1, operator, arg2, output):
    '''Runs the IRAF imarith routine.

    This function will take image or value given as arg 1, and then perform a
    binary operation defined by the operator string onto arg2 with it. The
    output image will then be placed at output.
    '''
    try:
        os.remove(output)
    except OSError:
        pass
    iraf.images()
    iraf.imutil()
    iraf.imarith(arg1, operator, arg2, output)
