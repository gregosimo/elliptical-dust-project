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
