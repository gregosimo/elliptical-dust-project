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

def generate_mask(image, weights, threshold, config="wise.sex", 
        thresh_type="RELATIVE", minimum_area=15, maskoutput="mask.fits"):
    '''
    Creates a mask for an image.

    Uses a threshold value to determine whether an object is a foreground
    objects or part of the galaxy.
    '''
