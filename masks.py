import os
import os.path
import subprocess
import shutil
import math
import itertools

import numpy as np
from pyraf import iraf
from astropy.io import fits
from astropy.table import Table

import photometry as phot
from mask_trimmer import MaskCMD

def build_masks(BASEDIR, WISETable, maskband, threshold=5,
        output="foreground.fits", maskconfigbase="default"):
    '''Builds masks for specified objects.

    The objects to be built should be specified in WISETable, which should be a
    valid parameter construction file. The mask will be based on the image in
    the band called maskband. As a result, there should only be one mask to be
    used per image type.

    Parameters sent to SExtractor should be the config file, specified in
    maskconfigbase, along with the threshold of the measurement, specified in
    threshold.

    The resulting mask will written to $BASEDIR/output.
    '''
    phot.allMasks(BASEDIR, WISETable, maskband, threshold=threshold, 
            output=output, maskconfigbase=maskconfigbase)

def run_sextractor(image, config, **options):
    '''Runs SExtractor on an image.

    The configuration file to be used can be specified through the config
    keyword.

    Other keywords to overwrite the configuration file parameters can be
    specified as additional keyword arguments.
    '''
    command = ["sex", image, "-c", config]
    for key, value in options.iteritems():
        command.append("-"+key)
        command.append(str(value))
    returncode = subprocess.call(command)
    if returncode:
        print returncode
        raise ValueError("Fatal Error in SExtractor.")

def sextractor_background(image, config, **sexargs):
    '''Creates a background for an image.

    Gives a mesh with size of mesh_size to estimate the background, along with
    filtering it with filter_size.

    Relevant keyword arguments are:
    BACK_SIZE: Specifies the size of the background mesh.
    BACK_FILTERSIZE: Specifies how much filtering of the background is done.
    CHECKIMAGE_NAME: The location of the background file.
    '''
    sexargs["CHECKIMAGE_TYPE"] = "BACKGROUND"
    run_sextractor(image, config, **sexargs)

def sextractor_subtracted_background(image, config,
        output="subtracted.fits", mesh_size=480, filter_size=7):
    '''Creates a background-subtracted image.

    Gives a mesh with size of mesh_size to estimate the background, along with
    filtering it with filter_size.

    Relevant keyword arguments are:
    BACK_SIZE: Specifies the size of the background mesh.
    BACK_FILTERSIZE: Specifies how much filtering of the background is done.
    CHECKIMAGE_NAME: The location of the background file.
    '''
    sexargs["CHECKIMAGE_TYPE"] = "-BACKGROUND"
    run_sextractor(image, config, **sexargs)

def sextractor_mask(image, threshold, config, **sexargs):
    '''
    Creates a mask for an image.

    Uses a threshold value to determine whether an object is a foreground
    objects or part of the galaxy.

    Relevant keyword arguments are:
    DETECT_THRESH: The detection threshold for objects.
    THRESH_TYPE: How the detection threshold is defined.
    DETECT_MINAREA: The minimum area for an object to be detected.
    CHECKIMAGE_NAME: The location of the mask.
    '''
    sexargs["CHECKIMAGE_TYPE"] = "SEGMENTATION"
    sexargs["DETECT_THRESH"] = threshold
    run_sextractor(image, config, **sexargs)

def mask_algorithm(BASEDIR, WISErow, maskband="W1", threshold=50, 
        output="foregroundmask.fits", ellipsebase="ellipsepars", spreadpix=5,
        maskconfigbase="default"):
    '''Creates a mask file for the object in WISErow.

    The general algorithm for the mask creation algorithm is to find bright
    stars outside of a radius given by r_high. It then finds stars outside of a
    radius given by the isophotal aperture times lowfrac.
    '''
    galaxydir = phot.change_to_galaxy_dir(BASEDIR, WISErow["objstr_01"])
    objectcoords = phot.getpixelcoords(phot.match_filter(galaxydir, maskband),
            WISErow["ra"], WISErow["dec"])
    mask_elliptical_galaxy(galaxydir, threshold, maskband, objectcoords,
            maskfile=output, ellipsebase=ellipsebase, configbase=maskconfigbase,
            spreadpix=5)

def mask_elliptical_galaxy(galaxydir, threshold, maskband, objectcoords,
        maskfile="foregroundmask.fits", ellipsebase="ellipsepars", 
        configbase="default", segment="rawsegment.fits",
        clearedsegment="segment_nogalaxy.fits",
        procsegment="foreground_unnormalized.fits",
        prespreadfile="foreground_normalized.fits", spreadpix=25):
    '''Creates a foreground mask for an elliptical galaxy.

    This function uses the ellipse output to find the location of the ellipse
    and then remove it from the segmentation map, leaving us with a segmentation
   map of just the foreground objects.
   '''
    image = phot.match_filter(galaxydir, maskband)
    configfile = os.path.join(os.path.split(galaxydir)[0],
            phot.format_band_dependence(configbase, maskband, "sex"))
    masked_image = os.path.join(galaxydir, segment)
    fullmask = os.path.join(galaxydir, maskfile)
    config = os.path.join(galaxydir, configfile)
    galaxy_removed = os.path.join(galaxydir, clearedsegment)
    segment_needs_normalization = os.path.join(galaxydir, procsegment)
    normalized_segment = os.path.join(galaxydir, prespreadfile)

    segmentation_mask(config, image, threshold, masked_image,
            segment_needs_normalization, objectcoords)
    # We'll interactively generate masks.
    print "Please remove object {0}.".format(os.path.basename(galaxydir))
    maskprog = MaskCMD(segment_needs_normalization, galaxy_removed, objectcoords)
    maskprog.cmdloop()
    normalize_segmentation_map(galaxy_removed, normalized_segment)
    spreadmask(galaxydir, spreadpix, normalized_segment, outputfile=fullmask)

def spreadmask(workdir, pixels, maskimage, outputfile="foreground.fits", 
        tempfolder="offsets", tempfilebase="foreground_shifted"):
    '''Masks a square around each original pixel

    This function takes each of the pixels in maskimage and then masks a box
    with side length pixels around them.'''
    # Use reduce for this. It'll be awesome!
    pixelrange = range(-(pixels-1)/2, (pixels+1)/2)
    permutations = list(itertools.product(pixelrange, pixelrange))
    # If the directory doesn't exist, then make it!
    try:
        os.mkdir(os.path.join(workdir, tempfolder))
    except OSError:
        pass
    temppath = os.path.join(workdir, tempfolder)
    outputpath = os.path.join(temppath, outputfile)
    filenames = []
    for i,j in permutations:
        filename = os.path.join(temppath,
                "{0}{1:+d}{2:+d}.fits".format(tempfilebase, i, j))
        filenames.append(filename)
        run_imshift(os.path.join(workdir, maskimage), filename, i, j)
    shutil.copy(filenames[0], outputpath)
    filenames[0] = outputpath

    # I don't like the way that this is done... but it's convenient.
    # For the sake of code clarity, the iteration here should be made clearer.
    reduce(combinemasks, filenames)

def combinemasks(basefile, additionfile):
    '''Adds a mask to a base mask.'''
    run_imcalc([basefile, additionfile], basefile, 'im1 || im2')
    os.remove(additionfile)
    return basefile

def subtractw3fromw1(config, w1image, w3image, w1output_nobackground, 
        w3output_nobackground, w3output_scaled, w1output_convolved, 
        subtracted_output, objcenter, central_radius):
    '''Subtracts the W3 background from W1 to accentuate point sources.'''
    # Generate W1 background-subtracted image.
    print "Subtracting W1 background."
    sextractor_subtracted_background(w1image, config,
            CHECKIMAGE_NAME=w1output_nobackground)
    # Generate W3 background-subtracted image.
    print "Subtracting W3 background."
    sextractor_subtracted_background(w3image, config,
            CHECKIMAGE_NAME=w3output_nobackground)
    # Convolve the W1 image.
    # Let's see if this causes the problems we expect it to.
    #print "Convolving W1 image."
    #run_gauss(w1output_nobackground, w1output_convolved, str(5 /
    #        phot.getPixelScale("W1")))
    shutil.copyfile(w1output_nobackground, w1output_convolved)
    # Find the mean values of the image centers
    # We're not actually using a radius. More of a box.
    bounds = "[{0}:{1},{2}:{3}]".format(objcenter[0] - central_radius,
            objcenter[0] + central_radius, objcenter[1] - central_radius,
            objcenter[1]+central_radius)
    w1mean = run_immean(w1output_convolved + bounds)
    w3mean = run_immean(w3output_nobackground + bounds)
    # Create the scaled W3 image.
    print "Scaling W3 image."
    run_imarith(w3output_nobackground, '*', str(w1mean / w3mean),
            w3output_scaled)
    # Subtract the two images.
    print "Subtracting images."
    run_imarith(w1output_convolved, '-', w3output_scaled, subtracted_output)

def segmentation_mask(config, image, threshold, masked_image, fullmask,
        coords):
    '''Creates a mask from a segmentation image and object coordinates.

    SExtractor is only run once, but is used to generate a segmentation map.
    After generating the segmentation map, the object is looked up in the pixel
    coordinate of the image, and all matching pixel values are removed.

    The raw segmentation image is stored at masked_image and the segmentation
    image with the galaxyy removed is saved in fullmask, which is the desirable
    product.
    '''
    sextractor_mask(image, threshold, config, 
            CHECKIMAGE_NAME=masked_image)
    remove_galaxy_mask(masked_image, fullmask, coords)

def remove_galaxy_mask(imagepath, newimagepath, coord):
    '''Removes a galaxy from a segmentation image.

    The coordinate of the galaxy in image pixels should be given. After that,
    the segmentation pixels corresponding to the galaxy will be removed from the
    image. The image without the galaxy segmentation patches will be saved to
    newimagepath.
    '''
    hdulist = fits.open(imagepath) 
    raw_mask = hdulist[0].data
    segment = find_segment(raw_mask, coord)
    new_mask = remove_segment(raw_mask, segment)
    newhdu = fits.PrimaryHDU(new_mask)
    try:
        newhdu.writeto(newimagepath)
    except IOError:
        os.remove(newimagepath)
        newhdu.writeto(newimagepath)
    


def remove_segment(image, number):
    '''Takes a FITS image and removes the region corresponding to the segment.

    The image should be a ndarray corresponding to the raw FITS image. The
    number should be the segmentation number we wish to remove from the image.
    It will be replaced with 0.
    '''
    segindices = np.where(image == number)
    image_copy = image.copy()
    image_copy[segindices] = 0
    return image_copy

def normalize_segmentation_map(image, output):
    '''Takes a segmentation map and sets all of the pixels to be either 1 or
    0.'''
    command = "if im1 then 1.0 else 0.0"
    run_imcalc(image, output, command, newformat="real")

def clip_image(image, output, threshold):
    '''Sets all pixels lower than the given threshold to zero.'''
    command = "if im1 < {0} then 0.0 else im1".format(threshold)
    run_imcalc(image, output, command)

def find_segment(segimage, coord):
    '''Finds the SExtractor segment at a pixel value.

    This function uses the segmentation image to find the segmentation number of
    an object. The image should be passed as an ndarray, along with the
    coordinates in image pixel values.

    THE COORDINATES SHOULD NOT BE SPECIFIED AS NUMPY INDICES!
    '''
    return segimage[np.round(coord[1])-1, np.round(coord[0])-1]

def w1w3simulmask(config, w1image, w3image, w1initthresh, w3initthresh, 
        decfactor, diffactor, w1endthresh, w3threshfloor, masked_image, w1mask, 
        w3mask, accummask, iteration="geometric"):
    '''Masking algorithm which runs SExtractor on the w3 image to mask out the
    galaxy, and then uses the mask generated from the w3 image on the w1 image
    in order to capture Rayleigh-Jeans foreground sources.

    The algorithm starts out with w1initthresh and w3initthresh being given to 
    DETECT_THRESH in SExtractor. With each iteration, the threshold for w1 will
    decrease either arithmetically or geometrically by the decfactor, while the 
    threshold for w3 will decrease by decfactor * diffactor. The iterating 
    continues until the threshold for w1 drops below w1endthresh. In order to 
    deal with cases where the threshold for w3 would drop below zero if 
    diffactor is greater than 1, there will be a w3threshfloor argument which 
    bottoms out the value of w3. This should probably be at around 5 sigma.

    Each iteration will have the mask add on to accummask to preserve point
    sources which may be embedded in the galaxy. The location of the 
    intermediate w1 and w3 masks can be specified through the w1mask and w3mask
    arguments. The masked w1 image will be stored in masked_image.

    The effects of the parameters on the resulting mask can be roughly described
    as follows:
    w1initthresh: How sensitive the initial detection is. If you notice that
        flux from the center of the galaxy is being incorporated into the max,
        you may want to raise this. Note that raising this will also result in
        a longer computation.
    w3initthresh: How large the initial w1 mask is. If you notice that flux from
        the center of the galaxy is being incorporated into the mask, you may
        want to lower this. Additionally, if you notice that foreground stars
        close to the nucleus do not appear in the final mask, you will want to
        raise this.
    decfactor: How finely the iteration occurs. If you notice that foreground
        objects within the galaxy should be detected but aren't, you should 
        lower this value. Note: Lowering this value will also result in a longer
        computation.
    diffactor: How quickly the w3 mask increases over the w1 image. If you
        notice that objects embedded within the galaxy aren't being masked, then
        you should lower this value. However, if you notice that somewhere in
        the middle of the computation, the galaxy flux starts to appear in the
        final mask, you'll want to raise this value.
    w1endthresh: The ending threshold for w1. If you are not detecting faint
        enough objects, you will want to lower this. However, you don't want to
        lower it so far that it gets to be extremely difficult to avoid
        measuring galaxy flux.
    '''
    w1thresh = w1initthresh
    w3thresh= w3initthresh
    try:
        os.remove(accummask)
    except OSError:
        pass
    while w1thresh >= w1endthresh:
        if w3thresh <= w3threshfloor:
            w3thresh = w3threshfloor
        print "W1 threshold: {0}, W3 threshold: {1}".format(w1thresh, w3thresh)
        # Generate the W3 mask.
        sextractor_mask(w3image, w3thresh, config, maskoutput=w3mask)
        # Create the Masked W1 image.
        apply_mask(w1image, w3mask, masked_image)
        # Get the point sources from the masked W1 image.
        sextractor_mask(masked_image, w1thresh, config, maskoutput=w1mask)
        # Add the current mask to the accumulated mask.

        # We'll want to find a way to determine a better location for tempfile.
        # Probably through an argument.
        tempfile = "/home/regulus/simonian/year1/wise/sextractor_demo/NGC6946/tempmask.fits"
        try:
            run_imarith(accummask, '+', w1mask, tempfile)
            os.rename(tempfile, accummask)
        except OSError:
            print "Creating {0}.".format(accummask)
            os.rename(w1mask, accummask)
        if iteration is "arithmetic":
            w1thresh -= decfactor
            w3thresh -= decfactor * diffactor
        else:
            w1thresh /= decfactor
            w3thresh /= decfactor * diffactor
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

def mask_ellipse(image, center, semimajor, semiminor, pa, outputfile):
    '''Masks an elliptical region of the image.

    This function takes an elliptical region of an image and sets it to zero,
    effectively masking it. The center and semimajor axis should be in units of
    pixels. The masked image will then be outputted into outputfile.
    '''
    xcenter, ycenter = center
    command = build_imcalc_ellipse(xcenter, ycenter, semimajor, semiminor, pa)
    run_imcalc(image, outputfile, command)

def build_imcalc_ellipse(xcenter, ycenter, semimajor, semiminor, pa):
    '''Builds the command for creating an ellipse in imcalc.'''
    command = ("if ((x-{xcen})*cos({pa}) + (y-{ycen})*sin({pa}))**2 / {a}**2" +
    " + ((x-{xcen})*sin({pa}) - (y-{ycen})*cos({pa}))**2 / {b}**2 " + 
    " < 1 then 1 else im1").format(xcen=xcenter, ycen=ycenter, a=semiminor,
            b=semimajor, pa=-pa/180.0*math.pi)
    return command


def build_imcalc_circle(xcenter, ycenter, radius):
    '''Builds the command for creating a circle in imcalc.'''
    command = "if (x-{0})**2 + (y-{1})**2 < {2}**2 then 1 else im1".format(
            xcenter, ycenter, radius)
    return command

def run_imcalc(image, output, command, overwrite=True, newformat="old"):
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
    # Weird stuff can happen when we're doing a reduction. I.e. the output file
    # is also one of the input files. We'll try to deal with this corner case
    # the best way I know how.
    if output is image:
        newimage = image.rpartition(".")[2] + ".tmp.fits"
        shutil.move(image, newimage)
        image = newimage
    elif output in image:
        reassignindex = image.index(output)
        reassignfile = image[reassignindex]
        newimage = reassignfile.rpartition(".")[2] + ".tmp.fits"
        shutil.move(reassignfile, newimage)
        image[reassignindex] = newimage
    if overwrite:
        try:
            backup = backup_file(output)
        except IOError:
            pass
    iraf.stsdas()
    iraf.toolbox()
    iraf.imgtools()
    if type(image) is list:
        imagestring = ','.join(image)
    else:
        imagestring = image
    try:
        iraf.imcalc.setParam("pixtype", newformat)
        iraf.imcalc(imagestring, output, command)
    except iraf.IrafError as e:
        restore_file(output)
        raise e

def run_imshift(image, output, xshift, yshift):
    '''Runs imshift on the image.'''
    try:
        os.remove(output)
    except OSError:
        pass
    iraf.images()
    iraf.imgeom()
    iraf.imshift(image, output, xshift, yshift)


def run_imarith(arg1, operator, arg2, output):
    '''Runs the IRAF imarith routine.

    This function will take image or value given as arg 1, and then perform a
    binary operation defined by the operator string onto arg2 with it. The
    output image will then be placed at output.
    '''
    if not os.path.isfile(arg1) and not os.path.isfile(arg2):
        raise OSError("Neither argument is a valid path.")
    try:
        os.remove(output)
    except OSError:
        pass
    iraf.images()
    iraf.kmutil()
    iraf.imarith(arg1, operator, arg2, output)

def run_gauss(input, output, sigma):
    '''Performs a gaussian convolution on the input image.'''
    try:
        os.remove(output)
    except OSError:
        pass
    iraf.images()
    iraf.imfilter()
    iraf.gauss(input, output, sigma)

def run_immean(input):
    '''Calculates the mean value of the image given.

    If you only want the mean value of a rectangular section of an image, you
    can pass a slice of the image to only sample the cutout.
    '''
    iraf.stsdas()
    iraf.playpen()
    print input
    iraf.immean(input)
    meanvalue = iraf.immean.getParam("mean")
    return meanvalue

def backup_file(filepath):
    '''Performs a backup of a file by appending .backup to it. 
    
    Returns the path of the new file.'''
    dest = filepath + ".backup"
    shutil.move(filepath, dest)



def restore_file(filepath):
    '''Restores a file backed up by backup_file().'''
    source = filepath + ".backup"
    os.rename(source, filepath)
