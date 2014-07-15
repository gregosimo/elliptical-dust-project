import glob
import urllib
import urlparse
import subprocess
import os
import os.path
import gzip

import requests
import astropy
from astropy.table import Table
import numpy as np

import photometry as phot

# We begin with an IPAC table which has object names and ra/dec coordinates. We
# must first query the WISE Image metadata server to get the images which
# correspond to those coordinates. The WISE Image metadata server is located at:
METADATA_SERVER ="http://irsa.ipac.caltech.edu/ibe/search/wise/allwise/p3am_cdd"
# Further queries should be placed after the url beginning with a ? and then
# parameters
#
# This query will return an IPAC table which contains the coaddgrp, coadd_ra,
# coadd_id and bands available for that location. We then place the images into
# the correct folder in the BASEDIR.
IMAGE_SERVER = "http://irsa.ipac.caltech.edu/ibe/data/wise/allwise/p3am_cdd/{coaddgrp:s}/{coadd_ra:s}/{coadd_id:s}/{coadd_id:s}-w{band:1d}-int-3.fits"

# This is the code which corresponds to the latest WISE catalog. In this case,
# it is for ALLWISE.
LATEST_WISE_CODE = "ac"

def batch_download_images(BASEDIR, objects, ras, decs, upgrade=False):
    '''Downloads all images for many objects.

    The objects, ras, and decs variables should be arrays with the same length.
    '''
    for object, ra, dec in zip(objects, ras, decs):
        coaddID = query_metadata(ra, dec)
        query_image(BASEDIR, object, coaddID, ra, dec)

def query_metadata(ra, dec):
    '''Queries the WISE Image Metadata service for image information.

    This function will send a GET query to the WISE Image Metadata service. It
    asks for the information about the images at the given RA and Dec. It will
    then return a dictionary containing the coaddgrp, coadd_ra, coadd_id.

    The RA and Dec should be given in decimal format. No sexagecimal stuff!
    '''
    # The value given to mcen is ignored, but it will cause the database to only
    # give the most centered tile anyway.
    payload = {"POS": "{0},{1}".format(ra, dec), "mcen": "1"}
    payloadget = urllib.urlencode(payload)
    metatable = Table.read(get_url(METADATA_SERVER, payloadget), 
        format="ascii.ipac")
    if len(metatable) < 4:
        raise ValueError("Not all WISE Colors found")
    elif len(metatable) > 4:
        raise ValueError("Multiple Entries Found")
    coaddID = np.unique(metatable["coadd_id"])
    if len(coaddID) != 1:
        raise ValueError("More than one coadd found")
    return coaddID[0]

def query_image(BASEDIR, objstr, coaddID, ra, dec, size=600, upgrade=False):
    '''Downloads WISE images into the correct directories.

    Objstr should be the full name of the object. If the directory corresponding
    to the object does no exist, this function will create it. If it does exist,
    this function will either skip the download, or replace it with a more
    recent version.

    All that's needed is the name of the object in order to correctly detect the
    directory, and the coaddID to fetch the images from the server.
    '''
    galaxydir = phot.change_to_galaxy_dir(BASEDIR, objstr)
    coadddic = {"coaddgrp": get_coaddgrp(coaddID), "coadd_ra":
        get_coadd_ra(coaddID), "coadd_id": coaddID}
    try:
        if not check_galaxy_images_version(galaxydir):
            if upgrade:
                print "Upgrading images for {0}".format(objstr)
                upgrade_images(galaxydir, coadddic, ra, dec, size)
        else:
            print "Skipping {0}: Folder exists.".format(objstr)
    except RuntimeError:
        os.mkdir(galaxydir)
        download_images(galaxydir, coadddic, ra, dec, size)
    print "Images for {0} downloaded".format(objstr)

def upgrade_images(galaxydir, coadddic, ra, dec, size=600):
    '''Performs an upgrade for images in a directory.
    
    This requies going through the WISE images, deleting them, and then
    downloading the new images.'''
    for band in phot.IRBANDS:
        os.remove(os.path.join(galaxydir, phot.match_filter(galaxydir, band)))
    download_images(galaxydir, coadddic, ra, dec, size)

def download_images(galaxydir, coadddic, ra, dec, size):
    '''Downloads the images into the given directory.
    
    This function will download cutouts. Therefore, it will need to know the RA,
    Dec, and the size of the cutout. The size which is generally returned by the
    IRSA web interface is 600 arcseconds.'''
    # Downloading all bands
    for i in range(1,5):
        coadddic["band"] = i
        image_url = IMAGE_SERVER.format(**coadddic)
        query_params = {"center": "{0},{1}".format(ra, dec), "size":
                "{0}arcsec".format(size)}
        image_query = get_url(image_url, urllib.urlencode(query_params))
        # The -P sets the prefix for the downloaded files. So we want them to be
        # located in galaxydir.
        command = "wget"
        wget_flags = "-P{0}".format(galaxydir)
        subprocess.call([command, wget_flags, image_query])
        # Once the image is downloaded, we want to uncompress it, and then
        # delete the compressed file.
        #
        # This function finds the filename for the url string, and then appends
        # that to the path with a .gz extension.
        urlparts = urlparse.urlsplit(image_query)
        downloaded_filename = glob.glob(os.path.join(galaxydir,
            "*w{0}*".format(i)))[0]
        compressed_image_filename = ".".join([os.path.split(urlparts.path)[1], 
            "gz"])
        compressed_path = os.path.join(os.path.split(downloaded_filename)[0],
                compressed_image_filename)
        os.rename(downloaded_filename, compressed_path)
        subprocess.call(["gunzip", compressed_path])
                

def check_galaxy_images_version(galaxydir):
    '''Checks if the WISE images are from the latest catalog.

    The code of the latest catalog is in the variable LATEST_WISE_CODE. The
    codes of images in the folder will be checked against this. If they are not
    more recent, then this function will return false.
    '''
    image_name = os.path.split(phot.match_filter(galaxydir, "W1"))[-1]
    return get_version(image_name) >= LATEST_WISE_CODE

def get_version(filename):
    '''Extracts the version code from the filename of a file'''
    return filename[9:11]

def get_coaddgrp(coaddID):
    '''Extracts the coaddgrp from the coaddID'''
    return coaddID[0:2]

def get_coadd_ra(coaddID):
    '''Extracts the coadd_ra from the coaddID'''
    return coaddID[0:4]

def get_url(baseurl, getstring):
    '''Transforms a url and a data string into a full http GET request.

    For some reason, there don't appear to be any functions taht can simply take
    a URL and a data to form a URL. The HTTP Request functions in urllib2 need
    to actually make the connection before a valid url is given. The same occurs
    with request. I just want the full HTTP GET URL, so it can be passed to an
    Astropy table.
    '''
    return "?".join([baseurl, getstring])

