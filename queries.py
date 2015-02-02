import glob
import urllib
import urlparse
import subprocess
import os
import os.path
import gzip

import requests
import astropy
from astropy.table import Table, vstack
from astroquery.ned import Ned
from astroquery.irsa import Irsa
import numpy as np

import photometry as phot

# This is the entry point for the catalog.
CATALOG_BASE = "http://irsa.ipac.caltech.edu/cgi-bin/Gator/nph-query"

# These are a bunch of lookup tables for the WISE catalog.
CATALOGS=["AllWISE", "All-Sky"]
ATLAS_CATALOG_NAMES = {"AllWISE": "wise_allwise_p3am_cdd", "All-Sky":
        "wise_allsky_4band_p3am_cdd"}
INVERTED_ATLAS_CATALOG_NAMES = {v: k for k,v in CATALOG_NAMES.items()}
# A bunch of helper functions to organize the dictionaries here.
# The dictionaries can probably be bypassed entirely in favor of these helper
# functions... but that's more architecture change than I'm currently willing to
# take.
def extract_mission_from_full_catalog_name(catalog):
    '''Extracts the mission part from a catalog name.'''
    return catalog[:catalog.index("_")] 

def extract_survey_from_full_catalog_name(catalog):
    '''Extracts the survey part from a catalog name.'''
    missionremoved = catalog[catalog.index("_")+1:]
    return missionremoved[:missionremoved.index("_")]

def extract_catalog_folder_from_full_catalog_name(catalog):
    '''Extracts the catalog folder from a catalog name.'''
    missionremoved = catalog[catalog.index("_")+1:]
    return missionremoved[missionremoved.index("_")+1:]
# I noticed that the data directories seem to be subdivided the same way the
# catalog is. Therefore, building up a directory catalog string doesn't seem to
# be too bad.
# Please kill me for doing things in this spaghettified way. I'm basically
# trying to split the CATALOG_NAMES into the component parts. For example,
# "wise_allwise_p3as_psd" becomes "wise", "allwise", "p3as_psd". There should
# just be a group of functions that extract the values...
MISSION_NAMES = {k: extract_mission_from_full_catalog_name(v) for (k,v) in 
        ATLAS_CATALOG_NAMES.iteritems()}
SURVEY_NAMES = {k: extract_survey_from_full_catalog_name(v) for (k,v) in 
        ATLAS_CATALOG_NAMES.iteritems()}
CATALOG_FOLDER_NAMES = {k: extract_catalog_folder_from_full_catalog_name(v) for 
        (k,v) in ATLAS_CATALOG_NAMES.iteritems()}

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
ALLWISE_BASE = "http://irsa.ipac.caltech.edu/ibe/sia/wise/allwise/p3am_cdd"
ALLSKY_BASE  = \
        "http://irsa.ipac.caltech.edu/ibe/sia/wise/allsky/4band_p3am_cdd"

IMAGE_SERVER = "http://irsa.ipac.caltech.edu/ibe/data/wise/allwise/p3am_cdd/{coaddgrp:s}/{coadd_ra:s}/{coadd_id:s}/{coadd_id:s}-w{band:1d}-int-3.fits.gz"
UNCERTAINTY_SERVER = "http://irsa.ipac.caltech.edu/ibe/data/wise/allwise/p3am_cdd/{coaddgrp:s}/{coadd_ra:s}/{coadd_id:s}/{coadd_id:s}-w{band:1d}-unc-3.fits.gz"

# This is the code which corresponds to the latest WISE catalog. In this case,
# it is for ALLWISE.
LATEST_WISE_CODE = "ab"



def construct_search_url(survey):
    '''Constructs a metadata search url for a survey.

    This url will *NOT* contain the query info, just the resource name. For
    example, the URL for the AllWISE database would be 

    http://irsa.ipac.caltech.edu/ibe/search/wise/allwise/p3am_cdd
    '''
    # Ehhh.... make a new function that does this line automatically?
    extension = CATALOG_EXTENSION.format(operation="search",
            mission=MISSION_NAMES[survey], survey=SURVEY_NAMES[survey],
            catalog=CATALOG_FOLDER_NAMES[survey])
    fullurl = urlparse.urljoin(IRSA_BASE, extension)
    return fullurl

def construct_image_url(survey, coaddid, band):
    '''Constructs an image URL.
    
    This URL will be the path to an image tile. An example of this would be:
        
    http://irsa.ipac.caltech.edu/ibe/data/wise/allwise/p3am_cdd/03/0390/0390p605_ac51/0390p605_ac51-w1-int-3.fits'''
    extension = CATALOG_EXTENSION.format(operation="data",
            mission=MISSION_NAMES[survey], survey=SURVEY_NAMES[survey],
            catalog=CATALOG_FOLDER_NAMES[survey])
    firstbase = urlparse.urljoin(IRSA_BASE, extension)
    coaddgrp, coaddra = parse_coaddID(coaddid)
    # Let both 2 and W2 be valid.
    band = str(band)
    if band.startswith("w") or band.startswith("W"):
        band = band[-1]
    fileextend = FILE_EXTENSION.format(coaddgrp=coaddgrp, coadd_ra=coaddra,
            coadd_id=coaddid, band=band)
    fullurl = urlparse.urljoin(firstbase, fileextend)
    return fullurl

def batch_download_images(BASEDIR, objects, ras, decs, surveys, size=600, 
        upgrade=False, uncertainty=True, overwrite=True):
    '''Downloads all images for many objects.

    The objects, ras, and decs variables should be arrays with the same length.
    It will download square cutouts with length given by size arcseconds. If the
    uncertainty keyword is given as true, it will additionally download the
    corresponding uncertainty files.
    '''
    # I'm not exactly sure how to deal with this, so I will hack away!
    catalogtablepath = "/tmp/table"
    objecttable = Table({"objstr": objects, "ra": ras, "dec": decs})
    objecttable.write(catalogtablepath, format="ascii.ipac")
    objectcatalog = get_WISE_catalog_entries(catalogtablepath)
    for object, ra, dec, survey in objectcatalog[["objstr_01", "ra", "dec",
            "cat"]]:
        coaddID = query_metadata(ra, dec, survey)
        query_image(BASEDIR, object, survey, coaddID, ra, dec, size=size,
                uncertainty=uncertainty, overwrite=overwrite)

def query_WISE_catalog_file_upload(inputpath, url=CATALOG_BASE, 
        catalog=CATALOG_NAMES["AllWISE"], radius=10, 
        cols=['ra', 'dec', 'w1rsemi', 'w1ba', 'w1pa', 'w1gmag', 
        'w1sat', 'w2rsemi', 'w2ba', 'w2pa', 'w2gmag', 'w2sat', 'w3rsemi', 
        'w3ba', 'w3pa', 'w3gmag', 'w3sat', 'w4rsemi', 'w4ba', 'w4pa', 
        'w4gmag', 'w4sat']):
    '''Queries IRSA for the objects found in the given catalog.
    The filename '''	
    data = {"catalog": catalog, "spatial": "Upload", "uradius": radius, 
            "outfmt": 1, 'selcols': ','.join(cols)}
    files = {'filename': open(inputpath, "rb")}
    ipac_output = requests.post(url, data=data, files=files)
    ##########################################################################
    # This section of code will be unnecessary when astropy 1.0.0 is released.
    # Remove at that point.
    filedir, filename = os.path.split(inputpath)
    outputpath = os.path.join(filedir, "output_{0}".format(filename))
    outputhandle = open(outputpath, 'w')
    outputhandle.write(ipac_output.content)
    outputhandle.close()
    expandedpath = os.path.join(filedir, "expanded_{0}".format(filename))
    expand_IPAC_table(outputpath, expandedpath)
    ipac_table = Table.read(expandedpath, format="ascii.ipac")
    ##########################################################################
    # Uncomment the line below in order in order to upgrade to astropy 1.0.0.
    #ipac_table = Table.read(ipac_output.content, format="ascii.ipac")
    ipac_table = clear_invalid_entries(ipac_table, ipac_table["w1rsemi"])
    ipac_table["cat"] = INVERTED_CATALOG_NAMES[catalog]
    return ipac_table
	
def clear_invalid_entries(fulltable, indexcolumn):
    '''Removes rows that are invalid in indexcolumn from the fulltable.

    Invalid entries in this case are entries with masked values.'''
    try:
        return fulltable[~indexcolumn.mask]
    # If the mask doesn't exist, then indexcolumn isn't a masked column, and we
    # shouldn't have a problem.
    except AttributeError:
        return fulltable

def get_WISE_catalog_entries(objectfile):
    '''Gets entries from objectfile and returns it as a table.
	
    This function first gets the AllWISE data for the objects in objectfile,
    and then gets the WISE All-Sky data for the objects in objectfile. For 
    objects which are saturated in the AllWISE data, it will replace them 
    with objects in the All-Sky data, thereby decreasing the effects of 
    saturation.'''
    allwiseTable = query_WISE_catalog_file_upload(objectfile, 
            catalog=CATALOG_NAMES["AllWISE"])
    allskyTable = query_WISE_catalog_file_upload(objectfile, 
            catalog=CATALOG_NAMES["All-Sky"])
    satobjects  = (allwiseTable["w1sat"] + allwiseTable["w2sat"] +
            allwiseTable["w3sat"] + allwiseTable["w4sat"])
    # I'm hoping there's a better way to iterate through lists, because we're
    # not guaranteed that there will be a correct number of valid entries. Some
    # weird poblems could occur where this is not the case.
    # I'd rather do something like joining and then processing or just
    # processing by objstr rather than index.
    for i, satpixels in enumerate(satobjects):
        if satpixels != 0:
            allwiseTable[i] = allskyTable[i]
    return allwiseTable

def query_metadata(ra, dec, survey):
    '''Queries the WISE Image Metadata service for image information.

    This function will send a GET query to the WISE Image Metadata service. It
    asks for the information about the images at the given RA and Dec. It will
    then return a dictionary containing the coaddgrp, coadd_ra, coadd_id.

    The RA and Dec should be given in decimal format. No sexagecimal stuff!

    NOTE: This method assumes the coaddID is the same for all surveys. This may
    not necessarily be the case. However, it's easier to assume that as a
    workaround.
    '''
    # The value given to mcen is ignored, but it will cause the database to only
    # give the most centered tile anyway.
    payload = {"POS": "{0},{1}".format(ra, dec), "mcen": "1"}
    payloadget = urllib.urlencode(payload)
    metatable = Table.read(get_url(construct_search_url(survey), payloadget), 
        format="ascii.ipac")
    if len(metatable) < 4:
        raise ValueError("Not all WISE Colors found")
    elif len(metatable) > 4:
        raise ValueError("Multiple Entries Found")
    coaddID = np.unique(metatable["coadd_id"])
    if len(coaddID) != 1:
        raise ValueError("More than one coadd found")
    return coaddID[0]

def query_image(BASEDIR, objstr, survey, coaddID, ra, dec, size=600, 
        upgrade=False, uncertainty=True, overwrite=True):
    '''Downloads WISE images into the correct directories.

    Objstr should be the full name of the object. If the directory corresponding
    to the object does no exist, this function will create it. If it does exist,
    this function will either skip the download, or replace it with a more
    recent version. If you want to overwrite an existing folder, use the
    overwrite flag.

    All that's needed is the name of the object in order to correctly detect the
    directory, and the coaddID to fetch the images from the server.

    If the uncertainty keyword is set to true, then uncertainty images will also
    be downloaded from the Atlas web site.
    '''
    galaxydir = phot.change_to_galaxy_dir(BASEDIR, objstr)
    coadddic = {"coaddgrp": get_coaddgrp(coaddID), "coadd_ra":
        get_coadd_ra(coaddID), "coadd_id": coaddID}
    # If the folder exists, check to see if we want to upgrade. If we do, then
    # check if the images are up to date. If they aren't, then download them
    # using upgrade_images.
    if os.path.isdir(galaxydir):
        if overwrite:
            download_images(galaxydir, survey, coadddic, ra, dec, size)
        elif upgrade and not check_galaxy_images_version(galaxydir):
            print "Upgrading images for {0}".format(objstr)
            upgrade_images(galaxydir, survey, coadddic, ra, dec, size)
        else:
            print "Skipping {0}: Folder exists.".format(objstr)
    # If the folder doesn't exist, make it and download the images into it.
    else:
        os.mkdir(galaxydir)
        download_images(galaxydir, survey, coadddic, ra, dec, size)
    print "Images for {0} downloaded".format(objstr)

def upgrade_images(galaxydir, survey, coadddic, ra, dec, size=600):
    '''Performs an upgrade for images in a directory.
    
    This requies going through the WISE images, deleting them, and then
    downloading the new images.'''
    for band in phot.IRBANDS:
        os.remove(os.path.join(galaxydir, phot.match_filter(galaxydir, band)))
    download_images(galaxydir, survey, coadddic, ra, dec, size)

#def construct_image_query(BASEURL, coadddic, ra, dec, size 

def download_images(galaxydir, survey, coadddic, ra, dec, size, uncertainty=True):
    '''Downloads the images into the given directory.
    
    This function will download cutouts. Therefore, it will need to know the RA,
    Dec, and the size of the cutout. The size which is generally returned by the
    IRSA web interface is 600 arcseconds.'''
    # Downloading all bands
    for i in range(1,5):
        coadddic["band"] = i
        imagebase = construct_image_url(survey, coadddic["coadd_id"], "w"+str(i))
        query_params = {"center": "{0},{1}".format(ra, dec), "size":
                "{0}arcsec".format(size)}
        image_query = get_url(imagebase, urllib.urlencode(query_params))
        download_image(galaxydir, image_query, "w{0}".format(i))
        if uncertainty:
            uncert_url = imagebase.replace("int", "unc") + ".tar.gz"
            uncert_query = get_url(uncert_url, urllib.urlencode(query_params))
            download_image(galaxydir, uncert_query, "w{0}".format(i))
                
def download_image(galaxydir, image_query, band):
    '''Downloads an image into galaxydir.

    The image_query argument can be any valid HTTP request which resolves into an
    image which can be downloaded. Band should be the name of the band e.g. W1.
    '''
    downloaded_filename = os.path.basename(image_query).split("?")[0]
    compressed_path = os.path.join(galaxydir, downloaded_filename)
    # The -P sets the prefix for the downloaded files. So we want them to be
    # located in galaxydir.
    # The better way of doing this will be to use --content-disposition to name
    # the file. However, the current version of wget installed on this machine
    # is 1.12, and I'm running into a bug with it. When wget is upgraded to
    # 1.15, we'll see if that is still a problem.
    wget_command = ["wget", "--directory-prefix={0}".format(galaxydir), 
            "--content-disposition", image_query]
    subprocess.call(wget_command)
    # Once the image is downloaded, we want to uncompress it, and then
    # delete the compressed file.
    subprocess.call(["gunzip", "--force", compressed_path])

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

def parse_coaddID(coaddID):
    '''Returns a tuple with the coaddgrp an coaddra'''
    return get_coaddgrp(coaddID), get_coadd_ra(coaddID)

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

def ned_resolve(objects):
    '''Takes a list of objects and resolves them into RA and Dec.

    The resolution takes place using NED. It will return an astropy table
    containing the original names of the objects, along with columns for their
    RAs and Decs.
    '''
    resolvedlist = [Ned.query_object(object) for object in objects]
    resolvedtable = vstack(resolvedlist)
    relevanttable = Table(resolvedtable[["Object Name", "RA(deg)", "DEC(deg)"]],
            names=("ID", "RA", "DEC"))
    relevanttable["ID"] = objects
    return relevanttable

def run_stilts(taskname, **taskargs):
    '''Wrapper function for the stilts program.

    Runs the STILTS program with the given taskname, and the arguments required
    for that task. The options for that task should be given in taskargs.
    '''
    command = ["stilts"] + [taskname] + ["{0}={1}".format(k,v) for k,v in 
            taskargs.items()]
    subprocess.check_call(command)

def expand_IPAC_table(inputfile, outputfile):
    '''De-abbreviates an IPAC file.

    Takes an abbreviated IPAC file at input, and then rewrites it to output,
    which will not be contracted.'''
    # Since "in" and "out" are reserved python keywords, I will have to
    # work around the fact that I can't use them as keyword args.
    run_stilts("tcopy", ifmt="ipac", ofmt="ipac", **{"in": inputfile, 
        "out": outputfile})
