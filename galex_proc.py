import os
import os.path
from collections import defaultdict
import subprocess
import tarfile
import glob
import gzip

from astropy.table import Table

import photometry as phot

def create_upload_file(ids, ras, decs, output):
    '''Creates a file that can be uploaded to MAST.

    This file will contain Object names, RAs, and Decs as a comma-separated
    list. The file will be saved to output, where it can manually be uploaded to
    the GALEX MAST database.
    '''
    if len(ids) is 0:
        return
    relevantTable = Table([ids, ras, decs], names=("ID", "RA", "DEC"))
    relevantTable.write(output, format="ascii.csv")

def select_best_surveys(inputfile, output_dir):
    '''Takes a file from the GALEX catalog and optimizes the exposures to use.

    There are numerous surveys which the GALEX mission has collected data from.
    They are categorized as AIS, MIS, DIS, NGS, and GII. Each has a different
    level of coverage and depth. AIS has shallow photometry of the whole sky,
    while GII often are PI projects which have very deep images of only a few
    areas. This function maximizes the exposure time for the images that will be
    downloaded.
    '''
    inputinfo = Table.read(inputfile, format="ascii.csv")

    surveys = ["AIS", "DIS", "GII", "GIS", "MIS", "NGS"]
    # Make a dictionary associated with each filename.
    filepaths = dict((survey, os.path.join(output_dir,
        "{0}.csv".format(survey))) for survey in surveys)
    # We want to clear out all files before resorting them.
    for surveyfile in filepaths:
        try:
            os.remove(surveyfile)
        except OSError:
            pass

    surveytables = {}
    for survey in surveys:
        surveytables[survey] = Table(inputinfo, copy=True)
        surveytables[survey].remove_rows(slice(len(inputinfo)))

    # These lists will be used to quickly sort the downloaded images into
    # their correct destinations
    objectlist = []
    NUVlist = []
    FUVlist = []
    sortdict = defaultdict(list)
    inputgroup = inputinfo.group_by("uploadID")
    for objectinfo in inputgroup.groups:
        objectinfo.sort('nuv_exptime')
        topnuv = objectinfo[-1]
        objectinfo.sort('fuv_exptime')
        topfuv = objectinfo[-1]
        # If one frame has both the highest NUV and FUV exposure, then only
        # download it once. If not, then put it on both lists.
        surveytables[topnuv["survey"]].add_row(topnuv)
        if topnuv["photoextractid"] != topfuv["photoextractid"]:
            print "Split survey between {0} and {1}!".format(topnuv["tilename"],
                    topfuv["tilename"])
            surveytables[topfuv["survey"]].add_row(topfuv)
        objectlist.append(topfuv["uploadID"])
        NUVlist.append(topnuv["tilename"])
        FUVlist.append(topfuv["tilename"])
            
    for survey in surveys: 
        tab = surveytables[survey]
        path = filepaths[survey]
        create_upload_file(tab["uploadID"], tab["uploadRA"], tab["uploadDEC"],
                path)       

    sortTable = Table([objectlist, NUVlist, FUVlist], names=("object",
        "NUV_Tile", "FUV_Tile"))
    sortTable.write(os.path.join(output_dir, "sorttable.csv"), 
            format="ascii.csv")

def process_GALEX_tarfile(BASEDIR, workfolder, sortTable, tempfolder="images"):
    """Processes a tarfile downloaded from GALEX using sortTable."""
    tempfolder = os.path.join(workfolder, tempfolder)
    # We first want to go through all of the tar archives and extract them into
    # tempfolder. This will make a single location that contains all of the
    # tiles.
    filelist = glob.glob(os.path.join(workfolder, "Galex*.tar"))
    for tarball in filelist:
        untar(tarball, tempfolder)
    # Next, go through the images and sort them into the correct directories in
    # BASEDIR.
    for entry in sortTable:
        galaxydir = phot.change_to_galaxy_dir(BASEDIR, entry["object"])
        matchstring = os.path.join(tempfolder, "{tile}*",
        "{tile}*-{band}-*.fits.gz")
        galexFUVfiles = glob.glob(matchstring.format(tile=entry["FUV_Tile"],
            band="fd")) 
        galexNUVfiles = glob.glob(matchstring.format(tile=entry["NUV_Tile"],
            band="nd") )
        for imagefile in galexFUVfiles + galexNUVfiles:
            gunzip(imagefile, galaxydir)
    
def untar(inputfile, outputdir):
    '''Extracts a tar file into a directory.'''
    tar = tarfile.open(inputfile)
    tar.extractall(outputdir)
    tar.close()

def gunzip(inputfile, outputdir):
    '''Extracts a gzipped file into a directory.'''
    compressedfile = gzip.open(inputfile)
    inputfilename = os.path.split(inputfile)[1]
    # Remove the .gz from the end.
    outputfilename = inputfilename[:-3]
    decompressedfile = open(os.path.join(outputdir, outputfilename), 'wb')
    decompressedfile.write(compressedfile.read())
    compressedfile.close()
    decompressedfile.close()
