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

def create_HYPERLEDA_upload_file(ids, output):
    '''Creates a file that can be uploaded to HYPERLEDA.

    The file will only contain object names without any columns in a way that
    can be immediately parseable by HYPERLEDA.'''
    # I'm a very naughty boy for doing this.
    # We don't want a header because that will cause problems with HYPERLEDA.
    # However, using the ascii.no_header writer will enclose the object names in
    # quotes, which also causes problems with HYPERLEDA. Therefore, the
    # workaround I've arrived at is to use a newline as the column name. When
    # writing the column name, it will instead make it into a blank line, which
    # is ignored by HYPERLEDA. It would be nice to have it actually be
    # configurable, though.
    hypertable = Table([ids], names=("\n",))
    hypertable.write(output, format="ascii.tab")

def select_best_surveys(inputfile, output_dir, keytable="sorttable.csv",
        blocklist="bad_images.csv"):
    '''Takes a file from the GALEX catalog and optimizes the exposures to use.

    There are numerous surveys which the GALEX mission has collected data from.
    They are categorized as AIS, MIS, DIS, NGS, and GII. Each has a different
    level of coverage and depth. AIS has shallow photometry of the whole sky,
    while GII often are PI projects which have very deep images of only a few
    areas. This function maximizes the exposure time for the images that will be
    downloaded.
    '''
    inputinfo = Table.read(inputfile, format="ascii.csv", fill_values=[("---",
        "0"), ("", "0")])
    inputinfo["fuv_exptime"].fill_value = 0.0
    inputinfo["nuv_exptime"].fill_value = 0.0
    inputinfo_filled = inputinfo.filled()

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
    inputgroup = inputinfo_filled.group_by("uploadID")
    for objectinfo in inputgroup.groups:
        # Weird bug where the -1 index cannot be given as an argument to rows.
        infolen = len(objectinfo)
        # We need to make copies because assignment only points to the position
        # in the table, not the row itself.
        objectinfo.sort('nuv_exptime')
        topnuv = Table(rows=objectinfo[infolen-1])[0]
        objectinfo.sort('fuv_exptime')
        topfuv = Table(rows=objectinfo[infolen-1])[0]
        # If one frame has both the highest NUV and FUV exposure, then only
        # download it once. If not, then put it on both lists.
        surveytables[topnuv["survey"]].add_row(topnuv)
        if topnuv["photoextractid"] != topfuv["photoextractid"]:
            surveytables[topfuv["survey"]].add_row(topfuv)
        objectlist.append(topfuv["uploadID"])
        NUVlist.append(galex_tilename(topnuv))
        FUVlist.append(galex_tilename(topfuv))
        print "{0}: {1}, {2}".format(objectlist[-1], NUVlist[-1], FUVlist[-1])
            
    for survey in surveys: 
        tab = surveytables[survey]
        path = filepaths[survey]
        create_upload_file(tab["uploadID"], tab["uploadRA"], tab["uploadDEC"],
                path)       
    sortTable = Table([objectlist, NUVlist, FUVlist], names=("object",
        "NUV_Tile", "FUV_Tile"))
    sortTable.write(os.path.join(output_dir, keytable), format="ascii.csv")

def galex_tilename(MASTrow):
    '''Transforms the MAST row into a full tilename.

    Simply using the "tilename" flag ignores the subtiles that are part of AIS
    images. This function will leave all other tilenames along, but append the
    subtile number to the tilename to avoid duplicate tiles.
    '''
    base_tilename = MASTrow["tilename"]
    subtile = MASTrow["subvis"]
    # Only AIS tiles appear to have non-negative subtile numbers.
    if subtile == -999:
        tilename = base_tilename
    else:
        tilename = "{0}_sg{1:02g}".format(base_tilename, subtile)
    return tilename

def process_GALEX_tarfile(BASEDIR, workfolder, sortTablepath, 
        tempfolder="images"):
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
    # We can potentially separate the extraction and the sorting into two
    # different functions. It might actually make more sense.
    sortTable = Table.read(sortTablepath, format="ascii.csv")
    for entry in sortTable:
        galaxydir = phot.change_to_galaxy_dir(BASEDIR, entry["object"])
        FUVstring = os.path.join(tempfolder, 
                folder_matchstring(entry["FUV_Tile"]), 
                "{tile}*-fd-*.fits.gz".format(tile=entry["FUV_Tile"]))
        NUVstring = os.path.join(tempfolder, 
                folder_matchstring(entry["NUV_Tile"]), 
                "{tile}*-nd-*.fits.gz".format(tile=entry["NUV_Tile"]))
        galexFUVfiles = glob.glob(FUVstring) 
        galexNUVfiles = glob.glob(NUVstring)
        if not galexFUVfiles:
            print "Could not match {0}.".format(entry["FUV_Tile"])
        for imagefile in galexFUVfiles + galexNUVfiles:
            try:
                gunzip(imagefile, galaxydir)
            except IOError:
                os.mkdir(galaxydir)
                gunzip(imagefile, galaxydir)

def folder_matchstring(filetile):
    '''Creates an approprite matchstring for a folder from a file tile.

    The AIS tiles are named hierarchically as:
    AIS_{tile}_*_sv{subtile}/AIS_{tile}_sg{subtile}*

    This function will take an AIS_{tile}_sg{subtile} string and transform it 
    so that it works for the folder tile. This involves breaking off the subtile
    to the right edge.

    For all other suveys, it will simply return the same thing except with an
    asterisk. e.g. GISAWEAJWA21q2*'''
    if filetile.startswith("AIS"):
        subtile = filetile[10:12]
        tilename = filetile[0:7]
        folderstring = "{0}_*_sv{1}".format(tilename, subtile)
    else:
        folderstring = "{0}*".format(filetile)
    return folderstring
    
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
