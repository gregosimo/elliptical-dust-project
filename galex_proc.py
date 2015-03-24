import os
import os.path
from collections import defaultdict
import subprocess
import tarfile
import glob
import gzip

from astropy.table import Table
import astropy.io.fits as fits
from pyraf import iraf

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

    surveys = ["AIS", "DIS", "GII", "GIS", "MIS", "NGS", "ETS", "CAS", "CAI"]
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
    RAlist = []
    DEClist = []
    # Now that I have RA and Dec, this seems like a weird way of doing things.
    # It leaves a bad taste in my mouth.
    sortdict = defaultdict(list)
    skipped_objects = []
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
        if topnuv["nuv_exptime"] > 0 and topfuv["fuv_exptime"] > 0:
            surveytables[topnuv["survey"]].add_row(topnuv)
            # If the two photos are in different surveys
            if topnuv["survey"] != topfuv["survey"]:
                surveytables[topfuv["survey"]].add_row(topfuv)
            objectlist.append(topfuv["uploadID"])
            RAlist.append(topfuv["uploadRA"])
            DEClist.append(topfuv["uploadDEC"])
            NUVlist.append(galex_tilename(topnuv))
            FUVlist.append(galex_tilename(topfuv))
            print "{0}: {1}, {2}".format(objectlist[-1], NUVlist[-1], 
                    FUVlist[-1])
        else:
            print "Excluding {0} from the list".format(topfuv["uploadID"])
            if topnuv["nuv_exptime"] <= 0:
                print "Did not have NUV exposure."
            if topfuv["fuv_exptime"] <= 0:
                print "Did not have FUV exposure."
            skipped_objects += topfuv["uploadID"]
            
    for survey in surveys: 
        tab = surveytables[survey]
        path = filepaths[survey]
        create_upload_file(tab["uploadID"], tab["uploadRA"], tab["uploadDEC"],
                path)       
    sortTable = Table([objectlist, RAlist, DEClist, NUVlist, FUVlist], 
            names=("object", "RA", "DEC", "NUV_Tile", "FUV_Tile"))
    sortTable.write(os.path.join(output_dir, keytable), format="ascii.csv")
    return skipped_objects

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

def extract_GALEX_folder(tarfolder, extractedfolder):
    '''Extracts GALEX tar files into a folder.

    The contents of the tar file is put into extractedfolder.
    '''
    # We first want to go through all of the tar archives and extract them into
    # tempfolder. This will make a single location that contains all of the
    # tiles.
    filelist = glob.glob(os.path.join(tarfolder, "Galex*.tar"))
    for tarball in filelist:
        untar(tarball, extractedfolder)


def process_GALEX_tarfile(BASEDIR, workfolder, sortTablepath, 
        tempfolder="images", sidelength=1000, replacement_path=""):
    """Processes a tarfile downloaded from GALEX using sortTable.
    
    BASEDIR is the directory where we want the image folders to be.
    
    Workfolder is the location which contains the tarfiles as well as the
    location that will have tempfolder. This does not necessarily have to be
    identical to BASEDIR, but often is.

    Sorttablepath is the path to the sorttable file. The sorttable file should
    be in the form of:
    NGCXXXX,RA,DEC,NUV_TILE_NAME_NUMBER,FUV_TILE_NAME_NUMBER.
    
    The replacement_path should lead to a table in the same form as sorttable,
    except with post-hoc additions which will be incorporated in the final
    steps. """
    # We'll go through the images and sort them into the correct directories in
    # BASEDIR.
    sortTable = Table.read(sortTablepath, format="ascii.csv", guess=False)
    for entry in sortTable:
        galaxydir = phot.change_to_galaxy_dir(BASEDIR, entry["object"])
        # Added this because Gil de Paz tables have the underscore replaced by a
        # hypen for reasons I have no idea about.
        fuvtile = entry["FUV_Tile"].replace("-", "_")
        FUVstring = os.path.join(
            tempfolder, folder_matchstring(fuvtile), 
            "*-fd-*.fits.gz".format(tile=fuvtile))
        nuvtile = entry["NUV_Tile"].replace("-", "_")
        NUVstring = os.path.join(
            tempfolder, folder_matchstring(nuvtile), 
            "*-nd-*.fits.gz".format(tile=nuvtile))
        galexFUVfiles = glob.glob(FUVstring) 
        galexNUVfiles = glob.glob(NUVstring)
        if not galexFUVfiles:
            print "Could not match {0}.".format(entry["FUV_Tile"])
        for imagefile in galexFUVfiles + galexNUVfiles:
            extractedimage = imagefile[:-3]
            imagedir, imagename = os.path.split(extractedimage)
            try:
                gunzip(imagefile, imagedir)
            except:
                print "Error with {0}".format(imagefile)
            if "-flags" not in imagefile:
                ra, dec = entry["RA"], entry["DEC"]
                # If it throws an IrafError, that means the destination
                # directory didn't exist, and we'd like to change that.
                try:
                    try:
                        extract_image_with_coordinates(
                            extractedimage, ra, dec, sidelength, sidelength, 
                            galaxydir)
                    except iraf.IrafError:
                        os.mkdir(galaxydir)
                        extract_image_with_coordinates(
                            extractedimage, ra, dec, sidelength, sidelength, 
                            galaxydir)
                # If an IOError is thrown, that means there was something
                # strange that occurred with the extraction and I'd like to
                # look into it.
                except IOError:
                    print
                    print "Error extracting {0}.".format(extractedimage)
                    print
                # Although, it seems like there just might have been issues on
                # the GALEX portion, not this portion.
                
def extract_image_with_coordinates(original, centerra, centerdec, arcsecwidth,
        arcsecheight, destination, band="NUV"):
    '''Copies a part of an image to a destination file.

    The coordinates will be given in celestial coordinates. Note that the actual
    computation is done on physical coordinates, so for particularl distortion
    portions of the image, weird geometries may occur.
    '''
    centerx, centery = phot.getpixelcoords(original, centerra, centerdec)
    width = arcsecwidth / phot.getPixelScale(band)
    height = arcsecheight / phot.getPixelScale(band)

    extract_from_image_with_height_width(original, centerx, centery, height,
            width, destination)

def extract_from_image_with_height_width(original, centerx, centery, height,
        width, destination):
    '''Copies a part of an image to a destination file.

    The part will have the center given by centerx, centery, and will have a
    given height and width'''
    lowerx = centerx - width
    upperx = centerx + width
    lowery = centery - width
    uppery = centery + width

    extract_from_image_with_bounds(original, lowerx, upperx, lowery, uppery,
            destination)

def extract_from_image_with_bounds(original, lowerx, upperx, lowery, uppery, 
        destination):
    '''Copies a part of an image to a destination file.

    The original image plus bounds in pixels will be written to the destination.
    The destination will be clobbered.
    '''
    fitsheader = fits.getheader(original)
    xsize = int(fitsheader["NAXIS2"])
    ysize = int(fitsheader["NAXIS1"])
    lowerx = fix_to_within_bounds(int(lowerx), xsize, 0)+1
    upperx = fix_to_within_bounds(int(upperx), xsize, 0)+1
    lowery = fix_to_within_bounds(int(lowery), ysize, 0)+1
    uppery = fix_to_within_bounds(int(uppery), ysize, 0)+1
    subimage = "{0}[{1}:{2},{3}:{4}]".format(
        original, lowerx, upperx, lowery, uppery)
    run_imcopy(subimage, destination)

def fix_to_within_bounds(val, valmax, valmin=0):
    '''Fixes bounds such that val lies between valmin and valmax.

    "Between" takes on the traditional Python definition of inclusive for min
    and exclusive for max. If val is outside of those bounds, it is reset to be
    one of the edge values.'''
    if val < valmin:
        val = valmin
    elif val >= valmax:
        val = valmax-1
    return val

def run_imcopy(original, destination):
    '''A raw layer on top of imcopy.

    Imcopy has a lot of flexibility, such as with copying multiple images, or
    copying sections of an image, or pattern-matching. However, utilizing these
    features shouldn't be done by working directly with this file, but rather by
    having layers on top of it which are more pythonic.

    For simple image copying, this function should be simple enough.
    '''
    if os.path.isfile(destination):
        os.remove(destination)
    elif os.path.isdir(destination):
        extension = ".fits"
        basefile = os.path.basename(original)
        filename = basefile[:basefile.index(extension)+len(extension)]
        destination_path = os.path.join(destination, filename)
        # In the corner case where the directory exists, but we haven't put a
        # file in there yet, this will prevent failures to remove from causing
        # major problems.
        # If this doesn't work, simply do a os.path.isfile(destination_path)
        # before removing.
        try:
            os.remove(destination_path)
        except OSError:
            pass


    iraf.images()
    iraf.imutil()
    iraf.imcopy(original, destination)

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
        secondmarker = filetile.rindex("_")
        subtile = filetile[secondmarker+3:]
        tilename = filetile[:secondmarker]
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
