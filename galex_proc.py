import os
import os.path

from astropy.table import Table

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

    surveytables = dict((survey, Table(inputinfo, 
        copy=True).remove_rows(slice(-1, 0))) for survey in surveys)

    # These lists will be used to quickly sort the downloaded images into
    # their correct destinations
    objectlist = []
    NUVlist = []
    FUVlist = []

    inputgroup = inputinfo.group_by("uploadID")
    for objectinfo in inputgroup.groups:
        objectinfo.sort('nuv_exptime')
        topnuv = objectinfo[0]
        objectinfo.sort('fuv_exptime')
        topfuv = objectinfo[0]
        # If one frame has both the highest NUV and FUV exposure, then only
        # download it once. If not, then put it on both lists.
        surveytables[topnuv["survey"]].add_row(topnuv)
        if topnuv is not topfuv:
            surveytables[topfuv["survey"]].add_row(topfuv)
        objectlist.append(topfuv["uploadID"])
        NUVlist.append(topnuv["tilename"])
        FUVlist.append(topfuv["tilename"])
            
    for tab, path in zip(surveytables, filepaths):
        create_upload_file(tab["uploadID"], tab["uploadRA"], tab["uploadDEC"],
                path)       

    sortTable = Table([objectlist, NUVlist, FUVlist], names=("object",
        "NUV_Tile", "FUV_Tile"))
    sortTable.write("Sort Table")

