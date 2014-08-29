

from astropy.table import Table

def create_upload_file(ids, ras, decs, output):
    '''Creates a file that can be uploaded to MAST.

    This file will contain Object names, RAs, and Decs as a comma-separated
    list. The file will be saved to output, where it can manually be uploaded to
    the GALEX MAST database.
    '''
    relevantTable = Table([ids, ras, decs], names=("ID", "RA", "DEC"))
    relevantTable.write(output, format="ascii.csv")

def select_best_surveys(inputfile, output_dir, surveybase):
    '''Takes a file from the GALEX catalog and optimizes the exposures to use.

    There are numerous surveys which the GALEX mission has collected data from.
    They are categorized as AIS, MIS, DIS, NGS, and GII. Each has a different
    level of coverage and depth. AIS has shallow photometry of the whole sky,
    while GII often are PI projects which have very deep images of only a few
    areas. This function maximizes the exposure time for the images that will be
    downloaded.
    '''
