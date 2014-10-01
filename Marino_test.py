import os
import os.path

import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table

import photometry as phot
import queries as query
import galex_proc as galex

def prompt_with_default_value(default_value):
    '''Returns a default value if the user doesn't choose a newline.

    This function will prompt a user for a value and display a default value. If
    the user presses enter without typing in a value, this function will return
    that default value. Otherwise, it will return what the user entered.
    '''
    value = raw_input("[{0}] ".format(default_value))
    if not value:
        value = default_value 
    return value

def build_GALEX_atlas(BASEDIR):
    '''Guides the user through building the pipeline for Galex images.

    This function will handle getting the images and setting up the directory
    structure for aperture photometry in the given BASEDIR.'''
    HYPERLEDA_LIST = os.path.join(BASEDIR, "hyperleda.txt")
    MAST_LIST = os.path.join(BASEDIR, "MAST_LIST.csv")

    marino_objects = Table.read(os.path.join(BASEDIR, "..",
        "Marino_objects.txt"), format="ascii.csv")
    marino_objects.write(HYPERLEDA_LIST, format="ascii.tab",
        include_names=("Ident.",))

    terminal_string = '''Now creating a file to be submitted to HYPERLEDA in
    order to retrieve the D25 and axis ratio. The file is located at {0}. Please
    submit this file to http://atlas.obs-hp.fr/hyperleda/leda/upload.html. 
    
    The necessary columns are al2000, de2000, logd25, logr25, and pa.  Once the 
    file is downloaded, type in the filename of the output with respect to the 
    BASEDIR, or simply press enter to accept the displayed
    filename.'''.format(HYPERLEDA_LIST)
    print terminal_string
    galexellipse = prompt_with_default_value("hyperleda_output.txt")
    hyperledapars = Table.read(os.path.join(BASEDIR, galexellipse),
        format="ascii.basic", delimiter="|", comment="!")
    
    galex.create_upload_file(hyperledapars["name"], hyperledapars["al2000"], 
            hyperledapars["de2000"], MAST_LIST)
    terminal_string = '''
    The list of objects has now been written to {0}. The next step is go
    navigate to the MAST interface at http://galex.stsci.edu/GalexView/. Upload
    the list of objects and perform a Tiles search. Please enter the filename of
    of the downloaded csv file:'''.format(MAST_LIST)
    print terminal_string

    lookup_table = "sorttable.csv"
    lookup_table_path = os.path.join(BASEDIR, lookup_table)
    galex_list = raw_input("[galex_complete_tiles.csv] ")
    if not galex_list:
        galex_list = "galex_complete_tiles.csv"
    galex_list_location = os.path.join(BASEDIR, galex_list)
            
    galex.select_best_surveys(galex_list_location, BASEDIR, lookup_table)
    terminal_string = '''
    The best observations for each object will now be separated into many
    different .csv files under {0}. Please upload these files into MAST and
    download the images as one or more tarfiles that match the pattern
    "Galex*.tar". The table which holds which tiles correspond to which objects
    will be saved at {1}. Once this is done, press enter.'''.format(BASEDIR, 
            lookup_table_path)
    print terminal_string
    raw_input()
    galex.process_GALEX_tarfile(BASEDIR, BASEDIR, lookup_table_path)
    print "Finished Processing!"


if __name__ == "__main__":
    MARINO_DIR = "/home/regulus/simonian/year1/wise/Marino_DB"

    # Set up the database from scratch
    build_galaxy_atlas(MARINO_DIR)

    hyperleda = Table.read("../wise/Marino_DB/hyperleda_output.txt",
        format="ascii.basic", delimiter="|", comment="!")
    marpar = phot.Hyperleda_Table_to_WISE_Table(hyperleda)
    phot.build_pipeline(MARINO_DIR, marpar, runbands=phot.UVBANDS)

    # Now generate the photometric magnitudes for the objects.
    marinoapmags = phot.aperturePhotometryTable(MARINO_DIR, marpar["objstr_01"],
            runbands=phot.UVBANDS)

    marinomags = Table.read(os.path.join(MARINO_DIR, "Marino_Table3.csv"),
            format="ascii.csv")

    plt.figure()
    phot.createDifferencePlot(marinomags["NUV D25"], marinoapmags["NUVapmag"],
            marinomags["NUV D25 err"], marinoapmags["NUVaperr"], "Marino Mag",
            "Aperture Mag - Marino Mag", 
            "GALEX Photometry Magnitude Difference")
