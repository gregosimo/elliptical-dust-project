import os
import os.path

import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table

import photometry as phot
import queries as query
import galex_proc as galex

def build_GALEX_atlas(BASEDIR):
    '''Guides the user through building the pipeline for Galex images.

    This function will handle getting the images and setting up the directory
    structure for aperture photometry in the given BASEDIR.'''
    MAST_LIST = os.path.join(BASEDIR, "MAST_LIST.csv")

    marino_objects = Table.read(os.path.join(BASEDIR, "..",
        "Marino_objects.txt"), format="ascii.csv")
    
    marino_objects_info = query.ned_resolve(marino_objects["Ident."])
    galex.create_upload_file(marino_objects["Ident."],
            marino_objects_info["RA"], marino_objects_info["DEC"], MAST_LIST)
    marino_objects_info.write(MAST_LIST, format="ascii.csv")
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


