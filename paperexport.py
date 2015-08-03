'''This module sets up the tables and figures necessary for the paper. 

Big tables which contain a large swath of data should be located at the
FULL_*_TABLE variables. These tables will be used to generate all of the
necessary plots and tables in the paper. When in doubt, include it.

'''
import os

import matplotlib
matplotlib.use("PDF")
import matplotlib.pyplot as plt
from astropy.table import Table, vstack
import numpy as np

import photometry as phot
import queries
import rampazzo_plots as rp
import fsps

BASEPATH = "/home/regulus/simonian/year1/wise"
FSPSPATH = "/home/regulus/simonian/year1/fsps"

ATLAS3DBASE = os.path.join(BASEPATH, "ATLAS3D_DB")
RAMPAZZOBASE = os.path.join(BASEPATH, "Rampazzo_DB")
JARRETTBASE = os.path.join(BASEPATH, "Jarrett_DB")

PAPERPATH = "/home/regulus/simonian/papers/wise14"
TABLEPATH = os.path.join(PAPERPATH, "tables")
FIGUREPATH = os.path.join(PAPERPATH, "fig")

FULL_ATLAS3D_TABLE = os.path.join(ATLAS3DBASE, "atlas3d.tbl")
FULL_RAMPAZZO_TABLE = os.path.join(RAMPAZZOBASE, "rampazzo.tbl")

# I'd like for the tables to just be loaded without worrying about writing to
# and from disk.
try:
    rampazzo_table = Table.read(FULL_RAMPAZZO_TABLE, format="ascii.ipac")
except IOError:
    rampazzo_table=[]

# I made this a csv because the astropy ipac routine doesn't believe in having
# slashes in ipac column names
try:
    atlas3d_table = Table.read(FULL_ATLAS3D_TABLE, format="ascii.csv")
except IOError:
    atlas3d_table=[]

# With a later version of astropy, I could just use the unique function...
fulltable = vstack([rampazzo_table, atlas3d_table])
groupedtable = fulltable.group_by('objstr_01')
fulltable = Table(rows=groupedtable[0])
for ix in xrange(1, len(groupedtable.groups)):
    fulltable.add_row(groupedtable.groups[ix][0])

#######################################################################
# Create Tables #
#######################################################################
def create_Rampazzo_sample_table(table=rampazzo_table,
                                 dest=os.path.join(TABLEPATH,
                                                   "rampazzotbl.tex")):
    '''Table corresponding to information about Rampazzo galaxies.'''
    caption = r"""Properties of the galaxies in the Rampazzo sample. Columns:
    (1) galaxy name; (2) Morphological type; (3) Distance; (4) MIR Class.
    References can be found as \citet{Rampazzo13}.
    \label{tab:rampazzosample}"""
    columns = ["objstr_01", "RSA_morph_type", "D", "MIR_class"]
    names = ["Galaxy", "Morph.", "Distance", "MIR Class"]
    units = {"Distance", "Mpc"}
    table1 = table[columns]
#    table1.write(dest, format="ascii.latex", names=names)
    table1.write(dest, format="ascii.aastex", names=names,
                 latexdict={"caption": caption, "units": units})

def create_ATLAS3D_sample_table(table=atlas3d_table,
                                dest=os.path.join(TABLEPATH,
                                                  "atlas3dtbl.tex")):
    caption = r"""Properties of galaxies in the \ATLAS sample.
    \label{tab:atlas3dsample}"""
    columns = ["objstr_01", "type", "D", "Age_SSP", "[Z/H]_SSP"]
    names = ["Galaxy", "Morph.", "Distance", "SSP Age", "SSP [Z/H]"]
    table1 = table[columns]
    table1.write(dest, format="ascii.aastex", names=names,
                 latexdict={"caption": caption})

def create_magnitude_table(table=fulltable, dest=os.path.join(TABLEPATH,
                                                              "mags.tex")):
    caption = r"""Magnitudes of galaxies in the \ATLAS and Rampazzo samples.
    \label{tab:magtable}"""
    tablefoot = r"""\tablecomments{Magnitudes are reported in the AB system. 
    They have not been corrected for extinction.}"""
    columns = ["objstr_01", "FUVapmag", "FUVaperr", "NUVapmag", "NUVaperr", 
               "w1apmag", "w1aperr", "w2apmag", "w2aperr", "w3apmag",
               "w3aperr", "w4apmag", "w4aperr"]
    names = ["Galaxy", "FUV", r"\(\sigma_{FUV}\)", "NUV", r"\(\sigma_{NUV}\)", 
             "W1", r"\(\sigma_{W1}\)", "W2", r"\(\sigma_{W2}\)", "W3",
             r"\(\sigma_{W3}\)", "W4", r"\(\sigma_{W4}\)"]
    formats = {"Galaxy": "%s",
               "FUV": format_mag, r"\(\sigma_{FUV}\)": format_mag_err, 
               "NUV": format_mag, r"\(\sigma_{NUV}\)": format_mag_err, 
               "W1": format_mag, r"\(\sigma_{W1}\)": format_mag_err, 
               "W2": format_mag, r"\(\sigma_{W2}\)": format_mag_err, 
               "W3": format_mag, r"\(\sigma_{W3}\)": format_mag_err, 
               "W4": format_mag, r"\(\sigma_{W4}\)": format_mag_err}
    export = table[columns]
    export.write(dest, format="ascii.aastex", names=names, formats=formats,
                 latexdict={"caption": caption, "tablefoot": tablefoot,
                 "tabletype": "deluxetable*"})

def create_NUV_J_histogram(table=rampazzo_table, dest=os.path.join(FIGUREPATH,
                               "uvsed.pdf")):
    '''Creates the histogram which plots Class 2/3 objects in NUV-J.'''
    title = "Class Histogram"
    xlabel = "NUV-J"
    rp.color_histogram_by_class(rampazzo_table["NUVunextmag"],
                                rampazzo_table["j_m_k20fe"],
                                rampazzo_table["MIR_class"], "NUV-J", 
                                "Class Histogram", 26, classes=[2,3],
                                colrange=(2.5, 9))
    plt.savefig(dest)
    plt.close()


def move_rampazzo():
    # First read in Table 1
    # Then Table 2
    # Concatenate them.
    # filter them for galaxies we have data for.
    pass


def read_Rampazzo_Table1(tablepath=os.path.join(BASEPATH,
                                                "Rampazzo_Table1.csv")):
    rampazzotable = Table.read(tablepath, format="ascii.csv", guess=False,
                               data_start=2, delimiter=":", comment="\s*#",
                               names=("Galaxy", "RSA_morph_type", "T", "Terr", 
                                      "D", "T88_Group", "MK", "re", "sigc"),
                               fill_values=[("", "0"), ('---', "0")])
    # First flag the entries which are starred.
    rampazzotable["H0D"] = np.char.endswith(rampazzotable["D"], '*')
    rampazzotable["H0MK"] = np.char.endswith(rampazzotable["MK"], '*')
    # Now remove the stars from the columns
    rampazzotable["D"] = np.char.rstrip(rampazzotable["D"], '*')
    rampazzotable["MK"] = np.char.rstrip(rampazzotable["MK"], '*')
    # The default fill value for string arrays is 'N/A', which cannot be easily
    # converted to and int. Therefore, I'm changing the fill value to 0.
    rampazzotable["D"].set_fill_value(0)
    rampazzotable["MK"].set_fill_value(0.0)
    # This is a workaround for the fact that I don't think tables copy over
    # very well. See the Astropy mailing list for various ways of doing this.
    tempcol = np.ma.asanyarray(rampazzotable["D"], np.int)
    del(rampazzotable["D"])
    rampazzotable["D"] = tempcol
    tempcol = np.ma.asanyarray(rampazzotable["MK"], np.float)
    del(rampazzotable["MK"])
    rampazzotable["MK"] = tempcol
    
    return rampazzotable

def read_Rampazzo_Table2(tablepath=os.path.join(BASEPATH,
                                                "Rampazzo_Table2.csv")):
    rampazzotable = Table.read(tablepath, format="ascii.csv", guess=False,
                               data_start=2, delimiter=":", comment="\s*#",
                               names=("Galaxy", "RSA_morph_type", "T", "Terr", 
                                      "D", "T88_Group", "MK", "re", "sigc"),
                               fill_values=[("", "0"), ('---', "0")])
    # First flag the entries which are starred.
    rampazzotable["H0D"] = np.char.endswith(rampazzotable["D"], '*')
    rampazzotable["H0MK"] = np.char.endswith(rampazzotable["MK"], '*')
    # Now remove the stars from the columns
    rampazzotable["D"] = np.char.rstrip(rampazzotable["D"], '*')
    rampazzotable["MK"] = np.char.rstrip(rampazzotable["MK"], '*')
    # The default fill value for string arrays is 'N/A', which cannot be easily
    # converted to and int. Therefore, I'm changing the fill value to 0.
    rampazzotable["D"].set_fill_value(0)
    rampazzotable["MK"].set_fill_value(0.0)
    # This is a workaround for the fact that I don't think tables copy over
    # very well. See the Astropy mailing list for various ways of doing this.
    tempcol = np.ma.asanyarray(rampazzotable["D"], np.int)
    del(rampazzotable["D"])
    rampazzotable["D"] = tempcol
    tempcol = np.ma.asanyarray(rampazzotable["MK"], np.float)
    del(rampazzotable["MK"])
    rampazzotable["MK"] = tempcol
    
    return rampazzotable

def read_Rampazzo_TableA1(tablepath=os.path.join(BASEPATH,
                                                 "Rampazzo_TableA1.csv")):
    rampazzotable = Table.read(tablepath, format="ascii.csv", guess=False,
                               data_start=2, delimiter=":", comment="\s*#",
                               names=("Galaxy", "6.22 um", "6.22 um err", 
                                      "7.7 um", "7.7 um err", "8.6 um", 
                                      "8.6 um err", "11.3 um", "11.3 um err", 
                                      "12.7 um", "12.7 um err", "17 um", 
                                      "17 um err"),
                               fill_values=[("", "0"), ("-", "0")])
    # If there are no detections, set the values to 0 when filled.
    rampazzotable["6.22 um"].fill_value = 0.0
    rampazzotable["7.7 um"].fill_value = 0.0
    rampazzotable["8.6 um"].fill_value = 0.0
    rampazzotable["11.3 um"].fill_value = 0.0
    rampazzotable["12.7 um"].fill_value = 0.0
    rampazzotable["17 um"].fill_value = 0.0
    # For non-detections, I'm setting the upper limits to be the uncertainty on
    # the weakest detection.
    rampazzotable["6.22 um err"].fill_value = 4.4
    rampazzotable["7.7 um err"].fill_value = 11.9
    rampazzotable["8.6 um err"].fill_value = 1.1
    rampazzotable["11.3 um err"].fill_value = 3.0
    rampazzotable["12.7 um err"].fill_value = 1.5
    rampazzotable["17 um err"].fill_value = 1.1
    return rampazzotable

def read_Cappellari11_Table_3(tablepath=os.path.join(ATLAS3DBASE,
                                                     "Cappellari11_Table_3.txt")):
    '''Reads in the third table from Cappellari 2011.'''
    names = ["Galaxy", "RA", "DEC", "SBF", "NED-D", "Virgo", "VHel", "D",
             "M_K", "A_B", "T-type", "log(Re)"]
    atlas3dsample = Table.read(tablepath, format="ascii.no_header", data_start=3,
                               names=names, guess=False)
    return atlas3dsample

def rampazzo_sample_list(table1=os.path.join(BASEPATH, "Rampazzo_Table1.csv"), 
                         table2=os.path.join(BASEPATH, "Rampazzo_Table2.csv"),
                         destination=os.path.join(PAPERPATH, 
                                                  "Rampazzo_sample.csv")):
    '''Exports the current sample list to the destination.

    This function builds up a sample list from the Rampazzo Tables and exports
    them to the destination (by default the folder where the paper will be).
    '''
    table_1 = read_Rampazzo_Table1(table1)
    table_2 = read_Rampazzo_Table2(table2)
    fulltable = vstack([table_1, table_2])

    sampletable = phot.filterTableforCompleteBands(RAMPAZZOBASE, fulltable)

    sampletable.write(destination, format="ascii.csv", delimiter=":")

def format_mag(mag):
    '''Format magnitudes so that they can be displayed on a table.

    This essentially breaks off the magnitude at the second decimal place. It
    is meant to be used with the format keyword in Table.write().
    '''
    magstr = "{0:.2f}".format(mag)
    return magstr

def format_mag_err(err):
    '''Format magnitude errors so that they can be displayed on a table.

    This essentially breaks off the error at the third decimal place. It is
    meant to be used with the format keyword in Table.write().
    '''
    magerrstr = "{0:.3f}".format(err)
    return magerrstr

if __name__ == "__main__":

    # Write Rampazzo parameters and magnitudes and MIR classes to paper
    #   directory.
    # Write ATLAS3D parameters and magnitudes to paper directory.
    # Write Jarrett fluxes (paper and calculated) to paper directory.
    # Move FSPS output to paper directory.
    pass
