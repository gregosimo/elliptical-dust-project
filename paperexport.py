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
import band_conversions as conv

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
FULL_JARRETT_TABLE = os.path.join(JARRETTBASE, "jarrett.tbl")

# If we want to change the type of file which is exported, just change this
# extension!
EXT = "pdf"

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

# This table contains the full ATLAS3D + RSA Sample.
fulltable = []

# I made this a csv because the astropy ipac routine doesn't believe in having
# periods in ipac column names.
try:
    jarrett_table = Table.read(FULL_JARRETT_TABLE, format="ascii.csv")
except IOError:
    jarrett_table=[]


def generate_fulltable(rampazzo=rampazzo_table, atlas3d=atlas3d_table):
    '''Function to take care of generating the full sample table.

    Since generating the full table is an expensive operation, this function is
    called to ensure it is only called once per load, and only when it is
    really needed.'''
    global fulltable
    if fulltable is []:
        # With a later version of astropy, I could just use the unique 
        # function...
        fulltable = vstack([rampazzo_table, atlas3d_table])
        groupedtable = fulltable.group_by('objstr_01')
        fulltable = Table(rows=groupedtable[0])
        for ix in xrange(1, len(groupedtable.groups)):
            fulltable.add_row(groupedtable.groups[ix][0])
    else:
        pass

def build_filepath(basepath, filename, extension=EXT):
    '''Builds a full file path of a file.

    This function takes a basepath, joins it to a filename, and intelligently
    adds a filename extension to the end. This is so that extensions can be
    specified independently of the filename.
    '''
    fullfilename = os.extsep.join((filename, extension))
    fullpath = os.path.join(basepath, fullfilename)
    return fullpath

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
    caption = r"""Properties of galaxies in the \ATLAS{} sample.
    \label{tab:atlas3dsample}"""
    columns = ["objstr_01", "type", "D", "Age_SSP", "[Z/H]_SSP"]
    names = ["Galaxy", "Morph.", "Distance", "SSP Age", "SSP [Z/H]"]
    table1 = table[columns]
    table1.write(dest, format="ascii.aastex", names=names,
                 latexdict={"caption": caption})

def create_param_table(table=fulltable, dest=os.path.join(TABLEPATH,
                                                          "params.tex")):
    generate_fulltable()
    caption = r"""Aperture photometry parameters for galaxies in the \ATLAS{}
    and Rampazzo samples
    \label{tab:params}"""
    columns = ["objstr_01", "w1rsemi", "w1ba", "w1pa", "cat", "NUV_Tile",
               "FUV_Tile"]
    names = ["Galaxy", "Semimajor Axis", "Axis Ratio", "Position Angle",
             "\WISE{} Survey", "NUV Tile", "FUV Tile"]

def create_magnitude_table(table=fulltable, dest=os.path.join(TABLEPATH,
                                                              "mags.tex")):
    generate_fulltable()
    caption = r"""Magnitudes of galaxies in the \ATLAS{} and Rampazzo samples.
    \label{tab:magtable}"""
    tablefoot = r"""\tablecomments{Magnitudes are reported in the AB system. 
    They have not been corrected for extinction.}"""
    columns = ["objstr_01", "FUVapmag", "FUVaperr", "NUVapmag", "NUVaperr", 
               "w1apmag", "w1aperr", "w2apmag", "w2aperr", "w3apmag",
               "w3aperr", "w4apmag", "w4aperr"]
    names = ["Galaxy", "FUV", r"\(\sigma_{FUV}\)", "NUV", r"\(\sigma_{NUV}\)", 
             "W1", r"\(\sigma_{W1}\)", "W2", r"\(\sigma_{W2}\)", "W3",
             r"\(\sigma_{W3}\)", "W4", r"\(\sigma_{W4}\)"]
    formats = {"Galaxy": format_reflect,
               "FUV": format_mag, r"\(\sigma_{FUV}\)": format_mag_err, 
               "NUV": format_mag, r"\(\sigma_{NUV}\)": format_mag_err, 
               "W1": format_mag, r"\(\sigma_{W1}\)": format_mag_err, 
               "W2": format_mag, r"\(\sigma_{W2}\)": format_mag_err, 
               "W3": format_mag, r"\(\sigma_{W3}\)": format_mag_err, 
               "W4": format_mag, r"\(\sigma_{W4}\)": format_mag_err}
    export = table[columns]
    export.write(dest, format="ascii.aastex", names=names, formats=formats,
                 latexdict={"caption": caption, "tablefoot": tablefoot,
                 "tabletype": "deluxetable"})

def create_W2_W3_histogram(table=rampazzo_table, 
                           dest=build_filepath(FIGUREPATH, "w2w3hist")):
    '''Creates the histogram which plots all of the objects in W2-W3.'''
    xlabel = "W2-W3"
    rp.color_histogram_by_class(rampazzo_table["w2unextmag"],
                                rampazzo_table["w3unextmag"],
                                rampazzo_table["MIR_class"], "W2-W3", 
                                "", 80, colrange=(-1.5, 2.5))
    plt.savefig(dest)
    plt.close()

def create_W1W2_W2W3_MIR_plot(table=rampazzo_table,
                              dest=build_filepath(FIGUREPATH, "w1w2w2w3_mir")):
    '''Plots W1-W2 vs W2-W3'''
    ylabel = "W1-W2"
    xlabel = "W2-W3"

    w1 = rampazzo_table["w1unextmag"]
    w2 = rampazzo_table["w2unextmag"]
    w3 = rampazzo_table["w3unextmag"]
    w1err = rampazzo_table["w1unexterr"]
    w2err = rampazzo_table["w2unexterr"]
    w3err = rampazzo_table["w3unexterr"]
    MIR = rampazzo_table["MIR_class"]

    w1w2, w1w2err = phot.calc_statistical_difference(w1, w2, w1err, w2err)
    w2w3, w2w3err = phot.calc_statistical_difference(w2, w3, w2err, w3err)

    rp.MIRplot(w2w3, w1w2, MIR, w1w2err, w2w3err, xrange(5), xlabel,
               ylabel, "", loc="upper left")
    plt.savefig(dest)
    plt.close()

def create_W2W3_W3W4_MIR_plot(table=rampazzo_table,
                              dest=build_filepath(FIGUREPATH, "w2w3w3w4_mir")):
    '''Plots W2-W3 vs W3-W4'''
    ylabel = "W2-W3"
    xlabel = "W3-W4"

    w2 = rampazzo_table["w2unextmag"]
    w3 = rampazzo_table["w3unextmag"]
    w4 = rampazzo_table["w4unextmag"]
    w2err = rampazzo_table["w2unexterr"]
    w3err = rampazzo_table["w3unexterr"]
    w4err = rampazzo_table["w4unexterr"]
    MIR = rampazzo_table["MIR_class"]

    w2w3, w2w3err = phot.calc_statistical_difference(w2, w3, w2err, w3err)
    w3w4, w3w4err = phot.calc_statistical_difference(w3, w4, w3err, w4err)

    rp.MIRplot(w3w4, w2w3, MIR, w2w3err, w3w4err, xrange(5), xlabel,
               ylabel, "", loc="upper left")
    plt.savefig(dest)
    plt.close()


def create_PAH113_17_PAH77_113_plot(table=rampazzo_table,
       dest=os.path.join(FIGUREPATH, "shortpahs.pdf")):
    pahtable = read_Rampazzo_TableA1()
    fulltable = phot.join_by_galaxy_name(rampazzo_table, pahtable,
                                    names=("objstr_01", "Galaxy"))

    # Restrict to only Class 2 and Class 3 objects
    fullgroups = fulltable.group_by("MIR_class")
    classtable = fullgroups.groups[0:2]

    pah77 = classtable["7.7 um"]
    pah77_err = classtable["7.7 um err"]
    pah113 = classtable["11.3 um"]
    pah113_err = classtable["11.3 um err"]
    pah17 = classtable["17 um"]
    pah17_err = classtable["17 um err"]

    xratio, xratioerr = phot.calc_statistical_quotient(pah113, pah17, 
                                                       pah113_err, pah17_err)
    yratio, yratioerr = phot.calc_statistical_quotient(pah77, pah113,
                                                       pah77_err, pah113_err)

    rp.MIRplot(xratio, yratio, classtable["MIR_class"], yerr=yratioerr,
               xerr=xratioerr, classes=(2,3), xlabel="11.3 um/17 um", 
               ylabel="7.7 um/11.3 um")

    plt.xlim([0, 6])
    plt.ylim([0, 6])
    plt.savefig(dest)
    plt.close()

def create_NUV_J_PAH77_113_plot(table=rampazzo_table, 
                                dest=os.path.join(FIGUREPATH, "uvpahs.pdf")):
    pahtable = read_Rampazzo_TableA1()
    fulltable = phot.join_by_galaxy_name(rampazzo_table, pahtable,
                                    names=("objstr_01", "Galaxy"))

    # Restrict to only Class 2 and Class 3 objects
    fullgroups = fulltable.group_by("MIR_class")
    classtable = fullgroups.groups[0:2]

    nuv = classtable["NUVunextmag"]
    nuv_err = classtable["NUVunexterr"]
    j = classtable["j_m_k20fe"]
    j_err = classtable["j_msig_k20fe"]
    pah77 = classtable["7.7 um"]
    pah77_err = classtable["7.7 um err"]
    pah113 = classtable["11.3 um"]
    pah113_err = classtable["11.3 um err"]

    xcolor, xcolorerr = phot.calc_statistical_difference(nuv, j, nuv_err, j_err)
    yratio, yratioerr = phot.calc_statistical_quotient(pah77, pah113,
                                                       pah77_err, pah113_err)

    rp.MIRplot(xcolor, yratio, classtable["MIR_class"], yerr=yratioerr, 
               xerr=xcolorerr, classes=(2,3), xlabel="NUV-J",
               ylabel="7.7 um / 11.3 um")
    plt.xlabel("NUV-J")
    plt.ylabel("7.7 um/11.3 um")
    plt.title("Correlation for PAH-detected galaxies")
    plt.savefig(dest)
    plt.close()

    return classtable

def create_jarrett_comparison_plot(table=jarrett_table,
                                   dest=build_filepath(FIGUREPATH, "jarrett",
                                                       EXT)):
    #smallindices = [0, 2, 3, 12, 13]
    smallindices = range(len(table))
    orig_fluxes = read_Jarrett_Table2()[smallindices]
    my_mags = table[smallindices]

    orig_w1 = conv.Jansky2Vegamag("W1", orig_fluxes["W1"])
    orig_w1_err = conv.Jansky_err_to_mag_err("W1", orig_fluxes["W1"], 
                                  orig_fluxes["W1_err"])
    myw1 = my_mags["w1apmag"]
    myw1err = my_mags["w1aperr"]
    orig_w2 = conv.Jansky2Vegamag("W2", orig_fluxes["W2"])
    orig_w2_err = conv.Jansky_err_to_mag_err("W2", orig_fluxes["W2"], 
                                  orig_fluxes["W2_err"])
    myw2 = my_mags["w2apmag"]
    myw2err = my_mags["w2aperr"]
    orig_w3 = conv.Jansky2Vegamag("W3", orig_fluxes["W3"])
    orig_w3_err = conv.Jansky_err_to_mag_err("W3", orig_fluxes["W3"], 
                                  orig_fluxes["W3_err"])
    myw3 = my_mags["w3apmag"]
    myw3err = my_mags["w3aperr"]
    orig_w4 = conv.Jansky2Vegamag("W4", orig_fluxes["W4"])
    orig_w4_err = conv.Jansky_err_to_mag_err("W4", orig_fluxes["W4"], 
                                  orig_fluxes["W4_err"])
    myw4 = my_mags["w4apmag"]
    myw4err = my_mags["w4aperr"]

    w1diff, w1differr = phot.calc_statistical_difference(myw1, orig_w1,
        myw1err, orig_w1_err)
    w2diff, w2differr = phot.calc_statistical_difference(myw2, orig_w2,
        myw2err, orig_w2_err)
    w3diff, w3differr = phot.calc_statistical_difference(myw3, orig_w3,
        myw3err, orig_w3_err)
    w4diff, w4differr = phot.calc_statistical_difference(myw4, orig_w4,
        myw4err, orig_w4_err)

    phot.doubleDifferencePlot(myw1, orig_w1, myw2, orig_w2, myw1err,
                              orig_w1_err, myw2err, orig_w2_err,
                              "W1 Difference", "W2 Difference", "")
    plt.savefig(dest)
    plt.close()

def move_rampazzo():
    # First read in Table 1
    # Then Table 2
    # Concatenate them.
    # filter them for galaxies we have data for.
    pass

def read_Jarrett_Table2(tablepath=os.path.join(BASEPATH,
        "WISE_Isophotal-aperture_Photometry.txt")):
    jarrett_table2 = Table.read(tablepath, format="ascii.csv")
    return jarrett_table2

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

def read_Rampazzo_Table5(tablepath=os.path.join(BASEPATH,
                                                "Rampazzo_Table5.csv")):
    rampazzotable = Table.read(tablepath, format="ascii.csv", guess=False,
                               data_start=1,
                               names=("Galaxy", "RSA_morph_type", "MIR_class"))
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

def read_Rampazzo_TableA2(tablepath=os.path.join(BASEPATH,
                                                 "Rampazzo_TableA2.csv")):
    rampazzotable = Table.read(tablepath, format="ascii.csv", guess=False, 
                               data_start=2, delimiter=":", comment="\s*#",
                               header_start=1,
                               names=("Galaxy", "H2 S(7)", "H2 S(7) err", 
                                      "H2 S(6)", "H2 S(6) err", "H2 S(5)", 
                                      "H2 S(5) err", "[Ar II]", "[Ar II] err",
                                      "H Pfa", "H Pfa err", "H2 S(4)", 
                                      "H2 S(4) err", "[Ar III] 9um", 
                                      "[Ar III] 9 um err", "H2 S(3)", 
                                      "H2 S(3) err", "[S IV]", "[S IV] err", 
                                      "H2 S(2)", "H2 S(2) err", "[Ne II]", 
                                      "[Ne II] err", "[Ne V] 14 um", 
                                      "[Ne V] 14 um err", "[Ne III] 16 um", 
                                      "[Ne III] 16 um err", "H2 S(1)", 
                                      "H2 S(1) err", "[Fe II] 18 um", 
                                      "[Fe II] 18 um err", "[S III] 19 um", 
                                      "[S III] 19 um err", "[Ar III] 22 um", 
                                      "[Ar III] 22 um err", "[Ne V] 24 um", 
                                      "[Ne V] 24 um err", "[O IV]", 
                                      "[O IV] err", "[Fe II] 26 um", 
                                      "[Fe II] 26 um err", "H2 S(0)", 
                                      "H2 S(0) err", "[S III] 33 um", 
                                      "[S III] 33 um err", "[Si II]", 
                                      "[Si II] err", "[Fe II] 35 um", 
                                      "[Fe II] 35 um err", "[Ne III] 36 um",
                                      "[Ne III] 36 um err"))
    return rampazzotable

def read_Cappellari11_Table_3(tablepath=os.path.join(ATLAS3DBASE,
                                                     "Cappellari11_Table_3.txt")):
    '''Reads in the third table from Cappellari 2011.'''
    names = ["Galaxy", "RA", "DEC", "SBF", "NED-D", "Virgo", "VHel", "D",
             "M_K", "A_B", "T-type", "log(Re)"]
    atlas3dsample = Table.read(tablepath, format="ascii.no_header", data_start=3,
                               names=names, guess=False)
    return atlas3dsample

def read_Diamond_Stanic_Table1(
    tablepath=os.path.join(BASEPATH, "Diamond_Stanic_Table1.txt")):
    '''Reads the first table from Diamond-Stanic 2010.'''
    names = ["Name", "6.2 um", "6.2 um err", "7.7 um", "7.7 um err", "8.6 um",
             "8.6 um err", "11.3 um", "11.3 um err", "12.7 um", "12.7 um err",
             "[Ne II]", "[Ne II] err", "H2 S(3)", "H2 S(3) err"]
    dstable = Table.read

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

def format_arcseconds(arc):
    '''Format a number so that it is displayed with arcsecond units'''
    arcstr = "{0:.1f}''".format(arc)
    return arcstr

def format_degrees(deg):
    '''Format a number so that it is displayed with degree units'''
    degstr = "{0:.1f}\\\\(^\\\\circ\\\\)".format(deg)
    return degstr

def format_axis_ratio(ax):
    '''Format a number so that it's displayed as a proper axis ratio.'''
    axstr = "{0:.2f}".format(ax)
    return axstr

def format_reflect(inp):
    '''Returns the same string.

    This is made primarily to make functions for the formats argument in
    Table.write() more transparent.'''
    return inp


if __name__ == "__main__":

    # Write Rampazzo parameters and magnitudes and MIR classes to paper
    #   directory.
    # Write ATLAS3D parameters and magnitudes to paper directory.
    # Write Jarrett fluxes (paper and calculated) to paper directory.
    # Move FSPS output to paper directory.
    pass
