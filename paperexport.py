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
from astropy.modeling import models, fitting
from astropy.io.ascii import masked
import numpy as np
import numpy.core.defchararray as npstr

import photometry as phot
import queries
import rampazzo_plots as rp
import fsps
import band_conversions as conv
import atlas3d
import statop as stat
import parsec

BASEPATH = "/home/regulus/simonian/year1/wise"
FSPSPATH = "/home/regulus/simonian/year1/fsps"
PARSECPATH = "/home/regulus/simonian/year1/parsec"

ATLAS3DBASE = os.path.join(BASEPATH, "ATLAS3D_DB")
RAMPAZZOBASE = os.path.join(BASEPATH, "Rampazzo_DB")
JARRETTBASE = os.path.join(BASEPATH, "Jarrett_DB")
GDPBASE = os.path.join(BASEPATH, "GdP_sample_test")

PAPERPATH = "/home/regulus/simonian/papers/wise14"
TABLEPATH = os.path.join(PAPERPATH, "tables")
FIGUREPATH = os.path.join(PAPERPATH, "fig")

FULL_ATLAS3D_TABLE = os.path.join(ATLAS3DBASE, "atlas3d.tbl")
FULL_RAMPAZZO_TABLE = os.path.join(RAMPAZZOBASE, "rampazzo.tbl")
FULL_JARRETT_TABLE = os.path.join(JARRETTBASE, "jarrett.tbl")
FULL_GIL_DE_PAZ_TABLE = os.path.join(GDPBASE, "gil_de_paz.tbl")

# If we want to change the type of file which is exported, just change this
# extension!
EXT = "pdf"

# This is the string which is displayed in LaTeX for a missing value.
LATEX_TABLE_MASKSTRING = r"--"

# I'd like for the tables to just be loaded without worrying about writing to
# and from disk.
try:
    rampazzo_table = Table.read(FULL_RAMPAZZO_TABLE, format="ascii.csv",
                                guess=False)
except IOError:
    rampazzo_table=[]

# I made this a csv because the astropy ipac routine doesn't believe in having
# slashes in ipac column names
try:
    atlas3d_table = Table.read(FULL_ATLAS3D_TABLE, format="ascii.csv",
                               guess=False)
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

# I made this a csv because the astropy ipac routine doesn't believe in having
# periods in ipac column names.
try:
    gdp_table = Table.read(FULL_GIL_DE_PAZ_TABLE, format="ascii.csv")
except IOError:
    gdp_table=[]



def generate_fulltable(rampazzo=rampazzo_table, atlas3d=atlas3d_table):
    '''Function to take care of generating the full sample table.

    Since generating the full table is an expensive operation, this function is
    called to ensure it is only called once per load, and only when it is
    really needed.'''
    rampazzosample = rampazzo.copy()
    rampazzosample["sample"] = "Rampazzo"
    rampazzosample.rename_column("RSA_morph_type", "type")
    atlas3dsample = atlas3d.copy()
    atlas3dsample["sample"] = "ATLAS3D"
    global fulltable
    if not fulltable:
        # With a later version of astropy, I could just use the unique 
        # function...
        fulltable = vstack([rampazzosample, atlas3dsample])
        groupedtable = fulltable.group_by('objstr_01')
        # This will be bad if the first galaxy is in both datasets.
        fulltable = Table(rows=groupedtable.groups[0][0])
        for ix in xrange(1, len(groupedtable.groups)):
            tablegroup = groupedtable.groups[ix]
            if len(tablegroup) == 1:
                addrow = tablegroup[0]
            elif len(tablegroup) == 2:
                # We want to add the row from Rampazzo because we want to
                # preserve the MIR class.
                addrow = tablegroup[np.where(tablegroup["sample"] ==
                                             "Rampazzo")][0]
                addrow["sample"] = "Both"
            elif len(tablegroup) > 2:
                raise ValueError("Some table has more than three entries.")
            fulltable.add_row(addrow)
            if addrow["sample"] == "ATLAS3D":
                fulltable["MIR_class"][ix] = np.ma.masked

        fulltable.sort("objstr_01")
        # For some reason, the masks keep getting messed up whenever I make the
        # full table.
        fulltable["NUV_Tile"].mask = fulltable["NUV_Tile"] == '0.0'
        fulltable["FUV_Tile"].mask = fulltable["FUV_Tile"] == '0.0'
        fulltable["NUVapmag"].mask = np.isnan(fulltable["NUVapmag"])
        fulltable["FUVapmag"].mask = np.isnan(fulltable["FUVapmag"])
        fulltable["w1apmag"].mask = np.isnan(fulltable["w1apmag"])
        fulltable["w2apmag"].mask = np.isnan(fulltable["w2apmag"])
        fulltable["w3apmag"].mask = np.isnan(fulltable["w3apmag"])
        fulltable["w4apmag"].mask = np.isnan(fulltable["w4apmag"])
        fulltable["NUVaperr"].mask = np.isnan(fulltable["NUVaperr"])
        fulltable["FUVaperr"].mask = np.isnan(fulltable["FUVaperr"])
        fulltable["w1aperr"].mask = np.isnan(fulltable["w1aperr"])
        fulltable["w2aperr"].mask = np.isnan(fulltable["w2aperr"])
        fulltable["w3aperr"].mask = np.isnan(fulltable["w3aperr"])
        fulltable["w4aperr"].mask = np.isnan(fulltable["w4aperr"])
    else:
        pass

def remove_duplicate_galaxies(table):
    '''If the table has duplicate galaxies, they will be removed.

    This function doesn't check if the rows are equal or not. It just
    determines the row by the name.
    '''
    groupedtable = table.group_by('objstr_01')
    fulltable = Table(rows=groupedtable.groups[0][0])
    for ix in xrange(1, len(groupedtable.groups)):
        tablegroup = groupedtable.groups[ix]
        addrow = tablegroup[0]
        fulltable.add_row(addrow)
    fulltable.sort("objstr_01")
    return fulltable

def build_filepath(basepath, filename, extension=EXT):
    '''Builds a full file path of a file.

    This function takes a basepath, joins it to a filename, and intelligently
    adds a filename extension to the end. This is so that extensions can be
    specified independently of the filename.
    '''
    fullfilename = os.extsep.join((filename, extension))
    fullpath = os.path.join(basepath, fullfilename)
    return fullpath

def remove_bad_galaxies(table, galcol="objstr_01", badgals=["NGC2974"]):
    '''Creates a new table without galaxies in badgals.

    This is done to sanitize plots without making any persistent changes to the
    raw tables, since there are other functions which still need the full
    tables. Also, bad galaxies for one plot may not necessarily be bad galaxies
    for other plots.
    '''
    badindices = phot.astropy_table_index(table, galcol, 

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
    columns = ["objstr_01", "RSA_morph_type", "D", "MIR_class", "w1rsemi",
               "w1ba", "w1pa", "cat", "NUV_Tile", "FUV_Tile"]
    names = ["Galaxy", "Morph.", "Distance", "MIR Class", "Semimajor Axis",
             "Axis Ratio", "Position Angle", "WISE Catalog", "NUV Tile", 
             "FUV Tile"]
    units = {"Distance": "Mpc", "Semimajor Axis": "''", "Position Angle": 
             r"\(^\ocirc\)"}
    table1 = table[columns]
#    table1.write(dest, format="ascii.latex", names=names)
    table1.write(dest, format="ascii.aastex", names=names,
                 latexdict={"caption": caption, "units": units})

def create_ATLAS3D_sample_table(table=atlas3d_table,
                                dest=os.path.join(TABLEPATH,
                                                  "atlas3dtbl.tex")):
    atlas3dprops = read_Cappellari11_Table_3()
    joinedtable = phot.multijoin_by_galaxy_name(
        table, atlas3dprops, names=("objstr_01", "Galaxy"))
    caption = r"""Properties of galaxies in the \ATLAS{} sample.
    \label{tab:atlas3dsample}"""
    columns = ["objstr_01", "type", "D", "w1rsemi",
               "w1ba", "w1pa", "cat", "NUV_Tile", "FUV_Tile"]
    names = ["Galaxy", "Morph", "Distance", "Semimajor Axis", "Axis Ratio", 
               "Position Angle", "WISE Catalog", "NUV Tile", "FUV Tile"]
    table1 = joinedtable[columns]
    table1.write(dest, format="ascii.aastex", names=names,
                 latexdict={"caption": caption},
                 formats={"NUV Tile": format_GALEX_tile, "FUV Tile":
                          format_GALEX_tile})

def create_param_table(table=fulltable[:10], dest=os.path.join(TABLEPATH,
                                                               "params.tex")):

    caption = r"""Aperture photometry parameters for galaxies in the \ATLAS{}
    and Rampazzo samples
    \label{tab:params}"""
    columns = ["objstr_01", "type", "D", "sample", "w1rsemi", "w4rsemi", 
               "w1ba", "w4ba", "w1pa", "cat", "NUV_Tile", "FUV_Tile"]
    names = ["Galaxy", "Morph", "D", "S", r"\(a_{W1}\)", r"\(a_{W4}\)",
             r"\(b/a_{W1}\)", r"\(b/a_{W4}\)", "PA", "Survey", "NUV Tile", 
             "FUV Tile"]
    preamble = r"""\tabletypesize{\scriptsize}"""
    writetable = table[columns]
    writetable.write(dest, format="ascii.aastex", names=names,
                     latexdict={"caption": caption, "preamble": preamble}, 
                     formats={"NUV Tile": format_GALEX_tile, "FUV Tile":
                              format_GALEX_tile, "S": format_sample, 
                              "PA": "%d"})

def create_magnitude_table(table=rampazzo_table[:10], 
                           dest=os.path.join(TABLEPATH, "mags.tex")):
    caption = r"""Magnitudes of galaxies in the \ATLAS{} and Rampazzo samples.
    \label{tab:magtable}"""
    tablefoot = r"""\tablecomments{Magnitudes are reported in the AB system. 
    They have not been corrected for extinction.}"""
    columns = ["objstr_01", "FUVaperr", "NUVaperr", "w1aperr", "w2aperr", 
               "w3aperr", "w4aperr"]
    column_order = ["objstr_01", "FUVapmag", "FUVaperr", "NUVapmag", "NUVaperr", 
                    "w1apmag", "w1aperr", "w2apmag", "w2aperr", "w3apmag", 
                    "w3aperr", "w4apmag", "w4aperr"]
    names = ["Galaxy", "FUV", r"\(\sigma_{FUV}\)", "NUV", r"\(\sigma_{NUV}\)", 
             "W1", r"\(\sigma_{W1}\)", "W2", r"\(\sigma_{W2}\)", "W3",
             r"\(\sigma_{W3}\)", "W4", r"\(\sigma_{W4}\)"]
    preamble = r"""\tabletypesize{\scriptsize}"""
    formats = {"Galaxy": format_reflect, r"\(\sigma_{FUV}\)": format_mag_err, 
               r"\(\sigma_{NUV}\)": format_mag_err, 
               r"\(\sigma_{W1}\)": format_mag_err, 
               r"\(\sigma_{W2}\)": format_mag_err, 
               r"\(\sigma_{W3}\)": format_mag_err, 
               r"\(\sigma_{W4}\)": format_mag_err}
    export = table[columns]
    export["FUVapmag"] = format_mag_column(table["FUVapmag"],
                                           table["FUVaplim"])
    export["NUVapmag"] = format_mag_column(table["NUVapmag"],
                                           table["NUVaplim"])
    export["w1apmag"] = format_mag_column(table["w1apmag"],
                                           table["w1aplim"])
    export["w2apmag"] = format_mag_column(table["w2apmag"],
                                           table["w2aplim"])
    export["w3apmag"] = format_mag_column(table["w3apmag"],
                                           table["w3aplim"])
    export["w4apmag"] = format_mag_column(table["w4apmag"],
                                           table["w4aplim"])
    export[column_order].write(
        dest, format="ascii.aastex", names=names, formats=formats,
        latexdict={"caption": caption, "tablefoot": tablefoot,
                   "tabletype": "deluxetable", "preamble": preamble})

def create_W2_W3_histogram(table=rampazzo_table, 
                           dest=build_filepath(FIGUREPATH, "w2w3hist")):
    '''Creates the histogram which plots all of the objects in W2-W3.'''
    xlabel = "W2-W3"
    rp.color_histogram_by_class(table["w2unextmag"],
                                table["w3unextmag"],
                                table["MIR_class"], "W2-W3", 
                                "", 80, colrange=(-1.5, 2.5))
    plt.savefig(dest)
    plt.close()

def create_W2_W3_cumulative_histogram(
        table=rampazzo_table, dest=build_filepath(FIGUREPATH,
                                                  "class023_cumhist")):
    '''Cumulative histogram illustrating difference between Class 0 and 2/3.'''
    xlabel = "W2-W3"
    rp.color_cumulative_histogram_by_class(
        table["w2unextmag"], table["w3unextmag"], table["MIR_class"], "W2-W3",
        "", 80, colrange=(-1.5, 2.5), classes=[0, 2, 3])
    plt.savefig(dest)
    plt.close()

def create_W1W2_W2W3_MIR_plot(table=rampazzo_table,
                              dest=build_filepath(FIGUREPATH, "w1w2w2w3_mir")):
    '''Plots W1-W2 vs W2-W3'''
    ylabel = "W1-W2"
    xlabel = "W2-W3"

    w1 = table["w1unextmag"]
    w2 = table["w2unextmag"]
    w3 = table["w3unextmag"]
    w1err = table["w1unexterr"]
    w2err = table["w2unexterr"]
    w3err = table["w3unexterr"]
    w1lim = table["w1unextlim"]
    w2lim = table["w2unextlim"]
    w3lim = table["w3unextlim"]
    MIR = table["MIR_class"]

    w1w2, w1w2err, w1w2lim = stat.subtract(w1, w2, w1err, w2err, w1lim, w2lim)
    w2w3, w2w3err, w2w3lim = stat.subtract(w2, w3, w2err, w3err, w2lim, w3lim)

    rp.MIRplot(w2w3, w1w2, MIR, w1w2err, w2w3err, xrange(5), xlabel,
               ylabel, "", loc="upper left")
    plt.savefig(dest)
    plt.close()

def create_W2W3_W3W4_MIR_plot(table=rampazzo_table,
                              dest=build_filepath(FIGUREPATH, "w2w3w3w4_mir")):
    '''Plots W2-W3 vs W3-W4'''
    ylabel = "W2-W3"
    xlabel = "W3-W4"

    w2 = table["w2unextmag"]
    w3 = table["w3unextmag"]
    w4 = table["w4unextmag"]
    w2err = table["w2unexterr"]
    w3err = table["w3unexterr"]
    w4err = table["w4unexterr"]
    MIR = table["MIR_class"]

    w2w3, w2w3err = stat.subtract(w2, w3, w2err, w3err)
    w3w4, w3w4err = stat.subtract(w3, w4, w3err, w4err)

    rp.MIRplot(w3w4, w2w3, MIR, w2w3err, w3w4err, xrange(5), xlabel,
               ylabel, "", loc="upper left")
    plt.axis([-1.5, 0, -1.5, 0])
    plt.savefig(dest)
    plt.close()

def create_sulfur_h2_plot(
    table=rampazzo_table, dest=build_filepath(FIGUREPATH, "lineratios")):
    linetable = read_Rampazzo_TableA2()
    rp.plot_rampazzo_line_ratios(linetable, table, "")
    plt.savefig(dest)
    plt.close()

def create_dual_panel_W1W2_W2W3_W3W4_MIR_plot(
    table=rampazzo_table, dest=build_filepath(FIGUREPATH, "dual_mir_plot")):
    w1 = table["w1unextmag"]
    w2 = table["w2unextmag"]
    w3 = table["w3unextmag"]
    w4 = table["w4unextmag"]
    w1err = table["w1unexterr"]
    w2err = table["w2unexterr"]
    w3err = table["w3unexterr"]
    w4err = table["w4unexterr"]
    w1lim = table["w1unextlim"]
    w2lim = table["w2unextlim"]
    w3lim = table["w3unextlim"]
    w4lim = table["w4unextlim"]
    MIR = table["MIR_class"]

    w1w2, w1w2err, w1w2lim = stat.subtract(w1, w2, w1err, w2err, w1lim, w2lim)
    w2w3, w2w3err, w2w3lim = stat.subtract(w2, w3, w2err, w3err, w2lim, w3lim)
    w3w4, w3w4err, w3w4lim = stat.subtract(w3, w4, w3err, w4err, w3lim, w4lim)

    w1w2label = "W1-W2"
    w2w3label = "W2-W3"
    w3w4label = "W3-W4"

    plt.subplot(1, 2, 1)
    rp.MIRplot(w2w3, w1w2, MIR, w1w2err, w2w3err, xrange(5), w2w3label,
               w1w2label, "", loc="lower right")
    plt.axis([-1.5, 0, -0.75, -0.6])

    plt.subplot(1, 2, 2)
    rp.MIRplot(w3w4, w2w3, MIR, w2w3err, w3w4err, xrange(5), w3w4label,
               w2w3label, "", loc=None)
    plt.axis([-1.5, 0, -1.5, 0])
    plt.tight_layout()
    plt.savefig(dest)
    plt.close()



def create_PAH113_17_PAH77_113_plot(table=rampazzo_table,
       dest=os.path.join(FIGUREPATH, "shortpahs.pdf")):
    pahtable = read_Rampazzo_TableA1()
    fulltable = phot.join_by_galaxy_name(table, pahtable,
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

    xratio, xratioerr = stat.divide(pah113, pah17, 
                                                       pah113_err, pah17_err)
    yratio, yratioerr = stat.divide(pah77, pah113,
                                                       pah77_err, pah113_err)

    rp.MIRplot(xratio, yratio, classtable["MIR_class"], yerr=yratioerr,
               xerr=xratioerr, classes=(2,3), xlabel="11.3 um/17 um", 
               ylabel="7.7 um/11.3 um")

    plt.xlim([0, 6])
    plt.ylim([0, 6])
    plt.savefig(dest)
    plt.close()

def create_stellar_mass_ATLAS3D_comparison(
        table=atlas3d_table, dest=os.path.join(FIGUREPATH, "stellarmass.pdf")):
    atlas3d_params = read_Cappellari11_Table_3()
    atlas3d_lums = read_Cappellari13a_Table_1()
    atlas3d_masstolight = read_Cappellari13b_Table_1()
    joinedtable = phot.multijoin_by_galaxy_name(
        table, atlas3d_params, atlas3d_masstolight, atlas3d_lums,
        names=("objstr_01", "Galaxy", "Galaxy", "Galaxy"))
    absmag_w1 = joinedtable["w1unextmag"] - (5 *
        np.log10(joinedtable["D"]*1e6/10))
    logluminosity_w1 = -0.4 * (
        absmag_w1 - conv.SOLAR_ABSOLUTE_MAGNITUDES_AB["W1"])
    mass_jarrett = 10**(joinedtable["logML_W1"] + logluminosity_w1)
    mass_atlas3d = 10**(joinedtable["logML_star"] + joinedtable["logLum"])
    plt.loglog(mass_jarrett, mass_atlas3d, 'b*')
    plt.xlabel("Jarrett M* (Msun)")
    plt.ylabel("ATLAS3D M* (Msun)")
    plt.savefig(dest)
    plt.close()

def create_stellar_mass_luminosity_relation(
        table=atlas3d_table, dest=os.path.join(FIGUREPATH, 
        "mass-luminosity.pdf")):
    joinedtable = table

    atlas3d_stellar_mass, atlas3d_stellar_mass_err, _ = stat.add(
        joinedtable["logML_star"], joinedtable["logLum"], 0.06/np.log(10),
        0.1/np.log(10))
    distance_modulus, distance_modulus_err, _ = stat.logarithm(
        joinedtable["D"]*1e5, joinedtable["D_err"]*1e5)
    absolute_w1, absolute_w1_err, _ = stat.subtract(
        joinedtable["w1unextmag"], 5 * distance_modulus, 
        joinedtable["w1unexterr"], 5 * distance_modulus_err)
    lum, lum_err = (-0.4 * (absolute_w1 - 
                           conv.SOLAR_ABSOLUTE_MAGNITUDES_AB["W1"]),
                    0.4 * absolute_w1_err)
    plt.errorbar(lum, atlas3d_stellar_mass, atlas3d_stellar_mass_err, lum_err, 
                 'bo', label="ATLAS3D")
    plt.xlabel("log L_W1 (Lsun)")
    plt.ylabel("log M* (Msun)")
    plt.savefig(dest)
    plt.close()

    p1 = models.Polynomial1D(1)
    pfit = fitting.LinearLSQFitter()
    mass_lum_model = pfit(p1, lum, atlas3d_stellar_mass)

    print "Best fit is  log M* = {0:.1f} log L_W1 + {1:.1f}".format(
        mass_lum_model.c1.value, mass_lum_model.c0.value)
    

def create_cutout_grid(
        table=atlas3d_table[:2], dest=os.path.join(FIGUREPATH, "cutouts.png"),
        BASEDIR=ATLAS3DBASE):
    gridfig = phot.ellipse_cutout_grid(
        BASEDIR, table, ignore_exception=True)
    gridfig.savefig(dest)
    plt.close(gridfig)


def create_mass_to_light_ATLAS3D_comparison(
        table=atlas3d_table, dest=os.path.join(FIGUREPATH, "masstolight.pdf")):
    # Calculate the mass-to-light ratio for the ATLAS3D points in WISE.
    w1w2, w1w2err, w1w2lim = stat.subtract(
        table["w1unextmag"], table["w2unextmag"], table["w1unexterr"], 
        table["w2unexterr"], table["w1unextlim"], table["w2unextlim"])
    atlas3d.plot_dustless_separation(
        table, w1w2, table["logML_W1"], table["logML_W1err"], w1w2err, 
        table["logML_W1lim"], 'bo', color=True, ms=5, label="ATLAS3D",
        linewidth=0.7)


    # Remember that these will be given in Vega mags.
    meidt_limits_IRAC = np.linspace(-0.12, -0.04, 2)
    meidt_limits = conv.IRAC2WISEcolor("[3.6]", "[4.5]", meidt_limits_IRAC,
                                       wisesystem="AB")
    meidt_ML = (3.98 * meidt_limits_IRAC + 0.13 - 
                np.log10(conv.IRACflux2WISEflux("[3.6]", 1)))
    meidt_ML_upper = ((3.98+0.98) * meidt_limits_IRAC + (0.13 + 0.08) -
                      np.log10(conv.IRACflux2WISEflux("[3.6]", 1)))
    meidt_ML_lower = ((3.98-0.98) * meidt_limits_IRAC + (0.13 - 0.08) -
                      np.log10(conv.IRACflux2WISEflux("[3.6]", 1)))
    plt.plot(meidt_limits, meidt_ML, 'g-', label="Meidt M/L")
    plt.plot(meidt_limits, meidt_ML_upper, 'g--')
    plt.plot(meidt_limits, meidt_ML_lower, 'g--')
    const_ML = np.log10(0.6 / conv.IRAC_TO_WISE_FACTOR["[3.6]"])
    plt.plot(meidt_limits, [const_ML, const_ML], 'g:')

    # Relations from Cluver et al( 2014)
    cluver_limits_VEGA = np.linspace(-0.3, 0.7, 2)
    cluver_limits = conv.Vega2ABcolor("W1", "W2", cluver_limits_VEGA)
    cluver_ML = -2.54 * (cluver_limits_VEGA) - 0.17
    plt.plot(cluver_limits, cluver_ML, 'r-', label="Cluver M/L")

    # FSPS output
    solmetmags = fsps.read_mags(os.path.join(
        fsps.OUTPUT_PATH, "imf_met", "solmet_chabrier.mags"))
    lowmetmags = fsps.read_mags(os.path.join(
        fsps.OUTPUT_PATH, "imf_met", "lowmet_chabrier.mags"))
    highmetmags = fsps.read_mags(os.path.join(
        fsps.OUTPUT_PATH, "imf_met", "highmet_chabrier.mags"))

    modelage = 1e10
    solmet = fsps.FSPS_data_at_age(solmetmags["log(age)"], solmetmags,
                                   modelage)
    lowmet = fsps.FSPS_data_at_age(lowmetmags["log(age)"], lowmetmags,
                                   modelage)
    highmet = fsps.FSPS_data_at_age(highmetmags["log(age)"], highmetmags,
                                   modelage)
    spsmodels = Table(np.vstack([lowmet, solmet, highmet]),
                   names=solmetmags.colnames)
    w1w2 = spsmodels["W1"] - spsmodels["W2"]
    masstolight = (spsmodels["log(mass)"] - np.log10(conv.ABabsmag2inbandLum(
        "W1", spsmodels["W1"])))
    plt.plot(w1w2, masstolight, 'c-', label="FSPS (c)")

    solmetsalmags = fsps.read_mags(os.path.join(
        fsps.OUTPUT_PATH, "imf_met", "solmet_salpeter.mags"))
    lowmetsalmags = fsps.read_mags(os.path.join(
        fsps.OUTPUT_PATH, "imf_met", "lowmet_salpeter.mags"))
    highmetsalmags = fsps.read_mags(os.path.join(
        fsps.OUTPUT_PATH, "imf_met", "highmet_salpeter.mags"))
    solmetsal = fsps.FSPS_data_at_age(solmetsalmags["log(age)"], solmetsalmags,
                                   modelage)
    lowmetsal = fsps.FSPS_data_at_age(lowmetsalmags["log(age)"], lowmetsalmags,
                                   modelage)
    highmetsal = fsps.FSPS_data_at_age(highmetsalmags["log(age)"], highmetsalmags,
                                   modelage)
    salmodels = Table(np.vstack([lowmetsal, solmetsal, highmetsal]),
                   names=solmetsalmags.colnames)
    w1w2sal = salmodels["W1"] - salmodels["W2"]
    masstolightsal = (salmodels["log(mass)"] - np.log10(conv.ABabsmag2inbandLum(
        "W1", salmodels["W1"])))
    plt.plot(w1w2sal, masstolightsal, 'cx-', label="FSPS (s)")

    # This will be a fit to the color-selected dustless galaxies.
    dustlesstable = atlas3d.color_cut_dustless_table(table)
    w1w2dustless, w1w2dustlesserr, w1w2dustlesslim = stat.subtract(
        dustlesstable["w1unextmag"], dustlesstable["w2unextmag"], 
        dustlesstable["w1unexterr"], dustlesstable["w2unexterr"], 
        dustlesstable["w1unextlim"], dustlesstable["w2unextlim"])
    # If I figure out how to get parameter estimates with error bars, I'll do
    # that for a variable slope.
    flatline = models.Linear1D(0, -0.006, fixed={"slope": True})
    fit_l = fitting.LevMarLSQFitter()
    flatfit = fit_l(flatline, w1w2dustless, dustlesstable["logML_W1"])
    fit_limits = np.linspace(-0.75, -0.6, 2)
    flatlogval = flatfit(0)
    # I don't want a sqrt(N) because I'm not looking for the error on the mean,
    # I'm looking for the width of the Gaussian fitted to the distribution of
    # galaxies.
    flatlogerr = (np.std(dustlesstable["logML_W1"] - flatlogval)) 
    #flatval, flaterr, _ = stat.exponentiate(10, flatlogval, 0, flatlogerr)
    print "Appropriate M/L is {0:.2f} +/- {1:.3f}".format(
        flatlogval, flatlogerr)
    plt.plot(fit_limits, [flatlogval, flatlogval], 'k--', label="This work")

    plt.xlabel("W1-W2 (AB)")
    plt.ylabel("log (M*/L)_W1")
    # Not shown is PGC029321 all the way to the right.
    plt.xlim([-0.75, -0.6])
    plt.ylim([-1.0, 0.5])
    plt.legend(loc="lower left")
    ax = plt.gca()
    add_Vega_axis("W1", "W2")
    plt.savefig(dest)
    plt.close()

def create_NUV_W1_abs_plot(rtable=rampazzo_table, atable=atlas3d_table, 
        dest=os.path.join(FIGUREPATH, "nuvw1.pdf")):
    rnuv = rtable[phot.name_photometry_column("NUV")]
    rnuv_e = rtable[phot.name_photometry_column("NUV", error=True)]
    rnuv_l = rtable[phot.name_photometry_column("NUV", limit=True)]
    rw1 = rtable[phot.name_photometry_column("W1")]
    rw1_e = rtable[phot.name_photometry_column("NUV", error=True)]
    rw1_l = rtable[phot.name_photometry_column("NUV", limit=True)]
    rd = rtable["D"]

    rnuvw1, rnuvw1_e, rnuvw1_l = stat.subtract(
        rnuv, rw1, rnuv_e, rw1_e, rnuv_l, rw1_l)
    rw1abs = conv.app2absmag(rw1, rd)
    # This until I figure out what to do with distance errors.
    rw1abs_e = rw1_e

    anuv = atable[phot.name_photometry_column("NUV")]
    anuv_e = atable[phot.name_photometry_column("NUV", error=True)]
    anuv_l = atable[phot.name_photometry_column("NUV", limit=True)]
    aw1 = atable[phot.name_photometry_column("W1")]
    aw1_e = atable[phot.name_photometry_column("NUV", error=True)]
    aw1_l = atable[phot.name_photometry_column("NUV", limit=True)]
    ad = atable["D"]
    ad_e = atable["D_err"]

    anuvw1, anuvw1_e, anuvw1_l = stat.subtract(
            anuv, aw1, anuv_e, aw1_e, anuv_l, aw1_l)
    aw1abs = conv.app2absmag(aw1, ad)
    # This until I figure out what to do with distance errors.
    aw1abs_e = conv.app_err_to_abs_err(aw1_e, ad, ad_e)

    plt.errorbar(aw1abs, anuvw1, anuvw1_e, aw1abs_e, 'kx', label="ATLAS3D")
    rp.MIRplot(rw1abs, rnuvw1, rtable["MIR_class"], rnuvw1_e, xlabel="M_{W1}",
               ylabel="NUV-W1")
    print rtable["objstr_01"][np.where(np.logical_and(rnuvw1 < 4, rw1abs <
                                                      -23))]

    plt.ylim([2, 7])
    plt.xlim([-18, -24])
    plt.legend(loc="lower right", bbox_to_anchor=(0.85, 0.01))
    plt.savefig(dest)
    plt.close()

def create_NUV_J_PAH77_113_plot(table=rampazzo_table, 
                                dest=os.path.join(FIGUREPATH, "uvpahs.pdf")):
    pahtable = read_Rampazzo_TableA1()
    fulltable = phot.join_by_galaxy_name(table, pahtable,
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

    xcolor, xcolorerr, _ = stat.subtract(nuv, j, nuv_err, j_err)
    yratio, yratioerr, _ = stat.divide(pah77, pah113,
                                                       pah77_err, pah113_err)

    rp.MIRplot(xcolor, yratio, classtable["MIR_class"], yerr=yratioerr, 
               xerr=xcolorerr, classes=(2,3), xlabel="NUV-J",
               ylabel="7.7 um / 11.3 um")
    plt.xlabel("NUV-J")
    plt.ylabel("7.7 um/11.3 um")
    plt.savefig(dest)
    plt.close()

    return classtable

def create_SED(
    table=fulltable, dest=build_filepath(FIGUREPATH, "sed", EXT), 
    ages=[1e8, 1e9, 5e9, 1e10], 
    modelbase="toggle_dust_met_bounds/dust_highmet"):

    fsps.plot_FSPS_SED_at_ages(
        FSPSPATH, plotquant="Flux", fmts=None, ages=ages, 
        modelbase=modelbase)

    passive_indices = atlas3d.ATLAS3D_dustless_galaxy_indices(table)
    active_indices = get_complement_indices(passive_indices, len(table))
    fsps.plot_data_SED(table[passive_indices], plotquant="Flux", fmt="rx", 
                       label="Passive galaxies", markersize=8)
    fsps.plot_data_SED(table[active_indices], plotquant="Flux", fmt="b.", 
                       label="Non-passive galaxies", markersize=10)

    plt.xlabel("Wavelength (um)")
    plt.ylabel("nu f_nu")
    plt.xlim([0.1, 50])
    plt.ylim([1e8, 1e16])
#    plt.gca().invert_yaxis()
    plt.legend(loc="lower left")
    plt.savefig(dest)
    plt.close()

def create_circumstellar_verification_plot(
    atable=atlas3d_table, rtable=rampazzo_table, 
    dest=build_filepath(FIGUREPATH, "dustless", EXT)):
    '''Plot showing where Class-0 and dustless ATLAS3D objects lie.'''

    # We want to set up our datasets of dustless ATLAS3D and Class-0 Rampazzo
    # galaxies.
    dcat_indices  = atlas3d.ATLAS3D_dustless_galaxy_indices(atable)
    dusty_indices = get_complement_indices(dcat_indices, len(atable))
    dcat = atable[dcat_indices]
    dcat_dusty = atable[dusty_indices]
    colorcut_dustless_indices = atlas3d.color_cut_dustless_indices(
        dcat[phot.name_photometry_column("W1")],
        dcat[phot.name_photometry_column("W3")],
        dcat[phot.name_photometry_column("W4")])
    colorcut_dusty_indices = get_complement_indices(
        colorcut_dustless_indices, len(dcat))
    # These represent all of the "dustless" galaxies which pass the color cut.
    colorcut_dustless = dcat[colorcut_dustless_indices]
    # These represent all the "dustless" galaxies which fail the color cut.
    colorcut_dusty = dcat[colorcut_dusty_indices]

    class0 = rp.extract_MIR_class_sample(rtable, 0, "MIR_class")

    # These are the Amblard et al 2014 objects. 
    amblard_objects = read_Amblard_Table_4()
    amblard_overlap = phot.join_by_galaxy_name(dcat, amblard_objects, 
                                               names=("objstr_01", "Name"))
    amblard_dusty = amblard_overlap[np.where(
        np.logical_and(
            amblard_overlap["250_mum"] > 5*amblard_overlap["250_mum_err"],
            amblard_overlap["350_mum"] > 5*amblard_overlap["350_mum_err"],
            amblard_overlap["500_mum"] > 5*amblard_overlap["500_mum_err"]))]

    dusty_atlas3d_indices = phot.astropy_table_indices(
        dcat, "objstr_01", amblard_dusty["objstr_01"])
    print "% of overlapped galaxies w/ FIR Dust detections: {0:.2f}".format(
        float(len(amblard_dusty["objstr_01"]))/len(amblard_overlap)*100)


    class0w1w3, class0w1w3_err, class0w1w3_lim = stat.subtract(
        class0["w1unextmag"], class0["w3unextmag"], class0["w1unexterr"],
        class0["w3unexterr"], class0["w1unextlim"], class0["w3unextlim"])
    class0w1w4, class0w1w4_err, class0w1w4_lim = stat.subtract(
        class0["w1unextmag"], class0["w4unextmag"], class0["w1unexterr"],
        class0["w4unexterr"], class0["w1unextlim"], class0["w4unextlim"])
    dustyw1w3, dustyw1w3_err, dustyw1w3_lim = stat.subtract(
        colorcut_dusty["w1unextmag"], colorcut_dusty["w3unextmag"], 
        colorcut_dusty["w1unexterr"], colorcut_dusty["w3unexterr"], 
        colorcut_dusty["w1unextlim"], colorcut_dusty["w3unextlim"])
    dustyw1w4, dustyw1w4_err, dustyw1w4_lim = stat.subtract(
        colorcut_dusty["w1unextmag"], colorcut_dusty["w4unextmag"], 
        colorcut_dusty["w1unexterr"], colorcut_dusty["w4unexterr"], 
        colorcut_dusty["w1unextlim"], colorcut_dusty["w4unextlim"])
    colorcutw1w3, colorcutw1w3_err, colorcutw1w3_lim = stat.subtract(
        colorcut_dustless["w1unextmag"], colorcut_dustless["w3unextmag"], 
        colorcut_dustless["w1unexterr"], colorcut_dustless["w3unexterr"], 
        colorcut_dustless["w1unextlim"], colorcut_dustless["w3unextlim"])
    colorcutw1w4, colorcutw1w4_err, colorcutw1w4_lim = stat.subtract(
        colorcut_dustless["w1unextmag"], colorcut_dustless["w4unextmag"], 
        colorcut_dustless["w1unexterr"], colorcut_dustless["w4unexterr"], 
        colorcut_dustless["w1unextlim"], colorcut_dustless["w4unextlim"])

    stat.errorbar(class0w1w3, class0w1w4, class0w1w4_err, class0w1w3_err,
                  class0w1w4_lim, label="Class 0", ufmt="kv", lfmt="k^", 
                  **rp.MIR_Symbols[0])
    stat.errorbar(dustyw1w3, dustyw1w4, dustyw1w4_err,
                  dustyw1w3_err, dustyw1w4_lim, 'gx', 
                  label="ATLAS3D (FIR Det)", ufmt="gv", lfmt="g^")
    stat.errorbar(colorcutw1w3, colorcutw1w4, colorcutw1w4_err,
                  colorcutw1w3_err, colorcutw1w4_lim,
                  label="ATLAS3D", fmt="mx", ufmt="mv", lfmt="m^")
    fsps.plot_FSPS_color_color(
        fsps.OUTPUT_PATH, "W1", "W3", "W1", "W4", 
        modelbase=os.path.join("toggle_dust_met_bounds", "dust_highmet"),
        label="[Z/H] = 0.2", fmt="r-", agecutoff=1e9)
    fsps.plot_FSPS_color_color(
        fsps.OUTPUT_PATH, "W1", "W3", "W1", "W4", 
        modelbase=os.path.join("toggle_dust_met_bounds", "dust_lowmet"),
        label="[Z/H] = -0.89", fmt="b-", agecutoff=1e9)
    fsps.plot_FSPS_color_color(
        fsps.OUTPUT_PATH, "W1", "W3", "W1", "W4", 
        modelbase=os.path.join("toggle_dust_met_bounds", "nodust_highmet"),
        label="[Z/H] = 0.2", fmt="r:", agecutoff=1e9)
    fsps.plot_FSPS_color_color(
        fsps.OUTPUT_PATH, "W1", "W3", "W1", "W4", 
        modelbase=os.path.join("toggle_dust_met_bounds", "nodust_lowmet"),
        label="[Z/H] = -0.89", fmt="b:", agecutoff=1e9)
#   parsec.plot_parsec_color_color(PARSECPATH, "marigo_highmet.dat", "W1", "W3",
#                                  "W1", "W4", label="PARSEC (high met)",
#                                  fmt="r--", agecutoff=1e9)
#   parsec.plot_parsec_color_color(PARSECPATH, "marigo_lowmet.dat", "W1", "W3",
#                                  "W1", "W4", label="PARSEC (low met)",
#                                  fmt="b--", agecutoff=1e9)
    plt.xlabel("W1-W3 (AB)")
    plt.ylabel("W1-W4 (AB)")
    add_Vega_axis("W1", "W3")
    add_Vega_axis("W1", "W4", "y")
    plt.legend(loc="lower right")
    plt.savefig(dest)
    plt.close()

def add_Vega_axis(band1, band2, axis="x"):
    '''Adds a matching axis for colors in the Vega system to the current figure.

    The necessary information are the two bands used for the color, as well as
    whether the additional axis should be an x-axis or y-axis.
    '''
    def transform(a):
        return a + (conv.AB2Vegamag(band1, 0) - conv.AB2Vegamag(band2, 0))
    def detransform(a):
        return a - transform(0)

    ax = plt.gca()

    if axis == "x":
        newax = ax.twiny()
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        newax.xaxis.tick_bottom()
        newax.xaxis.set_label_position('bottom')
        lowbound, upbound = ax.get_xlim()
        newax.set_xlim(transform(lowbound), transform(upbound))
        newax.set_xlabel(ax.get_xlabel().replace("(AB)", "(Vega)"))
    if axis == "y":
        ax.yaxis.tick_left()
        newax = ax.twinx() 
        lowbound, upbound = ax.get_ylim()
        newax.set_ylim(transform(lowbound), transform(upbound))
        newax.set_ylabel(ax.get_ylabel().replace("(AB)", "(Vega)"))

def create_circumstellar_dust_plot(table=atlas3d_table,
                                   dest=build_filepath(FIGUREPATH, "cdust",
                                                       EXT)):
    '''Dual-paneled W1-W3 and W1-W4 vs age plot.'''
    # Set up the data
    dustless_catalog = \
        atlas3d.filter_ATLAS3D_table_for_dustless_galaxies(table)
    w1w3color, w1w3err, w1w3lim = stat.subtract(
        dustless_catalog["w1unextmag"], dustless_catalog["w3unextmag"], 
        dustless_catalog["w1unexterr"], dustless_catalog["w3unexterr"],
        dustless_catalog["w1unextlim"], dustless_catalog["w3unextlim"])
    w1w4color, w1w4err, w1w4lim = stat.subtract(
        dustless_catalog["w1unextmag"], dustless_catalog["w4unextmag"], 
        dustless_catalog["w1unexterr"], dustless_catalog["w4unexterr"],
        dustless_catalog["w1unextlim"], dustless_catalog["w4unextlim"])
    atlas3d_ages = dustless_catalog["Age_SSP"]
    atlas3d_ages_err = dustless_catalog["Age_SSP_err"]
    atlas3d_metallicities = dustless_catalog["[Z/H]_SSP"]
    med_met = float(np.ma.median(atlas3d_metallicities))
                                                
    
    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    # Let's make the first plot: W1-W3
    plt.sca(ax1)
    fsps.plot_atlas3d_coded_by_metallicity(
        atlas3d_ages, w1w3color, w1w3err, atlas3d_ages_err, w1w3lim,
        atlas3d_metallicities, med_met)
    fsps.plot_dust_toggled_metallicity_bounds(
        os.path.join(fsps.OUTPUT_PATH, "toggle_dust_met_bounds"), "W1", "W3")
    plt.legend(loc="upper right", fontsize="small")
    plt.ylabel("W1-W3")

    # Now the second plot: W1-W4
    plt.sca(ax2)
    fsps.plot_atlas3d_coded_by_metallicity(
        atlas3d_ages, w1w4color, w1w4err, atlas3d_ages_err, w1w4lim,
        atlas3d_metallicities, med_met)
    fsps.plot_dust_toggled_metallicity_bounds(
        os.path.join(fsps.OUTPUT_PATH, "toggle_dust_met_bounds"), "W1", "W4")
    ax2.set_ylim([-4, 1])
    plt.ylabel("W1-W4")
    plt.xlabel("SSP Age (Gyr)")
    plt.tight_layout(0)
    plt.savefig(dest)
    plt.close()

def create_star_formation_plot(
        table=atlas3d_table, dest=build_filepath(FIGUREPATH, "circsfr", EXT)):
    '''Creates a plot similar to Figure 1 in Davis et al 2014.

    This will be a log-log plot of the W4 vs Ks band luminosities of the
    ATLAS3D sample. However, instead of separating H2 detected and
    non-detected, we will separate the dustless from dusty based on our color
    cut in the plane.
    '''
    w1mag = table[phot.name_photometry_column("W1")]
    w1err = table[phot.name_photometry_column("W1", error=True)]
    w1lim = table[phot.name_photometry_column("W1", limit=True)]
    w2mag = table[phot.name_photometry_column("W2")]
    w2err = table[phot.name_photometry_column("W2", error=True)]
    w2lim = table[phot.name_photometry_column("W2", limit=True)]
    w3mag = table[phot.name_photometry_column("W3")]
    w3err = table[phot.name_photometry_column("W3", error=True)]
    w3lim = table[phot.name_photometry_column("W3", limit=True)]
    w4mag = table[phot.name_photometry_column("W4")]
    w4err = table[phot.name_photometry_column("W4", error=True)]
    w4lim = table[phot.name_photometry_column("W4", limit=True)]

    dustless_indices = atlas3d.color_cut_dustless_indices(
        w1mag, w3mag, w4mag)
    dustless_table = table[dustless_indices]
    dustless_ks = dustless_table[phot.name_photometry_column("Ks")]
    dustless_ks_err = dustless_table[
        phot.name_photometry_column("Ks", error=True)]
    dustless_ks_lim = dustless_table[
        phot.name_photometry_column("Ks", limit=True)]
    dustless_w1 = dustless_table[phot.name_photometry_column("W1")]
    dustless_w1_err = dustless_table[
        phot.name_photometry_column("W1", error=True)]
    dustless_w1_lim = dustless_table[
        phot.name_photometry_column("W1", limit=True)]
    dustless_w2 = dustless_table[phot.name_photometry_column("W2")]
    dustless_w4 = dustless_table[phot.name_photometry_column("W4")]
    dustless_w4_err = dustless_table[
        phot.name_photometry_column("W4", error=True)]
    dustless_w4_lim = dustless_table[
        phot.name_photometry_column("W4", limit=True)]

    dustless_klum = conv.ABmag2inbandLum(
        "Ks", dustless_ks, dustless_table["D"])
    dustless_klum_err = conv.AB_mag_err_to_inband_lum_err(
        "Ks", dustless_ks, dustless_ks_err, dustless_table["D"],
        dustless_table["D_err"])
    dustless_klum_lim = stat.invert_limits(dustless_ks_lim)
    dustless_w1lum = conv.ABmag2inbandLum(
        "W1", dustless_w1, dustless_table["D"])
    dustless_w1lum_err = conv.AB_mag_err_to_inband_lum_err(
        "W1", dustless_w1, dustless_w1_err, dustless_table["D"],
        dustless_table["D_err"])
    dustless_w1lum_lim = stat.invert_limits(dustless_w1_lim)
    dustless_w4lum = conv.ABmag2specLum(
        "W4", dustless_w4, dustless_table["D"])
    dustless_w4lum_err = conv.AB_mag_err_to_spec_lum_err(
        "W4", dustless_w4, dustless_w4_err, dustless_table["D"],
        dustless_table["D_err"])
    dustless_w4lum_lim = stat.invert_limits(dustless_w4_lim)
    dustless_w1loglum, dustless_w1loglum_err, dustless_w1loglum_lim = \
        stat.logarithm(dustless_w1lum, dustless_w1lum_err, 
                       numlim=dustless_w1lum_lim)
    dustless_w4loglum, dustless_w4loglum_err, dustless_w4loglum_lim = \
        stat.logarithm(dustless_w4lum, dustless_w4lum_err, 
                       numlim=dustless_w4lum_lim)


    dusty_indices = get_complement_indices(dustless_indices, len(table))
    dusty_table = table[dusty_indices]
    dusty_ks = dusty_table[phot.name_photometry_column("Ks")]
    dusty_ks_err = dusty_table[
        phot.name_photometry_column("Ks", error=True)]
    dusty_ks_lim = dusty_table[
        phot.name_photometry_column("Ks", limit=True)]
    dusty_w1 = dusty_table[phot.name_photometry_column("W1")]
    dusty_w1_err = dusty_table[
        phot.name_photometry_column("W1", error=True)]
    dusty_w1_lim = dusty_table[
        phot.name_photometry_column("W1", limit=True)]
    dusty_w2 = dusty_table[phot.name_photometry_column("W2")]
    dusty_w4 = dusty_table[phot.name_photometry_column("W4")]
    dusty_w4_err = dusty_table[
        phot.name_photometry_column("W4", error=True)]
    dusty_w4_lim = dusty_table[
        phot.name_photometry_column("W4", limit=True)]

    dusty_klum = conv.ABmag2inbandLum(
        "Ks", dusty_ks, dusty_table["D"])
    dusty_klum_err = conv.AB_mag_err_to_inband_lum_err(
        "Ks", dusty_ks, dusty_ks_err, dusty_table["D"],
        dusty_table["D_err"])
    dusty_klum_lim = stat.invert_limits(dusty_ks_lim)
    dusty_w1lum = conv.ABmag2inbandLum(
        "W1", dusty_w1, dusty_table["D"])
    dusty_w1lum_err = conv.AB_mag_err_to_inband_lum_err(
        "W1", dusty_w1, dusty_w1_err, dusty_table["D"],
        dusty_table["D_err"])
    dusty_w1lum_lim = stat.invert_limits(dusty_w1_lim)
    dusty_w4lum = conv.ABmag2specLum(
        "W4", dusty_w4, dusty_table["D"])
    dusty_w4lum_err = conv.AB_mag_err_to_inband_lum_err(
        "W4", dusty_w4, dusty_w4_err, dusty_table["D"],
        dusty_table["D_err"])
    dusty_w4lum_lim = stat.invert_limits(dusty_w4_lim)
    dusty_w1loglum, dusty_w1loglum_err, dusty_w1loglum_lim = \
        stat.logarithm(dusty_w1lum, dusty_w1lum_err, 
                       numlim=dusty_w1lum_lim)
    dusty_w4loglum, dusty_w4loglum_err, dusty_w4loglum_lim = \
        stat.logarithm(dusty_w4lum, dusty_w4lum_err, 
                       numlim=dusty_w4lum_lim)
    
    stat.errorbar(
        dusty_w1loglum, dusty_w4loglum, dusty_w4loglum_err, 
        dusty_w1loglum_err, dusty_w4lum_lim, 'bo', label="Dusty color")
    stat.errorbar(
        dustless_w1loglum, dustless_w4loglum, dustless_w4loglum_err, 
        dustless_w1loglum_err, dustless_w4lum_lim, 'ro',
        label="Dustless color")

    logw1range_inband = np.linspace(9.5, 12.0, 2)
    w1w2color = conv.AB2Vegacolor("W1", "W2", np.mean(w1mag-w2mag))
    for logsSFR in [-10, -11, -12]:
        logw4lum = (logsSFR + 7.3 + (-2.54*w1w2color - 0.17) +
                    logw1range_inband)/0.82 + np.log10(conv.SOLAR_LUMINOSITY)
        plt.plot(logw1range_inband, logw4lum,
                 label="log sSFR={0}".format(logsSFR))

    p1 = models.Linear1D(1, 30)
    pfit = fitting.LinearLSQFitter()
    w1loglumdet, w1loglumdet_err = stat.get_lim(
        dustless_w1loglum, vallim=dustless_w4loglum_lim)
    w4loglumdet, w4loglumdet_err = stat.get_lim(
        dustless_w4loglum, vallim=dustless_w4loglum_lim)
    w1w4relation = pfit(p1, w1loglumdet, w4loglumdet)
    plt.plot(logw1range_inband, w1w4relation(logw1range_inband), 'k-')
    modelshow = "Fit: log L_W4 = {0:0.2f} log L_W1 + {1:0.2f}"
    print modelshow.format(w1w4relation.slope.value,
                           w1w4relation.intercept.value)
    modelw4data = w1w4relation(w1loglumdet)
    disp = np.std(modelw4data - w4loglumdet)
    print "Dispersion is: {0:0.2f}".format(disp)
    

    plt.xlim(9.5, 12)
    plt.ylim(39.9, 43.7)
    plt.xlabel("log L_W1 [Lsun]")
    plt.ylabel("log vL_W4 [erg/s]")
    plt.legend(loc="lower right")
    plt.savefig(dest)
    plt.close()

def create_comparison_plot(wisetable=jarrett_table, galextable=gdp_table,
                           dest=build_filepath(FIGUREPATH, "photocomp", EXT)):
    smallobjs = ["NGC584", "NGC777", "NGC4486"]

    usableindices = np.where(np.logical_and(
        wisetable["objstr_01"] != "NGC5194", wisetable["objstr_01"] !=
        "NGC5195"))
    usabletable = wisetable[usableindices]
    smallindices = np.searchsorted(usabletable["objstr_01"], smallobjs)
    orig_fluxes = usabletable
    my_mags = usabletable
    specindices = np.ones(len(orig_fluxes))*-2

    orig_w1 = conv.Jansky2Vegamag("W1", orig_fluxes["W1"], specindices)
    orig_w1_err = conv.Jansky_err_to_mag_err("W1", orig_fluxes["W1"], 
                                  orig_fluxes["W1_err"])
    myw1 = my_mags["w1apmag"]
    myw1err = my_mags["w1aperr"]
    orig_w2 = conv.Jansky2Vegamag("W2", orig_fluxes["W2"], specindices)
    orig_w2_err = conv.Jansky_err_to_mag_err("W2", orig_fluxes["W2"], 
                                  orig_fluxes["W2_err"])
    myw2 = my_mags["w2apmag"]
    myw2err = my_mags["w2aperr"]
    orig_w3 = conv.Jansky2Vegamag("W3", orig_fluxes["W3"], specindices)
    orig_w3_err = conv.Jansky_err_to_mag_err("W3", orig_fluxes["W3"], 
                                  orig_fluxes["W3_err"])
    myw3 = my_mags["w3apmag"]
    myw3err = my_mags["w3aperr"]
    orig_w4 = conv.Jansky2Vegamag("W4", orig_fluxes["W4"], specindices)
    orig_w4_err = conv.Jansky_err_to_mag_err("W4", orig_fluxes["W4"], 
                                  orig_fluxes["W4_err"])
    myw4 = my_mags["w4apmag"]
    myw4err = my_mags["w4aperr"]

    # I'm throwing away the limits because they're not useful.
    w1diff, w1differr, _ = stat.subtract(myw1, orig_w1,
        myw1err, orig_w1_err)
    w2diff, w2differr, _ = stat.subtract(myw2, orig_w2,
        myw2err, orig_w2_err)
    w3diff, w3differr, _ = stat.subtract(myw3, orig_w3,
        myw3err, orig_w3_err)
    w4diff, w4differr, _ = stat.subtract(myw4, orig_w4,
        myw4err, orig_w4_err)
    # The more equivalent dataset would be the Bai et al. dataset.
    nuvdiff, nuvdifferr, _ = stat.subtract(
        galextable["NUVunextmag"], galextable["D25NUV_2"],
        galextable["NUVunexterr"], galextable["e_D25NUV_2"])
    fuvdiff, fuvdifferr, _ = stat.subtract(
        galextable["FUVunextmag"], galextable["D25FUV_2"],
        galextable["FUVunexterr"], galextable["e_D25FUV_2"])

    plt.figure(figsize=(15,5))
    plt.subplot(1, 3, 1)
    phot.doubleDifferencePlot(
        myw1, orig_w1, myw2, orig_w2, myw1err, orig_w1_err, myw2err, 
        orig_w2_err, "W1 Difference", "W2 Difference", "", fmt="b.")
    phot.doubleDifferencePlot(
        myw1[smallindices], orig_w1[smallindices], myw2[smallindices],
        orig_w2[smallindices], myw1err[smallindices],
        orig_w1_err[smallindices], myw2err[smallindices],
        orig_w2_err[smallindices], "W1 Difference", "W2 Difference", "", 
        fmt="r.")
    plt.subplot(1, 3, 2)
    phot.doubleDifferencePlot(
        myw3, orig_w3, myw4, orig_w4, myw3err, orig_w3_err, myw4err, 
        orig_w4_err, "W1 Difference", "W2 Difference", "", fmt="b.")
    phot.doubleDifferencePlot(
        myw3[smallindices], orig_w3[smallindices], myw2[smallindices],
        orig_w2[smallindices], myw3err[smallindices],
        orig_w3_err[smallindices], myw4err[smallindices],
        orig_w4_err[smallindices], "W3 Difference", "W4 Difference", "", 
        fmt="r.")
    plt.subplot(1, 3, 3)
    phot.doubleDifferencePlot(
        galextable["NUVapmag"], galextable["D25NUV_1"], galextable["FUVapmag"],
        galextable["D25FUV_1"], galextable["NUVaperr"], 
        galextable["e_D25NUV_1"], galextable["FUVaperr"], 
        galextable["e_D25FUV_1"], "NUV (mine) - NUV (survey)", 
        "FUV (mine) - FUV (survey)", "", label="Gil de Paz", fmt="b.")
    phot.doubleDifferencePlot(
        galextable["NUVunextmag"], galextable["D25NUV_2"], 
        galextable["FUVunextmag"], galextable["D25FUV_2"], 
        galextable["NUVunexterr"], galextable["e_D25NUV_2"],
        galextable["FUVunexterr"], galextable["e_D25FUV_2"], 
        "NUV (mine) - NUV (survey)", "FUV (mine) - FUV (survey)", "", 
        label="Bai", fmt="r.")
    plt.legend(loc="upper left")
    plt.savefig(dest)
    plt.close()

    print "W1 Difference: {0:.2g}".format(np.std(w1diff))
    print "W2 Difference: {0:.2g}".format(np.std(w2diff))
    print "W3 Difference: {0:.2g}".format(np.std(w3diff))
    print "W4 Difference: {0:.2g}".format(np.std(w4diff))
    print "NUV Difference: {0:.2g}".format(np.std(nuvdiff))
    print "FUV Difference: {0:.2g}".format(np.std(fuvdiff))

def separate_passive_galaxies(totaltable):
    '''Removes galaxies which have indications of dust in them.

    Totaltable should contain only ATLAS3D and Rampazzo galaxies. It will then
    return only the galaxies which do not have dust in them according to
    lacking visible dust lanes or molecular hydrogen clouds for ATLAS3D 
    galaxies, or the Class-0 galaxies for Rampazzo galaxies.
    '''
    atlasgals = atlas3d.filter_ATLAS3D_table_for_dustless_galaxies(totaltable)
    try:
        rampgals = rp.extract_MIR_class_sample(totaltable, 0)
    except KeyError:
        pass

    newgals = vstack([atlasgals, rampgals])
    uniqgals = remove_duplicate_galaxies(newgals)
    return uniqgals

def get_complement_indices(initindices, tablelength):
    '''Returns the indices corresponding to rows not in partialtable.

    This function essenially creates indices which correspond to the rows in
    totaltable rows not in partialtable.
    '''
    compmask = np.ones(tablelength, np.bool)
    compmask[initindices] = 0
    return np.where(compmask)

def get_complement_table(partialtable, totaltable):
    '''Returns a subtable of total table without rows in partialtable.

    This is kinda like an operation to create a table which when stacked with
    partialtable and sorted by objstr_01, will create totaltable.
    '''
    partialindices = phot.astropy_table_indices(totaltable, "objstr_01",
                                                partialtable["objstr_01"])
    compmask = get_complement_indices(partialindices, len(totaltable))
    comp_sample = totaltable[compmask]
    return comp_sample

def find_excluded_galaxies(names, xvalues, yvalues):
    '''Will print out which galaxies in the sample were excluded from the plot.

    Names should be the names of the galaxies. Xvalues should be the x-values
    done in the plot, and yvalues shoul be the y-values done in the plot. It
    will then automatically find which galaxies are outsie of the boundaries of
    the plot.
    '''
    pass

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
    atlas3dsample = Table.read(tablepath, format="ascii.fixed_width", 
                               data_start=3, guess=False)
    return atlas3dsample

def read_Cappellari13a_Table_1(
    tablepath=os.path.join(ATLAS3DBASE, "Cappellari13_Table_1_XV.txt")):
    '''Reads in the first table from Cappellari 2013 (ATLAS3D XV).'''
    atlas3dsample = Table.read(tablepath, format="ascii.fixed_width",
                               data_start=3, guess=False, 
                               fill_values=[("", 0), ("--", 0)])
    return atlas3dsample

def read_Cappellari13b_Table_1(
    tablepath=os.path.join(ATLAS3DBASE, "Cappellari13_Table_1_XX.txt")):
    '''Reads in the first table from Cappellari 2013 (ATLAS3D XX).'''
    atlas3dsample = Table.read(tablepath, format="ascii.fixed_width",
                               data_start=3, guess=False,
                               fill_values=[("",0), ("----", 0)])
    return atlas3dsample

def read_Young_Table_1(
    tablepath=os.path.join(BASEPATH, "Young11_Table1.txt")):
    '''Reads in the CO measurement table from Young et al (2011).'''
    cosample = Table.read(
        tablepath, format="ascii.fixed_width", names=(
            "Galaxy", "rms(1-0)", "rms(2-1)", "Start", "End", "I(1-0)", 
            "I(1-0) unc", "I(2-1)", "I(2-1) unc", "2-1/1-0", "2-1/1-0 unc",
            "Src", "log M(H2)", "log M(H2) unc"), guess=False,
        fill_values=[("",0), ("...", 0), ("...   ...", 0)])
    separate_limit(cosample, ["2-1/1-0", "log M(H2)"])
    return cosample

def read_McDermid15_Table_3(
    tablepath=os.path.join(ATLAS3DBASE, 
                           "McDermid2015_Atlas3D_Paper30_Table3.txt")):
    '''Reads in the third table from McDermid 2015.'''
    atlas3dsample = Table.read(tablepath, format="ascii.fixed_width", 
                               data_start=3, guess=False)
    atlas3dsample = separate_errors_in_table(atlas3dsample)
    return atlas3dsample

def read_McDermid_Table_4(
        URL=("/home/regulus/simonian/year1/wise/ATLAS3D_DB/"
             "McDermid2015_Atlas3D_Paper30_Table4.txt")):
    '''Reads in the table from McDermid 2015

    This table contains all of the early-type galaxies in the ATLAS3D sample,
    as well as their properties as measured by the Re aperture.'''
    mcdermid_raw_table = Table.read(
        URL, format="ascii.basic", data_start=0, header_start=None, 
        fill_values=("--", "0"))
    mcdermid_table = Table(
        mcdermid_raw_table[[
            "col1", "col2", "col4", "col5", "col7", "col8", "col10"]],
        names=(
            "Name", "Age_SFH", "Age_SFH_err", "[Z/H]_SFH", "[Z/H]_SFH_err", 
            "t50", "t50_err"))

    return mcdermid_table

def read_Krajnovic_Table_D1(
        URL=("/home/regulus/simonian/year1/wise/ATLAS3D_DB/"
             "Krajnovic2011_Atlas3D_Paper2_TableD1.txt")):
    '''Reads the table from Krajnovich 2011

    In particular, this table contains information about dust.'''
    krajnovic_table = Table.read(
        URL, format="ascii.commented_header", guess=False, header_start=-5, 
        data_start=0)
    return krajnovic_table

def read_Gil_de_Paz_Table_1(
    URL=("/home/regulus/simonian/year1/wise/"
         "Gil_de_Paz_Table1.txt")):
    '''Reads in Table 1 from Gil de Paz (2007).

    This table mainly contains the coordinates and other basic information
    about the galaxy.
    '''
    gdptable = Table.read(URL, format="ascii.cds")
    return gdptable

def read_Gil_de_Paz_Table_2(
    URL=("/home/regulus/simonian/year1/wise/"
         "Gil_de_Paz_Table2.txt")):
    '''Reads in Table 2 from Gil de Paz (2007).

    This table mainly contains information about the GALEX observations.
    about the galaxy.
    '''
    gdptable = Table.read(URL, format="ascii.cds")
    return gdptable

def read_Gil_de_Paz_Table_3(
    URL=("/home/regulus/simonian/year1/wise/"
         "Gil_de_Paz_Table3.txt")):
    '''Reads in Table 3 from Gil de Paz (2007).

    This table contains the magnitudes of all of the galaxies.
    '''
    gdptable = Table.read(URL, format="ascii.cds")
    return gdptable

def read_Bai_Table_2(
    URL=("/home/regulus/simonian/year1/wise/Bai_Table2.txt")):
    '''Reads Table 2 from Bai et al. (2015).

    This table contains the magnitudes of all the galaxies.
    '''
    gdptable = Table.read(URL, format="ascii.cds")
    return gdptable

def read_Amblard_Table_4(
    URL=("/home/regulus/simonian/year1/wise/Amblard_Table4.txt")):
    '''Reads Table 4 from Amblard et al (2014).

    This table contains 250um, 350um, and 500um detections for galaxies from
    the Amblard et al. sample.
    '''
    amblard_table = Table.read(URL, format="ascii.tab", header_start=2,
                               data_start=4, guess=False)
    revised_amblard_table = separate_errors_in_table(amblard_table,
                                                     seperator="+or-")
    return revised_amblard_table

def separate_errors_in_table(fulltable, seperator="+/-", suffix="_err",
                             mask="--"):
    '''Formats table to have separate error column.
    
    The separator acts as a delimiter between the value and the error. The
    suffix is added to the end of the column name to create a new column.
    This returns another table.
    '''
    newtable = Table(masked=True)
    for colname in fulltable.colnames:
        col = fulltable[colname]
        if np.issubdtype(col.dtype, np.str):
            splitcol = npstr.split(col, sep=seperator)
            if len(splitcol[0]) == 1:
                newtable[colname] = collapse_list_nested_array(splitcol)
            elif len(splitcol[0]) == 2:
                # This takes advantage of the fact that supplying an array to
                # "array" yields the same array of lists. While adding a list to 
                # "array" yields a 2-d array.
                combinedarray = np.array(list(splitcol))
                maskedvalarray = npstr.replace(
                    combinedarray[:,0], mask, 'NaN')
                floatvalarray = maskedvalarray.astype(np.float)
                newtable[colname] = np.ma.masked_invalid(floatvalarray)
                errcolname = "{0}{1}".format(colname, suffix)
                maskederrarray = npstr.replace(
                    combinedarray[:,1], mask, 'NaN')
                floaterrarray = maskederrarray.astype(np.float)
                newtable[errcolname] = np.ma.masked_invalid(floaterrarray)
            else:
                raise ValueError("Can't parse errors in Table.")
        else:
            newtable[colname] = col
    return newtable

def collapse_list_nested_array(arr):
    '''Collapses an array of singleton lists.

    This will turn the array([1], [2], [3], [4], [5]) to array([1, 2, 3, 4,
    5]).'''
    return np.hstack(arr)

def separate_limit(table, limcols, updelim="<", lowdelim=">", eqdelim="=",
                   coltemplate="{0} lim"):
    '''Takes limcols from a table and splits them into limit columns.

    For all columns in the list of limcols, this function will split them into
    a limit column and a numerical value column. The column will change dtype
    to be numerical. The limit will have a column name as determined by
    coltemplate, which shoul be a format string which takes the column name as
    the first argument.
    '''
    for col in limcols:
        strcol = table[col]
        valcol, limcol = split_limit_col(strcol, updelim, lowdelim, eqdelim)
        del(table[col])
        table[col] = valcol
        table[coltemplate.format(col)] = limcol

def split_limit_col(initcol, updelim="<", lowdelim=">", eqdelim="=",
                    dtype=np.float):
    '''Splits a column into a limit and numerical value column.

    One problem with table representations of limits is that the symbols for
    limits cause the columns to be represented as a string, not as a numerical
    limit. Therefore, this function splits a string column into two arrays:
    one with a limit representation, another with the numerical values.
    '''
    limcol = stat.generate_limit(None, len(initcol))
    try:
        upperindices = np.where(npstr.startswith(initcol, updelim))
    except TypeError:
        print "{0} is not a string column. Ignoring.".format(initcol.name)
    initcol = npstr.lstrip(initcol, updelim)
    lowerindices = np.where(npstr.startswith(initcol, lowdelim))
    initcol = npstr.lstrip(initcol, lowdelim)
    if eqdelim is not "":
        eqindices = np.where(npstr.startswith(initcol, eqdelim))
        initcol = npstr.lstrip(initcol, eqdelim)
    newcol = np.asanyarray(initcol, dtype=dtype)
    limcol[upperindices] = stat.UPPER
    limcol[lowerindices] = stat.LOWER
    # If there is a mask, then we want to ensure that the masked values are
    # considered to be invalid data points.
    try:
        limcol[initcol.mask] = stat.NA
    except AttributeError:
        pass

    return newcol, limcol

def read_Diamond_Stanic_Table1(
    tablepath=os.path.join(BASEPATH, "Diamond_Stanic_Table1.txt")):
    '''Reads the first table from Diamond-Stanic 2010.'''
    names = ["Name", "6.2 um", "6.2 um err", "7.7 um", "7.7 um err", "8.6 um",
             "8.6 um err", "11.3 um", "11.3 um err", "12.7 um", "12.7 um err",
             "[Ne II]", "[Ne II] err", "H2 S(3)", "H2 S(3) err"]
    dstable = Table.read

def generate_ATLAS3D_distance_errors(atlas3d_table_3=None):
    '''Creates a Column of ATLAS3D distance errors.

    It will take an instance of Table 3 from Cappellari et al (2011). However,
    if none is provided, it will read it on its own.
    '''
    at3 = atlas3d_table_3
    if at3 is None:
        at3 = read_Cappellari11_Table_3()
    # These galaxies are going to be ordered from most-to-least precise
    # distance determinations
    at3_group = at3.group_by("SBF")
    # When SBF=2, then the distances come from Mei et al (2007)
    meigals = at3_group.groups[2]
    meigals["D_err"] = 0.03 * meigals["D"]
    # Next precise is for galaxies which are in Virgo, so we'll make a table
    # for the non-Mei galaxies
    nonacs = at3_group.groups[0:2]
    nonacs_group = nonacs.group_by("Virgo")
    # When SBF=0 and NED-D=0, then if the galaxy is in Virgo, it gets the
    # distance to Virgo.
    virgogals = nonacs_group.groups[1]
    virgogals["D_err"] = 0.07 * virgogals["D"]
    # Next up is the Tonry et al (2001) paper, which is signified by SBF=1.
    nonvirgo = nonacs_group.groups[0]
    nonvirgo_group = nonvirgo.group_by("SBF")
    # The Tonry et al (2001) galaxies are the ones where SBF=1
    tonrygals = nonvirgo_group.groups[1]
    tonrygals["D_err"] = 0.10 * tonrygals["D"]
    # When SBF=0, we have multiple cases.
    nonSBF = nonvirgo_group.groups[0]
    nonSBF_group = nonSBF.group_by("NED-D")
    # When SBF=0 and NED-D > 0, then distance was taken from NED-D catalog.
    # There are two sets of methods which are good to ~10 percent, an <~20
    # percent. I'm gonna choose 15 percent just for current simplicity's sake.
    NEDgals = nonSBF_group.groups[1:]
    NEDgals["D_err"] = 0.15 * NEDgals["D"]
    # We took care of the 5 cases. Now for the rest which are only avaialble
    # through cosmic flow velocities.
    nonNEDgals = nonSBF_group.groups[0]
    nonNEDgals["D_err"] = 0.21 * nonNEDgals["D"]

    distance_error_table = vstack([meigals, virgogals, tonrygals, NEDgals,
                                   nonNEDgals])[["Galaxy", "D_err"]]
    distance_error_table["D_err"] /= np.log(10)
    full_table = phot.join_by_galaxy_name(
        at3, distance_error_table, names=("Galaxy", "Galaxy"), join_type="left")
    return full_table



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

def format_GALEX_tile(tilename):
    '''Tilenames contain underscores which cause errors in LaTeX.

    This function will escape the underscores.
    '''
    if tilename is None:
        return "--"
    return tilename.replace("_", r"\_")

def format_sample(sampstr):
    '''Condenses the string corresponding to the sample to a single letter.'''

    if sampstr == "Rampazzo":
        return "R"
    elif sampstr == "ATLAS3D":
        return "A"
    elif sampstr == "Both":
        return "R,A"
    else:
        raise ValueError("Don't recognize the sample")


def format_mag(mag):
    '''Format magnitudes so that they can be displayed on a table.

    This essentially breaks off the magnitude at the second decimal place. It
    is meant to be used with the format keyword in Table.write().
    '''
    try:
        magstr = "{0:.2f}".format(mag)
    except ValueError:
        # May occur for masked values. 
        magstr = LATEX_TABLE_MASKSTRING
    return magstr

def format_mag_column(mags, limits):
    '''Convert mag column to column of strings formatted correctly for limits.

    This function basically converts the float array of magnitudes into string
    arrays of well-formatted magnitudes. By well-formatted, this means the
    numbers are truncated in the right location, and the limits are prepended
    with "<" or ">" appropriately.
    '''
    magstr = np.ma.asanyarray(mags, 'a5')
    upperlims = np.where(limits == stat.UPPER)
    lowerlims = np.where(limits == stat.LOWER)
    magstr[upperlims] = npstr.add("<", magstr[upperlims])
    magstr[lowerlims] = npstr.add(">", magstr[lowerlims])
    return magstr

def format_mag_err(err):
    '''Format magnitude errors so that they can be displayed on a table.

    This essentially breaks off the error at the third decimal place. It is
    meant to be used with the format keyword in Table.write().
    '''
    try:
        magerrstr = "{0:.3f}".format(err)
    except ValueError:
        # May occur for masked values. 
        magerrstr = LATEX_TABLE_MASKSTRING
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
