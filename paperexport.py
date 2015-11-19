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
from astropy.io.ascii import masked
import numpy as np

import photometry as phot
import queries
import rampazzo_plots as rp
import fsps
import band_conversions as conv
import atlas3d
import statop as stat

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
            addrow = tablegroup[0]
            if len(tablegroup) == 2:
                addrow["sample"] = "Both"
            elif len(tablegroup) > 2:
                raise ValueError("Some table has more than three entries.")
            fulltable.add_row(addrow)
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

    generate_fulltable()
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
    preamble = r"""\tabletypesize{\scriptsize}"""
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
    MIR = table["MIR_class"]

    w1w2, w1w2err = stat.subtract(w1, w2, w1err, w2err)
    w2w3, w2w3err = stat.subtract(w2, w3, w2err, w3err)

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
    MIR = table["MIR_class"]

    w1w2, w1w2err = stat.subtract(w1, w2, w1err, w2err)
    w2w3, w2w3err = stat.subtract(w2, w3, w2err, w3err)
    w3w4, w3w4err = stat.subtract(w3, w4, w3err, w4err)

    w1w2label = "W1-W2"
    w2w3label = "W2-W3"
    w3w4label = "W3-W4"

    plt.subplot(1, 2, 1)
    rp.MIRplot(w2w3, w1w2, MIR, w1w2err, w2w3err, xrange(5), w2w3label,
               w1w2label, "", loc="upper left")
    plt.axis([-1.5, 0, -0.75, -0.6])

    plt.subplot(1, 2, 2)
    rp.MIRplot(w3w4, w2w3, MIR, w2w3err, w3w4err, xrange(5), w3w4label,
               w2w3label, "", loc="upper left")
    plt.axis([-1.5, 0, -1.5, 0])
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
    atlas3d_params = read_Cappellari11_Table_3()
    atlas3d_params = generate_ATLAS3D_distance_errors(atlas3d_params)
    atlas3d_lums = read_Cappellari13a_Table_1()
    atlas3d_masstolight = read_Cappellari13b_Table_1()
    joinedtable = phot.multijoin_by_galaxy_name(
        table, atlas3d_params, atlas3d_masstolight, atlas3d_lums,
        names=("objstr_01", "Galaxy", "Galaxy", "Galaxy"))

    atlas3d_stellar_mass, atlas3d_stellar_mass_err = stat.add(
        joinedtable["logML_star"], joinedtable["logLum"], 0.06/np.log(10),
        0.1/np.log(10))
    distance_modulus, distance_modulus_err = stat.logarithm(
        joinedtable["D"]*1e5, joinedtable["D_err"]*1e5)
    absolute_w1, absolute_w1_err = stat.subtract(
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

def create_cutout_grid(
        table=atlas3d_table[:2], dest=os.path.join(FIGUREPATH, "cutouts.png")):
    gridfig = phot.ellipse_cutout_grid(
        ATLAS3DBASE, table, ignore_exception=True)
    gridfig.savefig(dest)
    plt.close(gridfig)


def create_mass_to_light_ATLAS3D_comparison(
        table=atlas3d_table, dest=os.path.join(FIGUREPATH, "masstolight.pdf")):
    atlas3d_params = read_Cappellari11_Table_3()
    atlas3d_params = generate_ATLAS3D_distance_errors(atlas3d_params)
    atlas3d_lums = read_Cappellari13a_Table_1()
    atlas3d_masstolight = read_Cappellari13b_Table_1()
    joinedtable = phot.multijoin_by_galaxy_name(
        table, atlas3d_params, atlas3d_masstolight, atlas3d_lums,
        names=("objstr_01", "Galaxy", "Galaxy", "Galaxy"))
    # Calculate the mass-to-light ratio for the ATLAS3D points in WISE.
    atlas3d_masstolight_w1, atlas3d_masstolight_w1_err = \
        atlas3d.atlas3d_ml_to_wise_ml(
            joinedtable["logML_star"], joinedtable["logLum"],
            joinedtable["w1unextmag_atlas3d"], joinedtable["D"],
            0.06/np.log(10), 0.1/np.log(10), joinedtable["w1unexterr_atlas3d"],
            joinedtable["D_err"])
    w1w2, w1w2_err = stat.subtract(
        joinedtable["w1unextmag"], joinedtable["w2unextmag"],
        joinedtable["w1unexterr"], joinedtable["w2unexterr"])
    plt.errorbar(w1w2, atlas3d_masstolight_w1, atlas3d_masstolight_w1_err,
                 w1w2_err, 'k*', label="ATLAS3D")

    # Remember that these will be given in Vega mags.
    w1w2_limits = np.linspace(-0.8+0.01, -0.2-0.01, 2)
    w1w2_limits_VEGA = w1w2_limits +0.64
    # Relations from Jarrett et al 2013
    jarrett_relation = -0.31 + 3.42 * w1w2_limits_VEGA
    jarrett_fit_relation = -0.246 - 2.100 * w1w2_limits_VEGA
    jarrett_fit_lower = (-0.246-0.027) - (2.100+0.238) * w1w2_limits_VEGA
    jarrett_fit_upper = (-0.246+0.027) - (2.100-0.238) * w1w2_limits_VEGA
    plt.plot(w1w2_limits, jarrett_relation, 'r-', label="Jarrett M/L")
    plt.plot(w1w2_limits, jarrett_fit_relation, 'm-', label="Jarrett M/L fit")
    plt.plot(w1w2_limits, jarrett_fit_upper, 'm--')
    plt.plot(w1w2_limits, jarrett_fit_lower, 'm--')
    # Relations from Meidt et al 2014
    meidt_relation = 0.07 + 3.98 * w1w2_limits_VEGA
    meidt_upper = 0.15 + 4.96 * w1w2_limits_VEGA
    meidt_lower = 0.01 + 3.00 * w1w2_limits_VEGA
    plt.plot(w1w2_limits, meidt_relation, 'g-', label="Meidt M/L")
    plt.plot(w1w2_limits, meidt_upper, 'g--')
    plt.plot(w1w2_limits, meidt_lower, 'g--')
    # Finally Eskew
    eskew_relation = 0.28 - 0.74 * w1w2_limits_VEGA
    plt.plot(w1w2_limits, eskew_relation, 'b-', label="Eskew M/L") 

    plt.xlabel("W1-W2 (AB)")
    plt.ylabel("(M/L)_W1")
    # Not shown is PGC029321 all the way to the right.
    plt.xlim([-0.8, -0.2])
    plt.ylim([-1.0, 1.0])
    plt.legend(loc="upper right")
    plt.savefig(dest)
    plt.close()

def create_NUV_W1_abs_plot(rtable=rampazzo_table, atable=atlas3d_table, 
        dest=os.path.join(FIGUREPATH, "nuvw1.pdf")):
    rnuv = rtable["NUVunextmag"]
    rnuv_e = rtable["NUVunexterr"]
    rw1 = rtable["w1unextmag"]
    rw1_e = rtable["w1unexterr"]
    rd = rtable["D"]

    rnuvw1, rnuvw1_e = stat.subtract(
            rnuv, rw1, rnuv_e, rw1_e)

    rw1abs = conv.app2absmag(rw1, rd)
    # This until I figure out what to do with distance errors.
    rw1abs_e = rw1_e

    anuv = atable["NUVunextmag"]
    anuv_e = atable["NUVunexterr"]
    aw1 = atable["w1unextmag"]
    aw1_e = atable["w1unexterr"]
    ad = atable["D"]
    ad_e = atable["D_err"]

    anuvw1, anuvw1_e = stat.subtract(
            anuv, aw1, anuv_e, aw1_e)
    aw1abs = conv.app2absmag(aw1, ad)
    # This until I figure out what to do with distance errors.
    aw1abs_e = conv.app_err_to_abs_err(aw1_e, ad, ad_e)

    plt.errorbar(aw1abs, anuvw1, anuvw1_e, aw1abs_e, 'kx', label="ATLAS3D")
    rp.MIRplot(rw1abs, rnuvw1, rtable["MIR_class"], rnuvw1_e, xlabel="M_{W1}",
               ylabel="NUV-W1")

    #plt.ylim([2, 7])
    #plt.xlim([-18, -24])
    plt.legend(loc="lower right", bbox_to_anchor=(0.85, 0.01))
    print "Not shown are NGC 4406 and NGC 4429"
    #plt.savefig(dest)
    #plt.close()

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

    xcolor, xcolorerr = stat.subtract(nuv, j, nuv_err, j_err)
    yratio, yratioerr = stat.divide(pah77, pah113,
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

def create_SED(table=atlas3d_table, dest=build_filepath(FIGUREPATH, "sed",
                                                        EXT)):
    fluxtable = Table(table["objstr_01"])
    table["w1unextmag"] = conv.ABmag2Jansky("W1", table["w1unextmag"])
    table["w1unexterr"] = conv.Mag_err_to_Jansky_err(
        "W1", table["w1unextmag"], table["w1unexterr"])
    table["w2unextmag"] = conv.ABmag2Jansky("W2", table["w2unextmag"])
    table["w2unexterr"] = conv.Mag_err_to_Jansky_err(
        "W2", table["w2unextmag"], table["w2unexterr"])
    table["w3unextmag"] = conv.ABmag2Jansky("W3", table["w3unextmag"])
    table["w3unexterr"] = conv.Mag_err_to_Jansky_err(
        "W3", table["w3unextmag"], table["w3unexterr"])
    table["w4unextmag"] = conv.ABmag2Jansky("W4", table["w4unextmag"])
    table["w4unexterr"] = conv.Mag_err_to_Jansky_err(
        "W4", table["w4unextmag"], table["w4unexterr"])
    table["NUVunextmag"] = conv.ABmag2Jansky("NUV", table["NUVunextmag"])
    table["NUVunexterr"] = conv.Mag_err_to_Jansky_err(
        "NUV", table["NUVunextmag"], table["NUVunexterr"])
    table["FUVunextmag"] = conv.ABmag2Jansky("FUV", table["FUVunextmag"])
    table["FUVunexterr"] = conv.Mag_err_to_Jansky_err(
        "FUV", table["FUVunextmag"], table["FUVunexterr"])

    normval = fsps.plot_FSPS_SED(FSPSPATH)
    fsps.plot_data_SED(fluxtable, normvalue=normval)

    plt.xlabel("Wavelength (um)")
    plt.ylabel("AB magnitude")
    plt.ylim([5, 17])
    plt.gca().invert_yaxis()
    plt.legend(loc="upper left")
    plt.savefig(dest)
    plt.close()

def create_circumstellar_dust_plot(table=atlas3d_table,
                                   dest=build_filepath(FIGUREPATH, "cdust",
                                                       EXT)):
    '''Dual-paneled W1-W3 and W1-W4 vs age plot.'''
    # Set up the data
    dustless_catalog = \
        atlas3d.filter_ATLAS3D_table_for_dustless_galaxies(table)
    w1w3color, w1w3err = stat.subtract(
        dustless_catalog["w1unextmag"], dustless_catalog["w3unextmag"], 
        dustless_catalog["w1unexterr"], dustless_catalog["w3unexterr"])
    w1w4color, w1w4err = stat.subtract(
        dustless_catalog["w1unextmag"], dustless_catalog["w4unextmag"], 
        dustless_catalog["w1unexterr"], dustless_catalog["w4unexterr"])
    atlas3d_ages = dustless_catalog["Age_SSP"]
    atlas3d_ages_err = dustless_catalog["Age_SSP_err"]
    atlas3d_metallicities = dustless_catalog["[Z/H]_SSP"]
    med_met = float(np.ma.median(atlas3d_metallicities))
                                                
    
    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    # Let's make the first plot: W1-W3
    plt.sca(ax1)
    fsps.plot_atlas3d_coded_by_metallicity(
        atlas3d_ages, w1w3color, w1w3err, atlas3d_ages_err,
        atlas3d_metallicities, med_met)
    fsps.plot_dust_toggled_metallicity_bounds(
        os.path.join(fsps.OUTPUT_PATH, "toggle_dust_met_bounds"), "W1", "W3")
    plt.legend(loc="upper right")
    plt.ylabel("W1-W3")

    # Now the second plot: W1-W4
    plt.sca(ax2)
    fsps.plot_atlas3d_coded_by_metallicity(
        atlas3d_ages, w1w4color, w1w4err, atlas3d_ages_err,
        atlas3d_metallicities, med_met)
    fsps.plot_dust_toggled_metallicity_bounds(
        os.path.join(fsps.OUTPUT_PATH, "toggle_dust_met_bounds"), "W1", "W4")
    plt.ylabel("W1-W4")
    plt.xlabel("SSP Age (Gyr)")
    plt.tight_layout(0)
    plt.close()
    plt.savefig(dest)
    # I don't think I need anything below this. But just to be sure, let me
    # check. If it works, delete EVERYTHING here.
    # Set up the FSPS models
    median_met_SSP_file = os.path.join(
        fsps.OUTPUT_PATH, "SSP_med.out.mags")
    low_met_SSP_file = os.path.join(
        fsps.OUTPUT_PATH, "toggle_dust_met_bounds", "dust_lowmet.mags")
    high_met_SSP_file = os.path.join(
        fsps.OUTPUT_PATH, "toggle_dust_met_bounds", "dust_highmet.mags")
    low_met_SSP_nodust_file = os.path.join(
        fsps.OUTPUT_PATH, "toggle_dust_met_bounds", "nodust_lowmet.mags")
    high_met_SSP_nodust_file = os.path.join(
        fsps.OUTPUT_PATH, "toggle_dust_met_bounds", "nodust_highmet.mags")

    median_met_SSP = fsps.read_mags(median_met_SSP_file)
    low_met_SSP = fsps.read_mags(low_met_SSP_file)
    high_met_SSP = fsps.read_mags(high_met_SSP_file)
    low_met_SSP_nodust = fsps.read_mags(low_met_SSP_nodust_file)
    high_met_SSP_nodust = fsps.read_mags(high_met_SSP_nodust_file)
    
    # W1-W3 tracks
    median_met_SSP_w1w3_track = median_met_SSP["W1"] - median_met_SSP["W3"]
    low_met_SSP_w1w3_track = low_met_SSP["W1"] - low_met_SSP["W3"]
    high_met_SSP_w1w3_track = high_met_SSP["W1"] - high_met_SSP["W3"]
    low_met_SSP_nodust_w1w3_track = (low_met_SSP_nodust["W1"] - 
                                     low_met_SSP_nodust["W3"])
    high_met_SSP_nodust_w1w3_track = (high_met_SSP_nodust["W1"] - 
                                      high_met_SSP_nodust["W3"])

    # W1-W4 tracks
    median_met_SSP_w1w4_track = median_met_SSP["W1"] - median_met_SSP["W4"]
    low_met_SSP_w1w4_track = low_met_SSP["W1"] - low_met_SSP["W4"]
    high_met_SSP_w1w4_track = high_met_SSP["W1"] - high_met_SSP["W4"]
    low_met_SSP_nodust_w1w4_track = (low_met_SSP_nodust["W1"] -
                                     low_met_SSP_nodust["W4"])
    high_met_SSP_nodust_w1w4_track = (high_met_SSP_nodust["W1"] -
                                      high_met_SSP_nodust["W4"])

    

def create_jarrett_comparison_plot(table=jarrett_table,
                                   dest=build_filepath(FIGUREPATH, "jarrett",
                                                       EXT)):
    smallobjs = ["NGC584", "NGC777"]

    smallindices = np.searchsorted(table["objstr_01"], smallobjs)
    orig_fluxes = table
    my_mags = table

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

    w1diff, w1differr = stat.subtract(myw1, orig_w1,
        myw1err, orig_w1_err)
    w2diff, w2differr = stat.subtract(myw2, orig_w2,
        myw2err, orig_w2_err)
    w3diff, w3differr = stat.subtract(myw3, orig_w3,
        myw3err, orig_w3_err)
    w4diff, w4differr = stat.subtract(myw4, orig_w4,
        myw4err, orig_w4_err)

    plt.figure(figsize=(10,5))
    plt.subplot(1, 2, 1)
    phot.doubleDifferencePlot(
        myw1, orig_w1, myw2, orig_w2, myw1err, orig_w1_err, myw2err, 
        orig_w2_err, "W1 Difference", "W2 Difference", "", fmt="b.")
    phot.doubleDifferencePlot(
        myw1[smallindices], orig_w1[smallindices], myw2[smallindices],
        orig_w2[smallindices], myw1err[smallindices],
        orig_w1_err[smallindices], myw2err[smallindices],
        orig_w2_err[smallindices], "W1 Difference", "W2 Difference", "", 
        fmt="r.")
    plt.subplot(1, 2, 2)
    phot.doubleDifferencePlot(
        myw3, orig_w3, myw4, orig_w4, myw3err, orig_w3_err, myw4err, 
        orig_w4_err, "W1 Difference", "W2 Difference", "", fmt="b.")
    phot.doubleDifferencePlot(
        myw3[smallindices], orig_w3[smallindices], myw2[smallindices],
        orig_w2[smallindices], myw3err[smallindices],
        orig_w3_err[smallindices], myw4err[smallindices],
        orig_w4_err[smallindices], "W3 Difference", "W4 Difference", "", 
        fmt="r.")
    plt.savefig(dest)
    plt.close()

    print "W1 Difference: {0:.2g}".format(np.std(w1diff))
    print "W2 Difference: {0:.2g}".format(np.std(w2diff))
    print "W3 Difference: {0:.2g}".format(np.std(w3diff))
    print "W4 Difference: {0:.2g}".format(np.std(w4diff))

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
            splitcol = np.core.defchararray.split(col, sep=seperator)
            if len(splitcol[0]) == 1:
                newtable[colname] = collapse_list_nested_array(splitcol)
            elif len(splitcol[0]) == 2:
                # This takes advantage of the fact that supplying an array to
                # "array" yields the same array of lists. While adding a list to 
                # "array" yields a 2-d array.
                combinedarray = np.array(list(splitcol))
                maskedvalarray = np.core.defchararray.replace(
                    combinedarray[:,0], mask, 'NaN')
                floatvalarray = maskedvalarray.astype(np.float)
                newtable[colname] = np.ma.masked_invalid(floatvalarray)
                errcolname = "{0}{1}".format(colname, suffix)
                maskederrarray = np.core.defchararray.replace(
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
