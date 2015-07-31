
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table, join

import photometry as phot
import fsps

MIR_Symbols = {0: {"marker": 'o', "markerfacecolor": 'white', "ls": ' ', 
                   "markeredgewidth": 1.5},
               1: {"marker": '^', "markerfacecolor": 'orange', "ls": ' '},
               2: {"marker": 'o', "markerfacecolor": 'green', "ls": ' '},
               3: {"marker": '*', "markerfacecolor": 'blue', "ls": ' '},
               4: {"marker": 'D', "markerfacecolor": 'white', "ls": ' ',
                   "markeredgecolor": 'red', "markeredgewidth": 1.5}}


def color_histogram_by_class(band1, band2, groupcol, xlabel, title, bins, 
                             colrange=(-4, 4), classes=np.arange(0,5)):
    '''Creates a histogam for colors for different classes.
    '''
    color = band1 - band2
    colorgroup = color.group_by(groupcol)
    plotcolors = ["black", "yellow", "green", "blue", "red"]
    plt.hist([colorgroup.groups[i] for i in classes], bins, range=colrange, 
             label=["Class {0}".format(i) for i in classes], 
             color= [plotcolors[i] for i in classes],
             histtype="bar", rwidth=1)
    plt.xlabel(xlabel)
    plt.ylabel("N")
    plt.title(title)
    plt.legend()

def plot_color_PAH_flux_ratio(pahtable, magtable, title, colorlabel):
    '''Plots color vs a PAH flux ratio.'''
    filledpahs = pahtable.filled(0)
    fulltable = join_by_galaxy_name(magtable, filledpahs, names=("objstr_01",
                                                               "Galaxy"))
    # Arithmetic to take care of the PAH ratio along with the uncertainties.
    pah_numerator = fulltable["7.7 um"] + fulltable["8.6 um"]
    pah_numerator_err = np.sqrt(fulltable["7.7 um err"]**2 + 
                                fulltable["8.6 um err"]**2)
    pah_denominator = (fulltable["11.3 um"] + fulltable["12.7 um"] +
                       fulltable["17 um"])
    pah_denominator_err = np.sqrt(fulltable["11.3 um err"]**2 + 
                                  fulltable["12.7 um err"]**2 +
                                  fulltable["17 um err"]**2)
    pahratio = pah_numerator / pah_denominator
    pahratio_err = (pahratio * 
                    (np.sqrt((pah_numerator_err / pah_numerator)**2 + 
                             (pah_denominator_err / pah_denominator)**2)))

    color = fulltable["w2unextmag"] - fulltable["w3unextmag"]
    color_err = np.sqrt(fulltable["w2unexterr"]**2 + fulltable["w3unexterr"]**2)

    plt.errorbar(color, pahratio, pahratio_err, color_err, '*')
    plt.xlabel(colorlabel)
    plt.ylabel("(f(7.7)+f(8.6))/(f(11.3)+f(12.7)+f(17))")
    plt.title(title)

def MIRplot(x, y, groupkey, xlabel, ylabel, title, yerr=None, xerr=None, 
            loc='upper right'):
    '''Makes a plot that automatically differentiates between MIR classes.

    The x and y data need to be columns which have the same length as 
    groupkey. Groupkey should be the list of MIR classes which are in the same
    order as x and y. Labels can also be added as desired.
    '''
    xgroup = x.group_by(groupkey)
    ygroup = y.group_by(groupkey)
    if yerr is not None:
        yerrgroup = yerr.group_by(groupkey)
    if xerr is not None:
        xerrgroup = xerr.group_by(groupkey)

    for MIRclass in xgroup.groups.keys:
        # Adding errors to this is proving to be much more difficult than
        # expected...
#        try:
#            plt.errorbar(xgroup.groups[MIRclass], ygroup.groups[MIRclass],
#                         yerrgroup.groups[MIRclass], xerrgroup.groups[MIRclass], 
#                         label="Class {0}".format(MIRclass), 
#                         **MIR_Symbols[MIRclass])
#        except UnboundLocalError:
            plt.plot(xgroup.groups[MIRclass], ygroup.groups[MIRclass], 
                     label="Class {0}".format(MIRclass), 
                     **MIR_Symbols[MIRclass])

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc=loc)


def generateCMDs(magtable):
    '''Generates permutations of Color-Magnitude Diagrams.

    The diagrams that are generated should be considered "sensible",
    which means a color between UV and IR, with a magnitude that's 
    either UV or IR.'''
    MIRclass = magtable["MIR class"]
    W1 = magtable["w1unextmag"]
    W2 = magtable["w2unextmag"]
    W3 = magtable["w3unextmag"]
    FUV = magtable["FUVunextmags"]
    NUV = magtable["NUVunextmags"]
    F1color = FUV - W1
    F2color = FUV - W2
    F3color = FUV - W3
    N1color = NUV - W1
    N2color = NUV - W2
    N3color = NUV - W3

    title = "WISE CMD"
    plt.figure()
    MIRplot(W1, F1color, MIRclass, "W1", "FUV-W1", title, loc="lower left")
    plt.figure()
    MIRplot(W2, F2color, MIRclass, "W2", "FUV-W2", title, loc="lower left")
    plt.figure()
    MIRplot(W3, F3color, MIRclass, "W3", "FUV-W3", title)
    plt.figure()
    MIRplot(W1, N1color, MIRclass, "W1", "NUV-W1", title, loc="lower left")
    plt.figure()
    MIRplot(W2, N2color, MIRclass, "W2", "NUV-W2", title, loc="lower left")
    plt.figure()
    MIRplot(W3, N3color, MIRclass, "W3", "NUV-W3", title)

def plot_SSP_color_color(DIR, xblueband, xredband, yblueband, yredband, 
                         fileformat="SSP.out.mags", label="SSP"):
    '''Plots an SSP on a color-color plot.'''
    filename = os.path.join(DIR, fileformat)
    magtable = fsps.read_mags(filename)
    xcolor = magtable[xblueband] - magtable[xredband]
    ycolor = magtable[yblueband] - magtable[yredband]
    plt.plot([xcolor[0]], [ycolor[0]], 'bs')
    plt.plot(xcolor, ycolor, 'b-', label=label)

def generate_color_color(magtable, SSPpath=fsps.OUTPUT_PATH, 
                         SSPfile="SSP.out.mags"):
    '''Generates all color-color plots which could be potentially useful.

    This currently consists of: FUV-NUV, NUV-J, J-Ks, Ks-W1, W1-W2, W2-W3,
    W3-W4.

    In addition to the plots according to MIR class, there will also be SSP
    tracks added to the plots from FSPS
    '''
    MIRclass = magtable["MIR class"]
    W1 = magtable["w1unextmag"]
    W2 = magtable["w2unextmag"]
    W3 = magtable["w3unextmag"]
    W4 = magtable["w4unextmag"]
    J = magtable["j_m_k20fe"]
    H = magtable["h_m_k20fe"]
    K = magtable["k_m_k20fe"]
    FUV = magtable["FUVunextmag"]
    NUV = magtable["NUVunextmag"]

    FNcolor = FUV - NUV
    NJcolor = NUV - J
    JKcolor = J - K
    KW1color = K - W1
    W1W2color = W1 - W2
    W2W3color = W2 - W3
    W3W4color = W3 - W4

    title = "MIR class correlations"
    plt.figure()
    fsps.plot_SSP_color_color(SSPpath, "NUV", "J", "FUV", "NUV",
                              fileformat=SSPfile)
    MIRplot(NJcolor, FNcolor, MIRclass, "NUV-J", "FUV-NUV", title,
            loc="upper left")
    plt.figure()
    fsps.plot_SSP_color_color(SSPpath, "J", "Ks", "FUV", "NUV",
                              fileformat=SSPfile)
    MIRplot(JKcolor, FNcolor, MIRclass, "J-Ks", "FUV-NUV", title,
            loc="upper right")
    plt.figure()
    fsps.plot_SSP_color_color(SSPpath, "Ks", "W1", "FUV", "NUV",
                              fileformat=SSPfile)
    MIRplot(KW1color, FNcolor, MIRclass, "Ks-W1", "FUV-NUV", title,
            loc="upper left")
    plt.figure()
    fsps.plot_SSP_color_color(SSPpath, "W1", "W2", "FUV", "NUV",
                              fileformat=SSPfile)
    MIRplot(W1W2color, FNcolor, MIRclass, "W1-W2", "FUV-NUV", title,
            loc="upper right")
    plt.figure()
    fsps.plot_SSP_color_color(SSPpath, "W2", "W3", "FUV", "NUV",
                              fileformat=SSPfile)
    MIRplot(W2W3color, FNcolor, MIRclass, "W2-W3", "FUV-NUV", title,
            loc="upper right")
    plt.figure()
    fsps.plot_SSP_color_color(SSPpath, "W3", "W4", "FUV", "NUV",
                              fileformat=SSPfile)
    MIRplot(W3W4color, FNcolor, MIRclass, "W3-W4", "FUV-NUV", title,
            loc="upper right")
 
    plt.figure()
    fsps.plot_SSP_color_color(SSPpath, "J", "Ks", "NUV", "J",
                              fileformat=SSPfile)
    MIRplot(JKcolor, NJcolor, MIRclass, "J-Ks", "NUV-J", title,
            loc="lower right")
    plt.figure()
    fsps.plot_SSP_color_color(SSPpath, "Ks", "W1", "NUV", "J",
                              fileformat=SSPfile)
    MIRplot(KW1color, NJcolor, MIRclass, "Ks-W1", "NUV-J", title,
            loc="lower left")
    plt.figure()
    fsps.plot_SSP_color_color(SSPpath, "W1", "W2", "NUV", "J",
                              fileformat=SSPfile)
    MIRplot(W1W2color, NJcolor, MIRclass, "W1-W2", "NUV-J", title,
            loc="lower right")
    plt.figure()
    fsps.plot_SSP_color_color(SSPpath, "W2", "W3", "NUV", "J",
                              fileformat=SSPfile)
    MIRplot(W2W3color, NJcolor, MIRclass, "W2-W3", "NUV-J", title,
            loc="lower left")
    plt.figure()
    fsps.plot_SSP_color_color(SSPpath, "W3", "W4", "NUV", "J",
                              fileformat=SSPfile)
    MIRplot(W3W4color, NJcolor, MIRclass, "W3-W4", "NUV-J", title,
            loc="lower left")
 
    plt.figure()
    fsps.plot_SSP_color_color(SSPpath, "Ks", "W1", "J", "Ks",
                              fileformat=SSPfile)
    MIRplot(KW1color, JKcolor, MIRclass, "Ks-W1", "J-Ks", title,
            loc="upper left")
    plt.figure()
    fsps.plot_SSP_color_color(SSPpath, "W1", "W2", "J", "Ks",
                              fileformat=SSPfile)
    MIRplot(W1W2color, JKcolor, MIRclass, "W1-W2", "J-Ks", title,
            loc="lower right")
    plt.figure()
    fsps.plot_SSP_color_color(SSPpath, "W2", "W3", "J", "Ks",
                              fileformat=SSPfile)
    MIRplot(W2W3color, JKcolor, MIRclass, "W2-W3", "J-Ks", title,
            loc="lower right")
    plt.figure()
    fsps.plot_SSP_color_color(SSPpath, "W3", "W4", "J", "Ks",
                              fileformat=SSPfile)
    MIRplot(W3W4color, JKcolor, MIRclass, "W3-W4", "J-Ks", title,
            loc="upper left")

    plt.figure()
    fsps.plot_SSP_color_color(SSPpath, "W1", "W2", "Ks", "W1",
                              fileformat=SSPfile)
    MIRplot(W1W2color, KW1color, MIRclass, "W1-W2", "Ks-W1", title,
            loc="lower right")
    plt.figure()
    fsps.plot_SSP_color_color(SSPpath, "W2", "W3", "Ks", "W1",
                              fileformat=SSPfile)
    MIRplot(W2W3color, KW1color, MIRclass, "W2-W3", "Ks-W1", title,
            loc="lower right")
    plt.figure()
    fsps.plot_SSP_color_color(SSPpath, "W3", "W4", "Ks", "W1",
                              fileformat=SSPfile)
    MIRplot(W3W4color, KW1color, MIRclass, "W3-W4", "Ks-W1", title,
            loc="lower right")

    plt.figure()
    fsps.plot_SSP_color_color(SSPpath, "W2", "W3", "W1", "W2",
                              fileformat=SSPfile)
    MIRplot(W2W3color, W1W2color, MIRclass, "W2-W3", "W1-W2", title,
            loc="upper left")
    plt.figure()
    fsps.plot_SSP_color_color(SSPpath, "W3", "W4", "W1", "W2",
                              fileformat=SSPfile)
    MIRplot(W3W4color, W1W2color, MIRclass, "W3-W4", "W1-W2", title,
            loc="upper left")

    plt.figure()
    fsps.plot_SSP_color_color(SSPpath, "W3", "W4", "W2", "W3",
                              fileformat=SSPfile)
    MIRplot(W3W4color, W2W3color, MIRclass, "W3-W4", "W2-W3", title,
            loc="upper left")

def jk_color_vs_wise_colors(magtable):
    '''Generates J-Ks vs wise colors.

    Although J and Ks aren't adjacent bands, they are usually put together as a
    representative color for the 2MASS infrared region. This function makes
    plots with them instead of J-H and H-Ks separately.
    '''
    MIRclass = magtable["MIR class"]
    W1 = magtable["w1unextmag"]
    W2 = magtable["w2unextmag"]
    W3 = magtable["w3unextmag"]
    W4 = magtable["w4unextmag"]
    J = magtable["j_m_k20fe"]
    K = magtable["k_m_k20fe"]
    #FUV = magtable["FUVunextmags"]
    #NUV = magtable["NUVunextmags"]

    #FNcolor = FUV - NUV
    #NJcolor = NUV - J
    JKcolor = J - K
    KW1color = K - W1
    W1W2color = W1 - W2
    W2W3color = W2 - W3
    W3W4color = W3 - W4

    title = "MIR Class Correlations"
#   plt.figure()
#   MIRplot(JKcolor, FNcolor, MIRclass, "J-K", "FUV-NUV", title,
#           loc="lower left")
#   plt.figure()
#   MIRplot(JKcolor, NJcolor, MIRclass, "J-H", "NUV-J", title,
#           loc="lower left")
    plt.figure()
    MIRplot(KW1color, JKcolor, MIRclass, "Ks-W1", "J-Ks", title,
            loc="lower left")
    plt.figure()
    MIRplot(W1W2color, JKcolor, MIRclass, "W1-W2", "J-Ks", title,
            loc="lower right")
    plt.figure()
    MIRplot(W2W3color, JKcolor, MIRclass, "W2-W3", "J-Ks", title,
            loc="upper left")
    plt.figure()
    MIRplot(W3W4color, JKcolor, MIRclass, "W3-W4", "J-Ks", title,
            loc="upper left")

def generateColorColors2MASS(magtable):
    '''Generates permutations of Color-Color Diagrams.
  
    The produced diagrams considered sensible involve a cross-band
    with a intra-band color. For example, a UV-IR vs. an IR-IR color.
    '''
    MIRclass = magtable["MIR class"]
    W1 = magtable["j_m_k20fe"]
    W2 = magtable["h_m_k20fe"]
    W3 = magtable["k_m_k20fe"]
    FUV = magtable["FUVunextmags"]
    NUV = magtable["NUVunextmags"]
    F1color = FUV - W1
    F2color = FUV - W2
    F3color = FUV - W3
    N1color = NUV - W1
    N2color = NUV - W2
    N3color = NUV - W3
    W12color = W1-W2
    W23color = W2-W3
    FNcolor = FUV - NUV

    title = "2MASS CMD"
    plt.figure()
    MIRplot(F1color, W12color, MIRclass, "FUV-J", "J-H", title,
            loc="lower left")
    plt.figure()
    MIRplot(F1color, W23color, MIRclass, "FUV-J", "H-K", title,
            loc="lower right")
    plt.figure()
    MIRplot(F1color, FNcolor, MIRclass, "FUV-J", "FUV-NUV", title,
            loc="left")
    plt.figure()
    MIRplot(F2color, W12color, MIRclass, "FUV-H", "J-H", title,
            loc="lower left")
    plt.figure()
    MIRplot(F2color, W23color, MIRclass, "FUV-H", "H-K", title,
            loc="lower right")
    plt.figure()
    MIRplot(F2color, FNcolor, MIRclass, "FUV-H", "FUV-NUV", title,
            loc="lower right")
    plt.figure()
    MIRplot(F3color, W12color, MIRclass, "FUV-K", "J-H", title,
            loc="lower left")
    plt.figure()
    MIRplot(F3color, W23color, MIRclass, "FUV-K", "H-K", title,
            loc="lower right")
    plt.figure()
    MIRplot(F3color, FNcolor, MIRclass, "FUV-K", "FUV-NUV", title,
            loc="upper left")
    plt.figure()
    MIRplot(N1color, W12color, MIRclass, "NUV-J", "J-H", title,
            loc="lower right")
    plt.figure()
    MIRplot(N1color, W23color, MIRclass, "NUV-J", "H-K", title,
            loc="lower right")
    plt.figure()
    MIRplot(N1color, FNcolor, MIRclass, "NUV-J", "FUV-NUV", title,
            loc="upper left")
    plt.figure()
    MIRplot(N2color, W12color, MIRclass, "NUV-H", "J-H", title,
            loc="lower right")
    plt.figure()
    MIRplot(N2color, W23color, MIRclass, "NUV-H", "H-K", title,
            loc="lower right")
    plt.figure()
    MIRplot(N2color, FNcolor, MIRclass, "NUV-H", "FUV-NUV", title,
            loc="upper right")
    plt.figure()
    MIRplot(N3color, W12color, MIRclass, "NUV-K", "J-H", title,
            loc="lower right")
    plt.figure()
    MIRplot(N3color, W23color, MIRclass, "NUV-K", "H-K", title, 
            loc="upper left")
    plt.figure()
    MIRplot(N3color, FNcolor, MIRclass, "NUV-K", "FUV-NUV", title, 
    loc="lower left")

def generateColorColors(magtable):
    '''Generates permutations of Color-Color Diagrams.
  
    The produced diagrams considered sensible involve a cross-band
    with a intra-band color. For example, a UV-IR vs. an IR-IR color.
    '''
    MIRclass = magtable["MIR class"]
    W1 = magtable["w1unextmag"]
    W2 = magtable["w2unextmag"]
    W3 = magtable["w3unextmag"]
    FUV = magtable["FUVunextmags"]
    NUV = magtable["NUVunextmags"]
    F1color = FUV - W1
    F2color = FUV - W2
    F3color = FUV - W3
    N1color = NUV - W1
    N2color = NUV - W2
    N3color = NUV - W3
    W12color = W1-W2
    W23color = W2-W3
    FNcolor = FUV - NUV

    title = "WISE CMD"
    plt.figure()
    MIRplot(F1color, W12color, MIRclass, "FUV-W1", "W1-W2", title)
    plt.figure()
    MIRplot(F1color, W23color, MIRclass, "FUV-W1", "W2-W3", title,
            loc="lower left")
    plt.figure()
    MIRplot(F1color, FNcolor, MIRclass, "FUV-W1", "FUV-NUV", title,
            loc="upper left")
    plt.figure()
    MIRplot(F2color, W12color, MIRclass, "FUV-W2", "W1-W2", title,
            loc="upper left")
    plt.figure()
    MIRplot(F2color, W23color, MIRclass, "FUV-W2", "W2-W3", title,
            loc="lower left")
    plt.figure()
    MIRplot(F2color, FNcolor, MIRclass, "FUV-W2", "FUV-NUV", title,
            loc="upper left")
    plt.figure()
    MIRplot(F3color, W12color, MIRclass, "FUV-W3", "W1-W2", title,
            loc="upper left")
    plt.figure()
    MIRplot(F3color, W23color, MIRclass, "FUV-W3", "W2-W3", title,
            loc="lower right")
    plt.figure()
    MIRplot(F3color, FNcolor, MIRclass, "FUV-W3", "FUV-NUV", title)
    plt.figure()
    MIRplot(N1color, W12color, MIRclass, "NUV-W1", "W1-W2", title,
            loc="upper left")
    plt.figure()
    MIRplot(N1color, W23color, MIRclass, "NUV-W1", "W2-W3", title,
            loc="lower left")
    plt.figure()
    MIRplot(N1color, FNcolor, MIRclass, "NUV-W1", "FUV-NUV", title,
            loc="upper left")
    plt.figure()
    MIRplot(N2color, W12color, MIRclass, "NUV-W2", "W1-W2", title,
            loc="upper left")
    plt.figure()
    MIRplot(N2color, W23color, MIRclass, "NUV-W2", "W2-W3", title,
            loc="lower left")
    plt.figure()
    MIRplot(N2color, FNcolor, MIRclass, "NUV-W2", "FUV-NUV", title,
            loc="upper left")
    plt.figure()
    MIRplot(N3color, W12color, MIRclass, "NUV-W3", "W1-W2", title,
            loc="upper left")
    plt.figure()
    MIRplot(N3color, W23color, MIRclass, "NUV-W3", "W2-W3", title, 
            loc="lower right")
    plt.figure()
    MIRplot(N3color, FNcolor, MIRclass, "NUV-W3", "FUV-NUV", title)
