
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table, join

import photometry as phot
import fsps

MIR_Symbols = {0: {"marker": 'o', "markerfacecolor": 'white', "ls": ' ', 
                   "markeredgewidth": 1.5, 
                   "ecolor": "black", "elinewidth": 0.7, "capthick": 1.0},
               1: {"marker": '^', "markerfacecolor": 'orange', "ls": ' ',
                   "ecolor": "orange", "elinewidth": 0.7, "capthick": 1.0},
               2: {"marker": 'o', "markerfacecolor": 'green', "ls": ' ',
                   "ecolor": "green", "elinewidth": 0.7, "capthick": 1.0},
               3: {"marker": '*', "markerfacecolor": 'blue', "ls": ' ',
                   "ecolor": "blue", "elinewidth": 0.7, "capthick": 1.0},
               4: {"marker": 'D', "markerfacecolor": 'white', "ls": ' ',
                   "markeredgecolor": 'red', "markeredgewidth": 1.5, 
                   "ecolor": "red", "elinewidth": 0.7, "capthick": 1.0}}


def color_histogram_by_class(band1, band2, groupcol, xlabel, title, bins, 
                             colrange=(-4, 4), classes=np.arange(0,5)):
    '''Creates a histogam for colors for different classes.
    '''
    color = band1 - band2
    colorgroup = color.group_by(groupcol)
    plotcolors = ["black", "orange", "green", "blue", "red"]
    plt.hist([colorgroup.groups[i] for i in classes], bins, range=colrange, 
             label=["Class {0}".format(i) for i in classes], 
             color= [plotcolors[i] for i in classes],
             histtype="bar", rwidth=1, lw=0)
    plt.xlabel(xlabel)
    plt.ylabel("N")
    plt.title(title)
    plt.legend()

def color_cumulative_histogram_by_class(
        band1, band2, groupcol, xlabel, title, bins, colrange=(-4, 4), 
        classes=np.arange(0,5), loc="lower right"):
    '''Creates a histogam for colors for different classes.
    '''
    color = band1 - band2
    colorgroup = color.group_by(groupcol)
    plotcolors = ["black", "orange", "green", "blue", "red"]
    plt.hist([colorgroup.groups[i] for i in classes], bins, range=colrange, 
             label=["Class {0}".format(i) for i in classes], 
             color= [plotcolors[i] for i in classes],
             histtype="step", rwidth=1, cumulative=True, normed=True)
    plt.xlabel(xlabel)
    plt.ylabel("N")
    plt.title(title)
    plt.legend(loc=loc)

def plot_color_PAH_flux_ratio(pahtable, magtable, title, colorlabel):
    '''Plots color vs a PAH flux ratio.'''
    filledpahs = pahtable.filled(0)
    fulltable = phot.join_by_galaxy_name(magtable, filledpahs, 
                                         names=("objstr_01", "Galaxy"))

    pahvalues = [fulltable["7.7 um"], fulltable["8.6 um"], 
                 fulltable["11.3 um"], fulltable["12.7 um"],
                 fulltable["17 um"]]
    paherrs = [fulltable["7.7 um err"], fulltable["8.6 um err"],
               fulltable["11.3 um err"], fulltable["12.7 um err"],
               fulltable["17 um err"]]
    pahratio, pahratio_err = phot.calc_statistical_fraction_of_sums(pahvalues,
        paherrs, (1, 1, 0, 0, 0), (0, 0, 1, 1, 1))
    # Arithmetic to take care of the PAH ratio along with the uncertainties.
    #pah_numerator = fulltable["7.7 um"] + fulltable["8.6 um"]
    #pah_numerator_err = np.sqrt(fulltable["7.7 um err"]**2 + 
    #                            fulltable["8.6 um err"]**2)
    #pah_denominator = (fulltable["11.3 um"] + fulltable["12.7 um"] +
    #                   fulltable["17 um"])
    #pah_denominator_err = np.sqrt(fulltable["11.3 um err"]**2 + 
    #                              fulltable["12.7 um err"]**2 +
    #                              fulltable["17 um err"]**2)
    #pahratio = pah_numerator / pah_denominator
    #pahratio_err = (pahratio * 
    #                (np.sqrt((pah_numerator_err / pah_numerator)**2 + 
    #                         (pah_denominator_err / pah_denominator)**2)))

    color = fulltable["w2unextmag"] - fulltable["w3unextmag"]
    color_err = np.sqrt(fulltable["w2unexterr"]**2 + fulltable["w3unexterr"]**2)

    plt.errorbar(color, pahratio, pahratio_err, color_err, '*')
    plt.xlabel(colorlabel)
    plt.ylabel("(f(7.7)+f(8.6))/(f(11.3)+f(12.7)+f(17))")
    plt.title(title)

def plot_PAH_flux_ratios(pahtable, mirtable, title):
    '''Plots two PAH intensity ratios.'''
    linetable = phot.join_by_galaxy_name(pahtable, mirtable, 
                                         names=("Galaxy", "objstr_01"))
    shortratio = linetable["7.7 um"] / linetable["11.3 um"]
    shortratio_err = shortratio * np.sqrt((linetable["7.7 um err"] / 
        linetable["7.7 um"])**2 + (linetable["11.3 um err"] / 
        linetable["11.3 um"])**2)
    longratio = linetable["17 um"] / linetable["11.3 um"]
    longratio_err = longratio * np.sqrt((linetable["17 um err"] / 
        linetable["17 um"])**2 + (linetable["11.3 um err"] / 
        linetable["11.3 um"])**2)

    linetable["short"] = shortratio
    linetable["short err"] = shortratio_err
    linetable["long"] = longratio
    linetable["long err"] = longratio_err
    linegroups = linetable.group_by("MIR_Class")

    class2table = linegroups.groups[0]
    class3table = linegroups.groups[1]

    print class2table[["objstr_01", "short"]][np.where(class2table["short"] > 2.3)]

    plt.errorbar(class2table["short"], class2table["long"], 
                 class2table["long err"], class2table["short err"], 
                 label="Class 2", **MIR_Symbols[2])
    plt.errorbar(class3table["short"], class3table["long"], 
                 class3table["long err"], class3table["short err"], 
                 label="Class 3", **MIR_Symbols[3])

    plt.xlabel("7.7um/11.3um")
    plt.ylabel("17um/11.3um")
    plt.legend()

def plot_rampazzo_line_ratios(linetable, mirtable, title):
    '''Plots line ratios of a galaxy with respect to each other.
    
    NOTE: Make this more flexible by adding arguments: ynum, ydenom, xnum,
    xdenom, xlabel, ylabel. All of these are strings.'''
    linetable = phot.join_by_galaxy_name(linetable, mirtable, 
                                         names=("Galaxy", "objstr_01"))
    sulfurlines = linetable["[S III] 19 um"] / linetable["[S III] 33 um"]
    sulfurlineerrs = sulfurlines * np.ma.sqrt(
        (linetable["[S III] 19 um err"]/linetable["[S III] 19 um"])**2 +
        (linetable["[S III] 33 um err"]/linetable["[S III] 33 um"])**2)
    hydrogenlines = linetable["H2 S(3)"] / linetable["H2 S(1)"]
    hydrogenlineerrs = hydrogenlines * np.ma.sqrt(
        (linetable["H2 S(3) err"]/linetable["H2 S(3)"])**2 +
        (linetable["H2 S(1) err"]/linetable["H2 S(1)"])**2)

    # This should make grouping more straightforward.
    linetable["Sratio"] = sulfurlines
    linetable["Sratioerr"] = sulfurlineerrs
    linetable["Hratio"] = hydrogenlines
    linetable["Hratioerr"] = hydrogenlineerrs
    linegroups = linetable.group_by("MIR_Class")
    
    class2table = linegroups.groups[1]
    class3table = linegroups.groups[2]
    plt.errorbar(class2table["Hratio"], class2table["Sratio"], 
                 class2table["Sratioerr"], class2table["Hratioerr"], 
                 label="Class 2", **MIR_Symbols[2])
    plt.errorbar(class3table["Hratio"], class3table["Sratio"], 
                 class3table["Sratioerr"], class3table["Hratioerr"], 
                 label="Class 3", **MIR_Symbols[3])

    print "Class 2"
    print class2table[["Galaxy", "Hratio", "Sratio"]]
    print "Class 3"
    print class3table[["Galaxy", "Hratio", "Sratio"]]

    plt.xlabel("H2S(3)/H2S(1)")
    plt.ylabel("[SIII]18.7/[SIII]33.5")
    plt.legend()
    plt.title(title)

def MIRplot(x, y, mirindex, yerr=None, xerr=None, classes=xrange(5), xlabel="", 
            ylabel="", title="", loc='upper right'):
    '''Makes a plot that automatically differentiates between MIR classes.

    The x and y data need to be columns which have the same length as 
    mirindex. Groupkey should be the list of MIR classes which are in the same
    order as x and y. Labels can also be added as desired.
    '''
    xgroup = x.group_by(mirindex)
    ygroup = y.group_by(mirindex)
    try:
        yerrgroup = yerr.group_by(mirindex)
    except AttributeError:
        if yerr is None:
            yerrgroup = None
        else:
            raise ValueError("yerr must be a Table or None")
    try:
        xerrgroup = xerr.group_by(mirindex)
    except AttributeError:
        if xerr is None:
            xerrgroup = None
        else:
            raise ValueError("xerr must be a Table or None")

    for MIRclass in classes:
        try:
            xvals = xgroup.groups[xgroup.groups.keys==MIRclass]
            yvals = ygroup.groups[ygroup.groups.keys==MIRclass]
            try:
                yerrvals = yerrgroup.groups[yerrgroup.groups.keys==MIRclass]
            except AttributeError:
                # Since yerrgroup was defined in this function. It can't be
                # anything other than a table or None. So I don't need the if..else
                # statement above.
                yerrvals = None
            try:
                xerrvals = xerrgroup.groups[yerrgroup.groups.keys==MIRclass]
            except AttributeError:
                xerrvals = None
        except IndexError:
            # If the given class is not in the groups, ignore it.
            print "Class {0} not detected.".format(MIRclass)
            raise
        

        plt.errorbar(xvals, yvals, yerrvals, xerrvals,
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
    MIRplot(W1, F1color, MIRclass, xlabel="W1", ylabel="FUV-W1", title=title, 
            loc="lower left")
    plt.figure()
    MIRplot(W2, F2color, MIRclass, xlabel="W2", ylabel="FUV-W2", title=title, 
            loc="lower left")
    plt.figure()
    MIRplot(W3, F3color, MIRclass, xlabel="W3", ylabel="FUV-W3", title=title)
    plt.figure()
    MIRplot(W1, N1color, MIRclass, xlabel="W1", ylabel="NUV-W1", title=title, 
            loc="lower left")
    plt.figure()
    MIRplot(W2, N2color, MIRclass, xlabel="W2", ylabel="NUV-W2", title=title, 
            loc="lower left")
    plt.figure()
    MIRplot(W3, N3color, MIRclass, xlabel="W3", ylabel="NUV-W3", title=title)

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
    MIRplot(NJcolor, FNcolor, MIRclass, xlabel="NUV-J", ylabel="FUV-NUV",
            title=title, loc="upper left")
    plt.figure()
    fsps.plot_SSP_color_color(SSPpath, "J", "Ks", "FUV", "NUV",
                              fileformat=SSPfile)
    MIRplot(JKcolor, FNcolor, MIRclass, xlabel="J-Ks", ylabel="FUV-NUV", 
            title=title, loc="upper right")
    plt.figure()
    fsps.plot_SSP_color_color(SSPpath, "Ks", "W1", "FUV", "NUV",
                              fileformat=SSPfile)
    MIRplot(KW1color, FNcolor, MIRclass, xlabel="Ks-W1", ylabel="FUV-NUV", 
            title=title, loc="upper left")
    plt.figure()
    fsps.plot_SSP_color_color(SSPpath, "W1", "W2", "FUV", "NUV",
                              fileformat=SSPfile)
    MIRplot(W1W2color, FNcolor, MIRclass, xlabel="W1-W2", ylabel="FUV-NUV", 
            title=title, loc="upper right")
    plt.figure()
    fsps.plot_SSP_color_color(SSPpath, "W2", "W3", "FUV", "NUV",
                              fileformat=SSPfile)
    MIRplot(W2W3color, FNcolor, MIRclass, xlabel="W2-W3", ylabel="FUV-NUV", 
            title=title, loc="upper right")
    plt.figure()
    fsps.plot_SSP_color_color(SSPpath, "W3", "W4", "FUV", "NUV",
                              fileformat=SSPfile)
    MIRplot(W3W4color, FNcolor, MIRclass, xlabel="W3-W4", ylabel="FUV-NUV", 
            title=title, loc="upper right")
 
    plt.figure()
    fsps.plot_SSP_color_color(SSPpath, "J", "Ks", "NUV", "J",
                              fileformat=SSPfile)
    MIRplot(JKcolor, NJcolor, MIRclass, xlabel="J-Ks", ylabel="NUV-J", 
            title=title, loc="lower right")
    plt.figure()
    fsps.plot_SSP_color_color(SSPpath, "Ks", "W1", "NUV", "J",
                              fileformat=SSPfile)
    MIRplot(KW1color, NJcolor, MIRclass, xlabel="Ks-W1", ylabel="NUV-J", 
            title=title, loc="lower left")
    plt.figure()
    fsps.plot_SSP_color_color(SSPpath, "W1", "W2", "NUV", "J",
                              fileformat=SSPfile)
    MIRplot(W1W2color, NJcolor, MIRclass, xlabel="W1-W2", ylabel="NUV-J", 
            title=title, loc="lower right")
    plt.figure()
    fsps.plot_SSP_color_color(SSPpath, "W2", "W3", "NUV", "J",
                              fileformat=SSPfile)
    MIRplot(W2W3color, NJcolor, MIRclass, xlabel="W2-W3", ylabel="NUV-J", 
            title=title, loc="lower left")
    plt.figure()
    fsps.plot_SSP_color_color(SSPpath, "W3", "W4", "NUV", "J",
                              fileformat=SSPfile)
    MIRplot(W3W4color, NJcolor, MIRclass, xlabel="W3-W4", ylabel="NUV-J", 
            title=title, loc="lower left")
 
    plt.figure()
    fsps.plot_SSP_color_color(SSPpath, "Ks", "W1", "J", "Ks",
                              fileformat=SSPfile)
    MIRplot(KW1color, JKcolor, MIRclass, xlabel="Ks-W1", ylabel="J-Ks", 
            title=title, loc="upper left")
    plt.figure()
    fsps.plot_SSP_color_color(SSPpath, "W1", "W2", "J", "Ks",
                              fileformat=SSPfile)
    MIRplot(W1W2color, JKcolor, MIRclass, xlabel="W1-W2", ylabel="J-Ks", 
            title=title, loc="lower right")
    plt.figure()
    fsps.plot_SSP_color_color(SSPpath, "W2", "W3", "J", "Ks",
                              fileformat=SSPfile)
    MIRplot(W2W3color, JKcolor, MIRclass, xlabel="W2-W3", ylabel="J-Ks", 
            title=title, loc="lower right")
    plt.figure()
    fsps.plot_SSP_color_color(SSPpath, "W3", "W4", "J", "Ks",
                              fileformat=SSPfile)
    MIRplot(W3W4color, JKcolor, MIRclass, xlabel="W3-W4", ylabel="J-Ks", 
            title=title, loc="upper left")

    plt.figure()
    fsps.plot_SSP_color_color(SSPpath, "W1", "W2", "Ks", "W1",
                              fileformat=SSPfile)
    MIRplot(W1W2color, KW1color, MIRclass, xlabel="W1-W2", ylabel="Ks-W1", 
            title=title, loc="lower right")
    plt.figure()
    fsps.plot_SSP_color_color(SSPpath, "W2", "W3", "Ks", "W1",
                              fileformat=SSPfile)
    MIRplot(W2W3color, KW1color, MIRclass, xlabel="W2-W3", ylabel="Ks-W1", 
            title=title, loc="lower right")
    plt.figure()
    fsps.plot_SSP_color_color(SSPpath, "W3", "W4", "Ks", "W1",
                              fileformat=SSPfile)
    MIRplot(W3W4color, KW1color, MIRclass, xlabel="W3-W4", ylabel="Ks-W1", 
            title=title, loc="lower right")

    plt.figure()
    fsps.plot_SSP_color_color(SSPpath, "W2", "W3", "W1", "W2",
                              fileformat=SSPfile)
    MIRplot(W2W3color, W1W2color, MIRclass, xlabel="W2-W3", ylabel="W1-W2", 
            title=title, loc="upper left")
    plt.figure()
    fsps.plot_SSP_color_color(SSPpath, "W3", "W4", "W1", "W2",
                              fileformat=SSPfile)
    MIRplot(W3W4color, W1W2color, MIRclass, xlabel="W3-W4", ylabel="W1-W2", 
            title=title, loc="upper left")

    plt.figure()
    fsps.plot_SSP_color_color(SSPpath, "W3", "W4", "W2", "W3",
                              fileformat=SSPfile)
    MIRplot(W3W4color, W2W3color, MIRclass, xlabel="W3-W4", ylabel="W2-W3", 
            title=title, loc="upper left")

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
    MIRplot(KW1color, JKcolor, MIRclass, xlabel="Ks-W1", ylabel="J-Ks", 
            title=title, loc="lower left")
    plt.figure()
    MIRplot(W1W2color, JKcolor, MIRclass, xlabel="W1-W2", ylabel="J-Ks", 
            title=title, loc="lower right")
    plt.figure()
    MIRplot(W2W3color, JKcolor, MIRclass, xlabel="W2-W3", ylabel="J-Ks", 
            title=title, loc="upper left")
    plt.figure()
    MIRplot(W3W4color, JKcolor, MIRclass, xlabel="W3-W4", ylabel="J-Ks", 
            title=title, loc="upper left")

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
    MIRplot(F1color, W12color, MIRclass, xlabel="FUV-J", ylabel="J-H", 
            title=title, loc="lower left")
    plt.figure()
    MIRplot(F1color, W23color, MIRclass, xlabel="FUV-J", ylabel="H-K", 
            title=title, loc="lower right")
    plt.figure()
    MIRplot(F1color, FNcolor, MIRclass, xlabel="FUV-J", ylabel="FUV-NUV", 
            title=title, loc="left")
    plt.figure()
    MIRplot(F2color, W12color, MIRclass, xlabel="FUV-H", ylabel="J-H", 
            title=title, loc="lower left")
    plt.figure()
    MIRplot(F2color, W23color, MIRclass, xlabel="FUV-H", ylabel="H-K", 
            title=title, loc="lower right")
    plt.figure()
    MIRplot(F2color, FNcolor, MIRclass, xlabel="FUV-H", ylabel="FUV-NUV", 
            title=title, loc="lower right")
    plt.figure()
    MIRplot(F3color, W12color, MIRclass, xlabel="FUV-K", ylabel="J-H", 
            title=title, loc="lower left")
    plt.figure()
    MIRplot(F3color, W23color, MIRclass, xlabel="FUV-K", ylabel="H-K", 
            title=title, loc="lower right")
    plt.figure()
    MIRplot(F3color, FNcolor, MIRclass, xlabel="FUV-K", ylabel="FUV-NUV", 
            title=title, loc="upper left")
    plt.figure()
    MIRplot(N1color, W12color, MIRclass, xlabel="NUV-J", ylabel="J-H", 
            title=title, loc="lower right")
    plt.figure()
    MIRplot(N1color, W23color, MIRclass, xlabel="NUV-J", ylabel="H-K", 
            title=title, loc="lower right")
    plt.figure()
    MIRplot(N1color, FNcolor, MIRclass, xlabel="NUV-J", ylabel="FUV-NUV", 
            title=title, loc="upper left")
    plt.figure()
    MIRplot(N2color, W12color, MIRclass, xlabel="NUV-H", ylabel="J-H", 
            title=title, loc="lower right")
    plt.figure()
    MIRplot(N2color, W23color, MIRclass, xlabel="NUV-H", ylabel="H-K", 
            title=title, loc="lower right")
    plt.figure()
    MIRplot(N2color, FNcolor, MIRclass, xlabel="NUV-H", ylabel="FUV-NUV", 
            title=title, loc="upper right")
    plt.figure()
    MIRplot(N3color, W12color, MIRclass, xlabel="NUV-K", ylabel="J-H", 
            title=title, loc="lower right")
    plt.figure()
    MIRplot(N3color, W23color, MIRclass, xlabel="NUV-K", ylabel="H-K", 
            title=title, loc="upper left")
    plt.figure()
    MIRplot(N3color, FNcolor, MIRclass, xlabel="NUV-K", ylabel="FUV-NUV", 
            title=title, loc="lower left")

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
    MIRplot(F1color, W12color, MIRclass, xlabel="FUV-W1", ylabel="W1-W2", 
            title=title)
    plt.figure()
    MIRplot(F1color, W23color, MIRclass, xlabel="FUV-W1", ylabel="W2-W3", 
            title=title, loc="lower left")
    plt.figure()
    MIRplot(F1color, FNcolor, MIRclass, xlabel="FUV-W1", ylabel="FUV-NUV", 
            title=title, loc="upper left")
    plt.figure()
    MIRplot(F2color, W12color, MIRclass, xlabel="FUV-W2", ylabel="W1-W2", 
            title=title, loc="upper left")
    plt.figure()
    MIRplot(F2color, W23color, MIRclass, xlabel="FUV-W2", ylabel="W2-W3", 
            title=title, loc="lower left")
    plt.figure()
    MIRplot(F2color, FNcolor, MIRclass, xlabel="FUV-W2", ylabel="FUV-NUV", 
            title=title, loc="upper left")
    plt.figure()
    MIRplot(F3color, W12color, MIRclass, xlabel="FUV-W3", ylabel="W1-W2", 
            title=title, loc="upper left")
    plt.figure()
    MIRplot(F3color, W23color, MIRclass, xlabel="FUV-W3", ylabel="W2-W3", 
            title=title, loc="lower right")
    plt.figure()
    MIRplot(F3color, FNcolor, MIRclass, xlabel="FUV-W3", ylabel="FUV-NUV", 
            title=title)
    plt.figure()
    MIRplot(N1color, W12color, MIRclass, xlabel="NUV-W1", ylabel="W1-W2", 
            title=title, loc="upper left")
    plt.figure()
    MIRplot(N1color, W23color, MIRclass, xlabel="NUV-W1", ylabel="W2-W3", 
            title=title, loc="lower left")
    plt.figure()
    MIRplot(N1color, FNcolor, MIRclass, xlabel="NUV-W1", ylabel="FUV-NUV", 
            title=title, loc="upper left")
    plt.figure()
    MIRplot(N2color, W12color, MIRclass, xlabel="NUV-W2", ylabel="W1-W2", 
            title=title, loc="upper left")
    plt.figure()
    MIRplot(N2color, W23color, MIRclass, xlabel="NUV-W2", ylabel="W2-W3", 
            title=title, loc="lower left")
    plt.figure()
    MIRplot(N2color, FNcolor, MIRclass, xlabel="NUV-W2", ylabel="FUV-NUV", 
            title=title, loc="upper left")
    plt.figure()
    MIRplot(N3color, W12color, MIRclass, xlabel="NUV-W3", ylabel="W1-W2", 
            title=title, loc="upper left")
    plt.figure()
    MIRplot(N3color, W23color, MIRclass, xlabel="NUV-W3", ylabel="W2-W3", 
            title=title, loc="lower right")
    plt.figure()
    MIRplot(N3color, FNcolor, MIRclass, xlabel="NUV-W3", ylabel="FUV-NUV", 
            title=title)
