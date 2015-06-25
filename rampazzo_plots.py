
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table

def plot_color_PAH_flux_ratio(pahtable, magtable, title, colorlabel):
    '''Plots color vs a PAH flux ratio.'''
    filledpahs = pahtable.filled(0)
    # Arithmetic to take care of the PAH ratio along with the uncertainties.
    pah_numerator = filledpahs["7.7 um"] + filledpahs["8.6 um"]
    pah_numerator_err = np.sqrt(filledpahs["7.7 um err"]**2 + 
                                filledpahs["8.6 um er"]**2)
    pah_denominator = (filledpahs["11.3um"] + filledpahs["12.7um"] +
                       filledpahs["17 um"])
    pah_denominator_err = np.sqrt(filledpahs["11.3 um err"]**2 + 
                                  filledpahs["12.7 um err"]**2 +
                                  filledpahs["17 um err"]**2)
    pahratio = pah_numerator / pah_denominator
    pahratio_err = (pahratio * 
                    (np.sqrt((pah_numerator_err / pah_numerator)**2 + 
                             (pah_denominator_err / pah_denominator)**2)
    plt.errorbar
