import os

from astropy.table import Table
import numpy as np

import photometry as phot

BASEPATH = "/home/regulus/simonian/year1/wise"
PAPERPATH = "/home/regulus/simonian/papers/wise14"
FSPSPATH = "/home/regulus/simonian/year1/fsps"

ATLAS3DBASE = os.path.join(BASEPATH, "ATLAS3D_DB")
RAMPAZZOBASE = os.path.join(BASEPATH, "Rampazzo_DB")
JARRETTBASE = os.path.join(BASEPATH, "Jarrett_DB")

def move_rampazzo():
    # First read in Table 1
    # Then Table 2
    # Concatenate them.
    # filter them for galaxies we have data for.
    pass


def read_Rampazzo_Table1(tablepath=os.path.join(BASEPATH,
                                                "Rampazzo_Table1.csv")):
    rampazzotable = Table.read(tablepath, format="ascii.csv", guess=False,
                               data_start=2, delimiter=":", 
                               names=("Galaxy", "RSA morph. type", "T", "Terr", 
                                      "D", "T88 Group", "MK", "re", "sigc"))
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
                               data_start=2, delimiter=":", 
                               names=("Galaxy", "RSA morph. type", "T", "Terr", 
                                      "D", "T88 Group", "MK", "re", "sigc"))
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



if __name__ == "__main__":

    # Write Rampazzo parameters and magnitudes and MIR classes to paper
    #   directory.
    # Write ATLAS3D parameters and magnitudes to paper directory.
    # Write Jarrett fluxes (paper and calculated) to paper directory.
    # Move FSPS output to paper directory.
    pass
