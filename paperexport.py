import os

from astropy.table import Table

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
                               data_start=3, names = ("Galaxy",
                                                      "RSA morph. type", "T",
                                                      "Terr", "D", "H0D",
                                                      "T88 Group", "MK", "H0M",
                                                      "re", "sigc"))

                               and)

if __name__ == "__main__":

    # Write Rampazzo parameters and magnitudes and MIR classes to paper
    #   directory.
    # Write ATLAS3D parameters and magnitudes to paper directory.
    # Write Jarrett fluxes (paper and calculated) to paper directory.
    # Move FSPS output to paper directory.

