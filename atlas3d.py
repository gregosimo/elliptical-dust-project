
import matplotlib.pyplot as plt

import photometry as phot

# W1, W2, W3, W4
WISE_wavelengths = np.array([3.4e-6, 4.6e-6, 12e-6, 22e-6])*1e6
# NUV, FUV
GALEX_wavelengths = np.array([2267e-10, 1516e-10])*1e6
# J, H, Ks
TWOMASS_wavelengths = np.array([1.24e-6, 1.66e-6, 2.16e-6])*1e6

WISE_frequencies = 3e10 / (WISE_wavelengths * 1e-4)
GALEX_frequencies = 3e10 / (GALEX_wavelengths * 1e-4)
TWOMASS_frequencies = 3e10 / (TWOMASS_wavelengths * 1e-4)
TWOMASS_ZP = np.array([1594, 1024, 666.7])

def plot_Conroy_SED(ATLAS3DBASE, atlas3d_sample_row
    '''Plots an SED against the Conroy model for ATLAS3D targets.'''
    # Generate photometry.
    W1, W1_err = phot.galaxy_photometry(ATLAS3DBASE, 
        atlas3d_sample_row["objstr_01"], "W1", brightness="flux")
    W2, W2_err = phot.galaxy_photometry(ATLAS3DBASE, 
        atlas3d_sample_row["objstr_01"], "W2", brightness="flux")
    W3, W3_err = phot.galaxy_photometry(ATLAS3DBASE, 
        atlas3d_sample_row["objstr_01"], "W3", brightness="flux")
    W4, W4_err = phot.galaxy_photometry(ATLAS3DBASE, 
        atlas3d_sample_row["objstr_01"], "W4", brightness="flux")
    NUV, NUV_err = phot.galaxy_photometry(ATLAS3DBASE, 
        atlas3d_sample_row["objstr_01"], "NUV", brightness="flux")
    FUV, FUV_err = phot.galaxy_photometry(ATLAS3DBASE, 
        atlas3d_sample_row["objstr_01"], "FUV", brightness="flux")

    WISE_flux = np.array([W1, W2, W3, W4])
    WISE_flux_err = np.array([W1_err, W2_err, W3_err, W4_err])
    GALEX_flux = np.array([NUV, FUV])
    GALEX_flux_err = np.array([NUV_err, FUV_err])
    TWOMASS = np.array([atlas3d_sample_row["j_m_k20fe"], 
        atlas3d_sample_row["h_m_k20fe"], atlas3d_sample_row["k_m_k20fe"]])
    TWOMASS_err = np.array([atlas3d_sample_row["j_msig_k20fe"], 
        atlas3d_sample_row["h_msig_k20fe"], atlas3d_sample_row["k_msig_k20fe"]])
    TWOMASS_flux = conv.mag2flux(TWOMASS, 0, TWOMASS_ZP)
    TWOMASS_flux_err = conv.magerr2fluxerr(TWOMASS, TWOMASS_err, 0, 0, 
    TWOMASS_ZP, 0)

    # This is normalized to H-band
    fsps = Table.read(os.path.join(ATLAS3dBASE, "fsps-egals.txt", format="ascii", 
                      names=["Wave", "F(0.2Gyr)", "F(2Gyr)", "F(5Gyr)", "F(10Gyr)"])
    fsps_02 = fsps["F(0.2Gyr)"] * TWOMASS_flux[1] / fsps["F(0.2Gyr)"][802]
    fsps_2 = fsps["F(2Gyr)"] * TWOMASS_flux[1] / fsps["F(2Gyr)"][802]
    fsps_5 = fsps["F(5Gyr)"] * TWOMASS_flux[1] / fsps["F(5Gyr)"][802]
    fsps_10 = fsps["F(10Gyr)"] * TWOMASS_flux[1] / fsps["F(10Gyr)"][802]
    fsps_frequencies = 3e10 / (fsps["Wave"] * 1e-4)

    plt.plot(fsps["Wave"], fsps_frequencies * fsps_02, 'b-', label="0.2 Gyr")
    plt.plot(fsps["Wave"], fsps_frequencies * fsps_2, 'g-', label="2 Gyr")
    plt.plot(fsps["Wave"], fsps_frequencies * fsps_5, 'r-', label="5 Gyr")
    plt.plot(fsps["Wave"], fsps_frequencies * fsps_10, 'c-', label="10 Gyr")

    plt.errorbar(GALEX_wavelengths, GALEX_flux * GALEX_frequencies, 
        GALEX_flux_err, fmt='o')
    plt.errorbar(WISE_wavelengths, WISE_flux * WISE_frequencies, WISE_flux_err, 
        fmt='ro')
    plt.errorbar(TWOMASS_wavelengths, TWOMASS_flux * TWOMASS_frequencies, 
        TWOMASS_flux_err, fmt='ro')
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.xlabel("Wavelength (um)")
    plt.ylabel("vFv (Jansky Hz)")
    objname = atlas3d_sample["objstr_01"][i]
    plt.title("SED for {0}; SSP Age: {1} Gyr".format(objname, 
        atlas3d_table["Age_SSP"][phot.astropy_table_index(atlas3d_sample, 
        "objstr_01", objname)][0]))
