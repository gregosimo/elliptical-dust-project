 PROGRAM EARLY_CSP

  !set up modules
  USE sps_vars; USE sps_utils  
  IMPLICIT NONE

  INTEGER :: i
  !define variable for SSP spectrum
  !This variable is essentially a time-series of spectra.
  !
  !ntfull comes from sps_vars and is the number of timesteps with which the
  !population will be evaluated.
  !nspec is the number of points in the stellar spectrum.
  REAL(SP), DIMENSION(ntfull,nspec)  :: spec_ssp
  !define variables for Mass and Lbol info
  !Mass over time and luminosity over time.
  REAL(SP), DIMENSION(ntfull)    :: mass_ssp,lbol_ssp
  ! The file to be written to by COMPSP
  CHARACTER(100) :: file1,file2
  !structure containing all necessary parameters
  ! This is complicated. See sps_vars for full definition...
  ! But it's basically a structured array with SSP parameters inside.
  TYPE(PARAMS) :: pset
  !define structure for CSP spectrum
  !This will be a timeseries which contains masses and stuff about the star
  !formation history. I'm not entirely sure what this is about.
  TYPE(COMPSPOUT), DIMENSION(ntfull) :: ocompsp

  !---------------------------------------------------------------!
  !---------------------------------------------------------------!
  
  ! Lets compute an SSP, solar metallicity, with a Salpeter IMF
  ! with no dust, and the 'default' assumptions regarding the 
  ! locations of the isochrones

  imf_type  = 0             !define the IMF (1=Chabrier 2003)
                            !see sps_vars.f90 for details of this var
  pset%zmet = 11            !define the metallicity (see the manual)
                            !20 = solar metallacity
  add_agb_dust_model = 1    !Toggle the Villaume dust model. Good to determine
                            !how much of the changes we see are caused by dust.

  ! Reads in all of the isochrones/libraries for a given metallicity. The
  ! metallicity should be specified as the argument. In order to read in all
  ! metallicities, -1 should be provided.
  CALL SPS_SETUP(pset%zmet) !read in the isochrones and spectral libraries


  !These are parameters for the delayed-tau model.
  pset%sfh      = 1     !set SFH to SSP
  pset%tau      = 0.1   !Timescale for the suppression of star formation
  pset%const    = 0.0
  pset%sf_start = 0.0   !When 
  !pset%tburst   = 11.0  !When the additional burst of star formation occurs.
  pset%fburst   = 0.1   ! CHANGE: Set some fraction of stars to form later.

  !define the parameter set.  These are the default values, specified 
  !in sps_vars.f90, but are explicitly included here for transparency
  pset%zred  = 0.0   !redshift  
  pset%dust1 = 0.0   !dust parameter 1
  pset%dust2 = 0.0   !dust parameter 2

  pset%dell  = 0.0   !shift in log(L) for TP-AGB stars
  pset%delt  = 0.0   !shift in log(Teff) for TP-AGB stars
  pset%fbhb  = 0.0   !fraction of blue HB stars
  pset%sbss  = 0.0   !specific frequency of BS stars

  !compute the CSP
  DO i=1,14
    pset%tburst = i
    ! Input is the parameter set (pset)
    ! Outputs are the time-dependent mass, bolometric luminosity, and spectra.
    CALL SSP_GEN(pset,mass_ssp,lbol_ssp,spec_ssp)
    !compute mags and write out mags and spec for SSP
    ! Compute the composite stellar population given a star-formation history, and
    ! write out the magnitudes.
    ! NOTE: WHEN CHANGING THE FILENAME, CHANGE THE LENGTH OF THE FORMAT!!!
    WRITE (file1, "(A25,I0.2)") "tburst_change/noagbdust_t", i
    CALL COMPSP(3,1,file1,mass_ssp,lbol_ssp,spec_ssp,pset,ocompsp)
  END DO

  ! Now make an SSP for comparison.
  pset%sfh  = 0 
  CALL SSP_GEN(pset,mass_ssp,lbol_ssp,spec_ssp)
  file2 = "tburst_change/ssp"
  CALL COMPSP(3,1,file2,mass_ssp,lbol_ssp,spec_ssp,pset,ocompsp)

 END PROGRAM EARLY_CSP
