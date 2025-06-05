
## Preface

A collection of Eureka! Control Files (.ecf files) and a Eureka! Parameter File (.epf file) are copied into the cells of this notebook. These parameters are setup with generally reasonable and performant settings that should give a good first-look at the data to determine whether or not the observations were successful. To expedite the process, quicklook analyses will start with _rateints.fits files downloaded from MAST and will then be processed through Stages 2--5 of Eureka!.

For the quicklook processing, only the Stage 5 Eureka! Parameter File will need to be adjusted based on the parameters of the particular planet that your data observed.

All told, running Stages 2--5 should take several minutes (e.g., ~9 minutes for a TRAPPIST-1b eclipse observation on a laptop with an M3 Max processor).

## Context behind the Stage 5 EPF

For all observations, we must specify some important details that will determine the setup of our fitted model including specifying any parameters we want to fix to a specific value and specifying Bayesian priors on any parameters we want fitted. These model settings are specified through a Eureka! Parameter File (EPF; file extension of .epf); we will specify the contents of that file in the next cell of the notebook and then write that to a file.

In particular, you will need to specify the planet-to-star radius ratio (`rp`; unitless), planet-to-star flux ratio (`fp`; unitless; also called the eclipse depth), orbital period (`per`; units of days), expected mid-time of the eclipse (`t_secondary`; units of BMJD_TDB), orbital inclination (`inc`; units of degrees), orbital semi-major axis to stellar radius ratio (`a`; unitless). If relevant to your particular system, you may also need to fit for the eccentricity (`ecc`; unitless) and longitude of periastron (`w`; units of degrees), or alternatively fit for `ecosw` and `esinw` which allow for better behaved fits with less of a bias towards `e > 0`.

The JWST Scheduling Team of the CIT provides the JWST Data Analysis Team of the CIT with the expected values for each of these parameters.

For our quicklook analysis, we will "fix" all astrophysical parameters to the values provided by the observation planning team with the exception of `fp` (which is our main parameter of interest) and `t_secondary` (which will be one of the most uncertain orbital parameters, a priori). To fix parameters, you must simply specify their expected values in the second column of the EPF file, under the column header "Value". To allow `fp` and `t_secondary` to vary within some reasonable level of uncertainty, we must provide Bayesian priors to the model, and we will base these on the expectations provided by the observation planning team. In particular, we will set the "Value" and "PriorPar1" columns to the expected values provided by the observation planning team; the "Value" column sets the starting point of your model fit, while the "PriorPar1" column sets the mean of the Gaussian prior (which "PriorType" is set to "N", which stands for "Normal"). We will then set the "PriorPar2" column of the `t_secondary` parameter to the 1-sigma uncertainty on the mid-eclipse time provided by the observation planning team. However, for `fp` we want to minimally bias our conclusions, so we will set "PriorPar2" to be `2000e-6` which will allow for a very broad range of eclipse depths to be considered by the fitter.

All other variables starting from the "Systematic variables" header have already been setup to have reasonable starting points and priors that should work well enough for all of our observations.