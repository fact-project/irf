# FACT irf
Tools to calculate IRFs (instrument response functions) for the FACT telescope.

# Joint Crab Data and the FACT Open Data Release

The FACT open data release contains 17.7 hours of Crab Nebula observations taken in October 2013 by the First G-APD Cherenkov Telescope (FACT)
FACT is located on the Roque des los Muchachos on the Canary island of La Palma off the west coast of Africa. It is an imaging atmospheric
cherenkov telescope protoyping silicon-photomultipliers.
FACT data is stored in FITS files from the raw data level up to higher levels containing the results of our analysis.
All of the data used here is publicly available, including the simulated data used for gauging the instrument response. For more information about the data visit [https://fact-project.org/data](https://fact-project.org/data) or read the corresponding section in the paper.

All of FACT's software is open source and can be accessed at [https://github.com/fact-project/](https://github.com/fact-project/).

For more information about FACT's data do not hesitate to contact

 * Max Nöthe (maximilian.noethe@tu-dortmund.de)
 * Kai Brügge (kai.bruegge@tu-dortmund.de)

Or contact the fact-online mailing list at

 * fact-online@lists.phys.ethz.ch

 For the joint crab publication we manually selected a smaller dataset 
 due to some runs having bad weather and getting a dataset with roughly equal 
 number of excess events for each experiment.

```
python irf/scripts/fact_dl3_to_gadf.py -c 0.8 -t 0.03 ../gamma_diffuse_showers.hdf5 ../gamma_diffuse_precuts.hdf5 ../crab_dl3.hdf5 fact_dl3 --start '2013/11/04 00:00' --end '2013/11/07' -e 20131104_196 -e 20131105_175 -e 20131105_176 -e 20131105_199 -e 20131105_201

```
