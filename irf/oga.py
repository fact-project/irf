'''
This module contains methods to help handle the fits tables
required to work with the open gamma-ray astro formats (oga):

https://gamma-astro-data-formats.readthedocs.io/en/latest/

'''

import fact.io
import astropy.units as u
from astropy.io import fits
from pandas.api.types import is_datetime64_any_dtype
import pandas as pd
from astropy.table import Table
from astropy.time import Time
import numpy as np
from astropy.coordinates.angle_utilities import angular_separation


# the timestamps in the fits files need a reference MJD time according to some (ogip?) standard.
# The exact value is not important for anything.
MJDREF = 55835  # MJD sometime near FACT's first light. 2011-10-01T00:00:00 UTC


def extend_hdu_header(header, values_dict):
    '''
    Extend existing hdu header by values in a dict.
    It automatically strips units from any value in the passed dict.
    '''
    for k, v in values_dict.items():
        if not np.isscalar(v):
            if len(v) > 1:
                raise TypeError('Can only add scalar values to a header keyword')
            v = v[0]
        try:
            header[k] = v.value
        except AttributeError:
            header[k] = v


def calculate_fov_offset(df):
    '''
    Calculate the `offset` aka the `angular_separation` between the pointing and
    the source position.
    '''
    pointing_lat = (90 - df.aux_pointing_position_zd.values) * u.deg
    pointing_lon = df.aux_pointing_position_az.values * u.deg

    source_lat = (90 - df.source_position_zd.values) * u.deg
    source_lon = df.source_position_az.values * u.deg

    return angular_separation(pointing_lon, pointing_lat, source_lon, source_lat).to('deg')


def observation_id(night, run):
    '''
    Creates a single integer from the night and run numbers.
    '''
    obs_id = (night * 1E3 + run).values
    return obs_id.astype(int)


def file_names_from_run_table(runs):
    '''
    Creates a file name from the `run_id` and `night` entries in the runs table.
    returns a list of filenames of the form <night>_<run_id>_dl3.fits
    '''
    run_ids = runs.run_id.values.astype(str)
    nights = runs.night.values.astype(str)
    names = [f'{n}_{r}_dl3.fits' for n, r in zip(nights, run_ids)]
    return names



def timestamp_to_mjdref(timestamp, mjdref=MJDREF, location=fact.instrument.constants.LOCATION):
    '''
    Convert a timestamp (or many timestamps in a pandas series) to seconds relative to the MJDREF
    keyword which has to be given in the FITS header.

    I convert this to a pandas datetime thing before creating a astropy Time object.

    See https://github.com/astropy/astropy/issues/6428
    '''
    timestamp = pd.Series(pd.to_datetime(timestamp))

    time = Time(timestamp.asobject, location=location)
    time = time - Time(mjdref, scale='utc', format='mjd')

    return np.array(time.to(u.s).value, ndmin=1) * u.s


def create_gti_hdu(run):
    '''
    Creates the GTI hdu which gets added to the dl3 files.
    See https://gamma-astro-data-formats.readthedocs.io/en/latest/events/gti.html
    '''

    start_time = timestamp_to_mjdref(run.run_start)
    stop_time = timestamp_to_mjdref(run.run_stop)

    t = Table({'START': start_time, 'STOP': stop_time})
    hdu = fits.table_to_hdu(t)
    add_time_information_to_hdu(hdu)

    hdu.header['EXTNAME'] = 'GTI'
    hdu.header['CREATOR'] = 'FACT IRF'
    hdu.header['TELESCOP'] = 'FACT'
    return hdu


def create_primary_hdu():
    '''
    Creates a primary fits HDU common to all FACT fits files.
    '''
    header = fits.Header()

    header['OBSERVER'] = 'The non-insane FACT guys '
    header['COMMENT'] = 'FACT OGA.'
    header['COMMENT'] = 'See https://gamma-astro-data-formats.readthedocs.io/en/latest/'
    header['COMMENT'] = 'This file was created by https://github.com/fact-project/irf'
    header['COMMENT'] = 'See our full analysis here https://github.com/fact-project/open_crab_sample_analysis'

    now = Time.now().iso
    header['COMMENT'] = f'This file was created on {now}'

    return fits.PrimaryHDU(header=header)


def create_dl3_hdu(dl3, run):
    '''
    Takes a pandas dataframe which contains the DL3 Information and creates an hdu
    according to the standard found here:
    http://gamma-astro-data-formats.readthedocs.io/en/latest/events/events.html

    returns a fits hdu object
    '''
    # the format expects to see some event id. I just take the index
    event_id = dl3.index

    # convert from hour angles to deg
    ra = dl3.ra_prediction.values * u.hourangle
    ra = ra.to('deg')

    dec = dl3.dec_prediction.values * u.deg

    # convert to TeV energy
    energy = dl3.gamma_energy_prediction.values * u.GeV
    energy = energy.to('TeV')

    timestamp = timestamp_to_mjdref(dl3.timestamp)
    t = Table({'EVENT_ID': event_id, 'ENERGY': energy, 'DEC': dec, 'RA': ra, 'TIME': timestamp})
    hdu = fits.table_to_hdu(t)
    add_time_information_to_hdu(hdu)
    hdu.header['EUNIT'] = 'TeV'
    hdu.header['EXTNAME'] = 'EVENTS'
    hdu.header['CREATOR'] = 'FACT IRF'
    hdu.header['TELESCOP'] = 'FACT'
    d = ontime_info_from_runs(run)
    extend_hdu_header(hdu.header, d)

    return hdu


def create_index_hdu(runs, path_to_irf_file='fact_irf.fits'):
    '''
    The index hdu contains the paths and filenames to other fits files containing
    EVENTS, EFFECTIVE AREA, ENERGY DISPERSION.

    See http://gamma-astro-data-formats.readthedocs.io/en/latest/data_storage/hdu_index/index.html

    returns a fits hdu object
    '''

    hdu_type = np.repeat(['events', 'aeff', 'edisp'], len(runs))
    hdu_class = np.repeat(['events', 'aeff_2d', 'edisp_2d'], len(runs))
    file_dir = np.repeat('./', 3 * len(runs))

    f = file_names_from_run_table(runs)
    p = np.repeat(path_to_irf_file, 2 * len(runs))
    file_name = np.append(f, p)
    hdu_name = np.repeat(['EVENTS', 'EFFECTIVE AREA', 'ENERGY DISPERSION'], len(runs))

    obs_id = np.tile(observation_id(runs.night, runs.run_id), 3)

    d = {
        'OBS_ID': obs_id,
        'HDU_TYPE': hdu_type,
        'HDU_CLASS': hdu_class,
        'FILE_DIR': file_dir,
        'FILE_NAME': file_name,
        'HDU_NAME': hdu_name,
    }

    hdu = fits.table_to_hdu(Table(d))
    add_time_information_to_hdu(hdu)
    hdu.header['EUNIT'] = 'TeV'
    hdu.header['EXTNAME'] = 'HDU_INDEX'
    hdu.header['CREATOR'] = 'FACT IRF'
    hdu.header['TELESCOP'] = 'FACT'
    return hdu


def ontime_info_from_runs(runs):

    # to make this work for a single run (a pandas series) convert to table
    if isinstance(runs, pd.core.series.Series):
        runs = runs.to_frame().T


    if not is_datetime64_any_dtype(runs.run_start):
        runs.run_start = pd.to_datetime(runs.run_start, infer_datetime_format=True)

    if not is_datetime64_any_dtype(runs.run_stop):
        runs.run_stop = pd.to_datetime(runs.run_stop, infer_datetime_format=True)

    # in the format specified here
    # http://gamma-astro-data-formats.readthedocs.io/en/latest/data_storage/obs_index/index.html
    # ontime is the total observation time (including deadtime)
    ontime = runs.run_stop - runs.run_start
    ontime = ontime.dt.seconds.values * u.s
    livetime = runs.ontime.values * u.s  # the runs table contains the thing FACT calls ontime
    deadc = livetime / ontime
    deadc = deadc.value

    tstart = Time(runs.run_start.asobject, location=fact.instrument.constants.LOCATION)
    tstart = tstart - Time(MJDREF, scale='utc', format='mjd')
    tstart = tstart.to('s')

    tstop = Time(runs.run_stop.asobject, location=fact.instrument.constants.LOCATION)
    tstop = tstop - Time(MJDREF, scale='utc', format='mjd')
    tstop = tstop.to('s')

    date_obs = runs.run_start.dt.strftime('%Y-%m-%d').values.astype(np.str)
    date_end = runs.run_stop.dt.strftime('%Y-%m-%d').values.astype(np.str)

    time_obs = runs.run_start.dt.strftime('%H:%M:%S').values.astype(np.str)
    time_end = runs.run_stop.dt.strftime('%H:%M:%S').values.astype(np.str)

    return {
        'ONTIME': ontime,
        'LIVETIME': livetime,
        'DEADC': deadc,
        'TSTART': tstart,
        'TSTOP': tstop,
        'DATE-OBS': date_obs,
        'TIME-OBS': time_obs,
        'DATE-END': date_end,
        'TIME-END': time_end,
    }


def create_observation_index_hdu(runs):
    '''
    Takes a pandas dataframe which contains the information about runs.
    Take all the mandatory keywords found here:
    http://gamma-astro-data-formats.readthedocs.io/en/latest/data_storage/obs_index/index.html

    returns a fits hdu object
    '''
    obs_id = observation_id(runs.night, runs.run_id)
    ra_pnt = runs.right_ascension.values * u.hourangle
    ra_pnt = ra_pnt.to('deg')

    dec_pnt = runs.declination.values * u.deg

    zen_pnt = runs.zenith.values * u.deg
    alt_pnt = (90 - runs.zenith.values) * u.deg
    az_pnt = runs.azimuth.values * u.deg


    n_tels = np.ones_like(obs_id)
    tellist = np.ones_like(obs_id).astype(np.str)
    quality = np.zeros_like(obs_id)

    d = {
        'OBS_ID': obs_id,
        'RA_PNT': ra_pnt,
        'DEC_PNT': dec_pnt,
        'ZEN_PNT': zen_pnt,
        'ALT_PNT': alt_pnt,
        'AZ_PNT': az_pnt,
        'N_TELS': n_tels,
        'TELLIST': tellist,
        'QUALITY': quality,
        **ontime_info_from_runs(runs)
    }

    hdu = fits.table_to_hdu(Table(d))
    hdu.header['EUNIT'] = 'TeV'
    hdu.header['EXTNAME'] = 'OBS_INDEX'
    add_time_information_to_hdu(hdu)
    return hdu


def add_time_information_to_hdu(hdu):
    '''
    Takes an hdu object and adds information about FACTs time reference to it.
    These values are constants like the location of the telescope and its altitude ASL.
    As well as the TIMESYS keyword which is always fixed to 'utc' for FACT.
    '''
    hdu_header = hdu.header
    hdu_header['MJDREFI'] = MJDREF
    hdu_header['MJDREFF'] = 0
    hdu_header['TIMEREF'] = 'local'
    hdu_header['TIMESYS'] = 'utc'
    hdu_header['GEOLON'] = -17.890701
    hdu_header['GEOLAT'] = 28.761795
    hdu_header['ALTITUDE'] = 2199.4
