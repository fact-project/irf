import fact.io
import astropy.units as u
from astropy.io import fits
from pandas.api.types import is_datetime64_any_dtype
import pandas as pd
from astropy.table import Table
from astropy.time import Time
import numpy as np

MJDREF = 55835  # MJD sometime near FACT's first light. 2011-10-01T00:00:00 UTC


def observation_id(night, run):
    obs_id = (night * 1E3 + run).values
    return obs_id.astype(int)


def file_names_from_runs(runs):
    run_ids = runs.run_id.values.astype(str)
    nights = runs.night.values.astype(str)
    names = [f'{n}_{r}_dl3.fits' for n, r in zip(nights, run_ids)]
    return names



def create_primary_hdu():
    header = fits.Header()

    header['OBSERVER'] = 'The non-insane FACT guys '
    header['COMMENT'] = 'FACT OGA. Very preliminary'
    header['COMMENT'] = 'See https://gamma-astro-data-formats.readthedocs.io/en/latest/'
    header['COMMENT'] = 'This file was created by https://github.com/fact-project/irf'
    header['COMMENT'] = 'See our full analysis here https://github.com/fact-project/open_crab_sample_analysis'

    now = Time.now().iso
    header['COMMENT'] = f'This file was created on {now}'

    return fits.PrimaryHDU(header=header)


def create_dl3_hdu(dl3):
    '''
    Takes a pandas dataframe which contains the DL3 Information and creates an hdu
    accroding to the standard found here:
    http://gamma-astro-data-formats.readthedocs.io/en/latest/events/events.html
    return a fits hdu object
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

    # convert to timestamps. older pyfact versions don't do this automagically
    if not is_datetime64_any_dtype(dl3.timestamp):
        dl3.timestamp = pd.to_datetime(dl3.timestamp, infer_datetime_format=True)

    time = Time(dl3.timestamp.asobject, location=fact.instrument.constants.LOCATION)
    time = time - Time(MJDREF, scale='utc', format='mjd')

    t = Table({'EVENT_ID': event_id, 'ENERGY': energy, 'DEC': dec, 'RA': ra, 'TIME': time.to('s')})
    hdu = fits.table_to_hdu(t)
    add_time_information_to_hdu(hdu)
    hdu.header['EUNIT'] = 'TeV'
    hdu.header['EXTNAME'] = 'EVENTS'
    hdu.header['CREATOR'] = 'FACT IRF'
    hdu.header['TELESCOP'] = 'FACT'
    return hdu


def create_index_hdu(runs, path_to_irf_file='fact_irf.fits'):

    hdu_type = np.repeat(['events', 'aeff', 'edisp'], len(runs))
    hdu_class = np.repeat(['events', 'aeff_2d', 'edisp_2d'], len(runs))
    file_dir = np.repeat('./', 3 * len(runs))

    f = file_names_from_runs(runs)
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
        'ONTIME': ontime,
        'LIVETIME': livetime,
        'DEADC': deadc,
        'TSTART': tstart,
        'TSTOP': tstop,
        'DATE-OBS': date_obs,
        'TIME-OBS': time_obs,
        'DATE-END': date_end,
        'TIME-END': time_end,
        'N_TELS': n_tels,
        'TELLIST': tellist,
        'QUALITY': quality,
    }

    hdu = fits.table_to_hdu(Table(d))
    hdu.header['EUNIT'] = 'TeV'
    hdu.header['EXTNAME'] = 'OBS_INDEX'
    add_time_information_to_hdu(hdu)
    return hdu


def add_time_information_to_hdu(hdu):
    hdu_header = hdu.header
    hdu_header['MJDREFI'] = MJDREF
    hdu_header['MJDREFF'] = 0
    hdu_header['TIMEREF'] = 'local'
    hdu_header['TIMESYS'] = 'utc'
    hdu_header['GEOLON'] = -17.890701
    hdu_header['GEOLAT'] = 28.761795
    hdu_header['ALTITUDE'] = 2199.4
