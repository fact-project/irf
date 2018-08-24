'''
This module contains methods to help handle the fits tables
required to work with the gamma-astro-data-formats (gadf):

https://gamma-astro-data-formats.readthedocs.io/en/latest/
'''

import astropy.units as u
from astropy.io import fits
import pandas as pd
from astropy.table import Table
from astropy.time import Time
import numpy as np
import datetime
from irf.gadf.time import timestamp_to_mjdref, ontime_info_from_runs, TIME_INFO


def create_gti_hdu(run):
    '''
    Creates the GTI hdu which gets added to the dl3 files.
    See https://gamma-astro-data-formats.readthedocs.io/en/latest/events/gti.html
    '''

    start_time = timestamp_to_mjdref(run.run_start)
    stop_time = timestamp_to_mjdref(run.run_stop)

    t = Table({'START': start_time, 'STOP': stop_time})
    hdu = fits.table_to_hdu(t)
    add_meta_information_to_hdu(hdu)
    hdu.header['EXTNAME'] = 'GTI'
    hdu.header['HDUCLAS1'] = 'GTI'
    return hdu


def create_primary_hdu():
    '''
    Creates a primary fits HDU common to all FACT fits files.
    '''
    header = fits.Header()

    header['OBSERVER'] = 'The FACT collaboration'
    header['COMMENT'] = 'FACT OGA.'
    header['COMMENT'] = 'See https://gamma-astro-data-formats.readthedocs.io/en/latest/'
    header['COMMENT'] = 'This file was created by https://github.com/fact-project/irf'
    header['COMMENT'] = 'See our full analysis on GitHub'
    header['COMMENT'] = 'https://github.com/fact-project/open_crab_sample_analysis'

    now = Time.now().iso
    header['COMMENT'] = f'This file was created on {now}'

    return fits.PrimaryHDU(header=header)


def create_events_hdu(dl3, run):
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

    # add information to HDU header
    add_meta_information_to_hdu(hdu)
    hdu.header['EXTNAME'] = 'EVENTS'
    hdu.header['HDUCLAS1'] = 'EVENTS'
    hdu.header['OBS_ID'] = _observation_ids(run)
    hdu.header['OBJECT'] = run.source
    d = ontime_info_from_runs(run)
    _extend_hdu_header(hdu.header, d)

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

    f = _file_names_from_run_table(runs)
    p = np.repeat(path_to_irf_file, 2 * len(runs))
    file_name = np.append(f, p)
    hdu_name = np.repeat(['EVENTS', 'EFFECTIVE AREA', 'ENERGY DISPERSION'], len(runs))

    obs_id = np.tile(_observation_ids(runs), 3)

    d = {
        'OBS_ID': obs_id,
        'HDU_TYPE': hdu_type,
        'HDU_CLASS': hdu_class,
        'FILE_DIR': file_dir,
        'FILE_NAME': file_name,
        'HDU_NAME': hdu_name,
    }

    hdu = fits.table_to_hdu(Table(d))
    add_meta_information_to_hdu(hdu)
    hdu.header['EXTNAME'] = 'HDU_INDEX'
    return hdu


def create_observation_index_hdu(runs):
    '''
    Takes a pandas dataframe which contains the information about runs.
    Take all the mandatory keywords found here:
    http://gamma-astro-data-formats.readthedocs.io/en/latest/data_storage/obs_index/index.html

    returns a fits hdu object
    '''
    obs_id = _observation_ids(runs)
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

    hdu.header['EXTNAME'] = 'OBS_INDEX'

    add_meta_information_to_hdu(hdu)
    return hdu


def add_meta_information_to_hdu(hdu, **kwargs):
    '''
    Takes an hdu object and adds meta information as required by the open gamma ray astro
    format to its header.
    '''
    hdu.header['CREATOR'] = ('FACT IRF', 'See https://github.com/fact-project/irf')
    hdu.header['TELESCOP'] = ('FACT', 'The First G-APD Cherenkov Telescope.')
    hdu.header['HDUCLASS'] = ('GADF', 'Gamma-Ray Astro Data Format')
    hdu.header['HDUDOC'] = 'https://gamma-astro-data-formats.readthedocs.io'
    hdu.header['HDUVERS'] = '0.2'
    hdu.header['EQUINOX'] = '2000.0'
    hdu.header['RADECSYS'] = 'ICRS'
    hdu.header['EUNIT'] = 'TeV'
    hdu.header['DATE'] = datetime.datetime.now().replace(microsecond=0).isoformat()
    _extend_hdu_header(hdu.header, TIME_INFO)
    if kwargs:
        _extend_hdu_header(hdu.header, kwargs)


def _extend_hdu_header(header, values_dict):
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


def _observation_ids(runs):
    '''
    Creates a single integer from the night and run numbers for given run(s)
    '''

    if isinstance(runs, pd.core.series.Series) and 'night' not in runs:
        return int(runs.name[0] * 1E3 + runs.name[1])

    return np.array(runs.night * 1E3 + runs.run_id).astype(np.int)


def _file_names_from_run_table(runs):
    '''
    Creates a file name from the `run_id` and `night` entries in the runs table.
    returns a list of filenames of the form <night>_<run_id>_dl3.fits
    '''
    run_ids = runs.run_id.values.astype(str)
    nights = runs.night.values.astype(str)
    names = [f'{n}_{r}_dl3.fits' for n, r in zip(nights, run_ids)]
    return names