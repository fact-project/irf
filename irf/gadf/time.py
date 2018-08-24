from pandas.api.types import is_datetime64_any_dtype
import pandas as pd
from astropy.time import Time
import astropy.units as u
import numpy as np
import fact

# the timestamps in the fits files need a reference MJD time according to some (ogip?) standard.
# The exact value is not important for anything.
MJDREF = 55835  # MJD sometime near FACT's first light. 2011-10-01T00:00:00 UTC



# These values are constants like the location of the telescope and its altitude ASL.
# As well as the TIMESYS keyword which is always fixed to 'utc' for FACT.
TIME_INFO = {
    'MJDREFI': MJDREF,
    'MJDREFF': 0,
    'TIMEREF': 'local',
    'TIMESYS': 'utc',
    'GEOLON': -17.890701,
    'GEOLAT': 28.761795,
    'ALTITUDE': 2199.4,
    'TIMEUNIT': 's',
}


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




def ontime_info_from_runs(runs):
    '''
    Calculates ontime information from (one or more) FACT runs as
    stored in the dl2 FACT hdf5 files.

    Parameters
    ----------
    runs : pd.DataFrame or pd.Series
        the runs as given in FACTS DL2 hdf5 files

    Returns
    -------
    dict
        returns dict containing necessary ontime Information
        as described here:
        https://gamma-astro-data-formats.readthedocs.io/en/latest/events/events.html
    '''

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
