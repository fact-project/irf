import fact.io
import os
import pytest
from irf import oga
import astropy.units as u
import pandas as pd
import numpy as np


FIXTURE_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'test_files',
)


@pytest.fixture
def events():
    return fact.io.read_data(
        os.path.join(FIXTURE_DIR, 'crab_dl3_small.hdf5'), key='events')


def test_timestamp_conversion_from_events(events):
    timestamp = oga.timestamp_to_mjdref(events.timestamp)
    assert timestamp.unit == u.s


def test_timestamp_conversion_from_pandas(events):
    ts = pd.Series(['01-01-2013', '01-02-2013'], name='foo')
    timestamp = oga.timestamp_to_mjdref(ts)
    assert len(timestamp) == 2
    assert timestamp.unit == u.s


def test_timestamp_conversion_from_numpy(events):
    ts = np.array(['2014-01-01T12:00', '2015-02-03T13:56:03.172'], dtype='datetime64')
    timestamp = oga.timestamp_to_mjdref(ts)
    assert len(timestamp) == 2
    assert timestamp.unit == u.s
