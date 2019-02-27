import fact.io
import os
import pytest
from irf import gadf
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


@pytest.fixture
def predictions():
    return fact.io.read_data(
        os.path.join(FIXTURE_DIR, 'gamma_predictions_dl2.hdf5'), key='events')


@pytest.fixture
def showers():
    return fact.io.read_data(
        os.path.join(FIXTURE_DIR, 'showers.hdf5'), key='showers')


def test_timestamp_conversion_from_events(events):
    timestamp = gadf.time.timestamp_to_mjdref(events.timestamp)
    assert timestamp.unit == u.s


def test_timestamp_conversion_from_pandas(events):
    ts = pd.Series(['01-01-2013', '01-02-2013'], name='foo')
    timestamp = gadf.time.timestamp_to_mjdref(ts)
    assert len(timestamp) == 2
    assert timestamp.unit == u.s


def test_timestamp_conversion_from_numpy(events):
    ts = np.array(['2014-01-01T12:00', '2015-02-03T13:56:03.172'], dtype='datetime64')
    timestamp = gadf.time.timestamp_to_mjdref(ts)
    assert len(timestamp) == 2
    assert timestamp.unit == u.s


def test_edisp_irf_writing(predictions):
    t = gadf.response.energy_dispersion_hdu(predictions, bins=5, theta_bins=1)
    assert t.data['MATRIX'].shape == (1, 1, 5, 5)
    assert t.header['HDUCLAS1'] == 'RESPONSE'
    assert t.columns['ENERG_LO'].unit == u.TeV


def test_aeff_irf_writing(showers, predictions, tmpdir):
    t = gadf.response.effective_area_hdu(showers, predictions, bins=6)
    assert t.data['ENERG_LO'].shape == (1, 6)
    assert t.columns['ENERG_LO'].unit == u.TeV
    assert t.data['EFFAREA'].shape == (1, 4, 6)
    assert t.columns['EFFAREA'].unit == u.m**2
