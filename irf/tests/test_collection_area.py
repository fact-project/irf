import fact.io
import os
import pytest
from irf import collection_area, collection_area_to_irf_table
import astropy.units as u
import numpy as np

FIXTURE_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'test_files',
)


@pytest.fixture
def showers():
    return fact.io.read_data(
        os.path.join(FIXTURE_DIR, 'showers.hdf5'), key='showers')


@pytest.fixture
def predictions():
    return fact.io.read_data(
        os.path.join(FIXTURE_DIR, 'gamma_predictions_dl2.hdf5'), key='events')


def test_units(showers, predictions):
    assert len(showers) > 0
    assert len(predictions) > 0
    predictions['energy'] = predictions['corsika_evt_header_total_energy']
    r = collection_area(
        showers.energy * u.TeV, predictions.energy * u.TeV, bins=10, impact=270 * u.m)
    area = r[0]
    assert area.si.unit == u.m**2


def test_irf_writing(showers, predictions, tmpdir):
    predictions['energy'] = predictions['corsika_evt_header_total_energy']

    shower_energy = (showers.energy.values * u.GeV).to('TeV')
    true_event_energy = (predictions.corsika_evt_header_total_energy.values * u.GeV).to('TeV')
    offsets = np.ones_like(true_event_energy.value) * u.deg
    t = collection_area_to_irf_table(shower_energy, true_event_energy, offsets, bins=20)
    assert t['ENERG_LO'].shape == (1, 20)
    assert t['ENERG_LO'].unit == u.TeV
    assert t['EFFAREA'].shape == (1, 3, 20)
    assert t['EFFAREA'].unit == u.m**2
    assert len(t) == 1
