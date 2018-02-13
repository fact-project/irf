import fact.io
import os
import pytest
from irf import collection_area, collection_area_to_irf_table
import astropy.units as u


FIXTURE_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'test_files',
)


@pytest.fixture
def showers():
    return fact.io.read_data(os.path.join(FIXTURE_DIR, 'showers.hdf5'), key='table')


@pytest.fixture
def predictions():
    return fact.io.read_data(os.path.join(FIXTURE_DIR, 'gamma_predictions_dl2.hdf5'), key='events')


def test_units(showers, predictions):
    assert len(showers) > 0
    assert len(predictions) > 0
    predictions['energy'] = predictions['corsika_evt_header_total_energy']
    r = collection_area(showers.energy, predictions.energy, bins=10, impact=270 * u.m)
    area = r[0]
    assert area.si.unit == u.m**2


def test_irf_writing(showers, predictions, tmpdir):
    predictions['energy'] = predictions['corsika_evt_header_total_energy']

    r = collection_area(showers.energy, predictions.energy, bins=10, impact=270 * u.m,)
    area, bin_center, bin_width, lower_conf, upper_conf = r

    t = collection_area_to_irf_table(area, bin_center, bin_width)

    assert len(t) == 1
