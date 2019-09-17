import fact.io
import os
import pytest
from irf import collection_area
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
    predictions['energy'] = predictions['corsika_event_header_total_energy']
    r = collection_area(
        showers.energy * u.TeV,
        predictions.energy * u.TeV,
        bins=6,
        impact=270 * u.m
    )
    area = r[0]
    assert area.si.unit == u.m**2


def test_sample_fraction(showers, predictions):
    predictions['energy'] = predictions['corsika_event_header_total_energy']
    r = collection_area(
        showers.energy * u.TeV,
        predictions.energy * u.TeV,
        bins=6,
        impact=270 * u.m,
        sample_fraction=1,
    )
    area_lower = r[0]

    r = collection_area(
        showers.energy * u.TeV,
        predictions.energy * u.TeV,
        bins=6,
        impact=270 * u.m,
        sample_fraction=0.5,
    )
    area_higher = r[0]

    # effective area should be twice as high if sample fraction is 0.5
    np.testing.assert_array_equal(2 * area_lower.value, area_higher.value)



def test_scatter_radius(showers, predictions):
    predictions['energy'] = predictions['corsika_event_header_total_energy']
    r = collection_area(
        showers.energy * u.TeV,
        predictions.energy * u.TeV,
        bins=6,
        impact=1 * u.m,
    )
    area_lower = r[0]

    r = collection_area(
        showers.energy * u.TeV,
        predictions.energy * u.TeV,
        bins=6,
        impact=2 * u.m,
    )
    area_higher = r[0]

    # effective area should be four times as high if impact is twice as high
    np.testing.assert_array_equal(4 * area_lower.value, area_higher.value)
