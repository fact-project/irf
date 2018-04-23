import fact.io
import os
import pytest
from irf.models import MCSpectrum
from irf import collection_area, collection_area_to_irf_table
from irf.oga import calculate_fov_offset
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


def test_units(predictions):

    n_showers = 10E6
    area = (270 * u.m)**2 * np.pi
    mc_production_spectrum = MCSpectrum(200 * u.GeV, 50 * u.TeV, n_showers, area, index=-2.7)

    energies = predictions['corsika_event_header_total_energy'].values * u.GeV

    area, e_low, e_high = collection_area(
        mc_production_spectrum,
        energies,
        bins=10,
    )

    assert area.si.unit == u.m**2
    assert e_low.si.unit == u.m**2
    assert e_high.si.unit == u.m**2


def test_sample_fraction(predictions):

    n_showers = 10E6
    area = (270 * u.m)**2 * np.pi
    mc_production_spectrum = MCSpectrum(200 * u.GeV, 50 * u.TeV, n_showers, area, index=-2.7)

    energies = predictions['corsika_event_header_total_energy'].values * u.GeV

    area_lower, _, _ = collection_area(
        mc_production_spectrum,
        energies,
        bins=10,
    )


    area_higher, _, _ = collection_area(
        mc_production_spectrum,
        energies,
        bins=10,
        sample_fraction=0.5
    )

    # effective area should be twice as high if sample fraction is 0.5
    np.testing.assert_array_equal(2 * area_lower.value, area_higher.value)



def test_scatter_radius(predictions):
    energies = predictions['corsika_event_header_total_energy'].values * u.GeV

    n_showers = 10E6
    area = (270 * u.m)**2 * np.pi
    mc_production_spectrum = MCSpectrum(200 * u.GeV, 50 * u.TeV, n_showers, area, index=-2.7)

    area_lower, _, _ = collection_area(
        mc_production_spectrum,
        energies,
        bins=10,
    )

    area = (2 * 270 * u.m)**2 * np.pi
    mc_production_spectrum = MCSpectrum(200 * u.GeV, 50 * u.TeV, n_showers, area, index=-2.7)
    area_higher, _, _ = collection_area(
        mc_production_spectrum,
        energies,
        bins=10,
    )

    # effective area should be four times as high if impact is twice as high
    np.testing.assert_array_equal(4 * area_lower.value, area_higher.value)




def test_irf_writing(predictions, tmpdir):

    energies = predictions['corsika_event_header_total_energy'].values * u.GeV
    offsets = calculate_fov_offset(predictions)

    n_showers = 10E6
    area = (270 * u.m)**2 * np.pi
    mc_production_spectrum = MCSpectrum(200 * u.GeV, 50 * u.TeV, n_showers, area, index=-2.7)


    t = collection_area_to_irf_table(
        mc_production_spectrum,
        event_energies=energies,
        event_fov_offsets=offsets,
        bins=6
    )


    assert t['ENERG_LO'].shape == (1, 6)
    assert t['ENERG_LO'].unit == u.TeV
    assert t['EFFAREA'].shape == (1, 4, 6)
    assert t['EFFAREA'].unit == u.m**2
    assert len(t) == 1
