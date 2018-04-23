import fact.io
import os
import pytest
from irf import energy_dispersion, energy_dispersion_to_irf_table
from irf.oga import calculate_fov_offset

import astropy.units as u
import numpy as np

FIXTURE_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'test_files',
)


@pytest.fixture
def predictions():
    return fact.io.read_data(
        os.path.join(FIXTURE_DIR, 'gamma_predictions_dl2.hdf5'), key='events')


def test_dispersion(predictions):
    energy_true = predictions['corsika_event_header_total_energy'].values * u.GeV
    energy_prediction = predictions['gamma_energy_prediction'].values * u.GeV

    hist, bins_e_true, bins_e_prediction = energy_dispersion(
        energy_true,
        energy_prediction,
        bins=5
    )

    assert hist.shape == (5, 5)
    assert bins_e_true[0] < bins_e_true[-1]
    np.testing.assert_array_equal(bins_e_true, bins_e_prediction)
    assert bins_e_true.unit == u.GeV


def test_irf_writing(predictions):
    energy_true = predictions['corsika_event_header_total_energy'].values * u.GeV
    energy_prediction = predictions['gamma_energy_prediction'].values * u.GeV
    offsets = calculate_fov_offset(predictions)
    t = energy_dispersion_to_irf_table(energy_true, energy_prediction, offsets, bins=5, theta_bins=1)
    assert len(t) == 1
    assert t['MATRIX'].data.shape == (1, 1, 5, 5)
