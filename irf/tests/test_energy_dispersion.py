import fact.io
import os
import pytest
from irf import energy_dispersion, energy_migration
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


def test_migration(predictions):
    energy_true = predictions['corsika_event_header_total_energy'].values * u.GeV
    energy_prediction = predictions['gamma_energy_prediction'].values * u.GeV

    hist, bins_e_true, bins_mu = energy_migration(
        energy_true,
        energy_prediction,
        bins_energy=5,
        bins_mu=10
    )

    assert hist.shape == (5, 10)
    assert bins_e_true.unit == u.GeV


def test_normalization(predictions):
    energy_true = predictions['corsika_event_header_total_energy'].values * u.GeV
    energy_prediction = predictions['gamma_energy_prediction'].values * u.GeV

    hist, bins_e_true, bins_mu = energy_migration(
        energy_true,
        energy_prediction,
        bins_energy=5,
        bins_mu=10,
        normalize=True,
    )

    assert hist.shape == (5, 10)
    assert bins_e_true.unit == u.GeV
    # check columns are normalized
    assert np.allclose(hist.sum(axis=1), np.ones(5))
