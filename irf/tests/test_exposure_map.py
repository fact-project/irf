import os
import pytest
import fact.io as fio
import pandas as pd
from irf import estimate_exposure_time


FIXTURE_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'test_files',
)


@pytest.fixture
def events():
    return fio.read_data(os.path.join(FIXTURE_DIR, 'crab_dl3_sample.hdf5'), key='events')


@pytest.fixture
def runs():
    return fio.read_data(os.path.join(FIXTURE_DIR, 'crab_dl3_sample.hdf5'), key='runs')



def test_exposure_time(runs, events):
        n_crab_sample = 687226
        n_test_sample = len(events)

        live_time_crab = runs.ontime.sum() / 3600  # to hours

        expected_live_time_sample = n_test_sample / n_crab_sample * live_time_crab

        timestamps = pd.to_datetime(events.timestamp).values
        exposure = estimate_exposure_time(timestamps).to('h').value

        assert expected_live_time_sample == pytest.approx(exposure, 0.05)
