import os
from irf.scripts import fact_dl3_to_oga
from click.testing import CliRunner

FIXTURE_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'test_files',
)


def test_roundtrip():
    runner = CliRunner()

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:

        output_path = tmpdir
        dl3 = os.path.join(FIXTURE_DIR, 'crab_dl3_small.hdf5')
        predictions = os.path.join(FIXTURE_DIR, 'gamma_predictions_dl2.hdf5')

        result = runner.invoke(fact_dl3_to_oga.main, [predictions, dl3, output_path])
        assert result.exit_code == 0
