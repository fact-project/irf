import os
from irf.scripts import fact_dl3_to_fits
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
        input_path = os.path.join(FIXTURE_DIR, 'crab_dl3.hdf5')

        result = runner.invoke(fact_dl3_to_fits.main, [input_path, output_path, '--threshold', 0.6])
        assert result.exit_code == 0
