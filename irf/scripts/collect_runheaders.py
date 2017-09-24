import pandas as pd
import click
import os
import gzip
import re
import joblib
import numpy as np
import h5py

from fact.io import to_h5py

from ..corsika import read_corsika_headers

event_header_columns = [
    'run_number',
    'event_number',
    'particle_id',
    'total_energy',
    'first_interaction_height',
    'momentum_x',
    'momentum_y',
    'momentum_minus_z',
    'zenith',
    'azimuth',
    'n_reuse',
]

event_header_renames = {
    'total_energy': 'energy'
}

event_header_types = {
    'particle_id': int,
    'event_number': int,
    'run_number': int,
    'n_reuse': int,
}

run_header_columns = [
    'run_number',
    'date',
    'version',
    'energy_spectrum_slope',
    'energy_min',
    'energy_max',
    'n_showers',
    'x_scatter',
    'y_scatter',
]

run_header_types = {
    'run_number': int,
    'n_showers': int,
}


@click.command()
@click.argument(
    'directories',
    nargs=-1,
    type=click.Path(file_okay=False, dir_okay=True, exists=True),
)
@click.argument(
    'outputfile',
    type=click.Path(file_okay=True, dir_okay=False, exists=False),
)
@click.option(
    '-p', '--pattern',
    default='cer\d{6}.*',
    help='Regex pattern for the inputfiles',
)
@click.option(
    '-n', '--n-jobs',
    default=-1,
    type=int,
    help='How may cores to use in parallel',
)
def main(directories, outputfile, pattern, n_jobs):
    '''
    This tool collects information from Corsika (MMCs) outputfile needed to create
    IRFs (effectice area and such).
    This tool iterates over the files in the DIRECTORIES given in the first argument.
    Note that you can pass many directories as the first argument such as

        /fhgfs/groups/app/fact/simulated/corsika/gamma_*


    The output ist stored in the OUTPUTFILE as pandas hdf5 data.
    '''

    with h5py.File(outputfile, 'a') as f:
        if 'corsika_runs' in f.keys() or 'corsika_events' in f.keys():
            click.confirm(
                'Group "corsika_runs" exists in outputfile, overwrite? ',
                abort=True,
            )

            del f['corsika_runs']
            del f['corsika_events']

    filename_re = re.compile(pattern)
    pool = joblib.Parallel(n_jobs=n_jobs, verbose=10)

    for directory in directories:
        mc_files = filter(filename_re.match, os.listdir(directory))
        mc_files = [os.path.join(directory, f) for f in mc_files]

        print('Found {} files to read in directory {}'.format(
            len(mc_files), directory
        ))

        results = pool(
            joblib.delayed(read_mmc_headers)(f)
            for f in mc_files
        )

        event_headers = pd.concat(r[1] for r in results)
        run_headers = pd.DataFrame([r[0] for r in results])
        run_headers = run_headers.astype(run_header_types)

        print('number of runs parsed: {}'.format(len(run_headers)))
        print('number of events parsed: {}'.format(len(event_headers)))

        to_h5py(outputfile, event_headers, key='corsika_events')
        to_h5py(outputfile, run_headers, key='corsika_runs')


def open_file(f):
    basename, ext = os.path.splitext(f)

    if ext == '.gz':
        return gzip.open(f, mode='rb')
    else:
        return open(f, mode='rb')


def read_mmc_headers(data_file):

    with open_file(data_file) as f:
        run_header, event_headers = read_corsika_headers(f)

    event_df = pd.DataFrame(event_headers[event_header_columns])
    event_df.rename(columns=event_header_renames, inplace=True)
    event_df = event_df.astype(event_header_types)
    event_df.set_index('event_number', inplace=True)

    n_reuse = int(event_headers['n_reuse'][0])
    assert np.all(n_reuse == event_headers['n_reuse'])

    impact_x = event_headers['reuse_x'][:, :n_reuse].ravel()
    impact_y = event_headers['reuse_y'][:, :n_reuse].ravel()
    reuse_id = np.tile(np.arange(n_reuse) + 1, len(event_headers))

    reuse_df = pd.DataFrame({
        'impact_x': impact_x,
        'impact_y': impact_y,
        'reuse_id': reuse_id,
        'event_number': np.repeat(event_headers['event_number'].astype(int), n_reuse),
    })

    run_series = pd.Series(
        dict(zip(
            run_header[run_header_columns].dtype.names,
            run_header[run_header_columns][0],
        ))
    )
    return run_series, reuse_df.join(event_df, on=['event_number'])


if __name__ == '__main__':
    main()
