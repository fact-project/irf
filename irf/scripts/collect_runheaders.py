import numpy as np
import pandas as pd
import click
from eventio.iact.parse_corsika_data import parse_corsika_event_header
import struct
import os
import gzip
import re
import itertools
import joblib

from ..corsika import read_corsika_headers


BLOCK_SIZE = 273 * 4


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

    filename_re = re.compile(pattern)
    pool = joblib.Parallel(n_jobs=n_jobs, verbose=10)

    results = []

    for directory in directories:
        mc_files = filter(filename_re.match, os.listdir(directory))
        mc_files = [os.path.join(directory, f) for f in mc_files]

        print('Found {} files to read in directory {}'.format(
            len(mc_files), directory
        ))

        results += pool(
            joblib.delayed(read_mmc_headers)(f)
            for f in mc_files
        )

    df = pd.concat(itertools.chain(results))
    print('number of events parsed: {}'.format(len(df)))
    df.to_hdf(outputfile, key='table')


def open_file(f):
    basename, ext = os.path.splitext(f)

    if ext == '.gz':
        return gzip.open(f, mode='rb')
    else:
        return open(f, mode='rb')


def read_mmc_headers(data_file):

    with open_file(data_file) as f:
        headers = read_corsika_headers(f)

    energies = [e.total_energy for e in headers['event_headers']]
    zeniths = [e.zenith_angle for e in headers['event_headers']]

    return pd.DataFrame({'energy': energies, 'zenith': zeniths})


if __name__ == '__main__':
    main()
