import numpy as np
import pandas as pd
import click
from eventio.iact.parse_corsika_data import parse_corsika_event_header
import struct
import os
import gzip
import re
import multiprocessing
import itertools


HEADER_SIZE = 273 * 4


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
def main(directories, outputfile):
    '''
    This tool collects information from Corsika (MMCs) outputfile needed to create
    IRFs (effectice area and such).
    This tool iterates over the files in the DIRECTORIES given in the first argument.
    Note that you can pass many directories as the first argument such as

        /fhgfs/groups/app/fact/simulated/corsika/gamma_*


    The output ist stored in the OUTPUTFILE as csv data.
    The filenames are expected to be named according to this pattern:

        'cer\d{6}.gz'

    which should be the case for FACT simulations.
    '''

    pool = multiprocessing.Pool()
    results = []
    for directory in directories:
        mc_files = filter(lambda name:  re.match(
            'cer\d{6}.gz', name) is not None, os.listdir(directory))
        mc_files = [os.path.join(directory, f) for f in mc_files]
        print('Found {} files to read in directory {}'.format(
            len(mc_files), directory))
        results += [df for df in pool.map(read_mmc_headers, mc_files)]

    df = pd.concat(itertools.chain(results))
    print('number of events parsed: {}'.format(len(df)))
    df.to_csv(outputfile, index=False)


def read_mmc_headers(data_file):
    energies = []
    zenith = []

    with gzip.open(data_file, mode='rb') as f:
        c = f.read()

        skip = HEADER_SIZE
        while True:
            try:
                event_header = np.array(struct.unpack(
                    '273f', c[skip:skip + HEADER_SIZE]))
                e = parse_corsika_event_header(event_header)
                zenith.append(e.zenith_angle)
                energies.append(e.total_energy)
            except ValueError as e:
                pass
            except struct.error:
                break
            skip += HEADER_SIZE

    return pd.DataFrame({'energy': energies, 'zenith': zenith})


if __name__ == '__main__':
    main()
