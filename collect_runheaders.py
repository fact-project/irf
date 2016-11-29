import numpy as np
import pandas as pd
from functools import namedtuple
import click
from eventio.iact.parse_corsika_data import *
import struct
import os
import gzip
import re
import multiprocessing
import itertools


HEADER_SIZE = 273 * 4

@click.command()
@click.argument('directories', nargs=-1, type=click.Path(file_okay=False, dir_okay=True, exists=True))
@click.argument('outputfile', type=click.Path(file_okay=True, dir_okay=False, exists=False))
def bla(directories, outputfile):
    '''
    This tool iterates over the files in the DIRECTORIES given in the firs argument.
    The output ist stored in the OUTPUTFILE as csv data.
    '''

    pool = multiprocessing.Pool()
    results = []
    for directory in directories:
        mc_files = filter(lambda name:  re.match('cer\d{6}.gz',name) is not None, os.listdir(directory))
        mc_files = [os.path.join(directory, f) for f in mc_files ]
        print('Found {} files to read in directory {}'.format(len(mc_files), directory))
        results += [df for df in pool.map(read_mmc_headers, mc_files)]

    df = pd.concat(itertools.chain(results))
    print('number of events parsed: {}'.format(len(df)))
    df.to_csv(outputfile, index=False)

def read_mmc_headers(data_file):
    energies =[]
    zenith = []

    with gzip.open(data_file, mode='rb') as f:
        c = f.read()
        run_header =  struct.unpack('273f', c[0:HEADER_SIZE])
        h = parse_corsika_run_header(run_header)
        # print('Reading Corsika Run: {} started on {}'.format(h.run_id, h.date_of_begin_run))

        skip = HEADER_SIZE
        while True:
            try:
                event_header =  np.array(struct.unpack('273f', c[skip:skip + HEADER_SIZE]))
                e = parse_corsika_event_header(event_header)
                zenith.append(e.zenith_angle)
                energies.append(e.total_energy)
            except ValueError as e:
                pass
            except struct.error:
                break
            skip += HEADER_SIZE

    return pd.DataFrame({'energy':energies, 'zenith':zenith})

if __name__ == '__main__':
    bla()
