from eventio import EventIOFile
from eventio.iact import RunHeader, EventHeader, RunEnd
from fact.io import to_h5py
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import os
import click
import pandas as pd
import numpy as np
import re


def get_headers(f):
    run_header = None
    run_end = None
    event_headers = []

    with EventIOFile(f) as cf:
        for o in cf:
            if isinstance(o, EventHeader):
                event_headers.append(o.parse)
            elif isinstance(o, RunHeader):
                run_header = o.parse()
            elif isinstance(o, RunEnd):
                run_end = o.parse()

    return run_header, event_headers, run_end


event_columns = [
    'run_number',
    'event_number',
    'particle_id',
    'total_energy',
    'starting_altitude',
    'first_target_id',
    'first_interaction_height',
    'momentum_x',
    'momentum_y',
    'momentum_minus_z',
    'zenith',
    'azimuth',
]

run_header_columns = [
    'run_number',
    'date',
    'energy_spectrum_slope',
    'energy_min',
    'energy_max',
]


@click.command()
@click.argument('outputfile')
@click.argument(
    'inputdir',
     nargs=-1,
     type=click.Path(exists=True, file_okay=False, dir_okay=True),
)
@click.option('--infile-re', default=r'*.eventio(\.gz|\.zst)?')
def main(outputfile, inputdir, infile_re):
    inputfiles = []
    file_re = re.compile(infile_re)

    for d in tqdm(inputdir):
        for root, dirs, files in os.walk(os.path.abspath(inputdir)):
            for f in files:
                if file_re.match(f):
                    inputfiles.append(os.path.join(root, f))

    print('Processing', len(inputfiles), 'files')
    print(*inputfiles[:10], sep='\n')

    with Pool(cpu_count()) as pool:
        results = pool.imap_unordered(get_headers, inputfiles)

        run_headers = []
        run_ends = []

        for run_header, event_headers, run_end in tqdm(results, total=len(inputfiles)):

            run_headers.append(run_header)
            run_ends.append(run_end)

            df = pd.DataFrame(event_headers[event_columns])
            to_h5py(df, outputfile, key='corsika_events', mode='a')

        print('saving runwise information')
        runs = pd.DataFrame(np.array(run_headers)[run_header_columns])
        runs['n_events'] = np.array(run_ends)['n_events']

        to_h5py(runs, outputfile, key='corsika_runs', mode='a')
        print('done')


if __name__ == '__main__':
    main()
