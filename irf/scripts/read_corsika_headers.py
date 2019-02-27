from corsikaio import CorsikaFile
from fact.io import to_h5py
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import os
import click
import pandas as pd
import numpy as np
from glob import glob


def get_headers(f):
    with CorsikaFile(f) as cf:
        run_header, event_headers, run_end = cf.read_headers()
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
def main(outputfile, inputdir):
    inputfiles = []
    for d in inputdir:
        inputfiles.extend(glob(os.path.join(d, 'cer*')))

    for f in inputfiles[:]:
        if f + '.gz' in inputfiles:
            inputfiles.remove(f + '.gz')

    print('Processing', len(inputfiles), 'files')

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
