from eventio import EventIOFile
from eventio.iact import RunHeader, EventHeader, RunEnd
from fact.io import to_h5py
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed
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

    cf = EventIOFile(f)

    for o in cf:
        if isinstance(o, EventHeader):
            event_headers.append(o.parse())
        elif isinstance(o, RunHeader):
            run_header = o.parse()
        elif isinstance(o, RunEnd):
            run_end = o.parse()

    return run_header, np.array(event_headers), run_end


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
    'n_reuse',
    'viewcone_inner_angle',
    'viewcone_outer_angle',
]

run_header_columns = [
    'run_number',
    'date',
    'energy_spectrum_slope',
    'energy_min',
    'energy_max',
    'n_showers',
    'x_scatter',
    'y_scatter',
]


@click.command()
@click.argument('outputfile')
@click.argument(
    'inputdir',
     nargs=-1,
     type=click.Path(exists=True, file_okay=False, dir_okay=True),
)
@click.option('--infile-re', default=r'.*.eventio(\.gz|\.zst)?')
@click.option('--n-jobs', default=cpu_count(), type=int)
def main(outputfile, inputdir, infile_re, n_jobs):
    inputfiles = []
    file_re = re.compile(infile_re)

    for d in tqdm(inputdir):
        for root, dirs, files in os.walk(os.path.abspath(d)):
            for f in files:
                if file_re.match(f):
                    inputfiles.append(os.path.join(root, f))

    print('Processing', len(inputfiles), 'files')

    with ProcessPoolExecutor(n_jobs) as pool:
        futures = [pool.submit(get_headers, f) for f in inputfiles]

        run_headers = []
        run_ends = []

        reuses = []
        for future in tqdm(as_completed(futures), total=len(inputfiles)):
            run_header, event_headers, run_end = future.result()

            run_headers.append(run_header)
            run_ends.append(run_end)

            df = pd.DataFrame(event_headers[event_columns])
            to_h5py(df, outputfile, key='corsika_events', mode='a')
            reuses.append(df['n_reuse'].iloc[0])

        print('saving runwise information')
        runs = pd.DataFrame(np.array(run_headers)[run_header_columns])
        runs['n_events'] = np.array(run_ends)['n_events']
        runs['n_reuse'] = reuses

        to_h5py(runs, outputfile, key='corsika_runs', mode='a')
        print('done')


if __name__ == '__main__':
    main()
