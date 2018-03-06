import fact.io
from irf import oga
import click
import os
from astropy.io import fits
from tqdm import tqdm
import pandas as pd


@click.command()
@click.argument(
    'input_path',
    type=click.Path(file_okay=True, dir_okay=False),
)
@click.argument(
    'output_directory',
    type=click.Path(file_okay=False, dir_okay=True),
)
@click.option(
    '-t', '--threshold',
    default=0.85,
    help='Prediction threshold to apply.',
)
def main(input_path, output_directory, threshold):
    dl3 = fact.io.read_h5py(input_path, key='events')
    dl3 = dl3.query(f'gamma_prediction >= {threshold}').copy()

    runs = fact.io.read_h5py(input_path, key='runs')

    os.makedirs(output_directory, exist_ok=True)

    primary_hdu = oga.create_primary_hdu()
    index_hdu = oga.create_observation_index_hdu(runs)
    hdulist = fits.HDUList([primary_hdu, index_hdu])
    hdulist.writeto(os.path.join(output_directory, 'obs-index.fits.gz'), overwrite=True)


    primary_hdu = oga.create_primary_hdu()
    index_hdu = oga.create_index_hdu(runs)
    hdulist = fits.HDUList([primary_hdu, index_hdu])
    hdulist.writeto(os.path.join(output_directory, 'hdu-index.fits.gz'), overwrite=True)

    # import IPython; IPython.embed()
    obs_ids = []
    for n, g in tqdm(dl3.groupby(['night', 'run_id'])):

        obs_ids.append(int(n[0] * 1E3 + n[1]))

        run_data = g.copy()
        oga.create_dl3_hdu(run_data)
        primary_hdu = oga.create_primary_hdu()
        event_hdu = oga.create_dl3_hdu(run_data)
        hdulist = fits.HDUList([primary_hdu, event_hdu])
        fname = f'{n[0]}_{n[1]}_dl3.fits'
        hdulist.writeto(os.path.join(output_directory, fname), overwrite=True)

    pd.DataFrame({'OBS_ID': obs_ids}).to_csv(
        os.path.join(output_directory, 'observations.csv'),
        header=True,
        index=False
    )




if __name__ == '__main__':
    main()
