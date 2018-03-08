import fact.io
from irf import oga, collection_area_to_irf_table, energy_dispersion_to_irf_table
import click
import os
from astropy.io import fits
from tqdm import tqdm
import pandas as pd
import astropy.units as u
import numpy as np
from astropy.coordinates.angle_utilities import angular_separation


columns_to_read = [
    'theta_deg',
    'corsika_evt_header_total_energy',
    'gamma_prediction',
    'gamma_energy_prediction',
    'zd_pointing',
    'az_pointing',
    'az_source_calc',
    'zd_source_calc',
]


def calculate_fov_offset(df):
    pointing_lat = (90 - df.zd_pointing.values) * u.deg
    pointing_lon = df.az_pointing.values * u.deg

    source_lat = (90 - df.zd_source_calc.values) * u.deg
    source_lon = df.az_source_calc.values * u.deg

    return angular_separation(pointing_lon, pointing_lat, source_lon, source_lat).to('deg')


@click.command()
@click.argument(
    'showers',
    type=click.Path(file_okay=True, dir_okay=False),
)
@click.argument(
    'predictions',
    type=click.Path(file_okay=True, dir_okay=False),
)
@click.argument(
    'dl3',
    type=click.Path(file_okay=True, dir_okay=False),
)
@click.argument(
    'output_directory',
    type=click.Path(file_okay=False, dir_okay=True, exists=False),
)
@click.option('-c', '--prediction_threshold', type=click.FLOAT, default=0.85)
@click.option('-t', '--theta_square_cut', type=click.FLOAT, default=0.02)
@click.option(
    '-m', '--max_scat',
    default=270,
    help='Maximum scatter radius (meter) used during corsika simulations of gammas.',
)
def main(showers, predictions, dl3, output_directory, prediction_threshold, theta_square_cut, max_scat):

    os.makedirs(output_directory, exist_ok=True)

    dl3_events = fact.io.read_h5py(dl3, key='events')
    dl3_events = dl3_events.query(f'gamma_prediction >= {prediction_threshold}').copy()
    runs = fact.io.read_h5py(dl3, key='runs')

    write_dl3(output_directory, dl3_events, runs)

    showers = fact.io.read_data(showers, key='showers')
    predictions = fact.io.read_data(predictions, key='events', columns=columns_to_read)
    q = f'gamma_prediction >= {prediction_threshold} & theta_deg <= {np.sqrt(theta_square_cut)}'
    selected_events = predictions.query(q)
    
    write_irf(output_directory, showers, selected_events)


def write_irf(output_directory, showers, selected_events, irf_path='fact_irf.fits'):
    shower_energy = (showers.energy.values * u.GeV).to('TeV')
    true_event_energy = (selected_events.corsika_evt_header_total_energy.values * u.GeV).to('TeV')
    predicted_event_energy = (selected_events.gamma_energy_prediction.values * u.GeV).to('TeV')
    event_offset = calculate_fov_offset(selected_events)

    collection_table = collection_area_to_irf_table(shower_energy, true_event_energy, event_offset, bins=20)
    e_disp_table = energy_dispersion_to_irf_table(true_event_energy, predicted_event_energy, event_offset, n_bins=60)

    primary_hdu = oga.create_primary_hdu()
    collection_hdu = fits.table_to_hdu(collection_table)
    e_disp_hdu = fits.table_to_hdu(e_disp_table)

    hdulist = fits.HDUList([primary_hdu, collection_hdu, e_disp_hdu])
    hdulist.writeto(os.path.join(output_directory, 'fact_irf.fits'), overwrite=True)


def write_dl3(output_directory, dl3_events, runs):
    primary_hdu = oga.create_primary_hdu()
    index_hdu = oga.create_observation_index_hdu(runs)
    hdulist = fits.HDUList([primary_hdu, index_hdu])
    hdulist.writeto(os.path.join(output_directory, 'obs-index.fits.gz'), overwrite=True)


    primary_hdu = oga.create_primary_hdu()
    index_hdu = oga.create_index_hdu(runs)
    hdulist = fits.HDUList([primary_hdu, index_hdu])
    hdulist.writeto(os.path.join(output_directory, 'hdu-index.fits.gz'), overwrite=True)

    obs_ids = []
    for n, g in tqdm(dl3_events.groupby(['night', 'run_id'])):

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
