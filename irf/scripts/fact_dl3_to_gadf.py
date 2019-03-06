import fact.io
from irf.gadf import response, hdus
from irf import energy_migration, energy_dispersion
from irf.spectrum import MCSpectrum
import click
import os
from astropy.io import fits
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from matplotlib.colors import PowerNorm
from astropy.coordinates.angle_utilities import angular_separation

from dateutil.parser import parse


columns_to_read = [
    'theta_deg',
    'corsika_event_header_total_energy',
    'gamma_prediction',
    'gamma_energy_prediction',
    'aux_pointing_position_zd',
    'aux_pointing_position_az',
    'source_position_az',
    'source_position_zd',
]


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
@click.option(
    '-c', '--prediction_threshold',
    type=click.FLOAT,
    default=0.85,
    help='The prediction threshold cut to apply to the MC events before calculating the irf'
        'and to the DL3 data before writing it to fits'
)
@click.option(
    '-t', '--theta_square_cut',
     type=click.FLOAT,
     default=0.03,
     help='The theta square cut to apply to the MC events before calculating the irf',
)
@click.option(
    '-m', '--max_scat',
    default=270,
    help='Maximum scatter radius (meter) used during corsika simulations of gammas.',
)
@click.option(
    '-i', '--spectrum_index',
    default=-2.7,
    help='production spectrum index (-2.7)',
)
@click.option(
    '--start',
    help='Min datetime stamp for run selection.',
)
@click.option(
    '--end',
    help='max datetime stamp for run selection.',
)
@click.option(
    '-e', '--exclude',
    help='runs to exclude in YYMMDD_ID format',
    multiple=True,
)
def main(showers, predictions, dl3, output_directory, prediction_threshold, theta_square_cut, spectrum_index, max_scat, start, end, exclude):
    '''
    Takes FACT Corsika information (SHOWERS), FACT (diffuse) MC data (PREDICTIONS)
    and FACT observations (DL3) as input and writes DL3 data and IRFs according
    to the open gamma-ray astro data format (OGA) to OUTPUT_DIRECTORY.

    The (PREDICTIONS) file needs to have the following columns:
        'theta_deg',
        'corsika_event_header_total_energy',
        'gamma_prediction',
        'gamma_energy_prediction',
        'aux_pointing_position_zd',
        'aux_pointing_position_az',
        'source_position_az',
        'source_position_zd'.

    This script will additionally produce a file called 'plots.png' in the given output_directory
    for debugging purposes.

    '''

    os.makedirs(output_directory, exist_ok=True)

    dl3_events = fact.io.read_h5py(dl3, key='events')
    runs = fact.io.read_h5py(dl3, key='runs')

    fact_mc_spectrum = MCSpectrum(
        e_min=200*u.GeV,
        e_max=5000*u.GeV,
        total_showers_simulated=6000000,
        generation_area=(max_scat*u.m)**2*np.pi,
        index=-2.7,
        generator_opening_angle=6*u.deg
    )

    if start:
        dt = parse(start)
        m = pd.to_datetime(runs.run_start) >= dt
        runs = runs[m]

    if end:
        dt = parse(end)
        m = pd.to_datetime(runs.run_stop) < dt
        runs = runs[m]


    if exclude:
        exclusions = np.array([list(map(int, e.split('_'))) for e in exclude])
        excluded_runs = runs.night.isin(exclusions[:, 0]) & runs.run_id.isin(exclusions[:, 1])
        print(f'Removed {excluded_runs.sum()} runs from dataset')
        runs = runs[~excluded_runs]

    dl3_events = pd.merge(dl3_events, runs, on=['night', 'run_id'])
    gamma_events = fact.io.read_data(predictions, key='events', columns=columns_to_read)


    diagnostic_plots(gamma_events, dl3_events, theta_square_cut=theta_square_cut, prediction_threshold=prediction_threshold)
    plt.savefig(os.path.join(output_directory, 'plots.png'))

    print(f'Total ontime: {runs.ontime.sum()/60/60} hours')

    write_dl3(output_directory, dl3_events, runs, prediction_threshold=prediction_threshold)

    write_irf(output_directory, fact_mc_spectrum, gamma_events, prediction_threshold, theta_square_cut)


def diagnostic_plots(gamma_events, dl3_events, theta_square_cut, prediction_threshold):
    q = f'theta_deg <= {np.sqrt(theta_square_cut)} & gamma_prediction >= {prediction_threshold}'

    crab_events_on_region = dl3_events.query(q).copy()
    selected_gamma_events = gamma_events.query(q).copy()

    min_energy = selected_gamma_events.corsika_event_header_total_energy.min() * u.GeV
    max_energy = selected_gamma_events.corsika_event_header_total_energy.max() * u.GeV

    bins = np.logspace(np.log10(min_energy.to('TeV').value), np.log10(max_energy.to('TeV').value), endpoint=True, num=25 + 1)

    true_event_energy = (selected_gamma_events.corsika_event_header_total_energy.values * u.GeV).to('TeV')
    predicted_event_energy = (selected_gamma_events.gamma_energy_prediction.values * u.GeV).to('TeV')
    predicted_event_energy_crab = (crab_events_on_region.gamma_energy_prediction.values * u.GeV).to('TeV')
    fig, (top, bottom) = plt.subplots(2, 3, figsize=(12, 10), constrained_layout=True)

    ax1 = top[0]
    hist, bins_e_true, bins_e_prediction = energy_dispersion(true_event_energy, predicted_event_energy, bins=bins)
    im = ax1.pcolormesh(bins_e_true, bins_e_prediction, hist.T, cmap='GnBu', norm=PowerNorm(0.5))
    ax1.plot(ax1.get_xlim(), ax1.get_ylim(), ls='--', color='darkgray')
    fig.colorbar(im, ax=ax1)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel(r'$E_{\mathrm{True}} /  \mathrm{TeV}$')
    ax1.set_ylabel(r'$E_{\mathrm{Reco}} /  \mathrm{TeV}$')

    ax2 = top[1]
    hist, bins_e_true, bins_mu = energy_migration(true_event_energy, predicted_event_energy, bins_energy=bins, normalize=False)
    im = ax2.pcolormesh(bins_e_true, bins_mu, hist.T, cmap='GnBu', norm=PowerNorm(0.5))
    fig.colorbar(im, ax=ax2)
    ax2.set_xscale('log')
    ax2.set_ylabel(r'$E_{\mathrm{Reco}} / E_\mathrm{{True}}$')
    ax2.set_xlabel(r'$E_{\mathrm{True}} /  \mathrm{TeV}$')
    ax2.set_ylim([bins_mu.min(), bins_mu.max()])

    ax3 = top[2]
    ax3.hist(true_event_energy, bins=bins, histtype='step', label='true energies', density=True, color='lightgray', ls='--')
    ax3.hist(predicted_event_energy, bins=bins, histtype='step', label='predicted energies', color='0.6', lw=2, density=True)
    ax3.hist(predicted_event_energy_crab, bins=bins, histtype='step', label='predicted energies crab data on region', color='0.2', lw=2, density=True)
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.set_xlim([bins.min(), bins.max()])


    ax1 = bottom[0]
    hist, bins_e_true, bins_e_prediction = energy_dispersion(true_event_energy, predicted_event_energy, bins=bins, normalize=True, smoothing=1.25)
    im = ax1.pcolormesh(bins_e_true, bins_e_prediction, hist.T, cmap='GnBu', norm=PowerNorm(0.5))
    ax1.plot(ax1.get_xlim(), ax1.get_ylim(), ls='--', color='darkgray')
    fig.colorbar(im, ax=ax1)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel(r'$E_{\mathrm{True}} /  \mathrm{TeV}$')
    ax1.set_ylabel(r'$E_{\mathrm{Reco}} /  \mathrm{TeV}$')

    ax2 = bottom[1]
    hist, bins_e_true, bins_mu = energy_migration(true_event_energy, predicted_event_energy, bins_energy=bins, normalize=True, smoothing=1.25)
    im = ax2.pcolormesh(bins_e_true, bins_mu, hist.T, cmap='GnBu', norm=PowerNorm(0.5))
    fig.colorbar(im, ax=ax2)
    ax2.set_xscale('log')
    ax2.set_ylabel(r'$E_{\mathrm{Reco}} / E_\mathrm{{True}}$')
    ax2.set_xlabel(r'$E_{\mathrm{True}} /  \mathrm{TeV}$')
    ax2.set_ylim([bins_mu.min(), bins_mu.max()])

    ax3 = bottom[2]
    tb = np.linspace(0, 0.1, 20)
    for i in range(1, 6):
        ax3.hist(dl3_events.query(f'gamma_prediction >= {prediction_threshold}')[f'theta_deg_off_{i}']**2, bins=tb, histtype='step', color='gray', ls='--')
    ax3.hist(dl3_events.query(f'gamma_prediction >= {prediction_threshold}').theta_deg**2, bins=tb, histtype='step',color='black', )


def calculate_fov_offset(df):
    '''
    Calculate the `offset` aka the `angular_separation` between the pointing and
    the source position.
    '''
    pointing_lat = (90 - df.aux_pointing_position_zd.values) * u.deg
    pointing_lon = df.aux_pointing_position_az.values * u.deg

    source_lat = (90 - df.source_position_zd.values) * u.deg
    source_lon = df.source_position_az.values * u.deg

    return angular_separation(pointing_lon, pointing_lat, source_lon, source_lat).to('deg')

def write_irf(output_directory, mc_production, gamma_events, prediction_threshold, theta_square_cut, irf_path='fact_irf.fits'):
    rad_max = np.sqrt(theta_square_cut)
    q = f'theta_deg <= {rad_max} & gamma_prediction >= {prediction_threshold}'

    selected_gamma_events = gamma_events.query(q).copy()

    min_energy = selected_gamma_events.corsika_event_header_total_energy.min() * u.GeV
    max_energy = selected_gamma_events.corsika_event_header_total_energy.max() * u.GeV
    energy_bins = np.logspace(
        np.log10(min_energy.to_value('TeV')),
        np.log10(max_energy.to_value('TeV')),
        endpoint=True,
        num=25 + 1
    ) * u.TeV

    theta_bins = np.linspace(0, 2.25, 5) * u.deg

    true_event_energy = (selected_gamma_events.corsika_event_header_total_energy.values * u.GeV).to('TeV')
    event_offsets = calculate_fov_offset(selected_gamma_events)

    a_eff_hdu = response.create_effective_area_hdu(mc_production, true_event_energy, event_offsets, energy_bin_edges=energy_bins, theta_bin_edges=theta_bins, sample_fraction=1, smoothing=0.8,)
    hdus.add_fact_meta_information_to_hdu(a_eff_hdu)
    
    # a_eff_hdu = response.effective_area_hdu_for_fact(mc_production, selected_gamma_events, bins=energy_bins, sample_fraction=1, smoothing=0.8,)
    true_energy = selected_gamma_events.corsika_event_header_total_energy.values * u.GeV
    estimated_energy = selected_gamma_events.gamma_energy_prediction.values * u.GeV
    e_disp_hdu = response.create_energy_dispersion_hdu(true_energy, estimated_energy, event_offsets,  energy_bin_edges=energy_bins, theta_bin_edges=theta_bins, smoothing=0.8)

    a_eff_hdu.header['RAD_MAX'] = rad_max
    e_disp_hdu.header['RAD_MAX'] = rad_max

    a_eff_hdu.header['PRED_MAX'] = (prediction_threshold, 'prediction threshold used to select events')
    e_disp_hdu.header['PRED_MAX'] = (prediction_threshold, 'prediction threshold used to select events')

    primary_hdu = hdus.create_primary_hdu()

    hdulist = fits.HDUList([primary_hdu, a_eff_hdu, e_disp_hdu])
    hdulist.writeto(os.path.join(output_directory, 'fact_irf.fits'), overwrite=True)


def write_dl3(output_directory, dl3_events, runs, prediction_threshold=0.85):
    primary_hdu = hdus.create_primary_hdu()
    index_hdu = hdus.create_observation_index_hdu(runs)
    hdulist = fits.HDUList([primary_hdu, index_hdu])
    hdulist.writeto(os.path.join(output_directory, 'obs-index.fits.gz'), overwrite=True)


    primary_hdu = hdus.create_primary_hdu()
    index_hdu = hdus.create_index_hdu(runs)
    hdulist = fits.HDUList([primary_hdu, index_hdu])
    hdulist.writeto(os.path.join(output_directory, 'hdu-index.fits.gz'), overwrite=True)

    gammalike_data_events = dl3_events.query(f'gamma_prediction >= {prediction_threshold}')

    obs_ids = []
    runs = runs.set_index(['night', 'run_id'])
    for n, run_data in tqdm(gammalike_data_events.groupby(['night', 'run_id'])):
        file_name = f'{n[0]}_{n[1]}_dl3.fits'
        obs_ids.append(int(n[0] * 1E3 + n[1]))

        primary_hdu = hdus.create_primary_hdu()

        gti_hdu = hdus.create_gti_hdu(runs.loc[n])
        event_hdu = hdus.create_events_hdu(run_data, runs.loc[n])
        event_hdu.header['PRED_MAX'] = (prediction_threshold, 'prediction threshold used to select events')

        hdulist = fits.HDUList([primary_hdu, event_hdu, gti_hdu])
        hdulist.writeto(os.path.join(output_directory, file_name), overwrite=True)


    pd.DataFrame({'OBS_ID': obs_ids}).to_csv(
        os.path.join(output_directory, 'observations.csv'),
        header=True,
        index=False
    )


if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    main()
