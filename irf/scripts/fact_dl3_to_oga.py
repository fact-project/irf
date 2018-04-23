import fact.io
from irf import oga, collection_area_to_irf_table, energy_dispersion_to_irf_table, energy_migration, energy_dispersion
import click
import os
from astropy.io import fits
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from matplotlib.colors import PowerNorm
from irf.models import MCSpectrum


columns_to_read = [
    'theta_deg',
    'leakage1',
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
def main(predictions, dl3, output_directory, prediction_threshold, theta_square_cut):

    os.makedirs(output_directory, exist_ok=True)

    dl3_events = fact.io.read_h5py(dl3, key='events')
    gamma_events = fact.io.read_data(predictions, key='events', columns=columns_to_read)

    energy_bins = np.logspace(np.log10(0.2), np.log10(50), endpoint=True, num=25 + 1) * u.TeV

    # prodcution settings for diffuse gammas (gustav plus werner)
    n_showers = 12E6
    area = (270 * u.m)**2 * np.pi
    mc_production_spectrum = MCSpectrum(200 * u.GeV, 50 * u.TeV, n_showers, area, index=-2.7)

    diagnostic_plots(gamma_events, dl3_events, energy_bins, theta_square_cut=theta_square_cut, prediction_threshold=prediction_threshold)
    plt.savefig(os.path.join(output_directory, 'plots.png'))

    gammalike_data_events = dl3_events.query(f'gamma_prediction >= {prediction_threshold}').copy()
    runs = fact.io.read_h5py(dl3, key='runs')
    write_dl3(output_directory, gammalike_data_events, runs)

    write_irf(output_directory, mc_production_spectrum, gamma_events, prediction_threshold, theta_square_cut, energy_bins)


def diagnostic_plots(gamma_events, dl3_events, bins, theta_square_cut, prediction_threshold):
    q = f'theta_deg <= {np.sqrt(theta_square_cut)} & gamma_prediction >= {prediction_threshold}'
    crab_events_on_region = dl3_events.query(q).copy()
    selected_gamma_events = gamma_events.query(q).copy()

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
    hist, bins_e_true, bins_mu = energy_migration(true_event_energy, predicted_event_energy, bins=bins, normalize=False)
    im = ax2.pcolormesh(bins_e_true, bins_mu, hist.T, cmap='GnBu', norm=PowerNorm(0.5))
    fig.colorbar(im, ax=ax2)
    ax2.set_xscale('log')
    ax2.set_ylabel(r'$E_{\mathrm{Reco}} / E_\mathrm{{True}}$')
    ax2.set_xlabel(r'$E_{\mathrm{True}} /  \mathrm{TeV}$')
    ax2.set_ylim([0, 3])

    ax3 = top[2]
    ax3.hist(true_event_energy, bins=bins, histtype='step', label='true energies', density=True, color='lightgray', ls='--')
    ax3.hist(predicted_event_energy, bins=bins, histtype='step', label='predicted energies', color='0.6', lw=2, density=True)
    ax3.hist(predicted_event_energy_crab, bins=bins, histtype='step', label='predicted energies crab data on region', color='0.2', lw=2, density=True)
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.set_xlim([bins.min().value, bins.max().value])



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
    hist, bins_e_true, bins_mu = energy_migration(true_event_energy, predicted_event_energy, bins=bins, normalize=True, smoothing=1.25)
    im = ax2.pcolormesh(bins_e_true, bins_mu, hist.T, cmap='GnBu', norm=PowerNorm(0.5))
    fig.colorbar(im, ax=ax2)
    ax2.set_xscale('log')
    ax2.set_ylabel(r'$E_{\mathrm{Reco}} / E_\mathrm{{True}}$')
    ax2.set_xlabel(r'$E_{\mathrm{True}} /  \mathrm{TeV}$')
    ax2.set_ylim([0, 3])

    ax3 = bottom[2]
    tb = np.linspace(0, 0.1, 20)
    for i in range(1, 6):
        ax3.hist(dl3_events.query(f'gamma_prediction >= {prediction_threshold}')[f'theta_deg_off_{i}']**2, bins=tb, histtype='step', color='gray', ls='--')
    ax3.hist(dl3_events.query(f'gamma_prediction >= {prediction_threshold}').theta_deg**2, bins=tb, histtype='step',color='black', )



def write_irf(output_directory, mc_production_spectrum, gamma_events, prediction_threshold, theta_square_cut, energy_bins, irf_path='fact_irf.fits'):

    q = f'theta_deg <= {np.sqrt(theta_square_cut)} & gamma_prediction >= {prediction_threshold}'
    selected_gamma_events = gamma_events.query(q).copy()

    energies = selected_gamma_events['corsika_event_header_total_energy'].values * u.GeV
    offsets = oga.calculate_fov_offset(selected_gamma_events)


    collection_table = collection_area_to_irf_table(
        mc_production_spectrum,
        event_energies=energies,
        event_fov_offsets=offsets,
        bins=energy_bins,
        smoothing=1.25,
    )

    energy_prediction = selected_gamma_events['gamma_energy_prediction'].values * u.GeV
    e_disp_table = energy_dispersion_to_irf_table(
        energies,
        energy_prediction,
        offsets,
        bins=energy_bins,
        theta_bins=2
    )


    primary_hdu = oga.create_primary_hdu()
    collection_hdu = fits.table_to_hdu(collection_table)
    collection_hdu.header['HDUCLAS3'] = 'POINT-LIKE'
    collection_hdu.header['RAD_MAX'] = np.sqrt(theta_square_cut)

    e_disp_hdu = fits.table_to_hdu(e_disp_table)
    e_disp_hdu.header['HDUCLAS3'] = 'POINT-LIKE'
    e_disp_hdu.header['RAD_MAX'] = np.sqrt(theta_square_cut)

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
    runs = runs.copy().set_index(['night', 'run_id'])
    for n, g in tqdm(dl3_events.groupby(['night', 'run_id'])):

        obs_ids.append(int(n[0] * 1E3 + n[1]))

        run_data = g.copy()
        primary_hdu = oga.create_primary_hdu()
        event_hdu = oga.create_dl3_hdu(run_data, runs.loc[n])
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
