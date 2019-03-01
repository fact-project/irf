import fact.io
from irf.gadf import response, hdus
from irf import energy_migration, energy_dispersion
# from irf import gadf, collection_area_to_irf_table, energy_dispersion_to_irf_table, collection_area, point_spread_function, psf_vs_energy, psf_to_irf_table
from irf.spectrum import MCSpectrum
from irf.gadf.hdus import create_primary_hdu_cta
import click
import os
from astropy.io import fits
from tqdm import tqdm
import pandas as pd
import numpy as np
import astropy.units as u

from astropy.coordinates.angle_utilities import angular_separation
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

@u.quantity_input(pointing_altitude=u.deg, pointing_azimuth=u.deg, source_altitude=u.deg, source_azimuth=u.deg)
def calculate_fov_offset(pointing_altitude, pointing_azimuth, source_altitude, source_azimuth):
    pointing_lat = pointing_altitude.to(u.deg)
    pointing_lon = pointing_azimuth.to(u.deg)

    source_lat = source_altitude.to(u.deg)
    source_lon = source_azimuth.to(u.deg)

    return angular_separation(pointing_lon, pointing_lat, source_lon, source_lat).to('deg')



@click.command()
@click.argument(
    'cta_data',
    type=click.Path(file_okay=True, dir_okay=False),
)
@click.argument(
    'output_path',
    type=click.Path(exists=False),
)
@click.option('-p', '--pointing', nargs=2, type=float, default=(70, 180))
def main(cta_data, output_path, pointing):

    runs = fact.io.read_data(cta_data, key='runs')

    mc_production_spectrum = MCSpectrum.from_cta_runs(runs)
    energy_bins = np.logspace(-2, 2, endpoint=True, num=55 + 1) * u.TeV

    array_events = fact.io.read_data(cta_data, key='array_events')

    energies = array_events.mc_energy.values * u.TeV
    offsets = calculate_fov_offset(pointing[0] * u.deg, pointing[1] * u.deg, array_events.mc_alt.values * u.rad, array_events.mc_az.values * u.rad)

    collection_table = collection_area_to_irf_table(
        mc_production_spectrum,
        event_energies=energies,
        event_fov_offsets=offsets,
        bins=energy_bins,
        smoothing=2,
        offset_bins=4,
        fov=12 * u.deg,
    )

    energy_prediction = array_events.gamma_energy_prediction_mean.values * u.TeV
    e_disp_table = energy_dispersion_to_irf_table(
        energies,
        energy_prediction,
        offsets,
        bins=energy_bins,
        offset_bins=4,
        smoothing=2,
        fov=12 * u.deg,
    )

    alt = array_events.alt.values * u.deg
    mc_alt = array_events.mc_alt.values * u.deg

    az = array_events.az.values * u.deg
    mc_az = array_events.mc_az.values * u.deg

    psf_table = psf_to_irf_table(mc_alt, mc_az, alt, az, energies, event_fov_offsets=offsets, fov=12 * u.deg, energy_bins=energy_bins, psf_bins=30, smoothing=0)

    primary_hdu = create_primary_hdu_cta()

    collection_hdu = fits.table_to_hdu(collection_table)
    e_disp_hdu = fits.table_to_hdu(e_disp_table)
    psf_table_hdu = fits.table_to_hdu(psf_table)

    hdulist = fits.HDUList([primary_hdu, collection_hdu, e_disp_hdu, psf_table_hdu])
    hdulist.writeto(os.path.join(output_path), overwrite=True)

    diagnostic_plots(array_events, mc_production_spectrum, energy_bins, offsets)
    plt.savefig('diagnostics_cta.png')


def diagnostic_plots(array_events, mc_production_spectrum, energy_bins, offsets):

    fig, [(ax1, ax2), (ax3, ax4)] = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)

    bin_edges = energy_bins
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_widths = np.diff(bin_edges)

    energies = array_events.mc_energy.values * u.TeV

    area, lower, upper = collection_area(
        mc_production_spectrum,
        energies,
        bins=bin_edges,
    )

    ax1.errorbar(
            bin_centers.value,
            area.value,
            xerr=bin_widths.value / 2.0,
            yerr=[lower.value, upper.value],
            linestyle='',
    )
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_ylabel('Effective Area')


    ax2.hist(energies, bins=bin_edges)
    ax2.set_xscale('log')

    alt = array_events.alt.values * u.deg
    mc_alt = array_events.mc_alt.values * u.deg

    az = array_events.az.values * u.deg
    mc_az = array_events.mc_az.values * u.deg

    bins = np.linspace(0, 90, 30) * u.deg
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    psf, _ = point_spread_function(mc_alt, mc_az, alt, az, bins=bins)
    ax3.plot(bin_centers, psf, '.')
    ax3.set_xlabel('angular distance')

    psf_2d, energy_bins, psf_bins = psf_vs_energy(mc_alt, mc_az, alt, az, event_energies=energies, normalize=True)
    ax4.pcolormesh(energy_bins, psf_bins, psf_2d.T, norm=LogNorm())
    ax4.set_xscale('log')
    ax4.set_xlabel('energy')
    # offsets = calculate_fov_offset(70 * u.deg, 0 * u.deg, array_events.mc_alt.values * u.rad, array_events.mc_az.values * u.rad)




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
