import os

import astropy.units as u
import click
import fact.io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.coordinates.angle_utilities import angular_separation
from astropy.io import fits
from matplotlib.colors import LogNorm
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

from irf import energy_dispersion, energy_migration
from irf.gadf import hdus, response
from irf.gadf.hdus import create_primary_hdu_cta
# from irf import gadf, collection_area_to_irf_table, energy_dispersion_to_irf_table, collection_area, point_spread_function, psf_vs_energy, psf_to_irf_table
from irf.spectrum import MCSpectrum


@u.quantity_input(pointing_altitude=u.deg, pointing_azimuth=u.deg, source_altitude=u.deg, source_azimuth=u.deg)
def calculate_fov_offset(pointing_altitude, pointing_azimuth, source_altitude, source_azimuth):
    pointing_lat = pointing_altitude.to(u.deg)
    pointing_lon = pointing_azimuth.to(u.deg)

    source_lat = source_altitude.to(u.deg)
    source_lon = source_azimuth.to(u.deg)

    return angular_separation(pointing_lon, pointing_lat, source_lon, source_lat).to('deg')

def create_interpolated_function(energies, values, sigma=1):
    m  = ~np.isnan(values) # do not use nan values
    r = gaussian_filter1d(values[m], sigma=sigma)
    f = interp1d(energies[m], r, kind='cubic', bounds_error=False, fill_value='extrapolate')
    return f


def apply_cuts(df, cuts_path, sigma=1, prediction_cuts=True, multiplicity_cuts=True):
    cuts = pd.read_csv(cuts_path)
    bin_center = np.sqrt(cuts.e_min * cuts.e_max)

    m = np.ones(len(df)).astype(np.bool)

    # if theta_cuts:
    #     source_az = df.mc_az.values * u.deg
    #     source_alt = df.mc_alt.values * u.deg

    #     df['theta'] = (calculate_distance_to_point_source(df, source_alt=source_alt, source_az=source_az).to_value(u.deg))

    #     f_theta =  create_interpolated_function(bin_center, cuts.theta_cut)
    #     m &= df.theta < f_theta(df.gamma_energy_prediction_mean)

    if prediction_cuts: 
        f_prediction = create_interpolated_function(bin_center, cuts.prediction_cut)
        m &= df.gamma_prediction_mean >= f_prediction(df.gamma_energy_prediction_mean)

    if multiplicity_cuts:
        multiplicity = cuts.multiplicity[0]
        print('multi', multiplicity)
        m &= df.num_triggered_telescopes >= multiplicity
    
    return df[m]


@click.command()
@click.argument(
    'cta_data',
    type=click.Path(file_okay=True, dir_okay=False),
)
@click.argument(
    'cuts_path',
    type=click.Path(file_okay=True, dir_okay=False),
)
@click.argument(
    'output_path',
    type=click.Path(exists=False),
)
@click.option('-p', '--pointing', nargs=2, type=float, default=(70, 180))
def main(cta_data, cuts_path,  output_path, pointing):

    runs = fact.io.read_data(cta_data, key='runs')

    mc_production = MCSpectrum.from_cta_runs(runs)
    energy_bins = np.logspace(-2, 2, endpoint=True, num=55 + 1) * u.TeV

    array_events = fact.io.read_data(cta_data, key='array_events')
    array_events = apply_cuts(array_events, cuts_path)

    # energies = array_events.mc_energy.values * u.TeV
    offsets = calculate_fov_offset(pointing[0] * u.deg, pointing[1] * u.deg, array_events.mc_alt.values * u.rad, array_events.mc_az.values * u.rad)

    alt = array_events.alt.values * u.deg
    mc_alt = array_events.mc_alt.values * u.deg

    az = array_events.az.values * u.deg
    mc_az = array_events.mc_az.values * u.deg

    # psf_table = psf_to_irf_table(mc_alt, mc_az, alt, az, energies, event_fov_offsets=offsets, fov=12 * u.deg, energy_bins=energy_bins, psf_bins=30, smoothing=0)

    primary_hdu = create_primary_hdu_cta()

    # collection_hdu = fits.table_to_hdu(collection_table)
    # e_disp_hdu = fits.table_to_hdu(e_disp_table)
    # psf_table_hdu = fits.table_to_hdu(psf_table)

    # hdulist = fits.HDUList([primary_hdu, collection_hdu, e_disp_hdu, psf_table_hdu])
    # hdulist.writeto(os.path.join(output_path), overwrite=True)

    # diagnostic_plots(array_events, mc_production_spectrum, energy_bins, offsets)
    # plt.savefig('diagnostics_cta.png')



    # rad_max = np.sqrt(theta_square_cut)
    # q = f'theta_deg <= {rad_max} & gamma_prediction >= {prediction_threshold}'


    min_energy = array_events.mc_energy.min() * u.TeV
    max_energy = array_events.mc_energy.max() * u.TeV
    # max_energy = selected_gamma_events.corsika_event_header_total_energy.max() * u.GeV
    energy_bins = np.logspace(
        np.log10(min_energy.to('TeV').value),
        np.log10(max_energy.to('TeV').value),
        endpoint=True,
        num=25 + 1
    )

    a_eff_hdu = response.effective_area_hdu_for_cta(mc_production, array_events, bins=energy_bins, sample_fraction=1, smoothing=0.8,)
    e_disp_hdu = response.energy_dispersion_hdu(selected_gamma_events, bins=energy_bins, theta_bins=2, smoothing=0.8)


    primary_hdu = hdus.create_primary_hdu()

    hdulist = fits.HDUList([primary_hdu, a_eff_hdu, e_disp_hdu])
    hdulist.writeto(os.path.join(output_path, 'cta_irf.fits'), overwrite=True)


# def diagnostic_plots(array_events, mc_production_spectrum, energy_bins, offsets):

#     fig, [(ax1, ax2), (ax3, ax4)] = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)

#     bin_edges = energy_bins
#     bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
#     bin_widths = np.diff(bin_edges)

#     energies = array_events.mc_energy.values * u.TeV

#     area, lower, upper = collection_area(
#         mc_production_spectrum,
#         energies,
#         bins=bin_edges,
#     )

#     ax1.errorbar(
#             bin_centers.value,
#             area.value,
#             xerr=bin_widths.value / 2.0,
#             yerr=[lower.value, upper.value],
#             linestyle='',
#     )
#     ax1.set_xscale('log')
#     ax1.set_yscale('log')
#     ax1.set_ylabel('Effective Area')


#     ax2.hist(energies, bins=bin_edges)
#     ax2.set_xscale('log')

#     alt = array_events.alt.values * u.deg
#     mc_alt = array_events.mc_alt.values * u.deg

#     az = array_events.az.values * u.deg
#     mc_az = array_events.mc_az.values * u.deg

#     bins = np.linspace(0, 90, 30) * u.deg
#     bin_centers = 0.5 * (bins[:-1] + bins[1:])
#     psf, _ = point_spread_function(mc_alt, mc_az, alt, az, bins=bins)
#     ax3.plot(bin_centers, psf, '.')
#     ax3.set_xlabel('angular distance')

#     psf_2d, energy_bins, psf_bins = psf_vs_energy(mc_alt, mc_az, alt, az, event_energies=energies, normalize=True)
#     ax4.pcolormesh(energy_bins, psf_bins, psf_2d.T, norm=LogNorm())
#     ax4.set_xscale('log')
#     ax4.set_xlabel('energy')
#     # offsets = calculate_fov_offset(70 * u.deg, 0 * u.deg, array_events.mc_alt.values * u.rad, array_events.mc_az.values * u.rad)




# def write_irf(output_directory, mc_production_spectrum, gamma_events, prediction_threshold, theta_square_cut, energy_bins, irf_path='fact_irf.fits'):

#     q = f'theta_deg <= {np.sqrt(theta_square_cut)} & gamma_prediction >= {prediction_threshold}'
#     selected_gamma_events = gamma_events.query(q).copy()

#     energies = selected_gamma_events['corsika_event_header_total_energy'].values * u.GeV
#     offsets = oga.calculate_fov_offset(selected_gamma_events)


#     collection_table = collection_area_to_irf_table(
#         mc_production_spectrum,
#         event_energies=energies,
#         event_fov_offsets=offsets,
#         bins=energy_bins,
#         smoothing=1.25,
#     )

#     energy_prediction = selected_gamma_events['gamma_energy_prediction'].values * u.GeV
#     e_disp_table = energy_dispersion_to_irf_table(
#         energies,
#         energy_prediction,
#         offsets,
#         bins=energy_bins,
#         theta_bins=2
#     )


#     primary_hdu = oga.create_primary_hdu()
#     collection_hdu = fits.table_to_hdu(collection_table)
#     collection_hdu.header['HDUCLAS3'] = 'POINT-LIKE'
#     collection_hdu.header['RAD_MAX'] = np.sqrt(theta_square_cut)

#     e_disp_hdu = fits.table_to_hdu(e_disp_table)
#     e_disp_hdu.header['HDUCLAS3'] = 'POINT-LIKE'
#     e_disp_hdu.header['RAD_MAX'] = np.sqrt(theta_square_cut)

#     hdulist = fits.HDUList([primary_hdu, collection_hdu, e_disp_hdu])
#     hdulist.writeto(os.path.join(output_directory, 'fact_irf.fits'), overwrite=True)

if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    main()
