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
# from irf import gadf, collection_area_to_irf_table, energy_dispersion_to_irf_table, collection_area, point_spread_function, psf_vs_energy, psf_to_irf_table
from irf.spectrum import MCSpectrum, CTAProtonSpectrum, CTAElectronSpectrum


@u.quantity_input(pointing_altitude=u.deg, pointing_azimuth=u.deg, source_altitude=u.deg, source_azimuth=u.deg)
def calculate_fov_offset(pointing_altitude, pointing_azimuth, source_altitude, source_azimuth):
    pointing_lat = pointing_altitude.to(u.deg)
    pointing_lon = pointing_azimuth.to(u.deg)

    source_lat = source_altitude.to(u.deg)
    source_lon = source_azimuth.to(u.deg)

    return angular_separation(pointing_lon, pointing_lat, source_lon, source_lat).to(u.deg)



@u.quantity_input(true_altitude=u.deg, true_azimuth=u.deg, estimated_altitude=u.deg, estimated_azimuth=u.deg)
def calculate_angular_separation(true_altitude, true_azimuth, estimated_altitude, estimated_azimuth):
    true_lat = true_altitude.to(u.deg)
    true_lon = true_azimuth.to(u.deg)

    lat = estimated_altitude.to(u.deg)
    lon = estimated_azimuth.to(u.deg)

    return angular_separation(true_lon, true_lat, lon, lat).to(u.deg)

def create_interpolated_function(energies, values, sigma=1):
    m  = ~np.isnan(values) # do not use nan values
    r = gaussian_filter1d(values[m], sigma=sigma)
    f = interp1d(energies[m], r, kind='cubic', bounds_error=False, fill_value='extrapolate')
    return f


def apply_cuts(df, cuts_path, sigma=1, prediction_cuts=True, multiplicity_cuts=True):
    cuts = pd.read_csv(cuts_path)
    bin_center = np.sqrt(cuts.e_min * cuts.e_max)

    m = np.ones(len(df)).astype(np.bool)
    if prediction_cuts: 
        f_prediction = create_interpolated_function(bin_center, cuts.prediction_cut)
        m &= df.gamma_prediction_mean >= f_prediction(df.gamma_energy_prediction_mean)

    if multiplicity_cuts:
        multiplicity = cuts.multiplicity[0]
        m &= df.num_triggered_telescopes >= multiplicity
    
    return df[m]

columns  = ['alt', 'az', 'mc_energy', 'gamma_energy_prediction_mean', 'gamma_prediction_mean', 'mc_alt', 'mc_az', 'num_triggered_telescopes']

@click.command()
@click.argument(
    'gammas_diffuse_path',
    type=click.Path(file_okay=True, dir_okay=False),
)
@click.argument(
    'protons_path',
    type=click.Path(file_okay=True, dir_okay=False),
)
@click.argument(
    'electrons_path',
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
def main(gammas_diffuse_path, protons_path, electrons_path,  cuts_path,  output_path, pointing):

    fov = 12*u.deg
    energy_bins = np.logspace(-2, 2, endpoint=True, num=25 + 1) * u.TeV

    
    gamma_runs = fact.io.read_data(gammas_diffuse_path, key='runs')
    mc_production = MCSpectrum.from_cta_runs(gamma_runs)

    gamma_events = fact.io.read_data(gammas_diffuse_path, key='array_events', columns=columns)
    gamma_events = apply_cuts(gamma_events, cuts_path)

    mc_alt = gamma_events.mc_alt.values * u.deg
    mc_az = gamma_events.mc_az.values * u.deg

    alt = gamma_events.alt.values * u.deg
    az = gamma_events.az.values * u.deg
    
    event_energies = gamma_events.mc_energy.values * u.TeV
    estimated_event_energies = gamma_events.gamma_energy_prediction_mean.values * u.TeV
    event_offsets = calculate_fov_offset(pointing[0] * u.deg, pointing[1] * u.deg, mc_alt, mc_az)
    angular_distance = calculate_angular_separation(mc_alt, mc_az, alt, az)


    primary_hdu = hdus.create_primary_hdu_cta()

    # create effective area hdu
    a_eff_hdu = response.create_effective_area_hdu(mc_production, event_energies, event_offsets, energy_bins, fov=fov)
    hdus.add_cta_meta_information_to_hdu(a_eff_hdu)

    # create edisp hdu
    e_disp_hdu = response.create_energy_dispersion_hdu(
        event_energies,
        estimated_event_energies,
        event_offsets,
        bins_e_true=energy_bins,
        num_theta_bins=5,
        fov=fov
    )
    hdus.add_cta_meta_information_to_hdu(e_disp_hdu)

    # create psf hdu
    psf_hdu = response.create_psf_hdu(
        event_energies,
        angular_distance,
        event_offsets,
        bins_energy=energy_bins,
        fov=12 * u.deg,
        rad_bins=20,
        smoothing=1
    )
    hdus.add_cta_meta_information_to_hdu(psf_hdu)

    t_obs = 50 * u.h

    proton_runs = fact.io.read_data(protons_path, key='runs')
    mc_production_protons = MCSpectrum.from_cta_runs(proton_runs)
    
    proton_events = fact.io.read_data(protons_path, key='array_events', columns=columns)
    proton_events = apply_cuts(proton_events, cuts_path)
    
    energies = proton_events.gamma_energy_prediction_mean.values * u.TeV
    proton_events['weight'] = mc_production_protons.reweigh_to_other_spectrum(CTAProtonSpectrum(), energies, t_assumed_obs=t_obs)

    electron_runs = fact.io.read_data(electrons_path, key='runs')
    mc_production_electrons = MCSpectrum.from_cta_runs(electron_runs)
    
    electron_events = fact.io.read_data(electrons_path, key='array_events', columns=columns)
    electron_events = apply_cuts(proton_events, cuts_path)
    
    energies = electron_events.gamma_energy_prediction_mean.values * u.TeV
    electron_events['weight'] = mc_production_protons.reweigh_to_other_spectrum(CTAProtonSpectrum(), energies, t_assumed_obs=t_obs)

    background = pd.concat([proton_events, electron_events])

    mc_alt = background.mc_alt.values * u.deg
    mc_az = background.mc_az.values * u.deg
    event_energy = background.gamma_energy_prediction_mean.values * u.TeV
    weights = background.weight

    background_event_offsets = calculate_fov_offset(pointing[0] * u.deg, pointing[1] * u.deg, mc_alt, mc_az)
    
    # from IPython import embed; embed()
    bkg_hdu = response.create_bkg_hdu(event_energy, background_event_offsets, weights, energy_bins, theta_bins=5, fov=10*u.deg,  smoothing=1)
    hdus.add_cta_meta_information_to_hdu(bkg_hdu)
    # response.create_bkg_hdu()

    hdulist = fits.HDUList([primary_hdu, a_eff_hdu, e_disp_hdu, psf_hdu, bkg_hdu])
    hdulist.writeto(output_path, overwrite=True)


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
