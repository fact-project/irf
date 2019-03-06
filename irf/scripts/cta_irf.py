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
from irf.spectrum import MCSpectrum, CTAProtonSpectrum, CTAElectronSpectrum, CrabSpectrum


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

columns  = [
    'alt',
    'az',
    'mc_energy',
    'gamma_energy_prediction_mean',
    'gamma_prediction_mean',
    'mc_alt',
    'mc_az',
    'num_triggered_telescopes'
]


def load_data(path, cuts_path, pointing):
    runs = fact.io.read_data(path, key='runs')
    mc_production = MCSpectrum.from_cta_runs(runs)
    
    events = fact.io.read_data(path, key='array_events', columns=columns).dropna()
    events = apply_cuts(events, cuts_path)

    mc_alt = events.mc_alt.values * u.deg
    mc_az = events.mc_az.values * u.deg

    alt = events.alt.values * u.deg
    az = events.az.values * u.deg
    
    # TODO i think this offset should be calculated from estimated coordinates. I should make sure tho
    event_offsets = calculate_fov_offset(pointing[0] * u.deg, pointing[1] * u.deg, alt, az)
    events['fov_offset'] = event_offsets

    angular_distance = calculate_angular_separation(mc_alt, mc_az, alt, az)
    events['distance_to_source'] = angular_distance
    return events, mc_production


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
@click.option('-p', '--pointing', nargs=2, type=float, default=(70, -180))
def main(gammas_diffuse_path, protons_path, electrons_path, cuts_path,  output_path, pointing):

    fov = 10*u.deg # TODO make sure this is the entire FoV
    
    # energy_bins = np.logspace(-2, 2, num=100 + 1) * u.TeV
    # theta_bins = np.linspace(0, fov.to_value(u.deg) / 2, endpoint=True, num=8 + 1) * u.deg
    
    gamma_events, mc_production_gammas  = load_data(gammas_diffuse_path, cuts_path, pointing=pointing)
    gamma_event_energies = gamma_events.mc_energy.values * u.TeV
    gamma_estimated_event_energies = gamma_events.gamma_energy_prediction_mean.values * u.TeV
    
    gamma_event_offsets = gamma_events.fov_offset.values * u.deg
    gamma_event_distance_to_source = gamma_events.distance_to_source.values * u.deg
    
    primary_hdu = hdus.create_primary_hdu_cta()

    # create effective area hdu

    energy_bins = np.logspace(-2, 2, num=30 + 1) * u.TeV
    theta_bins = np.linspace(0, fov.to_value(u.deg) / 2, endpoint=True, num=8 + 1) * u.deg
    a_eff_hdu = response.create_effective_area_hdu(
        mc_production_gammas,
        gamma_event_energies,
        gamma_event_offsets,
        energy_bin_edges=energy_bins,
        theta_bin_edges=theta_bins,
        smoothing=1,
    )
    hdus.add_cta_meta_information_to_hdu(a_eff_hdu)

    # create edisp hdu
    energy_bins = np.logspace(-2, 2, num=125 + 1) * u.TeV
    theta_bins = np.linspace(0, fov.to_value(u.deg) / 2, endpoint=True, num=8 + 1) * u.deg
    e_disp_hdu = response.create_energy_dispersion_hdu(
        gamma_event_energies,
        gamma_estimated_event_energies,
        gamma_event_offsets,
        energy_bin_edges=energy_bins,
        theta_bin_edges=theta_bins,
        smoothing=0,
    )
    hdus.add_cta_meta_information_to_hdu(e_disp_hdu)

    # create psf hdu
    energy_bins = np.logspace(-2, 2.2, num=10 + 1) * u.TeV
    theta_bins = np.linspace(0, fov.to_value(u.deg) / 2, endpoint=True, num=6 + 1) * u.deg
    psf_hdu = response.create_psf_hdu(
        gamma_event_energies,
        gamma_event_distance_to_source,
        gamma_event_offsets,
        energy_bin_edges=energy_bins,
        theta_bin_edges=theta_bins,
        rad_bins=50,
        smoothing=1, 
    )
    hdus.add_cta_meta_information_to_hdu(psf_hdu)


    proton_events, mc_production_protons = load_data(protons_path, cuts_path, pointing)
    proton_estimated_energies = proton_events.gamma_energy_prediction_mean.values * u.TeV
    proton_alt = proton_events.alt.values * u.deg
    proton_az = proton_events.az.values * u.deg
    # proton_offsets = proton_events.fov_offset.values * u.deg
    # proton_distance_to_source = proton_events.distance_to_source.values * u.deg


    electron_events, mc_production_electrons = load_data(electrons_path, cuts_path, pointing)
    electron_estimated_energies = electron_events.gamma_energy_prediction_mean.values * u.TeV
    electron_alt = electron_events.alt.values * u.deg
    electron_az = electron_events.az.values * u.deg
    # electron_distance_to_source = electron_events.distance_to_source.values * u.deg

    energy_bins = np.logspace(-2, 2, num=20 + 1) * u.TeV

    bkg_hdu = response.create_bkg_hdu(
        mc_production_protons,
        proton_estimated_energies,
        proton_alt,
        proton_az,
        mc_production_electrons,
        electron_estimated_energies,
        electron_alt,
        electron_az,
        energy_bins,
        smoothing=0.1,
    )
    hdus.add_cta_meta_information_to_hdu(bkg_hdu)
 

    hdulist = fits.HDUList([primary_hdu, a_eff_hdu, e_disp_hdu, psf_hdu, bkg_hdu])
    hdulist.writeto(output_path, overwrite=True)


if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    main()
