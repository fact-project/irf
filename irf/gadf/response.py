import astropy.units as u
from astropy.io import fits
from astropy.table import Table
import numpy as np
from astropy.coordinates.angle_utilities import angular_separation
from irf.collection_area import collection_area
from irf.energy_dispersion import energy_migration
from irf.gadf.hdus import add_fact_meta_information_to_hdu, add_cta_meta_information_to_hdu


def _calculate_fov_offset(df):
    '''
    Calculate the `offset` aka the `angular_separation` between the pointing and
    the source position.
    '''
    pointing_lat = (90 - df.aux_pointing_position_zd.values) * u.deg
    pointing_lon = df.aux_pointing_position_az.values * u.deg

    source_lat = (90 - df.source_position_zd.values) * u.deg
    source_lon = df.source_position_az.values * u.deg

    return angular_separation(pointing_lon, pointing_lat, source_lon, source_lat).to('deg')

def _calculate_fov_offset_cta(pointing_altitude, pointing_azimuth, source_altitude, source_azimuth):
    pointing_lat = pointing_altitude.to(u.deg)
    pointing_lon = pointing_azimuth.to(u.deg)

    source_lat = source_altitude.to(u.deg)
    source_lon = source_azimuth.to(u.deg)

    return angular_separation(pointing_lon, pointing_lat, source_lon, source_lat).to('deg')

def _make_energy_bins(energy_true, energy_prediction, bins):
    e_min = min(
        min(energy_true),
        min(energy_prediction)
    )

    e_max = max(
        max(energy_true),
        max(energy_prediction)
    )

    low = np.log10(e_min.value)
    high = np.log10(e_max.value)
    bin_edges = np.logspace(low, high, endpoint=True, num=bins + 1)

    return bin_edges


@u.quantity_input(fov=u.deg, impact=u.m)
def effective_area_hdu_for_fact(
    mc_production,
    selected_diffuse_gammas,
    bins=10,
    sample_fraction=1.0,
    fov=4.5 * u.deg,
    smoothing=0,
):
    '''
    Creates the effective area hdu to be written into a fits file accroding to the format
    described here:
    http://gamma-astro-data-formats.readthedocs.io/en/latest/irfs/full_enclosure/aeff/index.html

    Parameters
    ----------

     : pd.DataFrame
        the corsika shower information as produced by the 'read_corsika_headers' script
        in this project.
    selected_diffuse_gammas :  pd.DataFrame
        FACT DL2 data for diffuse gamma showers.
    bins : int or arraylike
        the enrgy bins.
    impact : astropy.unit.Quantity (meter)
        the maximum simulated scatter radius
    fov :  astropy.unit.Quantity (degree)
        the field of view of the telescope
    smoothing : float
        Amount of smoothing to apply to the generated matrices.
        Equivalent to the sigma parameter in
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html

    Returns
    -------

    astropy.io.fits.hdu
        The fits hdu as required by the GADF

    '''

    true_event_energy = (selected_diffuse_gammas.corsika_event_header_total_energy.values * u.GeV).to('TeV')
    event_offsets = _calculate_fov_offset(selected_diffuse_gammas)

    hdu = create_effective_area_hdu(mc_production, true_event_energy, bins, event_offsets, fov, sample_fraction, smoothing)
    add_fact_meta_information_to_hdu(hdu)
    return hdu

def effective_area_hdu_for_cta(
    mc_production,
    selected_diffuse_gammas,
    bins=10,
    sample_fraction=1.0,
    fov=12 * u.deg,
    smoothing=0,
):
    '''
    Creates the effective area hdu to be written into a fits file accroding to the format
    described here:
    http://gamma-astro-data-formats.readthedocs.io/en/latest/irfs/full_enclosure/aeff/index.html

    Parameters
    ----------
    mc_production: MCSpectrum
        a MCSpectrum instance describibng the mc production
    selected_diffuse_gammas :  pd.DataFrame
        CTA DL2 data for diffuse gamma showers.
    bins : int or arraylike
        the enrgy bins.
    fov :  astropy.unit.Quantity (degree)
        the field of view of the telescope
    smoothing : float
        Amount of smoothing to apply to the generated matrices.
        Equivalent to the sigma parameter in
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html

    Returns
    -------

    astropy.io.fits.hdu
        The fits hdu as required by the GADF

    '''

    true_event_energy = selected_diffuse_gammas.mc_energy.values * u.TeV
    event_offsets = _calculate_fov_offset_cta(selected_diffuse_gammas)
    

    hdu = create_effective_area_hdu(mc_production, true_event_energy, bins, event_offsets, fov, sample_fraction, smoothing)
    add_cta_meta_information_to_hdu(hdu)
    return hdu

def create_effective_area_hdu(mc_production, true_event_energy, bin_edges, event_offsets, fov, sample_fraction=1.0, smoothing=0.0):

    energy_lo = bin_edges[np.newaxis, :-1]
    energy_hi = bin_edges[np.newaxis, 1:]

    theta_bin_edges = np.linspace(0, fov.to('deg').value / 2, endpoint=True, num=5)
    theta_lo = theta_bin_edges[np.newaxis, :-1]
    theta_hi = theta_bin_edges[np.newaxis, 1:]
    # from IPython import embed; embed()
    areas = []
    for lower, upper in zip(theta_lo[0], theta_hi[0]):
        m = (lower <= event_offsets.value) & (event_offsets.value < upper)
        f = (upper**2 - lower**2) / ((fov.value / 2) ** 2) * sample_fraction

        r = collection_area(
            mc_production,
            true_event_energy[m],
            bin_edges=bin_edges,
            sample_fraction=f,
            smoothing=smoothing,
        )

        area, bin_center, bin_width, lower_conf, upper_conf = r
        areas.append(area.value)

    area = np.vstack(areas)
    area = area[np.newaxis, :] * u.m**2

    t = Table(
        {
            'ENERG_LO': energy_lo * u.TeV,
            'ENERG_HI': energy_hi * u.TeV,
            'THETA_LO': theta_lo * u.deg,
            'THETA_HI': theta_hi * u.deg,
            'EFFAREA': area,
        }
    )

    t.meta['HDUCLAS1'] = 'RESPONSE'
    t.meta['HDUCLAS2'] = 'EFF_AREA'
    t.meta['HDUCLAS3'] = 'POINT-LIKE'
    t.meta['HDUCLAS4'] = 'AEFF_2D'
    t.meta['EXTNAME'] = 'EFFECTIVE AREA'
    hdu = fits.table_to_hdu(t)
    return hdu


@u.quantity_input(fov=u.deg)
def energy_dispersion_hdu(selected_diffuse_gammas, bins=10, fov=4.5 * u.deg, theta_bins=3, smoothing=0):
    '''
    Creates the effective area hdu to be written into a fits file accroding to the format
    described here:
    http://gamma-astro-data-formats.readthedocs.io/en/latest/irfs/full_enclosure/aeff/index.html

    Parameters
    ----------
    selected_diffuse_gammas :  pd.DataFrame
        FACT DL2 data for diffuse gamma showers.
    bins : int or arraylike
        the enrgy bins.
    fov :  astropy.unit.Quantity (degree)
        the field of view of the telescope
    theta_bins : int
        number of bins to use for the theta axis. (offset in FoV)
    smoothing : float
        Amount of smoothing to apply to the generated matrices.
        Equivalent to the sigma parameter in
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html

    Returns
    -------

    astropy.io.fits.hdu
        The fits hdu as required by the GADF

    '''

    true_event_energy = (selected_diffuse_gammas.corsika_event_header_total_energy.values * u.GeV).to('TeV')
    predicted_event_energy = (selected_diffuse_gammas.gamma_energy_prediction.values * u.GeV).to('TeV')
    event_offset = _calculate_fov_offset(selected_diffuse_gammas)

    if np.isscalar(bins):
        bins_e_true = _make_energy_bins(true_event_energy, predicted_event_energy, bins)
    else:
        bins_e_true = bins

    bins_mu = np.linspace(0, 6, endpoint=True, num=len(bins_e_true))

    energy_lo = bins_e_true[np.newaxis, :-1]
    energy_hi = bins_e_true[np.newaxis, 1:]

    migra_lo = bins_mu[np.newaxis, :-1]
    migra_hi = bins_mu[np.newaxis, 1:]

    theta_bin_edges = np.linspace(0, fov.to('deg').value / 2, endpoint=True, num=theta_bins + 1)
    theta_lo = theta_bin_edges[np.newaxis, :-1]
    theta_hi = theta_bin_edges[np.newaxis, 1:]

    migras = []
    for lower, upper in zip(theta_lo[0], theta_hi[0]):
        m = (lower <= event_offset.value) & (event_offset.value < upper)
        migra, bins_e_true, bins_mu = energy_migration(true_event_energy[m], predicted_event_energy[m], bins_energy=bins_e_true, bins_mu=bins_mu, normalize=True, smoothing=smoothing)
        migras.append(migra.T)

    matrix = np.stack(migras)[np.newaxis, :]

    t = Table(
        {
            'ENERG_LO': energy_lo * u.TeV,
            'ENERG_HI': energy_hi * u.TeV,
            'MIGRA_LO': migra_lo,
            'MIGRA_HI': migra_hi,
            'THETA_LO': theta_lo * u.deg,
            'THETA_HI': theta_hi * u.deg,
            'MATRIX': matrix,
        }
    )

    t.meta['HDUCLAS1'] = 'RESPONSE'
    t.meta['HDUCLAS2'] = 'EDISP'
    t.meta['HDUCLAS3'] = 'POINT-LIKE'
    t.meta['HDUCLAS4'] = 'EDISP_2D'
    t.meta['EXTNAME'] = 'ENERGY DISPERSION'

    hdu = fits.table_to_hdu(t)
    add_fact_meta_information_to_hdu(hdu)
    return hdu
