import astropy.units as u
from astropy.io import fits
from astropy.table import Table
import numpy as np
from astropy.coordinates.angle_utilities import angular_separation
from irf.collection_area import collection_area
from irf.energy_dispersion import energy_migration
from irf.gadf.hdus import add_fact_meta_information_to_hdu, add_cta_meta_information_to_hdu



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


def create_effective_area_hdu(mc_production, true_event_energy, event_offsets, bin_edges, fov, sample_fraction=1.0, smoothing=0.0):
    '''
    Creates the effective area hdu to be written into a fits file accroding to the format
    described here:
    http://gamma-astro-data-formats.readthedocs.io/en/latest/irfs/full_enclosure/aeff/index.html

    Parameters
    ----------

    mc_production : irf.spectrum.MCSpectrum
        Corsika shower production
    true_event_energy :  array-like astropy.unit.Quantity (energy)
        true energy of selected events
    event_offsets : astropy.unit.Quantity (deg)
        the offset from the pointing direction in degrees
    bin_edges : int or arraylike
        the energy bins.
    fov :  astropy.unit.Quantity (degree)
        the field of view of the telescope
    smoothing : float
        Amount of smoothing to apply to the generated matrices.
        Equivalent to the sigma parameter in
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html    
    sampling_fraction : float
        Ratio of events actually taken into account from the MC production. Usually this should remain at 1

    Returns
    -------

    astropy.io.fits.hdu
        The fits hdu as required by the GADF

    '''

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


def create_energy_dispersion_hdu(true_event_energy, predicted_event_energy,  event_offset, bins_e_true, num_theta_bins=5, fov=10*u.deg,  smoothing=1):
    '''
    Creates the effective area hdu to be written into a fits file accroding to the format
    described here:
    http://gamma-astro-data-formats.readthedocs.io/en/latest/irfs/full_enclosure/aeff/index.html

    Parameters
    ----------
    true_event_energy :  array-like astropy.unit.Quantity (energy)
        true energy of selected events
    predicted_event_energy :  array-like astropy.unit.Quantity (energy)
        estimated energy of selected events
    event_offsets : astropy.unit.Quantity (deg)
        the offset from the pointing direction in degrees
    bins_e_true : arraylike
        the energy bin edges.
    num_theta_bins : int
        the number of fov bins to use.
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

    if np.isscalar(bins_e_true):
        print('You have to provide the actual bin edges for the enrgy binning')
        raise ValueError
    
    bins_mu = np.linspace(0, 6, endpoint=True, num=len(bins_e_true))

    energy_lo = bins_e_true[np.newaxis, :-1]
    energy_hi = bins_e_true[np.newaxis, 1:]

    migra_lo = bins_mu[np.newaxis, :-1]
    migra_hi = bins_mu[np.newaxis, 1:]

    theta_bin_edges = np.linspace(0, fov.to_value(u.deg) / 2, endpoint=True, num=num_theta_bins + 1)
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
    return hdu


