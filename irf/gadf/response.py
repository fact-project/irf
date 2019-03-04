import astropy.units as u
from astropy.io import fits
from astropy.table import Table
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from astropy.coordinates.angle_utilities import angular_separation

from irf.collection_area import collection_area
from irf.energy_dispersion import energy_migration
from irf.psf import binned_psf_vs_energy
from irf.background import background_vs_energy
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


def create_effective_area_hdu(mc_production, true_event_energy, event_offsets, bin_edges, num_theta_bins=5, fov=10*u.deg, sample_fraction=1.0, smoothing=0.0):
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

    theta_bin_edges = np.linspace(0, fov.to('deg').value / 2, endpoint=True, num=num_theta_bins + 1) * u.deg
    theta_lo = theta_bin_edges[np.newaxis, :-1]
    theta_hi = theta_bin_edges[np.newaxis, 1:]
    areas = []
    for lower, upper in zip(theta_lo[0], theta_hi[0]):
        m = (lower <= event_offsets) & (event_offsets < upper)
        f = (upper.value**2 - lower.value**2) / ((fov.value / 2) ** 2) * sample_fraction

        r = collection_area(
            mc_production,
            true_event_energy[m],
            bin_edges=bin_edges,
            sample_fraction=f,
            smoothing=smoothing,
        )

        area, _, _, _, _ = r
        areas.append(area.value)

    area = np.vstack(areas)
    area = area[np.newaxis, :] * u.m**2
    print('AEFF', area.shape, energy_lo.shape, theta_lo.shape)
    t = Table(
        {
            'ENERG_LO': energy_lo,
            'ENERG_HI': energy_hi,
            'THETA_LO': theta_lo,
            'THETA_HI': theta_hi,
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


def create_energy_dispersion_hdu(true_event_energy, predicted_event_energy, event_offset, bins_e_true, num_theta_bins=5, fov=10*u.deg,  smoothing=1):
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
    
    bins_mu = np.linspace(0, 6, endpoint=True, num=len(bins_e_true)+2)

    energy_lo = bins_e_true[np.newaxis, :-1]
    energy_hi = bins_e_true[np.newaxis, 1:]

    migra_lo = bins_mu[np.newaxis, :-1]
    migra_hi = bins_mu[np.newaxis, 1:]

    theta_bin_edges = np.linspace(0, fov.to_value(u.deg) / 2, endpoint=True, num=num_theta_bins + 1) * u.deg
    theta_lo = theta_bin_edges[np.newaxis, :-1]
    theta_hi = theta_bin_edges[np.newaxis, 1:]

    migras = []
    for lower, upper in zip(theta_lo[0], theta_hi[0]):
        m = (lower <= event_offset) & (event_offset < upper)
        migra, bins_e_true, bins_mu = energy_migration(true_event_energy[m], predicted_event_energy[m], bins_energy=bins_e_true, bins_mu=bins_mu, normalize=True, smoothing=smoothing)
        migras.append(migra.T)

    matrix = np.stack(migras)[np.newaxis, :]
    print('EDISP', matrix.shape)
    t = Table(
        {
            'ENERG_LO': energy_lo,
            'ENERG_HI': energy_hi,
            'MIGRA_LO': migra_lo,
            'MIGRA_HI': migra_hi,
            'THETA_LO': theta_lo,
            'THETA_HI': theta_hi,
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



def create_psf_hdu(event_energy, angular_separation, event_offset, bins_energy, rad_bins=20, theta_bins=5, fov=10*u.deg,  smoothing=1):
    '''
    Creates the effective area hdu to be written into a fits file accroding to the format
    described here:
    http://gamma-astro-data-formats.readthedocs.io/en/latest/irfs/full_enclosure/aeff/index.html

    Parameters
    ----------
    event_energy :  array-like astropy.unit.Quantity (energy)
        energy of selected events
    angular_separation :  array-like astropy.unit.Quantity (deg)
        distance to true source position
    event_offsets : astropy.unit.Quantity (deg)
        the offset from the pointing direction in degrees
    bins_energy : astropy.unit.Quantity (deg)
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

    if np.isscalar(bins_energy):
        print('You have to provide the actual bin edges for the enrgy binning')
        raise ValueError
    
    if np.isscalar(rad_bins):
        rad_max = 1.5
        rad_min = 0
        rad_bins = np.linspace(rad_min, rad_max, rad_bins) * u.deg
        # rad_bins = d**2/d.max()

    if np.isscalar(theta_bins):
        theta_bins = np.linspace(0, fov.to_value(u.deg) / 2, endpoint=True, num=theta_bins + 1) * u.deg

    rad_lo = rad_bins[np.newaxis, :-1]
    rad_hi = rad_bins[np.newaxis, 1:]
    
    energy_lo = bins_energy[np.newaxis, :-1]
    energy_hi = bins_energy[np.newaxis, 1:]
    
    theta_lo = theta_bins[np.newaxis, :-1]
    theta_hi = theta_bins[np.newaxis, 1:]

    r = ((rad_bins[:-1] + rad_bins[1:]) / 2).to_value(u.deg)
    deg2sr = (np.pi/180)**2
    migras = []
    for lower, upper in zip(theta_lo[0], theta_hi[0]):
        m_offset = (lower <= event_offset) & (event_offset < upper)
        for lower_e, upper_e in zip(energy_lo[0], energy_hi[0]):
            m_energy = (lower_e <= event_energy[m_offset]) & (event_energy[m_offset] < upper_e)
            psf, _ = np.histogram(angular_separation[m_offset][m_energy], bins=rad_bins, density=True) 
            norm = 2 * np.pi * r
            migras.append((psf/norm) / deg2sr)
            # print(((psf/norm) * rad_bins.diff() * 2* np.pi * r*u.deg ).sum())

        # solid_angle = (1 - np.cos(upper - lower)) * 2 * np.pi
        # print(solid_angle)
        # psf = binned_psf_vs_energy(event_energy[m],  angular_separation[m],  rad_bins=rad_bins, energy_bin_edges=bins_energy, smoothing=0) / solid_angle

    # print((migras[0][10] * solid_angle).sum())

    matrix = np.stack(migras).reshape(len(theta_bins) - 1, len(bins_energy) - 1, -1)
    matrix = matrix[np.newaxis, :]
    matrix = np.transpose(matrix, [0, 3, 1, 2])
    if smoothing > 0:
        a = matrix.copy()
        matrix = gaussian_filter(a, sigma=smoothing)

    t = Table(
        {
            'ENERG_LO': energy_lo,
            'ENERG_HI': energy_hi,
            'RAD_LO': rad_lo,
            'RAD_HI': rad_hi,
            'THETA_LO': theta_lo,
            'THETA_HI': theta_hi,
            'RPSF': matrix / u.sr,
        }
    )

    t.meta['HDUCLAS1'] = 'RESPONSE'
    t.meta['HDUCLAS2'] = 'PSF'
    t.meta['HDUCLAS3'] = 'FULL-ENCLOSURE'
    t.meta['HDUCLAS4'] = 'PSF_2D'
    t.meta['EXTNAME'] = 'PSF'

    hdu = fits.table_to_hdu(t)
    return hdu



def create_bkg_hdu(event_energy, event_offset, weights, bins_energy, theta_bins=5, fov=10*u.deg,  smoothing=1):
    '''
    Creates the effective area hdu to be written into a fits file accroding to the format
    described here:
    http://gamma-astro-data-formats.readthedocs.io/en/latest/irfs/full_enclosure/aeff/index.html

    Parameters
    ----------
    event_energy :  array-like astropy.unit.Quantity (energy)
        energy of selected events
    event_offsets : astropy.unit.Quantity (deg)
        the offset from the pointing direction in degrees
    weights : astropy.unit.Quantity (deg)
        the weight of the events
    bins_energy : astropy.unit.Quantity (deg)
        the energy bin edges.
    theta_bins : int
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

    if np.isscalar(bins_energy):
        print('You have to provide the actual bin edges for the enrgy binning')
        raise ValueError
    
    if np.isscalar(theta_bins):
        theta_bins = np.linspace(0, fov.to_value(u.deg) / 2, endpoint=True, num=theta_bins + 1) * u.deg

    energy_lo = bins_energy[np.newaxis, :-1]
    energy_hi = bins_energy[np.newaxis, 1:]
    
    theta_lo = theta_bins[np.newaxis, :-1]
    theta_hi = theta_bins[np.newaxis, 1:]

    migras = []
    for lower, upper in zip(theta_lo[0], theta_hi[0]):
        m = (lower <= event_offset) & (event_offset < upper)
        bkg = background_vs_energy(event_energy[m], weights[m], energy_bins=bins_energy, smoothing=0)
        migras.append(bkg)

    # transpose here. See https://github.com/gammapy/gammapy/issues/2067
    matrix = np.stack(migras).T[np.newaxis, :]

    if smoothing > 0:
        a = matrix.copy()
        matrix = gaussian_filter(a, sigma=smoothing)

    print('BKG', matrix.shape, energy_lo.shape, theta_lo.shape)
    t = Table(
        {
            'ENERG_LO': energy_lo,
            'ENERG_HI': energy_hi,
            'THETA_LO': theta_lo,
            'THETA_HI': theta_hi,
            'BKG': matrix/u.sr/u.s/u.m**2,
        }
    )

    t.meta['HDUCLAS1'] = 'RESPONSE'
    t.meta['HDUCLAS2'] = 'BKG'
    t.meta['HDUCLAS3'] = 'FULL-ENCLOSURE'
    t.meta['HDUCLAS4'] = 'BKG_2D'
    t.meta['EXTNAME'] = 'BACKGROUND'

    hdu = fits.table_to_hdu(t)
    return hdu
