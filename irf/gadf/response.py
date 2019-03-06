import astropy.units as u
from astropy.io import fits
from astropy.table import Table
import numpy as np
from astropy.coordinates import SkyCoord, AltAz
from ctapipe.coordinates import NominalFrame
from scipy.ndimage.filters import gaussian_filter
# from astropy.coordinates.angle_utilities import angular_separation

from irf.collection_area import collection_area_vs_offset
from irf.energy_dispersion import energy_migration
from irf.psf import binned_psf_vs_energy
from irf.background import background_vs_offset
from irf.gadf.hdus import add_fact_meta_information_to_hdu, add_cta_meta_information_to_hdu
from irf.spectrum import CTAProtonSpectrum, CTAElectronSpectrum, CrabSpectrum

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


def create_effective_area_hdu(
    mc_production,
    event_energies,
    event_offsets,
    energy_bin_edges,
    theta_bin_edges,
    sample_fraction=1.0,
    smoothing=0.0
):
    '''
    Creates the effective area hdu to be written into a fits file accroding to the format
    described here:
    http://gamma-astro-data-formats.readthedocs.io/en/latest/irfs/full_enclosure/aeff/index.html

    Parameters
    ----------

    mc_production : irf.spectrum.MCSpectrum
        Corsika shower production
    event_energies :  array-like astropy.unit.Quantity (energy)
        true energy of selected events
    event_offsets : astropy.unit.Quantity (deg)
        the offset from the pointing direction in degrees
    energy_bin_edges : arraylike Quantity (TeV)
        the energy bin edges.
    theta_bin_edges : arraylike Quantity (deg)
        the bin edges for the event offsets
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

    area = collection_area_vs_offset(
        mc_production,
        event_energies,
        event_offsets,
        energy_bin_edges,
        theta_bin_edges,
        smoothing=smoothing,
        sample_fraction=sample_fraction,

    )
    area = area[np.newaxis, :]
    
    energy_lo = energy_bin_edges[np.newaxis, :-1]
    energy_hi = energy_bin_edges[np.newaxis, 1:]

    theta_lo = theta_bin_edges[np.newaxis, :-1]
    theta_hi = theta_bin_edges[np.newaxis, 1:]

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


def create_energy_dispersion_hdu(true_event_energy, predicted_event_energy, event_offset, energy_bin_edges, theta_bin_edges, smoothing=1):
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
    energy_bin_edges : arraylike
        the energy bin edges.
    theta_bin_edges : int
        the number of fov bins to use.
    smoothing : float
        Amount of smoothing to apply to the generated matrices.
        Equivalent to the sigma parameter in
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html    

    Returns
    -------

    astropy.io.fits.hdu
        The fits hdu as required by the GADF

    '''

    
    bins_mu = np.linspace(0, 3, endpoint=True, num=300)

    energy_lo = energy_bin_edges[np.newaxis, :-1]
    energy_hi = energy_bin_edges[np.newaxis, 1:]

    migra_lo = bins_mu[np.newaxis, :-1]
    migra_hi = bins_mu[np.newaxis, 1:]

    theta_lo = theta_bin_edges[np.newaxis, :-1]
    theta_hi = theta_bin_edges[np.newaxis, 1:]

    migras = []
    for lower, upper in zip(theta_lo[0], theta_hi[0]):
        m = (lower <= event_offset) & (event_offset < upper)
        migra, bins_e_true, bins_mu = energy_migration(true_event_energy[m], predicted_event_energy[m], bins_energy=energy_bin_edges, bins_mu=bins_mu, normalize=False, smoothing=0)
        migras.append(migra.T)

    matrix = np.stack(migras)[np.newaxis, :]
    if smoothing > 0:
        a = matrix.copy()
        matrix = gaussian_filter(a, sigma=smoothing)

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



def create_psf_hdu(event_energy, angular_separation, event_offset, energy_bin_edges,  theta_bin_edges, rad_bins=20, smoothing=1):
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
    energy_bin_edges : astropy.unit.Quantity (deg)
        the energy bin edges.
    theta_bin_edges : array-like (deg)
        theta bins to use
    smoothing : float
        Amount of smoothing to apply to the generated matrices.
        Equivalent to the sigma parameter in
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html    

    Returns
    -------

    astropy.io.fits.hdu
        The fits hdu as required by the GADF

    '''

    
    if np.isscalar(rad_bins):
        rad_bins = np.linspace(0, theta_bin_edges.max().to_value(u.deg), rad_bins + 1) * u.deg

    rad_lo = rad_bins[np.newaxis, :-1]
    rad_hi = rad_bins[np.newaxis, 1:]

    energy_lo = energy_bin_edges[np.newaxis, :-1]
    energy_hi = energy_bin_edges[np.newaxis, 1:]
    
    theta_lo = theta_bin_edges[np.newaxis, :-1]
    theta_hi = theta_bin_edges[np.newaxis, 1:]

    migras = []
    for lower, upper in zip(theta_lo[0], theta_hi[0]):
        m = (lower <= event_offset) & (event_offset < upper)
        psf = binned_psf_vs_energy(event_energy[m],  angular_separation[m],  rad_bins=rad_bins, energy_bin_edges=energy_bin_edges, smoothing=0)
        migras.append(psf)

    matrix = np.stack(migras)[np.newaxis, :]
    matrix = np.transpose(matrix, [0, 3, 1, 2])/u.sr
    if smoothing > 0:
        a = matrix.copy()
        matrix = gaussian_filter(a, sigma=smoothing) * matrix.unit

    print('PSF', matrix.shape, matrix.unit)
    t = Table(
        {
            'ENERG_LO': energy_lo,
            'ENERG_HI': energy_hi,
            'RAD_LO': rad_lo,
            'RAD_HI': rad_hi,
            'THETA_LO': theta_lo,
            'THETA_HI': theta_hi,
            'RPSF': matrix,
        }
    )

    t.meta['HDUCLAS1'] = 'RESPONSE'
    t.meta['HDUCLAS2'] = 'PSF'
    t.meta['HDUCLAS3'] = 'FULL-ENCLOSURE'
    t.meta['HDUCLAS4'] = 'PSF_2D'
    t.meta['EXTNAME'] = 'PSF'

    hdu = fits.table_to_hdu(t)
    return hdu

def create_bkg_hdu(
    mc_production_proton,
    proton_event_energies,
    proton_event_alt,
    proton_event_az,
    mc_production_electron,
    electron_event_energies,
    electron_event_alt,
    electron_event_az,
    energy_bin_edges,
    smoothing=1
):

    t_assumed_obs = 1*u.s

    # proton_collection_area = collection_area_vs_offset(
    #     mc_production_proton, 
    #     proton_event_energies, 
    #     proton_event_offset,
    #     energy_bin_edges,
    #     theta_bin_edges,
    # )
    proton_weights = mc_production_proton.reweigh_to_other_spectrum(
        CTAProtonSpectrum(),
        proton_event_energies,
        t_assumed_obs=t_assumed_obs
    )

    # electron_collection_area = collection_area_vs_offset(
    #     mc_production_electron, 
    #     electron_event_energies, 
    #     electron_event_offset,
    #     energy_bin_edges,
    #     theta_bin_edges,
    # )
    electron_weights = mc_production_electron.reweigh_to_other_spectrum(
        CTAElectronSpectrum(),
        electron_event_energies,
        t_assumed_obs=t_assumed_obs
    )

    det_bins = np.linspace(-6, 6, 40+1) * u.deg
    
    pointing_position = SkyCoord(alt=70 * u.deg, az=-180 * u.deg, frame=AltAz())
    nominal = NominalFrame(origin=pointing_position)
    
    proton_altaz = SkyCoord(alt=proton_event_alt, az=proton_event_az, frame=AltAz())
    proton_nominal = proton_altaz.transform_to(nominal)
    proton_daz = proton_nominal.delta_az.to_value(u.deg)
    proton_dalt = proton_nominal.delta_alt.to_value(u.deg)

    electron_altaz = SkyCoord(alt=electron_event_alt, az=electron_event_az, frame=AltAz())
    electron_nominal = electron_altaz.transform_to(nominal)
    electron_daz = electron_nominal.delta_az.to_value(u.deg)
    electron_dalt = electron_nominal.delta_alt.to_value(u.deg)
    
    proton_bkg, _ = np.histogramdd(
        (proton_event_energies, proton_dalt, proton_daz),
        bins=(energy_bin_edges, det_bins, det_bins),
        weights=proton_weights,
        # density=True,
    )

    electron_bkg, _ = np.histogramdd(
        (electron_event_energies, electron_dalt, electron_daz),
        bins=(energy_bin_edges, det_bins, det_bins),
        weights=electron_weights,
        # density=True,
    )
    
    solid_angle = (1 - np.cos(np.diff(det_bins).to_value(u.rad))) * 2 * np.pi * u.sr
    matrix = electron_bkg + proton_bkg
    matrix = matrix / solid_angle / u.s 
    matrix = matrix / energy_bin_edges.diff()[:, np.newaxis, np.newaxis]

    energy_lo = energy_bin_edges[np.newaxis, :-1]
    energy_hi = energy_bin_edges[np.newaxis, 1:]
    
    det_lo = det_bins[np.newaxis, :-1]
    det_hi = det_bins[np.newaxis, 1:]

    # this wants per MeV units for some reason.
    # See https://gamma-astro-data-formats.readthedocs.io/en/latest/irfs/full_enclosure/bkg/index.html
    matrix = matrix.to(1/u.s/u.sr/u.MeV)
    matrix = matrix[np.newaxis, :]
    
    
    if smoothing > 0:
        a = matrix.copy()
        matrix = gaussian_filter(a, sigma=smoothing) * matrix.unit

    t = Table(
        {
            'ENERG_LO': energy_lo,
            'ENERG_HI': energy_hi,
            'DETX_LO': det_lo,
            'DETX_HI': det_hi,
            'DETY_LO': det_lo,
            'DETY_HI': det_hi,
            'BKG': matrix,
        }
    )
    t.meta['HDUCLAS1'] = 'RESPONSE'
    t.meta['HDUCLAS2'] = 'BKG'
    t.meta['HDUCLAS3'] = 'FULL-ENCLOSURE'
    t.meta['HDUCLAS4'] = 'BKG_3D'
    t.meta['EXTNAME'] = 'BACKGROUND'

    hdu = fits.table_to_hdu(t)
    return hdu
