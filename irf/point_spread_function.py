import astropy.units as u
import numpy as np
import datetime
from astropy.table import Table
from scipy.ndimage.filters import gaussian_filter
from astropy.coordinates.angle_utilities import angular_separation
from astropy.coordinates import Angle


@u.quantity_input(event_energies=u.TeV, true_altitude=u.deg, true_azimuth=u.deg, predicted_altitude=u.deg, predicted_azimuth=u.deg )
def psf_to_irf_table(
    true_altitude,
    true_azimuth,
    predicted_altitude,
    predicted_azimuth,
    event_energies,
    event_fov_offsets,
    fov=4.5 * u.deg,
    energy_bins=10,
    offset_bins=2,
    psf_bins=20,
    smoothing=0,
):
    '''
    See here what that format is supposed to look like:
    http://gamma-astro-data-formats.readthedocs.io/en/latest/irfs/full_enclosure/aeff/index.html
    '''

    event_energies = event_energies.to('TeV')

    if np.isscalar(energy_bins):
        low = np.log10(event_energies.min().value)
        high = np.log10(event_energies.max().value)
        energy_bins = np.logspace(low, high, endpoint=True, num=energy_bins + 1) * event_energies.unit

    if np.isscalar(offset_bins):
        offset_bins = np.linspace(0, fov.to('deg') / 2, endpoint=True, num=offset_bins + 1)

    if np.isscalar(psf_bins):
        psf_bins = np.linspace(0, 20, endpoint=True, num=psf_bins + 1) * u.deg


    theta_lo = offset_bins[np.newaxis, :-1]
    theta_hi = offset_bins[np.newaxis, 1:]

    energy_lo = energy_bins[np.newaxis, :-1]
    energy_hi = energy_bins[np.newaxis, 1:]

    rad_lo = psf_bins[np.newaxis, :-1]
    rad_hi = psf_bins[np.newaxis, 1:]


    migras = []
    for t_l, t_u in zip(theta_lo[0], theta_hi[0]):
        psfs = []
        for e_l, e_u in zip(energy_lo[0], energy_hi[0]):
            m = (t_l <= event_fov_offsets) & (event_fov_offsets < t_u) & (e_l <= event_energies) & (event_energies < e_u)
            psf, _ = point_spread_function(true_altitude[m], true_azimuth[m], predicted_altitude[m], predicted_azimuth[m], bins=psf_bins, normalize=True, smoothing=smoothing)
            psfs.append(psf)

        migras.append(np.array(psfs))


    matrix = np.transpose(np.stack(migras), axes=[2, 0, 1])[np.newaxis, :] / u.deg**2
    # matrix = matrix / 180**2

    # import IPython; IPython.embed()

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

    t.meta['DATE'] = datetime.datetime.now().replace(microsecond=0).isoformat()
    t.meta['TELESCOP'] = 'FACT'
    t.meta['HDUCLASS'] = 'GADF'
    t.meta['HDUCLAS1'] = 'RESPONSE'
    t.meta['HDUCLAS2'] = 'PSF'
    t.meta['HDUCLAS3'] = 'FULL-ENCLOSURE'
    t.meta['HDUCLAS4'] = 'PSF_TABLE'
    t.meta['EXTNAME'] = 'POINT SPREAD FUNCTION'

    return t


@u.quantity_input(true_altitude=u.deg, true_azimuth=u.deg, predicted_altitude=u.deg, predicted_azimuth=u.deg,)
def point_spread_function(true_altitude, true_azimuth, predicted_altitude, predicted_azimuth, bins=10, normalize=True, smoothing=0):

    alt = Angle(predicted_altitude)
    mc_alt = Angle(true_altitude)

    az = Angle(predicted_azimuth).wrap_at(180 * u.deg)
    mc_az = Angle(true_azimuth).wrap_at(180 * u.deg)

    distance = angular_separation(mc_az, mc_alt, az, alt).to('deg')

    if np.isscalar(bins):
        limit = np.percentile(distance, 99)
        bins = np.linspace(0, limit, bins) * u.deg

    range = (bins.min().value, bins.max().value)
    h, _ = np.histogram(distance.value, bins=bins, density=normalize, range=range)

    if smoothing > 0:
        h = gaussian_filter(h, sigma=smoothing, )

    return h, bins


@u.quantity_input(true_altitude=u.deg, true_azimuth=u.deg, predicted_altitude=u.deg, predicted_azimuth=u.deg, event_energies=u.TeV)
def psf_vs_energy(true_altitude, true_azimuth, predicted_altitude, predicted_azimuth, event_energies, energy_bins=10, psf_bins=10, normalize=True, smoothing=0):

    alt = Angle(predicted_altitude)
    mc_alt = Angle(true_altitude)

    az = Angle(predicted_azimuth).wrap_at(180 * u.deg)
    mc_az = Angle(true_azimuth).wrap_at(180 * u.deg)

    distance = angular_separation(mc_az, mc_alt, az, alt).to('deg')

    if np.isscalar(psf_bins):
        limit = np.nanpercentile(distance, 99)
        psf_bins = np.linspace(0, limit, endpoint=True, num=psf_bins + 1) * u.deg

    if np.isscalar(energy_bins):
        low = np.log10(event_energies.min().value)
        high = np.log10(event_energies.max().value)
        energy_bins = np.logspace(low, high, endpoint=True, num=energy_bins + 1) * event_energies.unit


    range = [(energy_bins.min().value, energy_bins.max().value), (psf_bins.min().value, psf_bins.max().value)]

    hist, _, _ = np.histogram2d(event_energies.value, distance.value, bins=[energy_bins, psf_bins], range=range)
    if normalize:
        h = hist.T
        h = h / h.sum(axis=0)
        hist = np.nan_to_num(h).T

    if smoothing > 0:
        hist = gaussian_filter(hist, sigma=smoothing, )

    return hist, energy_bins, psf_bins
