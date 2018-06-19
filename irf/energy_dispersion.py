import astropy.units as u
import numpy as np
import datetime
from astropy.table import Table
from scipy.ndimage.filters import gaussian_filter


def make_energy_bins(energy_true, energy_prediction, bins):
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
    bin_edges = np.logspace(low, high, endpoint=True, num=bins + 1) * energy_true.unit

    return bin_edges


@u.quantity_input(energy_true=u.TeV, energy_prediction=u.TeV, event_offset=u.deg)
def energy_dispersion_to_irf_table(
    true_event_energies,
    predicted_event_energies,
    event_fov_offsets,
    fov=4.5 * u.deg,
    bins=10,
    offset_bins=3,
    smoothing=0,
):
    '''
    See here what that format is supposed to look like:
    http://gamma-astro-data-formats.readthedocs.io/en/latest/irfs/full_enclosure/aeff/index.html
    '''

    true_event_energies = true_event_energies.to('TeV')
    predicted_event_energies = predicted_event_energies.to('TeV')

    if np.isscalar(bins):
        bins_e_true = make_energy_bins(true_event_energies, predicted_event_energies, bins)
    else:
        bins_e_true = bins

    bins_mu = np.linspace(0, 3, endpoint=True, num=len(bins_e_true))

    energy_lo = bins_e_true[np.newaxis, :-1]
    energy_hi = bins_e_true[np.newaxis, 1:]

    migra_lo = bins_mu[np.newaxis, :-1]
    migra_hi = bins_mu[np.newaxis, 1:]

    if np.isscalar(offset_bins):
        theta_bin_edges = np.linspace(0, fov.to('deg') / 2, endpoint=True, num=offset_bins + 1)
    else:
        theta_bin_edges = offset_bins

    theta_lo = theta_bin_edges[np.newaxis, :-1]
    theta_hi = theta_bin_edges[np.newaxis, 1:]

    migras = []
    for lower, upper in zip(theta_lo[0], theta_hi[0]):
        m = (lower <= event_fov_offsets) & (event_fov_offsets < upper)
        migra, bins_e_true, bins_mu = energy_migration(true_event_energies[m], predicted_event_energies[m], bins=bins_e_true, normalize=True, smoothing=smoothing)
        migras.append(migra.T)

    matrix = np.stack(migras)[np.newaxis, :]

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

    t.meta['DATE'] = datetime.datetime.now().replace(microsecond=0).isoformat()
    t.meta['TELESCOP'] = 'FACT'
    t.meta['HDUCLASS'] = 'GADF'
    t.meta['HDUCLAS1'] = 'RESPONSE'
    t.meta['HDUCLAS2'] = 'EDISP'
    t.meta['HDUCLAS3'] = 'FULL-ENCLOSURE'
    t.meta['HDUCLAS4'] = 'EDISP_2D'
    t.meta['EXTNAME'] = 'ENERGY DISPERSION'

    return t




@u.quantity_input(energy_true=u.TeV, energy_prediction=u.TeV)
def energy_dispersion(energy_true, energy_prediction, bins=10, normalize=False, smoothing=0):

    if np.isscalar(bins):
        bins = make_energy_bins(energy_true, energy_prediction, bins)

    hist, bins_e_true, bins_e_prediction = np.histogram2d(
        energy_true.value,
        energy_prediction,
        bins=bins,
    )

    if normalize:
        h = hist.T
        h = h / h.sum(axis=0)
        hist = np.nan_to_num(h).T


    if smoothing > 0:
        hist = gaussian_filter(hist, sigma=1.25, )

    return hist, bins_e_true * energy_true.unit, bins_e_prediction * energy_true.unit



@u.quantity_input(energy_true=u.TeV, energy_prediction=u.TeV)
def energy_migration(energy_true, energy_prediction, bins=10, normalize=True, smoothing=0):

    if np.isscalar(bins):
        bins = make_energy_bins(energy_true, energy_prediction, bins)

    migra = (energy_prediction / energy_true).si.value

    bins_migra = np.linspace(0, 3, endpoint=True, num=len(bins))

    hist, bins_e_true, bins_migra = np.histogram2d(
        energy_true.value,
        migra,
        bins=[bins, bins_migra],
    )

    if normalize:
        h = hist.T
        h = h / h.sum(axis=0)
        hist = np.nan_to_num(h).T

    if smoothing > 0:
        hist = gaussian_filter(hist, sigma=1.25, )

    return hist, bins * energy_true.unit, bins_migra
