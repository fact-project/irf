import astropy.units as u
import numpy as np
import datetime
from astropy.table import Table
from irf.oga import calculate_fov_offset


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
    bin_edges = np.logspace(low, high, endpoint=True, num=bins + 1)

    return bin_edges


@u.quantity_input(energy_true=u.TeV, energy_prediction=u.TeV, event_offset=u.deg)
def energy_dispersion_to_irf_table(selected_events, fov=4.5 * u.deg, n_bins=10, theta_bins=3):
    '''
    See here what that format is supposed to look like:
    http://gamma-astro-data-formats.readthedocs.io/en/latest/irfs/full_enclosure/aeff/index.html
    '''

    true_event_energy = (selected_events.corsika_event_header_total_energy.values * u.GeV).to('TeV')
    predicted_event_energy = (selected_events.gamma_energy_prediction.values * u.GeV).to('TeV')
    event_offset = calculate_fov_offset(selected_events)

    bins_e_true = make_energy_bins(true_event_energy, predicted_event_energy, n_bins)
    bins_mu = np.linspace(0, 3, endpoint=True, num=n_bins + 1)

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
        migra, bins_e_true, bins_mu = energy_migration(true_event_energy[m], predicted_event_energy[m], bins=n_bins)
        migras.append(migra)

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
def energy_dispersion(energy_true, energy_prediction, bins=10):

    if np.isscalar(bins):
        bins = make_energy_bins(energy_true, energy_prediction, bins)

    hist, bins_e_true, bins_e_prediction = np.histogram2d(
        energy_true.value,
        energy_prediction,
        bins=bins,
    )

    return hist, bins_e_true * energy_true.unit, bins_e_prediction * energy_true.unit


@u.quantity_input(energy_true=u.TeV, energy_prediction=u.TeV)
def energy_migration(energy_true, energy_prediction, bins=10):

    if np.isscalar(bins):
        bins = make_energy_bins(energy_true, energy_prediction, bins)

    mu = (energy_prediction / energy_true).si.value

    bins_mu = np.linspace(0, 3, endpoint=True, num=len(bins))

    migra, bins_e_true, bins_mu = np.histogram2d(
        energy_true.value,
        mu,
        bins=[bins, bins_mu],
    )

    return migra, bins * energy_true.unit, bins_mu
