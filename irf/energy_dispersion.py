import astropy.units as u
import numpy as np
import datetime
from astropy.table import Table


@u.quantity_input(energy_true=u.GeV, energy_prediction=u.GeV)
def energy_dispersion_to_irf_table(energy_true, energy_prediction, fov=4.5 * u.deg, n_bins=10):
    '''
    See here what that format is supposed to look like:
    http://gamma-astro-data-formats.readthedocs.io/en/latest/irfs/full_enclosure/aeff/index.html
    '''

    e_min = min(
        min(energy_true),
        min(energy_prediction)
    )
    e_max = max(
        max(energy_true),
        max(energy_prediction)
    )
    bins_e_true = np.logspace(np.log10(e_min.value), np.log10(e_max.value), endpoint=True, num=n_bins + 1)

    energy_lo = bins_e_true[np.newaxis, :-1] * energy_true.unit
    energy_hi = bins_e_true[np.newaxis, 1:] * energy_true.unit

    mu = (energy_prediction / energy_true).si.value
    bins_mu = np.linspace(mu.min(), mu.max(), endpoint=True, num=n_bins + 1)


    migra_lo = bins_mu[np.newaxis, :-1]
    migra_hi = bins_mu[np.newaxis, 1:]

    # the irf format does not specify that it needs at least 2 entries here.
    # however the tools fail if theres just one bin.
    # but the tools are shit anyways
    theta_lo = np.array([0, fov.to('deg').value / 2], ndmin=2) * u.deg
    theta_hi = np.array([fov.to('deg').value / 2, fov.to('deg').value], ndmin=2) * u.deg

    disp, bins_e_true, bins_e_prediction = np.histogram2d(
        energy_true.value,
        mu,
        bins=[bins_e_true, bins_mu],
    )
    e_disp = np.stack([disp, disp])[np.newaxis, :]

    t = Table(
        {
            'ENERG_LO': energy_lo,
            'ENERG_HI': energy_hi,
            'MIGRA_LO': migra_lo,
            'MIGRA_HI': migra_hi,
            'THETA_LO': theta_lo,
            'THETA_HI': theta_hi,
            'MATRIX': e_disp,
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



@u.quantity_input(energy_true=u.GeV, energy_prediction=u.GeV)
def energy_dispersion(energy_true, energy_prediction, n_bins=10):

    e_min = min(
        min(energy_true),
        min(energy_prediction)
    )

    e_max = max(
        max(energy_true),
        max(energy_prediction)
    )

    limits = e_min.value, e_max.value

    bin_edges = np.logspace(np.log10(e_min.value), np.log10(e_max.value), endpoint=True, num=n_bins + 1)

    hist, bins_e_true, bins_e_prediction = np.histogram2d(
        energy_true.value,
        energy_prediction,
        bins=bin_edges,
        range=[limits, limits],
    )

    return hist, bins_e_true * e_min.unit, bins_e_prediction * e_min.unit
