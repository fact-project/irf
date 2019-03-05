import numpy as np

import astropy.units as u

from scipy.stats import binned_statistic
from scipy.ndimage.filters import gaussian_filter


def background_vs_offset(event_energies, event_offset, weights, energy_bin_edges, theta_bin_edges, smoothing=0):
    migras = []
    for lower, upper in zip(theta_bin_edges[:-1], theta_bin_edges[1:]):
        m = (lower <= event_offset) & (event_offset < upper)
        bkg = background_vs_energy(event_energies[m], weights[m], energy_bins=energy_bin_edges, smoothing=0)
        migras.append(bkg)

    # transpose here. See https://github.com/gammapy/gammapy/issues/2067
    matrix = np.stack(migras)

    if smoothing > 0:
        a = matrix.copy()
        matrix = gaussian_filter(a, sigma=smoothing)
    
    return matrix


def background_vs_energy(
    event_energies,
    weights, 
    energy_bins,
    smoothing=0,
):
    '''
    Calculate the binned psf for the given events.

    Parameters
    ----------
    event_enrgies: Quantutiy 
        energy of events.
    weight: array-like
        corepsionding weights.
    energy_bin_edges:array-like
        bin edges for the histogram
    smoothing: float
        The amount of smoothing to apply to the resulting matrix

    '''

    bkg, _ = np.histogram(event_energies.to_value(u.TeV), bins=energy_bins, weights=weights)

    if smoothing > 0:
        a = bkg.copy()
        bkg = gaussian_filter(a, sigma=smoothing)

    return bkg

