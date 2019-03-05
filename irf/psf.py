import numpy as np

import astropy.units as u

from scipy.stats import binned_statistic
from scipy.ndimage.filters import gaussian_filter

def binned_psf_vs_energy(
    event_energy,
    angular_seperation,
    energy_bin_edges,
    rad_bins,
    smoothing=0,
    fov=10*u.deg,
):
    '''
    Calculate the binned psf for the given events.

    Parameters
    ----------
    angular_seperation: array-like 
        distance to true source position.
    energy_bin_edges:array-like
        bin edges for the histogram
    smoothing: float
        The amount of smoothing to apply to the resulting matrix

    '''
    if np.isscalar(rad_bins):
        rad_bins = np.linspace(0, fov.to_value(u.deg), rad_bins)

    energy_lo = energy_bin_edges[:-1].to_value(u.TeV)
    energy_hi = energy_bin_edges[1:].to_value(u.TeV)
    
    matrix = []
    for lower, upper in zip(energy_lo, energy_hi):
        m = (lower <= event_energy.to_value(u.TeV)) & (event_energy.to_value(u.TeV) < upper)
        if m.sum() > 0:
            psf = binned_psf(angular_seperation[m], rad_bins=rad_bins, smoothing=0)
        else:
            psf = np.zeros(len(rad_bins) - 1)
        matrix.append(psf)
    matrix = np.array(matrix)
    if smoothing > 0:
        a = matrix.copy()
        matrix = gaussian_filter(a, sigma=smoothing)

    return matrix



def binned_psf(
    angular_seperation,
    rad_bins=20,
    smoothing=0,
):
    '''
    Calculate the binned psf for the given events.

    Parameters
    ----------
    angular_seperation: array-like 
        distance to true source position.
    energy_bin_edges:array-like
        bin edges for the histogram
    smoothing: float
        The amount of smoothing to apply to the resulting matrix

    '''
    if np.isscalar(rad_bins):
        rad_bins = np.linspace(0, 2, rad_bins)

    r = ((rad_bins[:-1] + rad_bins[1:]) / 2).to_value(u.deg)
    deg2sr = (np.pi/180)**2
    norm = 2 * np.pi * r

    psf, _ = np.histogram(angular_seperation.to_value(u.deg), bins=rad_bins, density=True)
    psf = psf / norm / deg2sr

    if smoothing > 0:
        a = psf.copy()
        psf = gaussian_filter(a, sigma=smoothing)

    return psf



def psf_percentile(
    event_energy,
    angular_seperation,
    energy_bin_edges,
    smoothing=0,
    percentile=68,
):
    '''
    Calculate the psf the given events.

    Parameters
    ----------
    event_energy: array-like
        event energies.
    angular_seperation: array-like 
        distance to true source position.
    energy_bin_edges:array-like
        bin edges for the histogram
    smoothing: float
        The amount of smoothing to apply to the resulting matrix
    percentile: float
        the percentile to use.

    Returns
    -------

    b_68: Astropy Quantity
        The percentile of the angluar resolution function for each energy bin.
    '''
    x = event_energy.to_value(u.TeV)
    y = angular_seperation.to_value(u.deg)
    b_68, _, _ = binned_statistic(x, y, statistic=lambda y: np.percentile(y, percentile), bins=energy_bin_edges)

    if smoothing > 0:
        a = b_68.copy()
        b_68 = gaussian_filter(a.value, sigma=smoothing, )

    return b_68 * u.deg
