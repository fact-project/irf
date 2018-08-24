import astropy.units as u
import numpy as np
from scipy.ndimage.filters import gaussian_filter



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



@u.quantity_input(energy_true=u.TeV, energy_prediction=u.TeV)
def energy_dispersion(energy_true, energy_prediction, bins=10, normalize=False, smoothing=0):
    '''
    Creates energy dispersion matrix i.e. a histogram of e_reco vs e_true.

    Parameters
    ----------

    energy_true : astropy.unit.Quantity (TeV)
        the true event energy
    energy_prediction: astropy.unit.Quantity (TeV)
        the predicted event energy
    bins_energy : int or arraylike
        the energy bins.
    normalize : bool
        Whether to normalize the matrix. The sum of each column will be equal to 1
    smoothing : float
        Amount of smoothing to apply to the generated matrices.
        Equivalent to the sigma parameter in
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html
    Returns
    -------

    array
        the migration matrix
    astropy.unit.Quantity
        the bin edges for the true energy axis
    astropy.unit.Quantity
        the bin edges for the predicted energy axis

    '''
    if np.isscalar(bins):
        bins = _make_energy_bins(energy_true, energy_prediction, bins)

    hist, bins_e_true, bins_e_prediction = np.histogram2d(
        energy_true.value,
        energy_prediction,
        bins=bins,
    )

    if smoothing > 0:
        hist = gaussian_filter(hist, sigma=smoothing)

    if normalize:
        with np.errstate(invalid='ignore'):
            h = hist.T
            h = h / h.sum(axis=0)
            hist = np.nan_to_num(h).T

    return hist, bins_e_true * energy_true.unit, bins_e_prediction * energy_true.unit


@u.quantity_input(energy_true=u.TeV, energy_prediction=u.TeV)
def energy_migration(energy_true, energy_prediction, bins_energy=10, bins_mu=10, normalize=True, smoothing=0):
    '''
    Creates energy migration matrix i.e. a histogram of e_reco/e_true vs e_trueself.

    Parameters
    ----------

    energy_true : astropy.unit.Quantity (TeV)
        the true event energy
    energy_prediction: astropy.unit.Quantity (TeV)
        the predicted event energy
    bins_energy : int or arraylike
        the energy bins.
    bins_mu : int or arraylike
        the bins to use for the y axis.
    normalize : bool
        Whether to normalize the matrix. The sum of each column will be equal to 1
    smoothing : float
        Amount of smoothing to apply to the generated matrices.
        Equivalent to the sigma parameter in
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html
    Returns
    -------

    array
        the migration matrix
    astropy.unit.Quantity
        the bin edges for the energy axis
    array
        the bin edges for the migration axis

    '''
    if np.isscalar(bins_energy):
        bins_energy = _make_energy_bins(energy_true, energy_prediction, bins_energy)

    migra = (energy_prediction / energy_true).si.value

    if np.isscalar(bins_mu):
        bins_mu = np.linspace(0, 6, endpoint=True, num=bins_mu + 1)

    hist, bins_e_true, bins_mu = np.histogram2d(
        energy_true.value,
        migra,
        bins=[bins_energy, bins_mu],
    )

    if smoothing > 0:
        hist = gaussian_filter(hist, sigma=smoothing, )

    if normalize:
        with np.errstate(invalid='ignore'):
            h = hist.T
            h = h / h.sum(axis=0)
            hist = np.nan_to_num(h).T

    return hist, bins_energy * energy_true.unit, bins_mu
