import astropy.units as u
import numpy as np
from scipy.ndimage.filters import gaussian_filter


@u.quantity_input(energy_true=u.TeV, energy_prediction=u.TeV)
def _make_energy_bins(energy_true, energy_prediction, bins, e_ref=1 * u.TeV):
    e_min = min(
        np.min(energy_true),
        np.min(energy_prediction)
    )

    e_max = max(
        np.max(energy_true),
        np.max(energy_prediction)
    )

    low = np.log10(e_min / e_ref)
    high = np.log10(e_max / e_ref)
    bin_edges = np.logspace(low, high, endpoint=True, num=bins + 1)

    return bin_edges * e_ref


@u.quantity_input(energy_true=u.TeV, energy_prediction=u.TeV)
def energy_dispersion(energy_true, energy_prediction, bins=10, normalize=False, smoothing=0, e_ref=1 * u.TeV):
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
    e_ref: astropy.unit.Quantity[energy]
        Reference energy
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
        bins = _make_energy_bins(energy_true, energy_prediction, bins, e_ref=e_ref)

    hist, bins_e_true, bins_e_prediction = np.histogram2d(
        (energy_true / e_ref).to_value(u.dimensionless_unscaled),
        (energy_prediction / e_ref).to_value(u.dimensionless_unscaled),
        bins=(bins / e_ref).to_value(u.dimensionless_unscaled),
    )

    if smoothing > 0:
        hist = gaussian_filter(hist, sigma=smoothing)

    if normalize:
        hist = _normalize_hist(hist)

    return hist, bins_e_true, bins_e_prediction


@u.quantity_input(energy_true=u.TeV, energy_prediction=u.TeV)
def energy_migration(energy_true, energy_prediction, bins_energy=10, bins_mu=10, normalize=True, smoothing=0, e_ref=1 * u.TeV):
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
    e_ref: astropy.unit.Quantity[energy]
        Reference energy
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
        bins_energy = _make_energy_bins(
            energy_true, energy_prediction, bins_energy, e_ref=e_ref
        )

    migra = (energy_prediction / energy_true).to_value(u.dimensionless_unscaled)

    if np.isscalar(bins_mu):
        bins_mu = np.linspace(0, 6, bins_mu + 1)

    hist, bins_e_true, bins_mu = np.histogram2d(
        (energy_true / e_ref).to_value(u.dimensionless_unscaled),
        migra,
        bins=[(bins_energy / e_ref).to_value(u.dimensionless_unscaled), bins_mu],
    )

    if smoothing > 0:
        hist = gaussian_filter(hist, sigma=smoothing)

    if normalize:
        hist = _normalize_hist(hist)

    return hist, bins_energy, bins_mu


def _normalize_hist(hist):
    with np.errstate(invalid='ignore'):
        h = hist.T
        h = h / h.sum(axis=0)
        return np.nan_to_num(h).T
