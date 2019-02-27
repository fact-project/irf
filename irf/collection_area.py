import numpy as np

from astropy.stats import binom_conf_interval
import astropy.units as u

from scipy.ndimage.filters import gaussian_filter

import datetime



@u.quantity_input(fov=u.deg, event_fov_offsets=u.deg)
def collection_area_to_irf_table(
    mc_production_spectrum,
    event_fov_offsets,
    event_energies,
    bins=10,
    fov=4.5 * u.deg,
    offset_bins=5,
    sample_fraction=1,
    smoothing=0,
):
    '''
    See here what that format is supposed to look like:
    http://gamma-astro-data-formats.readthedocs.io/en/latest/irfs/full_enclosure/aeff/index.html
    '''

    event_energies = event_energies.to('TeV')
    event_fov_offsets = event_fov_offsets.to('deg')

    if np.isscalar(bins):
        low = np.log10(mc_production_spectrum.e_min.value)
        high = np.log10(mc_production_spectrum.e_max.value)
        bin_edges = np.logspace(low, high, endpoint=True, num=bins + 1) * u.TeV
    else:
        low = bins.min()
        high = bins.max()
        bin_edges = bins

    energy_lo = bin_edges[np.newaxis, :-1]
    energy_hi = bin_edges[np.newaxis, 1:]

    theta_bin_edges = np.linspace(0, fov.to('deg') / 2, endpoint=True, num=offset_bins + 1)
    theta_lo = theta_bin_edges[np.newaxis, :-1]
    theta_hi = theta_bin_edges[np.newaxis, 1:]

    areas = []
    for lower, upper in zip(theta_lo[0], theta_hi[0]):
        m = (lower <= event_fov_offsets) & (event_fov_offsets < upper)
        f = (upper**2 - lower**2) / ((fov / 2) ** 2) * sample_fraction

        r = collection_area(
            mc_production_spectrum,
            event_energies[m],
            bins=bin_edges,
            sample_fraction=f,
            smoothing=smoothing,
        )

        area, lower_conf, upper_conf = r
        areas.append(area.value)

    area = np.vstack(areas)
    area = area[np.newaxis, :] * u.m**2

    t = Table(
        {
            'ENERG_LO': energy_lo,
            'ENERG_HI': energy_hi,
            'THETA_LO': theta_lo,
            'THETA_HI': theta_hi,
            'EFFAREA': area,
        }
    )

    t.meta['DATE'] = datetime.datetime.now().replace(microsecond=0).isoformat()
    t.meta['TELESCOP'] = 'FACT    '
    t.meta['HDUCLASS'] = 'OGIP    '
    t.meta['HDUCLAS1'] = 'RESPONSE'
    t.meta['HDUCLAS2'] = 'EFF_AREA'
    t.meta['HDUCLAS3'] = 'FULL-ENCLOSURE'
    t.meta['HDUCLAS4'] = '2D      '
    t.meta['EXTNAME'] = 'EFFECTIVE AREA'
    return t


@u.quantity_input(impact=u.meter)
def collection_area(
        mc_production_spectrum,
        event_energies,
        bins,
        smoothing=0,
        sample_fraction=1.0,
):
    '''
    Calculate the collection area for the given events.

    Parameters
    ----------
    mc_production_spectrum: MCSpectrum instance
        The production spectrum used for producing the monte carlos.
    event_energies: array-like
        Quantity which should be histogrammed for all selected events
    bins: int or array-like
        either number of bins or bin edges for the histogram
    smoothing: float
        if larger than 0, a gaussian filter will be applied to the collection area.
        the amount of smoothing to apply given here is sigma parameter
        in scipy.ndimage.filters.gaussian_filter
    '''

    event_energies = event_energies.to('TeV')
    if np.isscalar(bins):
        low = np.log10(mc_production_spectrum.e_min.value)
        high = np.log10(mc_production_spectrum.e_max.value)
        bins = np.logspace(low, high, endpoint=True, num=bins + 1) * u.TeV

    hist_all = mc_production_spectrum.expected_events_for_bins(energy_bins=bins) * sample_fraction
    hist_selected, _ = np.histogram(event_energies, bins=bins)

    invalid = hist_selected > hist_all
    hist_selected[invalid] = hist_all[invalid]
    # use astropy to compute errors on that stuff
    lower_conf, upper_conf = binom_conf_interval(hist_selected, hist_all)

    # scale confidences to match and split
    lower_conf = lower_conf * mc_production_spectrum.generation_area
    upper_conf = upper_conf * mc_production_spectrum.generation_area

    area = (hist_selected / hist_all) * mc_production_spectrum.generation_area

    if smoothing > 0:
        a = area.copy()
        area = gaussian_filter(a.value, sigma=smoothing, ) * area.unit

    return area, lower_conf, upper_conf
