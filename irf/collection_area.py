import numpy as np
from astropy.stats import binom_conf_interval
import astropy.units as u
from astropy.table import Table
import datetime


@u.quantity_input(fov=u.deg)
def collection_area_to_irf_table(area, bin_center, bin_width, fov=4.5 * u.deg):
    '''
    See here what that format is supposed to look like:
    http://gamma-astro-data-formats.readthedocs.io/en/latest/irfs/full_enclosure/aeff/index.html
    '''

    energy_lo = 10**(bin_center - bin_width / 2)
    energy_hi = 10**(bin_center + bin_width / 2)

    energy_lo = energy_lo[np.newaxis, :] * u.GeV
    energy_hi = energy_hi[np.newaxis, :] * u.GeV

    # the irf format does not specify that it needs at least 2 entries here.
    # however the tools fail if theres just one bin.
    # but the tools are shit anyways
    theta_lo = np.array([0, fov.to('deg').value / 2], ndmin=2) * u.deg
    theta_hi = np.array([fov.to('deg').value / 2, fov.to('deg').value], ndmin=2) * u.deg

    area = np.vstack([area.value, area.value])
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


def histograms(
        all_events,
        selected_events,
        bins,
        range=None,
        log=True,
):
    '''
    Create histograms in the given bins for two vectors.

    Parameters
    ----------
    all_events: array-like
        Quantity which should be histogrammed for all simulated events
    selected_events: array-like
        Quantity which should be histogrammed for all selected events
    bins: int or array-like
        either number of bins or bin edges for the histogram
    range: (float, float)
        The lower and upper range of the bins
    log: bool
        flag indicating whether log10 should be applied to the values.

    returns: hist_all, hist_selected,  bin_edges
    '''

    if log is True:
        all_events = np.log10(all_events)
        selected_events = np.log10(selected_events)

    hist_all, bin_edges = np.histogram(
        all_events,
        bins=bins,
        range=range,
    )

    hist_selected, _ = np.histogram(
        selected_events,
        bins=bin_edges,
    )

    return hist_all, hist_selected, bin_edges


@u.quantity_input(impact=u.meter)
def collection_area(
        all_events,
        selected_events,
        impact,
        bins,
        range=None,
        log=True,
        sample_fraction=1.0,
):
    '''
    Calculate the collection area for the given events.

    Parameters
    ----------
    all_events: array-like
        Quantity which should be histogrammed for all simulated events
    selected_events: array-like
        Quantity which should be histogrammed for all selected events
    bins: int or array-like
        either number of bins or bin edges for the histogram
    impact: astropy Quantity of type length
        The maximal simulated impact parameter
    log: bool
        flag indicating whether log10 should be applied to the quantity.
    sample_fraction: float
        The fraction of `all_events` that was analysed
        to create `selected_events`
    '''

    hist_all, hist_selected, bin_edges = histograms(
        all_events,
        selected_events,
        bins,
        range=range,
        log=log
    )

    hist_selected = (hist_selected / sample_fraction).astype(int)

    bin_width = np.diff(bin_edges)
    bin_center = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    invalid = hist_selected > hist_all
    hist_selected[invalid] = hist_all[invalid]
    # use astropy to compute errors on that stuff
    lower_conf, upper_conf = binom_conf_interval(hist_selected, hist_all)

    # scale confidences to match and split
    lower_conf = lower_conf * np.pi * impact**2
    upper_conf = upper_conf * np.pi * impact**2

    area = hist_selected / hist_all * np.pi * impact**2

    return area, bin_center, bin_width, lower_conf, upper_conf
