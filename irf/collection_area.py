import numpy as np

from astropy.stats import binom_conf_interval
import astropy.units as u

from scipy.ndimage.filters import gaussian_filter


def histograms(
        all_events,
        selected_events,
        bins,
        range=None,
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
    returns: hist_all, hist_selected,  bin_edges
    '''

    hist_all, bin_edges = np.histogram(
        all_events,
        bins=bins,
    )

    hist_selected, _ = np.histogram(
        selected_events,
        bins=bin_edges,
    )

    return hist_all, hist_selected, bin_edges


@u.quantity_input(impact=u.meter)
def collection_area(
        mc_production,
        selected_events,
        bin_edges,
        sample_fraction=1.0,
        smoothing=0,
):
    '''
    Calculate the collection area for the given events.

    Parameters
    ----------
    mc_production: MCSpectrum instance
        MCSpectrum instance describing the MC production
    selected_events: array-like
        Quantity which should be histogrammed for all selected events
    bin_edges:array-like
        bin edges for the histogram
    sample_fraction: float
        The fraction of `all_events` that was analysed
        to create `selected_events`
    sample_fraction: float
        The fraction of `all_events` that was analysed
        to create `selected_events`
    smoothing: float
        The amount of smoothing to apply to the resulting matrix
    '''

    scatter_radius = np.sqrt(mc_production.generation_area / np.pi) 


    hist_all = mc_production.expected_events_for_bins(bin_edges*u.TeV)
    hist_selected, _ = np.histogram(
        selected_events,
        bins=bin_edges,
    )

    hist_selected = (hist_selected / sample_fraction).astype(int)

    bin_width = np.diff(bin_edges)
    bin_center = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    invalid = hist_selected > hist_all
    hist_selected[invalid] = hist_all[invalid]
    # use astropy to compute errors on that stuff
    lower_conf, upper_conf = binom_conf_interval(hist_selected, hist_all)

    # scale confidences to match and split
    lower_conf = lower_conf * np.pi * scatter_radius**2
    upper_conf = upper_conf * np.pi * scatter_radius**2

    area = (hist_selected / hist_all) * np.pi * scatter_radius**2

    if smoothing > 0:
        a = area.copy()
        area = gaussian_filter(a.value, sigma=smoothing, ) * area.unit

    return area, bin_center, bin_width, lower_conf, upper_conf
