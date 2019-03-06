import numpy as np

from astropy.stats import binom_conf_interval
import astropy.units as u

from scipy.ndimage.filters import gaussian_filter

def collection_area_vs_offset(
    mc_production,
    event_energies,
    event_offsets,
    energy_bins,
    theta_bins,
    sample_fraction=1,
    smoothing=0,
):
    areas = []
    for lower, upper in zip(theta_bins[:-1], theta_bins[1:]):
        m = (lower <= event_offsets) & (event_offsets < upper)
        f = (upper**2 - lower**2) / (mc_production.generator_opening_angle**2) * sample_fraction
        r = collection_area(
            mc_production,
            event_energies[m],
            bin_edges=energy_bins,
            sample_fraction=f,
            smoothing=smoothing,
        )

        area, _, _, = r
        areas.append(area)

    # np.vstack does weird things to the units
    area = np.vstack([area.value for area in areas]) * areas[0].unit
    return area


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


@u.quantity_input(bin_edges=u.TeV)
def collection_area(
        mc_production,
        selected_event_energies,
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
    selected_event_energies: array-like
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

    hist_all = mc_production.expected_events_for_bins(bin_edges)
    hist_selected, _ = np.histogram(
        selected_event_energies,
        bins=bin_edges,
    )

    hist_selected = (hist_selected / sample_fraction).astype(int)

    invalid = hist_selected > hist_all
    hist_selected[invalid] = hist_all[invalid]
    # use astropy to compute errors on that stuff
    lower_conf, upper_conf = binom_conf_interval(hist_selected, hist_all)

    # scale confidences to match and split
    lower_conf = lower_conf * mc_production.generation_area
    upper_conf = upper_conf * mc_production.generation_area

    area = (hist_selected / hist_all) * mc_production.generation_area

    if smoothing > 0:
        a = area.copy()
        area = gaussian_filter(a.value, sigma=smoothing, ) * area.unit

    return area, lower_conf, upper_conf
