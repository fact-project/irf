import numpy as np
from astropy.stats import binom_conf_interval
import astropy.units as u


def histograms_energy_zenith(
        all_events,
        selected_events,
        bins_energy,
        bins_zenith,
        log=True,
        ):
    '''
    calculate the matrices from the analysed and the simulated events.
    when dividing these matrices you get the some response which,
    when normalised correctly, corresponds to the collection area.

    Parameters
    ----------
    all_events: pd.DataFrame
        DataFrame with all simulated events.
        Must contain columns named 'zenith' and 'energy'
    selected_events: pd.DataFrame
        DataFrame with events that survived event selection.
        Must contain columns named 'zenith' and 'energy'
    bins_energy: int or array-like
        either number of bins or bin edges for the histogram in energy
    log: bool
        flag indicating whether log10 should be applied to the energy.

    returns: hist_all, hist_selected,  energy_edges, zenith_edges
    '''

    if log:
        all_events_energy = all_events['energy'].apply(np.log10)
        selected_events_energy = selected_events['energy'].apply(np.log10)

    all_events_zenith = all_events['zenith'].apply(np.rad2deg)
    selected_events_zenith = selected_events['zenith'].apply(np.rad2deg)

    hist_all, energy_edges, zenith_edges = np.histogram2d(
        all_events_energy,
        all_events_zenith,
        bins=(bins_energy, bins_zenith)
    )

    hist_selected, _, _ = np.histogram2d(
        selected_events_energy,
        selected_events_zenith,
        bins=(energy_edges, zenith_edges)
    )

    return hist_all, hist_selected, energy_edges, zenith_edges


def histograms_energy(
        all_events,
        selected_events,
        bins_energy,
        target='corsika_evt_header_total_energy',
        log=True,
        ):
    '''
    calculate the matrices from the analysed and the simulated events.
    when dividing these matrices you get the some response which,
    when normalised correctly, corresponds to the collection area.

    Parameters
    ----------
    all_events: pd.DataFrame
        DataFrame with all simulated events.
        Must contain column named 'energy'
    selected_events: pd.DataFrame
        DataFrame with events that survived event selection.
        Must contain column named 'energy'
    bins_energy: int or array-like
        either number of bins or bin edges for the histogram in energy
    log: bool
        flag indicating whether log10 should be applied to the energy.

    returns: hist_all, hist_selected,  energy_edges
    '''

    if log is True:
        all_events_energy = all_events['energy'].apply(np.log10)
        selected_events_energy = selected_events['energy'].apply(np.log10)
    else:
        all_events_energy = all_events['energy']
        selected_events_energy = selected_events['energy']

    hist_all, energy_edges = np.histogram(
        all_events_energy,
        bins=bins_energy
    )

    hist_selected, _ = np.histogram(
        selected_events_energy,
        bins=energy_edges
    )

    return hist_all, hist_selected, energy_edges


@u.quantity_input(impact=u.meter)
def collection_area_energy(
        all_events,
        selected_events,
        bins_energy,
        impact,
        target='corsika_evt_header_total_energy',
        log=True,
        sample_fraction=None,
        ):
    '''
    Calculate the collection area for the given events.

    Parameters
    ----------
    all_events: pd.DataFrame
        DataFrame with all simulated events.
    selected_events: pd.DataFrame
        DataFrame with events that survived event selection.
    bins_energy: int or array-like
        either number of bins or bin edges for the histogram in energy
    impact: astropy Quantity of type length
        The maximal simulated impact parameter
    target: string
        The key name of the energy variable. Default 'corsika_evt_header_total_energy'
        for getting the collection area vs the true energy.
    log: bool
        flag indicating whether log10 should be applied to the energy.
    sample_fraction: float or None
        If not None, the fraction of `all_events` that was analysed
        to create `selected_events`
    '''

    selected_events['energy'] = selected_events[target].copy()
    hist_all, hist_selected, energy_edges = histograms_energy(
        all_events,
        selected_events,
        bins_energy,
        log=log
    )

    if sample_fraction is not None:
        hist_selected = (hist_selected / sample_fraction).astype(int)

    bin_width = np.diff(energy_edges)
    bin_center = 0.5 * (energy_edges[:-1] + energy_edges[1:])

    valid = hist_selected <= hist_all
    # use astropy to compute errors on that stuff
    lower_conf = np.full(len(bin_center), np.nan)
    upper_conf = np.full(len(bin_center), np.nan)
    lower_conf[valid], upper_conf[valid] = binom_conf_interval(
        hist_selected[valid], hist_all[valid]
    )

    # scale confidences to match and split
    lower_conf = lower_conf * np.pi * impact**2
    upper_conf = upper_conf * np.pi * impact**2

    area = hist_selected / hist_all * np.pi * impact**2

    return area, bin_center, bin_width, lower_conf, upper_conf
