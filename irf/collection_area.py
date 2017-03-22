import numpy as np
from astropy.stats import binom_conf_interval
import astropy.units as u


def histograms_energy_zenith(
        showers,
        predictions,
        bins_energy,
        bins_zenith,
        log=True,
        ):
    '''
    calculate the matrices from the analysed and the simulated events.
    when dividing these matrices you get the some response which,
    when normalised correctly, corresponds to the collection area.

    returns hist_all, hist_selected,  energy_edges, zenith_edges
    '''

    showers_energy = showers['energy'].apply(np.log10)
    showers_zenith = showers['zenith'].apply(np.rad2deg)

    predictions_energy = predictions['energy'].apply(np.log10)
    predictions_zenith = predictions['zenith'].apply(np.rad2deg)

    hist_all, energy_edges, zenith_edges = np.histogram2d(
        showers_energy,
        showers_zenith,
        bins=(bins_energy, bins_zenith)
    )

    hist_selected, _, _ = np.histogram2d(
        predictions_energy,
        predictions_zenith,
        bins=(energy_edges, zenith_edges)
    )

    return hist_all, hist_selected, energy_edges, zenith_edges


def histograms_energy(
        showers,
        predictions,
        bins_energy,
        ):
    '''
    calculate the matrices from the analysed and the simulated events.
    when dividing these matrices you get the some response which,
    when normalised correctly, corresponds to the collection area.

    returns hist_all, hist_selected,  energy_edges
    '''

    showers_energy = showers['energy'].apply(np.log10)
    predictions_energy = predictions['energy'].apply(np.log10)

    hist_all, energy_edges = np.histogram(
        showers_energy,
        bins=bins_energy
    )

    hist_selected, _ = np.histogram(
        predictions_energy,
        bins=energy_edges
    )

    return hist_all, hist_selected, energy_edges


@u.quantity_input(impact=u.meter)
def collection_area_energy(
        all_events,
        selected_events,
        bins_energy,
        impact,
        ):
    '''
    Calculate the collection area for the given events.

    Parameters
    ----------
    all_events: pd.DataFrame
        DataFrame with all simulated events, must contain column "energy"
    selected_events: pd.DataFrame
        DataFrame with events that survived event selection,
        must contain column "energy"
    bins_energy: int or array-like
        either number of bins or bin edges for the histogram in energy
    impact: astropy Quantity of type length
        The maximal simulated impact parameter
    '''

    hist_all, hist_selected, energy_edges = histograms_energy(
        all_events,
        selected_events,
        bins_energy
    )

    bin_width = energy_edges[1] - energy_edges[0]
    bin_center = energy_edges[1:] - 0.5 * bin_width

    # use astropy to compute errors on that stuff
    conf = binom_conf_interval(hist_selected, hist_all)
    # scale confidences to match and split
    lower_conf, upper_conf = conf * np.pi * impact**2

    area = hist_selected / hist_all * np.pi * impact**2

    return area, bin_center, bin_width, lower_conf, upper_conf
