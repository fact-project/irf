import numpy as np

from astropy.stats import binom_conf_interval
import astropy.units as u
from astropy.table import Table

from scipy.ndimage.filters import gaussian_filter

from irf.oga import calculate_fov_offset
import datetime



@u.quantity_input(fov=u.deg, impact=u.m)
def collection_area_to_irf_table(
    mc_production_spectrum,
    event_fov_offsets,
    event_energies,
    bins=10,
    fov=4.5 * u.deg
):
    '''
    See here what that format is supposed to look like:
    http://gamma-astro-data-formats.readthedocs.io/en/latest/irfs/full_enclosure/aeff/index.html
    '''

    shower_energy = (corsika_showers.energy.values * u.GeV).to('TeV')
    true_event_energy = (selected_diffuse_gammas.corsika_event_header_total_energy.values * u.GeV).to('TeV')
    offset = calculate_fov_offset(selected_diffuse_gammas)

    if np.isscalar(bins):
        low = np.log10(shower_energy.min().value)
        high = np.log10(shower_energy.max().value)
        bin_edges = np.logspace(low, high, endpoint=True, num=bins + 1)
    else:
        low = bins.min()
        high = bins.max()
        bin_edges = bins

    energy_lo = bin_edges[np.newaxis, :-1]
    energy_hi = bin_edges[np.newaxis, 1:]

    theta_bin_edges = np.linspace(0, fov.to('deg').value / 2, endpoint=True, num=5)
    theta_lo = theta_bin_edges[np.newaxis, :-1]
    theta_hi = theta_bin_edges[np.newaxis, 1:]

    areas = []
    for lower, upper in zip(theta_lo[0], theta_hi[0]):
        m = (lower <= offset.value) & (offset.value < upper)
        f = (upper**2 - lower**2) / ((fov.value / 2) ** 2) * sample_fraction

        r = collection_area(
            shower_energy,
            true_event_energy[m],
            impact=impact,
            bins=bin_edges,
            sample_fraction=f,
            smooth=True,
        )

        area, bin_center, bin_width, lower_conf, upper_conf = r
        areas.append(area.value)

    area = np.vstack(areas)
    area = area[np.newaxis, :] * u.m**2

    t = Table(
        {
            'ENERG_LO': energy_lo * u.TeV,
            'ENERG_HI': energy_hi * u.TeV,
            'THETA_LO': theta_lo * u.deg,
            'THETA_HI': theta_hi * u.deg,
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
        mc_production_spectrum,
        event_energies,
        bins=10,
        smooth=False,
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
    '''

    if np.isscalar(bins):
        low = np.log10(mc_production_spectrum.e_min().value)
        high = np.log10(mc_production_spectrum.e_max().value)
        bins = np.logspace(low, high, endpoint=True, num=bins + 1)

    hist_all = mc_production_spectrum.expected_events_for_bins(energy_bins=bins)
    hist_selected, _ = np.histogram(event_energies, bins=bins)

    invalid = hist_selected > hist_all
    hist_selected[invalid] = hist_all[invalid]
    # use astropy to compute errors on that stuff
    lower_conf, upper_conf = binom_conf_interval(hist_selected, hist_all)

    # scale confidences to match and split
    lower_conf = lower_conf * mc_production_spectrum.generation_area
    upper_conf = upper_conf * mc_production_spectrum.generation_area

    area = (hist_selected / hist_all) * mc_production_spectrum.generation_area

    if smooth:
        a = area.copy()
        area = gaussian_filter(a.value, sigma=1.25, ) * area.unit

    return area, lower_conf, upper_conf
